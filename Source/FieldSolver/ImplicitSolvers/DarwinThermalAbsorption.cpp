/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "DarwinThermalAdvance.H"
#include "DarwinVacuumJointSolve.H"
#include "ThetaImplicitHybrid.H"
#include "MassMatrixDensityProjection.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "WarpX.H"
#include "Utils/WarpXConst.H"
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>

namespace warpx::thermal {
using Real=amrex::Real;
using warpx::fields::FieldType;
using ablastr::fields::Direction;

bool DarwinThermalAdvance::PrepareAbsorbingBoundary ()
{
    // Preserve the established no-boundary-action path and summation order.
    // Reflection changes private endpoint ownership/momentum, but the native
    // folded trajectory density already equals its reflected endpoint deposit.
    if (!m_absorbed_particles && !m_reflected_particles) { return true; }
    int private_axial=m_joint_vacuum&&m_joint_vacuum->AllowsAxialReflection()&&
        !m_absorbed_particles&&!m_ion_exchange?1:0;
    int minimum=private_axial,maximum=private_axial;
    amrex::ParallelDescriptor::ReduceIntMin(minimum);amrex::ParallelDescriptor::ReduceIntMax(maximum);
    if(minimum!=maximum)return false;
    if(private_axial)return m_joint_vacuum->PrepareAxialReflection(m_reflected_particles);
    auto& particles=m_simulation.GetPartContainer();
    // Native loss charge currently uses each species' fixed charge. Field
    // ionization needs its own loss ledger and is outside this bounded split.
    for (auto const& pc:particles) {
        if (pc->DoFieldIonization() || pc->HasiAttrib("ionizationLevel")) {
            amrex::Print()<<"Eulerian absorption requires fixed species charge\n";
            return false;
        }
    }
    if (!m_survivor_charge.isDefined()) {
        auto const& rho=m_endpoint_charge;
        m_survivor_charge.define(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        m_absorbed_charge.define(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        m_survivor_energy.define(m_endpoint.boxArray(),m_endpoint.DistributionMap(),1,0);
    }
    using Subset=MassMatrixDensityProjection::EndpointSubset;
    if (!MassMatrixDensityProjection::DepositEndpointSubset(m_simulation,0,m_survivor_charge,Subset::Survivors) ||
        !MassMatrixDensityProjection::DepositEndpointSubset(m_simulation,0,m_absorbed_charge,Subset::Absorbed)) {
        return false;
    }
    // The virtual trajectory deposit is the source of the implicit midpoint.
    // A boundary loss is an explicit sink, never an invented current or a
    // clipped/renormalized replacement of this charge-conserving pair.
    amrex::MultiFab difference(m_endpoint_charge.boxArray(),m_endpoint_charge.DistributionMap(),1,0);
    amrex::MultiFab::LinComb(difference,1.,m_survivor_charge,0,1.,m_absorbed_charge,0,0,1,0);
    amrex::MultiFab::Subtract(difference,m_endpoint_charge,0,0,1,0);
    Real const reference=std::max({m_endpoint_charge.norminf(),m_survivor_charge.norminf(),
        m_absorbed_charge.norminf(),PhysConst::q_e*m_model.m_n_floor});
    Real const relative=difference.norminf()/reference;
    if (!std::isfinite(relative) || relative>1.e-11) {
        amrex::Print()<<"Eulerian absorption virtual/survivor/loss charge mismatch="<<relative<<"\n";
        return false;
    }
    KineticThermalStateView virtual_state{m_endpoint_charge};
    virtual_state.pedestal=m_model.DensityPedestal(0);
    virtual_state.energy=&m_endpoint;
    if (!m_endpoint_moments->Evaluate(virtual_state)) { return false; }
    KineticThermalMoments survivor(m_geometry,m_endpoint.boxArray(),m_endpoint.DistributionMap(),
        m_model.EulerianMomentOptions());
    KineticThermalStateView survivor_state{m_survivor_charge};
    survivor_state.pedestal=virtual_state.pedestal;
    if (!survivor.Evaluate(survivor_state)) { return false; }
    amrex::MultiFab lost_number(m_endpoint.boxArray(),m_endpoint.DistributionMap(),1,0);
    m_endpoint_moments->RestrictNativeMoment(m_absorbed_charge,0,lost_number);
    lost_number.mult(1./PhysConst::q_e,0,1,0);
    m_absorbing_inventory=RemapAbsorbingElectronInventory(m_geometry,m_endpoint,
        m_endpoint_moments->NumberDensity(),survivor.NumberDensity(),lost_number,m_survivor_energy);
    if (!m_absorbing_inventory.valid ||
        m_absorbing_inventory.max_temperature_relative_error>128*std::numeric_limits<Real>::epsilon() ||
        m_absorbing_inventory.max_survivor_density_relative_increase>1.e-11) { return false; }
    survivor_state.energy=&m_survivor_energy;
    if (!survivor.Evaluate(survivor_state)) { return false; }
    // Independent charge inventory in the deposition's native dual measure,
    // converted conservatively to physical cells by RestrictNativeMoment.
    auto const dx=m_geometry.CellSizeArray(), lo=m_geometry.ProbLoArray();
    auto const index_lo=m_geometry.Domain().smallEnd();
    bool const rz=m_geometry.IsRZ();
    Real const volume=AMREX_D_TERM(dx[0],*dx[1],*dx[2]);
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<Real> data(op);
    for (amrex::MFIter mfi(lost_number);mfi.isValid();++mfi) {
        auto const loss=lost_number.const_array(mfi);
        op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            Real const v=rz ? volume*2*MathConst::pi*(lo[0]+(i-index_lo[0]+.5)*dx[0]) : volume;
            return amrex::GpuTuple<Real>{PhysConst::q_e*loss(i,j,k)*v};
        });
    }
    m_absorbed_charge_integral=amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(m_absorbed_charge_integral);
    if (!std::isfinite(m_absorbed_charge_integral)) { return false; }
    for (auto const& pc:particles) {
        for (int d=0;d<AMREX_SPACEDIM;++d) for (int side=0;side<2;++side) {
            m_boundary_tallies_before.push_back({pc->GetBoundaryAbsorbedWeight(d,side),
                pc->GetBoundaryAbsorbedCharge(d,side),pc->GetBoundaryAbsorbedEnergy(d,side)});
        }
    }
    return true;
}

bool DarwinThermalAdvance::HasOwnedAxialReflection () const noexcept
{
    return m_open&&!m_absorbed_particles&&!m_ion_exchange&&m_reflected_particles>0&&
        m_joint_vacuum&&m_joint_vacuum->AxialReflectionFinished();
}

bool DarwinThermalAdvance::CommitParticleBoundaries ()
{
    AMREX_ALWAYS_ASSERT(m_open && m_receipt_ready && !m_particle_boundary_committed);
    auto& particles=m_simulation.GetPartContainer();
    if (!m_absorbed_particles && !m_reflected_particles) {
        particles.Redistribute();
        return true;
    }
    int private_axial=HasOwnedAxialReflection()?1:0,minimum=private_axial,maximum=private_axial;
    amrex::ParallelDescriptor::ReduceIntMin(minimum);amrex::ParallelDescriptor::ReduceIntMax(maximum);
    if(minimum!=maximum)return false;
    if(private_axial) {
        bool ready=!m_axial_boundary_completed;
        amrex::ParallelDescriptor::ReduceBoolAnd(ready);
        if(!ready||!m_joint_vacuum->CheckReceipt())return false;
        // Every predicted endpoint was checked against its actual home tile
        // before finishing. Preserve the same native Redistribute operation
        // and reject any unexpected storage/identity change afterwards.
        particles.Redistribute();
        if(!m_joint_vacuum->CheckReceipt())return false;
        amrex::Long population=0;
        for(auto const& pc:particles)population+=pc->TotalNumberOfParticles(true,true);
        amrex::ParallelDescriptor::ReduceLongSum(population);
        if(population!=m_endpoint_population)return false;
        amrex::MultiFab actual(m_endpoint_charge.boxArray(),m_endpoint_charge.DistributionMap(),
            1,m_endpoint_charge.nGrowVect());
        ablastr::fields::MultiLevelScalarField view{&actual};
        particles.DepositCharge(view,0.);m_simulation.SyncRho(view,{},{});
        m_simulation.ApplyRhofieldBoundary(0,&actual,PatchType::fine);
        actual.FillBoundary(m_geometry.periodicity());
        Real const reference=std::max({actual.norminf(),m_endpoint_charge.norminf(),PhysConst::q_e*m_model.m_n_floor});
        amrex::MultiFab::Subtract(actual,m_endpoint_charge,0,0,1,0);
        Real const density_error=actual.norminf()/reference;
        if(!std::isfinite(density_error)||density_error>1.e-11||!m_joint_vacuum->CheckReceipt())return false;
        m_axial_boundary_completed=true;
        amrex::Print()<<"Eulerian owned axial reflection: particles="<<m_reflected_particles
            <<" density_relative_error="<<density_error<<" boundary_heat_J=0 reversible=1\n";
        return true;
    }
    // This is the irreversibility barrier: native per-face tallies and scrape
    // buffers are mutated. Any following mismatch is terminal, before OU RNG.
    m_particle_boundary_committed=true;
    particles.ApplyBoundaryConditions();
    m_simulation.GetParticleBoundaryBuffer().gatherParticlesFromDomainBoundaries(particles,m_time+m_dt);
    particles.Redistribute();
    amrex::Long survivors=0;
    for (auto const& pc:particles) { survivors+=pc->TotalNumberOfParticles(true,true); }
    amrex::ParallelDescriptor::ReduceLongSum(survivors);
    if (survivors!=m_endpoint_population-m_absorbed_particles) {
        amrex::Print()<<"Eulerian native absorption count mismatch: expected="
            <<m_endpoint_population-m_absorbed_particles<<" actual="<<survivors<<"\n";
        return false;
    }
    // Independently use the normal accepted particle deposit, including its
    // native RZ volume scaling. Compare before a stochastic ion kick.
    amrex::MultiFab actual(m_survivor_charge.boxArray(),m_survivor_charge.DistributionMap(),
        1,m_survivor_charge.nGrowVect());
    ablastr::fields::MultiLevelScalarField view{&actual};
    particles.DepositCharge(view,0.);
    m_simulation.SyncRho(view,{},{});
    m_simulation.ApplyRhofieldBoundary(0,&actual,PatchType::fine);
    actual.FillBoundary(m_geometry.periodicity());
    Real const reference=std::max({actual.norminf(),m_survivor_charge.norminf(),PhysConst::q_e*m_model.m_n_floor});
    amrex::MultiFab::Subtract(actual,m_survivor_charge,0,0,1,0);
    Real const density_error=actual.norminf()/reference;
    if (!std::isfinite(density_error) || density_error>1.e-11) {
        amrex::Print()<<"Eulerian native absorption survivor density mismatch="<<density_error<<"\n";
        return false;
    }
    std::size_t index=0;
    Real deposited_charge_loss=0;
    for (auto const& pc:particles) {
        for (int d=0;d<AMREX_SPACEDIM;++d) for (int side=0;side<2;++side) {
            auto const& before=m_boundary_tallies_before[index++];
            Real delta[3]={pc->GetBoundaryAbsorbedWeight(d,side)-before[0],
                pc->GetBoundaryAbsorbedCharge(d,side)-before[1],
                pc->GetBoundaryAbsorbedEnergy(d,side)-before[2]};
            amrex::ParallelDescriptor::ReduceRealSum(delta,3);
            for (Real x:delta) { if (!std::isfinite(x)) { return false; } }
            if (!pc->do_not_deposit) { deposited_charge_loss+=delta[1]; }
            if (delta[0]!=0 || delta[1]!=0 || delta[2]!=0) {
                amrex::Print()<<"Eulerian accepted particle absorption: species="<<pc->getName()
                    <<" dimension="<<d<<" side="<<side<<" weight="<<delta[0]
                    <<" charge_C="<<delta[1]<<" kinetic_J="<<delta[2]<<"\n";
            }
        }
    }
    Real const charge_error=std::abs(deposited_charge_loss-m_absorbed_charge_integral)/
        std::max({std::abs(deposited_charge_loss),std::abs(m_absorbed_charge_integral),Real(1.e-30)});
    amrex::Print()<<"Eulerian accepted absorption audit: lost_particles="<<m_absorbed_particles
        <<" reflected_particles="<<m_reflected_particles
        <<" survivor_density_relative_error="<<density_error
        <<" native_loss_charge_C="<<deposited_charge_loss
        <<" deposited_loss_charge_C="<<m_absorbed_charge_integral
        <<" charge_relative_error="<<charge_error<<"\n";
    return std::isfinite(charge_error) && charge_error<=1.e-10;
}

void DarwinThermalAdvance::RefreshAbsorbingLongitudinalField ()
{
    AMREX_ALWAYS_ASSERT(m_absorbed_particles && m_particle_boundary_committed && !m_open);
    // The trajectory's displacement/Je history has already consumed its
    // interval states. Add only the change caused by absorption. Replacing the
    // finished field with a fresh projection would also change its time
    // discretization and reintroduce the last stage's inertial source.
    auto total=m_simulation.m_fields.get_alldirs(FieldType::Efield_fp,0);
    auto longitudinal=m_simulation.m_fields.get_alldirs("hybrid_E_long_fp",0);
    std::array<amrex::MultiFab,3> finished, virtual_projection;
    for (int c=0;c<3;++c) {
        auto const& e=*longitudinal[c];
        finished[c].define(e.boxArray(),e.DistributionMap(),e.nComp(),e.nGrowVect());
        virtual_projection[c].define(e.boxArray(),e.DistributionMap(),e.nComp(),e.nGrowVect());
        amrex::MultiFab::Copy(finished[c],e,0,0,e.nComp(),e.nGrowVect());
    }
    auto& phi=*m_simulation.m_fields.get("hybrid_phi_darwin_fp",0);
    amrex::MultiFab finished_phi(phi.boxArray(),phi.DistributionMap(),phi.nComp(),phi.nGrowVect());
    amrex::MultiFab virtual_phi(phi.boxArray(),phi.DistributionMap(),phi.nComp(),phi.nGrowVect());
    amrex::MultiFab::Copy(finished_phi,phi,0,0,phi.nComp(),phi.nGrowVect());
    m_model.SetElectronThermalTrial(0,m_endpoint_moments->NodalTemperature(),m_endpoint_moments->OhmPressure());
    m_model.ComputeDarwinELong({&m_endpoint_charge},m_time+m_dt,true);
    for (int c=0;c<3;++c) {
        amrex::MultiFab::Copy(virtual_projection[c],*longitudinal[c],0,0,
            longitudinal[c]->nComp(),longitudinal[c]->nGrowVect());
    }
    amrex::MultiFab::Copy(virtual_phi,phi,0,0,phi.nComp(),phi.nGrowVect());
    m_model.ClearElectronThermalTrials();
    m_model.ComputeDarwinELong({&m_survivor_charge},m_time+m_dt,true);
    amrex::MultiFab::Subtract(phi,virtual_phi,0,0,phi.nComp(),phi.nGrowVect());
    amrex::MultiFab::Add(phi,finished_phi,0,0,phi.nComp(),phi.nGrowVect());
    auto const dx=m_geometry.CellSizeArray(), lo=m_geometry.ProbLoArray();
    auto const index_lo=m_geometry.Domain().smallEnd();
    bool const rz=m_geometry.IsRZ();
    Real const volume=AMREX_D_TERM(dx[0],*dx[1],*dx[2]);
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<Real> data(op);
    for (int c=0;c<3;++c) {
        auto& delta=*longitudinal[c];
        amrex::MultiFab::Subtract(delta,virtual_projection[c],0,0,delta.nComp(),delta.nGrowVect());
        auto const nodal=delta.ixType().toIntVect();
        for (amrex::MFIter mfi(delta);mfi.isValid();++mfi) {
            auto const increment=delta.const_array(mfi), e=total[c]->const_array(mfi);
            auto const box=mfi.validbox();
            auto const blo=box.smallEnd(), bhi=box.bigEnd();
            op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                // Same geometric dual-cell measure as native FieldEnergy;
                // physical half volumes at nodal walls and shared box faces.
                amrex::IntVect const iv(AMREX_D_DECL(i,j,k));
                Real metric=1.;
                if (rz) {
                    Real const radius=lo[0]+(i-index_lo[0]+.5*(1-nodal[0]))*dx[0];
                    metric=radius==0. ? MathConst::pi*dx[0]/4. : 2*MathConst::pi*radius;
                    if (radius!=0. && nodal[0] && i==blo[0]) { metric*=.5; }
                    if (nodal[0] && i==bhi[0]) { metric*=.5; }
                }
                for (int d=int(rz);d<AMREX_SPACEDIM;++d) {
                    if (nodal[d] && iv[d]==blo[d]) { metric*=.5; }
                    if (nodal[d] && iv[d]==bhi[d]) { metric*=.5; }
                }
                Real const de=increment(i,j,k);
                return amrex::GpuTuple<Real>{PhysConst::epsilon_0*volume*metric*de*(e(i,j,k)+.5*de)};
            });
        }
        amrex::MultiFab::Add(*total[c],delta,0,0,delta.nComp(),delta.nGrowVect());
        amrex::MultiFab::Add(delta,finished[c],0,0,delta.nComp(),delta.nGrowVect());
    }
    Real field_jump=amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(field_jump);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(field_jump),"Nonfinite absorption field-energy jump");
    amrex::Print()<<"Eulerian accepted absorption field split [J]: electric_jump="<<field_jump
        <<"; interval displacement and Je history exclude this boundary increment\n";
    m_simulation.FillBoundaryE(total[0]->nGrowVect(),true);
}
} // namespace warpx::thermal
