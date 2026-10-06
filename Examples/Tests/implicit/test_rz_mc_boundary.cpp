/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include "Python/callbacks.H"
#include "Particles/WarpXParticleContainer.H"
#include <AMReX_ParticleReduce.H>

#include <ablastr/coarsen/sample.H>
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>

namespace
{
    using namespace amrex::literals;
    using Field = ablastr::fields::VectorField;
    using warpx::fields::FieldType;
    using amrex::MultiFab;

    struct Scratch
    {
        std::array<MultiFab,3> values;
        Field field;
        explicit Scratch (Field const& like)
        {
            for (int c = 0; c < 3; ++c) {
                values[c].define(like[c]->boxArray(),like[c]->DistributionMap(),1,
                                 like[c]->nGrowVect());
                field[c] = &values[c];
            }
        }
        void copy (Field const& source)
        {
            for (int c = 0; c < 3; ++c) {
                MultiFab::Copy(values[c],*source[c],0,0,1,values[c].nGrowVect());
            }
        }
    };

    amrex::Real product (Field const& a, Field const& b, amrex::Geometry const& geom)
    {
        amrex::Real result = 0.0_rt;
        for (int c = 0; c < 3; ++c) {
            MultiFab measured(a[c]->boxArray(),a[c]->DistributionMap(),1,0);
            auto const volume = MakeQdsmcVolumeElement(geom,a[c]->ixType());
            for (amrex::MFIter mfi(measured); mfi.isValid(); ++mfi) {
                auto const aa = a[c]->const_array(mfi), bb = b[c]->const_array(mfi);
                auto const out = measured.array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    out(i,j,k) = volume(i,j,k)*aa(i,j,k)*bb(i,j,k);
                });
            }
            result += measured.sum_unique(0,false,geom.periodicity());
        }
        return result;
    }

    amrex::Real electron_energy (WarpX& w, HybridPICModel& hp, MultiFab const& rho,
                                 Field const& Ji, Field const& B)
    {
        hp.CalculatePlasmaCurrent(B,w.GetEBUpdateEFlag()[0],0);
        auto const J = w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        auto const& geom = w.Geom(0);
        auto const* pedestal = hp.DensityPedestal(0);
        amrex::GpuArray<int,3> const node{1,1,1}, ratio{1,1,1};
        auto const floor = hp.m_n_floor*PhysConst::q_e;
        amrex::Real result = 0.0_rt;
        for (int c = 0; c < 3; ++c) {
            auto const it = Ji[c]->ixType().toIntVect();
            amrex::GpuArray<int,3> const stagger{it[0],it[1],1};
            auto const volume = MakeQdsmcVolumeElement(geom,Ji[c]->ixType());
            MultiFab measured(Ji[c]->boxArray(),Ji[c]->DistributionMap(),1,0);
            for (amrex::MFIter mfi(measured); mfi.isValid(); ++mfi) {
                auto const den = rho.const_array(mfi);
                auto const ped = pedestal ? pedestal->const_array(mfi)
                                         : amrex::Array4<amrex::Real const>{};
                auto const ji = Ji[c]->const_array(mfi), jtot = J[c]->const_array(mfi);
                auto const out = measured.array(mfi);
                bool const has_pedestal = pedestal != nullptr;
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const raw = ablastr::coarsen::sample::Interp(
                        den,node,stagger,ratio,i,j,k,0) + (has_pedestal ?
                        ablastr::coarsen::sample::Interp(ped,node,stagger,ratio,i,j,k,0) : 0.0_rt);
                    auto const q = ji(i,j,k)-jtot(i,j,k);
                    out(i,j,k) = volume(i,j,k)*PhysConst::m_e*q*q/
                        (2.0_rt*PhysConst::q_e*amrex::max(floor,raw));
                });
            }
            result += measured.sum_unique(0,false,geom.periodicity());
        }
        return result;
    }

    // Full represented thermal state, including the pedestal and floor.
    // QDSMCClassEnergy deliberately omits sub-floor nodes for its bulk/band
    // report, so it is not the total electron energy of a moving pedestal edge.
    amrex::Real thermal_energy (WarpX& w, HybridPICModel const& hp, MultiFab const& rho)
    {
        if (!hp.m_solve_electron_energy_equation) { return 0.0_rt; }
        auto const& te=*w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
        auto const* pedestal=hp.DensityPedestal(0);
        bool const has_ped=pedestal!=nullptr;
        auto const floor=hp.m_n_floor*PhysConst::q_e;
        auto const volume=MakeQdsmcVolumeElement(w.Geom(0),rho.ixType());
        auto const capacity=PhysConst::kb/(PhysConst::q_e*(hp.m_gamma-1.0_rt));
        MultiFab measured(rho.boxArray(),rho.DistributionMap(),1,0);
        for (amrex::MFIter mfi(measured);mfi.isValid();++mfi) {
            auto const den=rho.const_array(mfi), temp=te.const_array(mfi);
            auto const ped=has_ped ? pedestal->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            auto const out=measured.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                out(i,j,k)=volume(i,j,k)*capacity*temp(i,j,k)*
                    amrex::max(floor,den(i,j,k)+(has_ped ? ped(i,j,k) : 0.0_rt));
            });
        }
        return measured.sum_unique(0,false,w.Geom(0).periodicity());
    }

    struct Inventory
    {
        amrex::Real charge = 0.0_rt, kinetic = 0.0_rt, pz = 0.0_rt;
        amrex::Real escaped_charge = 0.0_rt, escaped_energy = 0.0_rt;
        amrex::Real escaped_lo = 0.0_rt, escaped_hi = 0.0_rt;
        amrex::Real incident_lo = 0.0_rt, incident_hi = 0.0_rt;
    };

    Inventory inventory (WarpX& w)
    {
        Inventory out;
        auto& particles = w.GetPartContainer();
        for (auto const& name : particles.GetSpeciesNames()) {
            auto& pc = particles.GetParticleContainerFromName(name);
            out.charge += pc.sumParticleCharge();
            out.kinetic += pc.sumParticleEnergy();
            auto const mass = pc.getMass(), charge = pc.getCharge();
            auto const zlo = w.Geom(0).ProbLo(1), zhi = w.Geom(0).ProbHi(1);
            using Particle = WarpXParticleContainer::SuperParticleType;
            amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> op;
            auto const momentum = amrex::ParticleReduce<
                amrex::ReduceData<amrex::Real,amrex::Real,amrex::Real>>(
                pc, [=] AMREX_GPU_DEVICE (Particle const& p)
                    -> amrex::GpuTuple<amrex::Real,amrex::Real,amrex::Real> {
                    auto const weight = p.rdata(PIdx::w);
                    return {mass*weight*p.rdata(PIdx::uz),
                            p.pos(1)<zlo ? charge*weight : 0.0_rt,
                            p.pos(1)>zhi ? charge*weight : 0.0_rt};
                }, op);
            amrex::Real sums[3] = {amrex::get<0>(momentum),amrex::get<1>(momentum),
                                   amrex::get<2>(momentum)};
            amrex::ParallelDescriptor::ReduceRealSum(sums,3);
            out.pz += sums[0]; out.incident_lo += sums[1]; out.incident_hi += sums[2];
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                for (int side = 0; side < 2; ++side) {
                    auto const lost_q = particles.GetBoundaryAbsorbedCharge(name,d,side);
                    out.escaped_charge += lost_q;
                    out.escaped_energy += particles.GetBoundaryAbsorbedEnergy(name,d,side);
                    if (d == 1) {
                        (side == 0 ? out.escaped_lo : out.escaped_hi) += lost_q;
                    }
                }
            }
        }
        return out;
    }
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        auto const& geom = w.Geom(0);
        AMREX_ALWAYS_ASSERT(!geom.isPeriodic(1));
        AMREX_ALWAYS_ASSERT(WarpX::field_gathering_algo == GatheringAlgo::MomentumConserving);
        AMREX_ALWAYS_ASSERT(hp.m_include_electron_inertia_elliptic &&
            hp.m_electron_inertia_convection && hp.m_electron_inertia_moment_terms &&
            hp.m_electron_inertia_momentum_flux && hp.m_energy_conserving_motional);
        auto& rho = *w.m_fields.get(FieldType::rho_fp,0);
        auto const Ji = w.m_fields.get_alldirs(FieldType::current_fp,0);
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp,0);
        auto const B = w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
        auto const& te = *w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
        Scratch push_e(E);
        MultiFab previous_rho(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        MultiFab residual(rho.boxArray(),rho.DistributionMap(),1,0);
        MultiFab regions(rho.boxArray(),rho.DistributionMap(),6,0);
        int steps = 24;
        bool require_escape = true;
        amrex::ParmParse test("mc_boundary");
        test.query("steps",steps);
        test.query("require_escape",require_escape);
        Inventory before, before_boundary, after_boundary;
        auto const initial = inventory(w);
        amrex::Real max_charge_error = 0.0_rt, max_continuity = 0.0_rt;
        amrex::Real work_mismatch = 0.0_rt, grid_work_sum = 0.0_rt;
        amrex::Real reflected_charge = 0.0_rt, incident_charge = 0.0_rt;
        amrex::Real boundary_energy_error = 0.0_rt, wall_impulse_z = 0.0_rt;
        // aftercollisions runs after velocity de-synchronization and immediately
        // before the actual explicit hybrid momentum/position push. No collisions
        // are enabled in this controlled fixture.
        InstallPythonCallback("aftercollisions",[&]() {
            before = inventory(w);
            push_e.copy(E);
            MultiFab::Copy(previous_rho,rho,0,0,1,rho.nGrowVect());
        });
        InstallPythonCallback("particlescraper",[&]() {
            // Native Evolve calls this immediately before ApplyBoundaryConditions.
            before_boundary = inventory(w);
        });
        InstallPythonCallback("beforeEsolve",[&]() {
            after_boundary = inventory(w);
            auto const lost_q = after_boundary.escaped_charge-before_boundary.escaped_charge;
            auto const incident = before_boundary.incident_lo+before_boundary.incident_hi;
            incident_charge += incident;
            reflected_charge += incident-lost_q;
            wall_impulse_z += before_boundary.pz-after_boundary.pz;
            auto const error = before_boundary.kinetic-after_boundary.kinetic
                -(after_boundary.escaped_energy-before_boundary.escaped_energy);
            boundary_energy_error = std::max(boundary_energy_error,
                std::abs(error)/std::max(before_boundary.kinetic,1.e-30_rt));
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(boundary_energy_error < 2.e-12_rt,
                "Specular wall handling must lose only the tallied absorbed kinetic energy");
        });
        // This callback runs after native boundary processing, re-deposition and
        // the field subcycles. Thus the sampled J includes filtering, physical
        // boundary handling and the MPI overlap sum used by the real field solve.
        InstallPythonCallback("afterEsolve",[&]() {
            auto const after = inventory(w);
            auto const dt = w.getdt(0);
            auto const ion_gain = after.kinetic-before.kinetic
                + after.escaped_energy-before.escaped_energy;
            auto const grid_work = dt*product(Ji,push_e.field,geom);
            work_mismatch += ion_gain-grid_work;
            grid_work_sum += grid_work;
            auto const charge_error = std::abs(after.charge+after.escaped_charge
                -initial.charge-initial.escaped_charge)/std::abs(initial.charge);
            max_charge_error = std::max(max_charge_error,charge_error);
            w.ComputeRZContinuityResidual(residual,previous_rho,rho,Ji,dt);
            auto const continuity = residual.norminf()*dt/
                std::max(previous_rho.norminf(),rho.norminf());
            max_continuity = std::max(max_continuity,continuity);
            auto const domain = geom.Domain();
            int const irlo = domain.smallEnd(0), irhi = domain.bigEnd(0)+1;
            int const izlo = domain.smallEnd(1), izhi = domain.bigEnd(1)+1;
            int const seam = (izlo+izhi)/2;
            for (amrex::MFIter mfi(regions);mfi.isValid();++mfi) {
                auto const src = residual.const_array(mfi);
                auto const dst = regions.array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const a = amrex::Math::abs(src(i,j,k));
                    dst(i,j,k,0) = i<=irlo+3 ? a : 0.0_rt;
                    dst(i,j,k,1) = i>=irhi-3 ? a : 0.0_rt;
                    dst(i,j,k,2) = j<=izlo+3 ? a : 0.0_rt;
                    dst(i,j,k,3) = j>=izhi-3 ? a : 0.0_rt;
                    dst(i,j,k,4) = j>=seam-2 && j<=seam+2 ? a : 0.0_rt;
                    dst(i,j,k,5) = i>irlo+3 && i<irhi-3 && j>izlo+3 && j<izhi-3
                        ? a : 0.0_rt;
                });
            }
            amrex::Real reg[6];
            for (int c=0;c<6;++c) { reg[c]=regions.norminf(c)*dt/
                std::max(previous_rho.norminf(),rho.norminf()); }
            bool finite = rho.is_finite() && te.is_finite();
            for (int c = 0; c < 3; ++c) {
                finite = finite && E[c]->is_finite() && B[c]->is_finite() && Ji[c]->is_finite();
            }
            auto const bulk = electron_energy(w,hp,rho,Ji,B);
            auto const thermal = thermal_energy(w,hp,rho);
            auto const magnetic = product(B,B,geom)/(2.0_rt*PhysConst::mu0);
            // The work discrepancy contains MC spatial and finite-step particle
            // quadrature/boundary-reconstruction errors. It is not booked as heat
            // or claimed to isolate the spatial gather from the particle pusher.
            amrex::Print() << std::setprecision(17)
                << "MC_BOUNDARY {\"step\":" << w.getistep(0)
                << ",\"time_s\":" << w.gett_new(0)
                << ",\"dt_s\":" << dt
                << ",\"particle_charge_C\":" << after.charge
                << ",\"escaped_charge_C\":" << after.escaped_charge
                << ",\"escaped_zlo_C\":" << after.escaped_lo
                << ",\"escaped_zhi_C\":" << after.escaped_hi
                << ",\"charge_inventory_relative\":" << charge_error
                << ",\"continuity_relative\":" << continuity
                << ",\"continuity_axis_relative\":" << reg[0]
                << ",\"continuity_rwall_relative\":" << reg[1]
                << ",\"continuity_zlo_relative\":" << reg[2]
                << ",\"continuity_zhi_relative\":" << reg[3]
                << ",\"continuity_seam_relative\":" << reg[4]
                << ",\"continuity_interior_relative\":" << reg[5]
                << ",\"ion_J\":" << after.kinetic
                << ",\"ion_pz_kg_m_s\":" << after.pz
                << ",\"wall_impulse_z_kg_m_s\":" << wall_impulse_z
                << ",\"boundary_energy_relative\":" << boundary_energy_error
                << ",\"incident_axial_charge_C\":" << incident_charge
                << ",\"reflected_axial_charge_C\":" << reflected_charge
                << ",\"escaped_ion_J\":" << after.escaped_energy
                << ",\"electron_bulk_J\":" << bulk
                << ",\"thermal_J\":" << thermal
                << ",\"magnetic_J\":" << magnetic
                << ",\"ion_gain_plus_escape_J\":" << ion_gain
                << ",\"grid_work_J\":" << grid_work
                << ",\"cumulative_grid_work_J\":" << grid_work_sum
                << ",\"cumulative_work_mismatch_J\":" << work_mismatch
                << ",\"Te_max_K\":" << te.norminf()
                << ",\"finite\":" << (finite ? "true" : "false") << "}\n";
            AMREX_ALWAYS_ASSERT(finite && std::isfinite(work_mismatch));
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(charge_error < 2.e-12_rt,
                "Particle charge plus tallied absorbed charge must remain constant");
        });
        w.Evolve(steps);
        ClearPythonCallback("aftercollisions");
        ClearPythonCallback("particlescraper");
        ClearPythonCallback("beforeEsolve");
        ClearPythonCallback("afterEsolve");
        auto const final = inventory(w);
        AMREX_ALWAYS_ASSERT(w.getistep(0) == steps);
        AMREX_ALWAYS_ASSERT(incident_charge > 0.0_rt);
        if (WarpX::particle_absorption_fraction < 1.0_rt) {
            AMREX_ALWAYS_ASSERT(reflected_charge > 0.0_rt);
        }
        if (require_escape) {
            AMREX_ALWAYS_ASSERT(final.escaped_lo > 0.0_rt && final.escaped_hi > 0.0_rt);
        }
        amrex::Print() << std::setprecision(17)
            << "MC_BOUNDARY_COMPLETE {\"steps\":" << steps
            << ",\"max_charge_inventory_relative\":" << max_charge_error
            << ",\"max_continuity_relative\":" << max_continuity
            << ",\"work_mismatch_J\":" << work_mismatch
            << ",\"scope\":\"particle inventory and finite native evolution; energy closure not asserted\"}\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
