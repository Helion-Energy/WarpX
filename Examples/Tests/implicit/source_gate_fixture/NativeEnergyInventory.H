/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_SOURCE_GATE_NATIVE_ENERGY_INVENTORY_H_
#define WARPX_SOURCE_GATE_NATIVE_ENERGY_INVENTORY_H_
#include "Diagnostics/ReducedDiags/FieldEnergy.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include <AMReX_Reduce.H>
#include <array>
#include <limits>
#include <memory>

// Fixture-only read-only endpoint inventory. There are no residual callbacks,
// particle writes, parser changes, random draws, or production acceptance gates.
class NativeEnergyInventory {
    using Real=amrex::Real;
    using FieldType=warpx::fields::FieldType;
    using Direction=ablastr::fields::Direction;
    WarpX& m_w;
    amrex::Geometry m_g;
    FieldEnergy m_native_norm{"source_gate_native_norm"};
    std::array<amrex::MultiFab,7> m_old;
    std::array<Real,7> m_before{}, m_before_clipped{};
    std::vector<Real> m_old_k, m_old_weight;
    Real m_old_lost=0;
    bool m_open=false;

    std::array<amrex::MultiFab const*,7> Fields() const {
        std::array<amrex::MultiFab const*,7> f{};
        for(int d=0;d<3;++d) {
            f[d]=m_w.m_fields.get(FieldType::Bfield_fp,Direction{d},0);
            f[3+d]=m_w.m_fields.get("hybrid_E_long_fp",Direction{d},0);
        }
        f[6]=m_w.m_fields.get(FieldType::hybrid_electron_energy_fp,0);
        return f;
    }
public: // NVCC extended device lambdas require public enclosing methods.
    // Sum native geometric dual energy and stable difference. Exact clipped
    // volume is an additional independent diagnostic: its outer nodal ring is
    // pi*(r*dr-dr²/4), unlike the stock half-ring pi*r*dr quadrature.
    std::array<Real,5> Integrate(amrex::MultiFab const& f,
        amrex::MultiFab const* old, bool squared, bool clipped) const {
        AMREX_ALWAYS_ASSERT(f.nComp()==1);
        auto const dx=m_g.CellSizeArray(), lo=m_g.ProbLoArray();
        auto const domain_lo=m_g.Domain().smallEnd();
        auto const nodal=f.ixType().toIntVect();
        auto const volume=MakeQdsmcVolumeElement(m_g,f.ixType());
        auto const owner=f.OwnerMask(m_g.periodicity());
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,
            amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> op;
        amrex::ReduceData<Real,Real,Real,Real,Real> data(op);
        for(amrex::MFIter it(f);it.isValid();++it) {
            auto const a=f.const_array(it), b=old?old->const_array(it):a;
            auto const own=owner->const_array(it);
            auto const box=it.validbox();auto const small=box.smallEnd(), big=box.bigEnd();
            op.eval(box,data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real measure;
                if(clipped) { measure=own(i,j,k)?volume(i,j,k):0.; }
                else {
                    Real const radius=lo[0]+(i-domain_lo[0]+.5*(1-nodal[0]))*dx[0];
                    measure=radius==0.?MathConst::pi*dx[0]/4.:2*MathConst::pi*radius;
                    if(radius!=0. && nodal[0] && i==small[0]) measure*=.5;
                    if(nodal[0] && i==big[0]) measure*=.5;
                    if(nodal[1] && j==small[1]) measure*=.5;
                    if(nodal[1] && j==big[1]) measure*=.5;
                    measure*=dx[0]*dx[1];
                }
                Real const x=a(i,j,k), y=b(i,j,k);
                return amrex::GpuTuple<Real,Real,Real,Real,Real>{
                    measure*(squared?x*x:x),
                    measure*(squared?(x-y)*(x+y):x-y),
                    measure,1.,measure*std::abs(squared?(x-y)*(x+y):x-y)};
            });
        }
        auto const v=data.value();
        std::array<Real,5> answer{amrex::get<0>(v),amrex::get<1>(v),
            amrex::get<2>(v),amrex::get<3>(v),amrex::get<4>(v)};
        amrex::ParallelDescriptor::ReduceRealSum(answer.data(),int(answer.size()));
        return answer;
    }
    Real WallTangentialE() const {
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<Real> data(op);
        int const wall=m_g.Domain().bigEnd(0)+1;
        for(int d=1;d<3;++d) {
            auto const& f=*m_w.m_fields.get(FieldType::Efield_fp,Direction{d},0);
            AMREX_ALWAYS_ASSERT(f.ixType().nodeCentered(0));
            for(amrex::MFIter it(f);it.isValid();++it) {
                auto box=it.validbox();if(wall<box.smallEnd(0)||wall>box.bigEnd(0)) continue;
                box.setSmall(0,wall);box.setBig(0,wall);auto const a=f.const_array(it);
                op.eval(box,data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    return amrex::GpuTuple<Real>{std::abs(a(i,j,k))};
                });
            }
        }
        Real value=amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(value);return amrex::max(0.,value);
    }
    Real LostEnergy() const {
        Real value=0,weight=0,charge=0;
        auto const& particles=m_w.GetPartContainer();
        for(int s=0;s<particles.nSpecies();++s) {
            auto const& species=particles.GetParticleContainer(s);
            for(int d=0;d<AMREX_SPACEDIM;++d) for(int side=0;side<2;++side) {
                value+=species.GetBoundaryAbsorbedEnergy(d,side);
                weight+=species.GetBoundaryAbsorbedWeight(d,side);
                charge+=species.GetBoundaryAbsorbedCharge(d,side);
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum({value,weight,charge});
        amrex::Print()<<"SOURCE_GATE_BOUNDARY step="<<m_w.getistep(0)
            <<" lost_energy="<<value<<" lost_weight="<<weight<<" lost_charge="<<charge
            <<" tangential_wall_E="<<WallTangentialE()<<"\n";
        return value;
    }
public:
    explicit NativeEnergyInventory(WarpX& w):m_w(w),m_g(w.get_pointer_HybridPICModel()->ElectronThermalGeometry()) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_g.IsRZ() && w.finestLevel()==0,
            "Native source energy fixture is qualified only for single-level RZ");
    }
    void Begin() {
        AMREX_ALWAYS_ASSERT(!m_open);
        auto fields=Fields();
        for(int n=0;n<7;++n) {
            auto const& f=*fields[n];
            if(!m_old[n].isDefined()) m_old[n].define(f.boxArray(),f.DistributionMap(),1,0);
            amrex::MultiFab::Copy(m_old[n],f,0,0,1,0);
            auto const v=Integrate(f,nullptr,n<6,false), c=Integrate(f,nullptr,n<6,true);
            Real const factor=n<3?.5/PhysConst::mu0:(n<6?.5*PhysConst::epsilon_0:1.);
            m_before[n]=factor*v[0];m_before_clipped[n]=factor*c[0];
            Real const native=n<6?factor*m_native_norm.ComputeNorm2(f,0)*m_g.CellSize(0)*m_g.CellSize(1):m_before[n];
            amrex::Print()<<"SOURCE_GATE_INVENTORY_OLD component="<<n
                <<" energy="<<m_before[n]<<" clipped="<<m_before_clipped[n]
                <<" native="<<native<<" volume="<<v[2]<<" clipped_volume="<<c[2]
                <<" entries="<<v[3]<<"\n";
        }
        auto const& particles=m_w.GetPartContainer();
        m_old_k.resize(particles.nSpecies());m_old_weight.resize(particles.nSpecies());
        for(int s=0;s<particles.nSpecies();++s) {
            auto const [energy,weight]=particles.GetParticleContainer(s).sumParticleWeightAndEnergy(false);
            m_old_k[s]=energy;m_old_weight[s]=weight;
        }
        m_old_lost=LostEnergy();m_open=true;
    }
    void End() {
        AMREX_ALWAYS_ASSERT(m_open);
        auto fields=Fields();Real field_delta=0,clipped_delta=0,internal_delta=0;
        for(int n=0;n<7;++n) {
            auto const& f=*fields[n];auto const v=Integrate(f,&m_old[n],n<6,false);
            auto const c=Integrate(f,&m_old[n],n<6,true);
            Real const factor=n<3?.5/PhysConst::mu0:(n<6?.5*PhysConst::epsilon_0:1.);
            Real const native=n<6?factor*m_native_norm.ComputeNorm2(f,0)*m_g.CellSize(0)*m_g.CellSize(1):factor*v[0];
            if(n<6) {field_delta+=factor*v[1];clipped_delta+=factor*c[1];}
            else internal_delta=factor*v[1];
            amrex::Print()<<"SOURCE_GATE_INVENTORY_NEW component="<<n
                <<" energy="<<factor*v[0]<<" clipped="<<factor*c[0]<<" native="<<native
                <<" stable_delta="<<factor*v[1]<<" difference="<<factor*v[0]-m_before[n]
                <<" clipped_delta="<<factor*c[1]<<" absolute_delta="<<factor*v[4]<<" entries="<<v[3]<<"\n";
        }
        Real kinetic_delta=0;
        auto const& particles=m_w.GetPartContainer();
        for(int s=0;s<particles.nSpecies();++s) {
            auto const& species=particles.GetParticleContainer(s);
            auto const [energy,weight]=species.sumParticleWeightAndEnergy(false);
            kinetic_delta+=energy-m_old_k[s];
            amrex::Print()<<"SOURCE_GATE_ION_INVENTORY species="<<s
                <<" old="<<m_old_k[s]<<" new="<<energy<<" delta="<<energy-m_old_k[s]
                <<" old_weight="<<m_old_weight[s]<<" new_weight="<<weight
                <<" particles="<<species.TotalNumberOfParticles()<<"\n";
        }
        Real const lost=LostEnergy()-m_old_lost;
        amrex::Print()<<"SOURCE_GATE_TOTAL_INVENTORY step="<<m_w.getistep(0)
            <<" time="<<m_w.gett_new(0)<<" field_delta="<<field_delta
            <<" clipped_field_delta="<<clipped_delta<<" internal_delta="<<internal_delta
            <<" kinetic_delta="<<kinetic_delta<<" boundary_loss="<<lost
            <<" actual_total_defect="<<field_delta+internal_delta+kinetic_delta+lost
            <<" clipped_total_defect="<<clipped_delta+internal_delta+kinetic_delta+lost<<"\n";
        m_open=false;
    }
};
#endif
