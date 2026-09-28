/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativePairedDarwinFields.H"
#include "NativePECPlasma.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "NativeCoilField.H"
#include "Circuit/CircuitCoupling.H"
#include "NativeEndpointField.H"
#include "NativeVacuumEndpoint.H"
#include "DarwinInitialRateSchur.H"
#include "DarwinRZGreenSolver.H"
#include "DarwinABoundary.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include "Utils/WarpXConst.H"
#include <AMReX_ParmParse.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>

namespace warpx::darwin {
using R=amrex::Real;using MF=amrex::MultiFab;
using V=ablastr::fields::VectorField;
using D=ablastr::fields::Direction;
using warpx::fields::FieldType;
namespace {
constexpr char saved_e[]="native_PEC_saved_E_fp";
constexpr char mass[]="native_PEC_mass_fp";
constexpr char unit[]="native_PEC_unit_fp";
void SyncPMC(WarpX& sim,V const& v) {
    amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
        hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
    }
    for(auto* f:v) {
        f->OverrideSync(sim.Geom(0).periodicity());
        f->FillBoundary(sim.Geom(0).periodicity());
        ApplyDarwinPMCVectorBoundary(*f,sim.Geom(0),lo,hi);
    }
}
}
bool NativePECPlasmaEnabled() {
    bool enabled=false;
    amrex::ParmParse("endpoint_diagnostic").query("pec_plasma_current",enabled);
    return enabled;
}
void CompleteNativeElectricGatherImages(WarpX& sim,V const& fields) {
    AMREX_ALWAYS_ASSERT(sim.maxLevel()==0);
    for(auto* field:fields) {
        field->setBndry(0.);
        field->OverrideSync(sim.Geom(0).periodicity());
        field->FillBoundary(sim.Geom(0).periodicity());
    }
    amrex::Vector<amrex::IntVect> ratios;
    PEC::ApplyPECtoEfield(fields,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PEC,sim.get_ng_fieldgather(),sim.Geom(0),0,PatchType::fine,ratios);
    PEC::ApplyPECtoBfield(fields,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PMC,sim.get_ng_fieldgather(),sim.Geom(0),0,PatchType::fine,ratios);
#if defined(WARPX_DIM_RZ)
    sim.ApplyFieldBoundaryOnAxis(fields[0],fields[1],fields[2],0);
#endif
    amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
        hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
    }
    for(auto* field:fields)ApplyDarwinPMCVectorBoundary(*field,sim.Geom(0),lo,hi);
}
bool NativeDrivenPECNormalFieldSupported(WarpX& sim) {
#if defined(WARPX_DIM_RZ)
    auto const& model=*sim.get_pointer_HybridPICModel();
    bool valid=model.m_add_external_fields && model.m_external_vector_potential &&
        sim.Geom(0).isPeriodic(1) && sim.m_fields.has(FieldType::Bfield_fp,D{0},0);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid)return false;
    auto const& ext=*model.m_external_vector_potential;
    int count_lo=ext.nFields(),count_hi=count_lo;
    int static_lo=sim.m_fields.has("hybrid_B_static_fp",D{0},0),static_hi=static_lo;
    amrex::ParallelDescriptor::ReduceIntMin(count_lo);amrex::ParallelDescriptor::ReduceIntMax(count_hi);
    amrex::ParallelDescriptor::ReduceIntMin(static_lo);amrex::ParallelDescriptor::ReduceIntMax(static_hi);
    if(count_lo!=count_hi || static_lo!=static_hi)return false;
    for(int f=0;f<count_lo;++f)valid=valid && sim.m_fields.has(ext.FieldName(f)+"_curlAext",D{0},0);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);if(!valid)return false;
    int const ir=sim.Geom(0).Domain().bigEnd(0)+1;
    auto zero_wall=[&](MF const& field) {
        bool native=field.ixType().nodeCentered(0) && field.nComp()==1;
        amrex::ParallelDescriptor::ReduceBoolAnd(native);if(!native)return false;
        MF check(field.boxArray(),field.DistributionMap(),1,0);
        for(amrex::MFIter mfi(check);mfi.isValid();++mfi) {
            auto const out=check.array(mfi);auto const in=field.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                out(i,j,k)=i==ir?in(i,j,k):0.;
            });
        }
        return check.is_finite() && check.norminf()==0.;
    };
    valid=zero_wall(*sim.m_fields.get(FieldType::Bfield_fp,D{0},0));
    if(static_lo)valid=zero_wall(*sim.m_fields.get("hybrid_B_static_fp",D{0},0)) && valid;
    for(int f=0;f<count_lo;++f)
        valid=zero_wall(*sim.m_fields.get(ext.FieldName(f)+"_curlAext",D{0},0)) && valid;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);return valid;
#else
    amrex::ignore_unused(sim);return false;
#endif
}
void CombineNativePECTangential(WarpX& sim,V const& out,V const& first,R a,V const& second,R b) {
#if defined(WARPX_DIM_RZ)
    int const ir=sim.Geom(0).Domain().bigEnd(0)+1;
    for(int c=1;c<3;++c)for(amrex::MFIter mfi(*out[c]);mfi.isValid();++mfi) {
        auto box=mfi.validbox();if(ir<box.smallEnd(0)||ir>box.bigEnd(0))continue;
        box.setSmall(0,ir);box.setBig(0,ir);
        auto const dst=out[c]->array(mfi);auto const x=first[c]->const_array(mfi),y=second[c]->const_array(mfi);
        amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int j,int k){dst(i,j,k)=a*x(i,j,k)+b*y(i,j,k);});
    }
#else
    amrex::ignore_unused(sim,out,first,a,second,b);amrex::Abort("Native PEC combination requires RZ");
#endif
}
void ValidateNativePECPlasma(WarpX& sim) {
    int flags_lo=(NativePECPlasmaEnabled()?1:0)+(NativeCoilCurrentEnabled()?2:0),flags_hi=flags_lo;
    amrex::ParallelDescriptor::ReduceIntMin(flags_lo);amrex::ParallelDescriptor::ReduceIntMax(flags_hi);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(flags_lo==flags_hi,"PEC plasma/current convention differs across ranks");
    if(!(flags_lo&1))return;
#if !defined(WARPX_DIM_RZ)
    amrex::Abort("PEC plasma continuation currently requires on-axis RZm0");
#else
    auto const& m=*sim.get_pointer_HybridPICModel();
    bool const driven=NativeCoilCurrentEnabled();
    if(driven) {
        bool split=false,vacuum_split=false,mmj=false,mmpc=false,compensated=false;
        amrex::ParmParse options("endpoint_diagnostic");options.query("split_ampere",split);options.query("vacuum_split_ampere",vacuum_split);options.query("compensated_longitudinal_increment",compensated);
        amrex::ParmParse implicit("implicit_evolve");implicit.query("use_mass_matrices_jacobian",mmj);implicit.query("use_mass_matrices_pc",mmpc);
        bool supported=NativeCircuitDriveEnabled() && m.m_add_external_fields &&
            m.m_external_unified && !m.m_has_external_current && !m.m_darwin_checkpoint_restored &&
            !split && !vacuum_split && !mmj && !mmpc && !compensated;
        amrex::ParallelDescriptor::ReduceBoolAnd(supported);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
            "Driven PEC current prerequisite requires native unified circuit, MM0 and no split/vacuum/restart/compensated-low");
        auto const* circuit=sim.get_pointer_CircuitCoupling();
        bool owned=circuit && circuit->Coupler() && m.m_external_vector_potential;
        amrex::ParallelDescriptor::ReduceBoolAnd(owned);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(owned,"Driven PEC requires initialized native circuit-owned fields");
        auto const& external=*m.m_external_vector_potential;
        auto const& coils=circuit->GetCoilSet();
        owned=external.nFields()==coils.size();
        for(int f=0;f<external.nFields();++f) {
            bool matched=false;
            for(auto const& coil:coils.coils())matched=matched || coil.field_name==external.FieldName(f);
            owned=owned && matched && external.UsesPythonScale(f);
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(owned);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(owned,
            "Driven PEC currently requires all external fields to be native circuit-owned affine segments");
        // Extra analytic profiles can carry total-current curvature; they need
        // a separately derived plasma-current curvature path before admission.
        bool const normal_supported=NativeDrivenPECNormalFieldSupported(sim);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(normal_supported,
            "Driven PEC current prerequisite requires exact zero native wall-normal B on every rank");
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NativeEndpointEnabled() &&
        NativeIonQuadratureEnabled() && NativeCorrelatedIncrementEnabled() &&
        NativeFullOhmLongitudinalEnabled() && m.UseCompatibleYeeInertia() &&
        m.UsesEulerianElectronEnergy() && sim.Geom(0).ProbLo(0)==0. &&
        WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC &&
        WarpX::particle_boundary_hi[0]==ParticleBoundaryType::Reflecting &&
        !m.m_resistive_wall && !m.HasResistivity() &&
        !m.m_include_electron_viscosity && !m.m_include_hyper_resistivity_term &&
        !m.m_end_region.holmstrom && !m.m_holmstrom_vacuum_region &&
        !m.m_darwin_vacuum_recovery && (!m.m_add_external_fields || driven) &&
        !NativePrescribedDriveEnabled() && (!NativeCircuitDriveEnabled() || driven) &&
        !m.m_include_joule_heating && !m.m_include_temperature_relaxation &&
        !m.m_joule_redirect_to_ions && !m.m_esolve_tensor &&
        !WarpX::do_single_precision_comms,
        "PEC plasma continuation requires positive RZm0 pressure/Hall-only full-Ohm native inertia, perfect reflecting PEC, no eta/end/viscosity/vacuum/OU, DP communication, and any circuit must use the guarded driven-current convention");
#endif
}
void AllocateNativePECPlasma(WarpX& sim) {
    if(!NativePECPlasmaEnabled())return;
    auto& fields=sim.m_fields;
    auto const current=fields.get_alldirs(FieldType::current_fp,0);
    auto const electric=fields.get_alldirs(FieldType::Efield_fp,0);
    for(int c=0;c<3;++c) {
        for(auto const* name:{PECWallCurrentName,PECWallStageName,PECPlasmaRateName,mass,unit})
            if(!fields.has(name,D{c},0))fields.alloc_init(name,D{c},0,
                current[c]->boxArray(),current[c]->DistributionMap(),1,amrex::IntVect(1),
                name==unit?1.:0.,true,true,name==PECWallCurrentName);
        for(auto const* name:{PECPressureHallName,saved_e})
            if(!fields.has(name,D{c},0))fields.alloc_init(name,D{c},0,
                electric[c]->boxArray(),electric[c]->DistributionMap(),1,electric[c]->nGrowVect(),0.);
    }
}
void RebuildNativePECStageCurrent(WarpX& sim,R dt,NativeCoilField* coil,std::optional<R> stage_time) {
    if(!NativePECPlasmaEnabled())return;
    AMREX_ALWAYS_ASSERT(dt>0. && std::isfinite(dt));
    auto& fields=sim.m_fields;
    auto& model=*sim.get_pointer_HybridPICModel();
    if(NativeCoilCurrentEnabled()) {
        AMREX_ALWAYS_ASSERT(coil && stage_time && std::isfinite(*stage_time));
        coil->CalculatePlasmaCurrent(fields.get_alldirs(FieldType::Bfield_fp,0),*stage_time);
    } else model.CalculatePlasmaCurrent(fields.get_alldirs(FieldType::Bfield_fp,0),sim.GetEBUpdateEFlag()[0],0);
    auto const jp=fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    auto const e=fields.get_alldirs("hybrid_E_long_fp",0),e0=fields.get_alldirs("hybrid_E_long_old_fp",0);
    auto const dbar=fields.get_alldirs("diagnostic_D_stage_fp",0);
    if(auto* paired=NativeActivePairedFields(sim)) { paired->SubtractDisplacement(jp);return; }
    R const factor=2.*PhysConst::epsilon_0/dt;
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*dbar[c]);mfi.isValid();++mfi) {
        auto const j=jp[c]->array(mfi),d=dbar[c]->array(mfi);
        auto const a=e[c]->const_array(mfi),b=e0[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int k,int l) {
            R const displacement=factor*(a(i,k,l)-b(i,k,l));
            d(i,k,l)=displacement;j(i,k,l)-=displacement;
        });
    }
}
void SetNativePECTangential(WarpX& sim,V const& out,V const* src,R scale) {
#if defined(WARPX_DIM_RZ)
    int const wall=sim.Geom(0).Domain().bigEnd(0)+1;
    for(int c=1;c<3;++c)for(amrex::MFIter mfi(*out[c]);mfi.isValid();++mfi) {
        auto box=mfi.validbox();if(!box.contains(amrex::IntVect(wall,box.smallEnd(1))))continue;
        box.setSmall(0,wall);box.setBig(0,wall);
        auto const dst=out[c]->array(mfi);
        auto const in=src?(*src)[c]->const_array(mfi):amrex::Array4<R const>{};
        amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            dst(i,j,k)=in?scale*in(i,j,k):0.;
        });
    }
#else
    amrex::ignore_unused(sim,out,src,scale);
    amrex::Abort("PEC tangential plasma support requires RZ");
#endif
}
bool PrepareNativePECStage(WarpX& sim,R dt,MF const& kappa,V const& oldI,V const& endI,V const& stage) {
    if(!NativePECPlasmaEnabled())return true;
#if defined(WARPX_DIM_RZ)
    auto& fields=sim.m_fields;auto& model=*sim.get_pointer_HybridPICModel();
    auto const e=fields.get_alldirs(FieldType::Efield_fp,0);
    auto const saved=fields.get_alldirs(saved_e,0);
    auto const raw=fields.get_alldirs(PECPressureHallName,0);
    auto const jp=fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    auto const q0=fields.get_alldirs("diagnostic_Je_endpoint_fp",0);
    auto const wall=fields.get_alldirs(PECWallStageName,0);
    auto const rate=fields.get_alldirs(PECPlasmaRateName,0);
    auto const ones=fields.get_alldirs(unit,0),m=fields.get_alldirs(mass,0);
    R const scale=model.m_electron_inertia_mass/(PhysConst::q_e*PhysConst::q_e*model.m_n0_ref);
    ApplyYeeInertiaMass(sim.Geom(0),kappa,{ones[0],ones[1],ones[2]},m,scale);
    for(int c=0;c<3;++c)MF::Copy(*saved[c],*e[c],0,0,1,e[c]->nGrowVect());
    auto& rho=*fields.get(FieldType::rho_fp,0);
    MF half(rho,amrex::make_alias,rho.nComp()/2,1);
    model.HybridPICSolveE(e,fields.get_alldirs(FieldType::current_fp,0),
        fields.get_alldirs(FieldType::Bfield_fp,0),half,sim.GetEBUpdateEFlag()[0],0,
        false,true,nullptr,nullptr,nullptr,&raw);
    for(int c=0;c<3;++c)MF::Copy(*e[c],*saved[c],0,0,1,e[c]->nGrowVect());
    for(auto* f:wall)f->setVal(0.);
    for(auto* f:rate)f->setVal(0.);
    int const ir=sim.Geom(0).Domain().bigEnd(0)+1;
    R const interval=.5*dt;
    bool const driven=NativeCoilCurrentEnabled();
    for(int c=1;c<3;++c)for(amrex::MFIter mfi(*stage[c]);mfi.isValid();++mfi) {
        auto box=mfi.validbox();if(ir<box.smallEnd(0)||ir>box.bigEnd(0))continue;
        box.setSmall(0,ir);box.setBig(0,ir);
        auto const j=jp[c]->array(mfi),q=stage[c]->array(mfi),w=wall[c]->array(mfi),dq=rate[c]->array(mfi);
        auto const f=raw[c]->const_array(mfi),mu=m[c]->const_array(mfi),old=q0[c]->const_array(mfi),i0=oldI[c]->const_array(mfi),i1=endI[c]->const_array(mfi);
        auto const electric=saved[c]->const_array(mfi);
        amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int k,int l) {
            R const derivative=driven?(electric(i,k,l)-f(i,k,l))/mu(i,k,l):-f(i,k,l)/mu(i,k,l);
            R const plasma=old(i,k,l)+interval*derivative;
            R const ions=.5*(i0(i,k,l)+i1(i,k,l));
            w(i,k,l)=j(i,k,l)-ions-plasma;
            q(i,k,l)=plasma;dq(i,k,l)=derivative;
            j(i,k,l)=ions+plasma;
        });
    }
    SyncPMC(sim,wall);SyncPMC(sim,rate);
    SyncPMC(sim,stage);
    // Complete the same physical wall row in the native grow1 band. The
    // current registry has wider storage, but those unowned outer guards
    // are neither copied nor read by this constitutive branch.
    for(int c=1;c<3;++c)for(amrex::MFIter mfi(*stage[c]);mfi.isValid();++mfi) {
        auto box=mfi.fabbox();if(ir<box.smallEnd(0)||ir>box.bigEnd(0))continue;
        box.setSmall(0,ir);box.setBig(0,ir);
        auto const j=jp[c]->array(mfi);
        auto const q=stage[c]->const_array(mfi);
        auto const i0=oldI[c]->const_array(mfi),i1=endI[c]->const_array(mfi);
        amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int k,int l) {
            j(i,k,l)=.5*(i0(i,k,l)+i1(i,k,l))+q(i,k,l);
        });
    }
    // Do not copy the undefined grow2/3 Ampere band into any companion.
    for(auto const* f:stage)if(!f->is_finite(0,1,1))return false;
    return true;
#else
    amrex::ignore_unused(sim,dt,kappa,oldI,endI,stage);return false;
#endif
}
void CompleteNativePECStageRate(WarpX& sim,V const& rate) {
    if(!NativePECPlasmaEnabled())return;
    auto const wall=sim.m_fields.get_alldirs(PECPlasmaRateName,0);
    SetNativePECTangential(sim,rate,&wall,1.);SyncPMC(sim,rate);
}
void RotateNativePECWall(WarpX& sim) {
    if(!NativePECPlasmaEnabled())return;
    auto const old=sim.m_fields.get_alldirs(PECWallCurrentName,0),stage=sim.m_fields.get_alldirs(PECWallStageName,0);
    for(int c=0;c<3;++c)MF::LinComb(*old[c],2.,*stage[c],0,-1.,*old[c],0,0,1,1);
    SyncPMC(sim,old);ValidateNativePECWallState(sim);
}
void ValidateNativePECWallState(WarpX& sim) {
    if(!NativePECPlasmaEnabled())return;
#if defined(WARPX_DIM_RZ)
    int const ir=sim.Geom(0).Domain().bigEnd(0)+1;
    auto const wall=sim.m_fields.get_alldirs(PECWallCurrentName,0);
    for(int c=0;c<3;++c) {
        auto const& f=*wall[c];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.nGrowVect()==amrex::IntVect(1) && f.is_finite(0,1,1),
            "Accepted PEC reaction requires finite exact grow1 storage");
        MF outside(f.boxArray(),f.DistributionMap(),1,0);
        for(amrex::MFIter mfi(outside);mfi.isValid();++mfi) {
            auto const d=outside.array(mfi);
            auto const v=f.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                d(i,j,k)=(c>0&&i==ir)?0.:v(i,j,k);
            });
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(outside.norminf()==0.,"PEC reaction has normal or unconstrained support");
    }
#endif
}
void PublishNativePECEventWall(WarpX& sim,ablastr::fields::ConstVectorField const& supplied) {
    AMREX_ALWAYS_ASSERT(NativePECPlasmaEnabled());
    auto const out=sim.m_fields.get_alldirs(PECWallCurrentName,0);
    for(int c=0;c<3;++c) {
        AMREX_ALWAYS_ASSERT(supplied[c] && supplied[c]->boxArray()==out[c]->boxArray() &&
            supplied[c]->DistributionMap()==out[c]->DistributionMap() && supplied[c]->nComp()==1);
        out[c]->setVal(0.);MF::Copy(*out[c],*supplied[c],0,0,1,0);
    }
    SyncPMC(sim,out);ValidateNativePECWallState(sim);
    auto const q=sim.m_fields.get_alldirs("diagnostic_Je_endpoint_fp",0);
    for(auto const* name:{"hybrid_Je_n_nodal","hybrid_Je_nm1_nodal","hybrid_Je_theta_nodal"})
        CopyNativeEndpointCurrentMirror(sim,q,*sim.m_fields.get(name,0));
}
}
