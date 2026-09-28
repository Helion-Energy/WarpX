/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeVacuumJouleError.H"
#include "EulerianThermalSources.H"
#include "NativeJouleWork.H"
#include "NativeInstantaneousForce.H"
#include "NativeVacuumEndpoint.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <ablastr/coarsen/sample.H>
#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::thermal {
namespace {
using Real=amrex::Real;
using MF=amrex::MultiFab;
using View=ablastr::fields::VectorField;
using FT=warpx::fields::FieldType;
using Dir=ablastr::fields::Direction;
constexpr Real source_envelope=1.e5, potential_envelope=1000., magnetic_envelope=.2;
Real gamma(Real n) {
    Real const x=n*std::numeric_limits<Real>::epsilon()/2.;
    return x/(1.-x);
}
bool bounded(MF const& f,Real bound,int ng=0) {
    return f.is_finite(0,1,ng) && f.norminf(0,1,ng)<=bound;
}
bool CaptureEnvelope(WarpX& w, MF const& rho, std::array<MF,3>& full,
                     std::array<MF,3>& nores) {
    auto e=w.m_fields.get_alldirs(FT::Efield_fp,0);
    for(int c=0;c<3;++c)for(auto* f:{&full[c],&nores[c]})
        f->define(e[c]->boxArray(),e[c]->DistributionMap(),1,e[c]->nGrowVect());
    View f{&full[0],&full[1],&full[2]},n{&nores[0],&nores[1],&nores[2]};
    bool ok=warpx::darwin::CaptureNativeInstantaneousForce(w,rho,f,n,nullptr);
    auto const& phi=*w.m_fields.get("hybrid_phi_darwin_fp",0);
    ok=bounded(phi,potential_envelope) && ok;
    for(int c=0;c<3;++c) {
        ok=bounded(full[c],source_envelope) && ok;
        ok=bounded(*w.m_fields.get(FT::Bfield_fp,Dir{c},0),magnetic_envelope,1) && ok;
        for(auto const* name:{"hybrid_E_long_fp","hybrid_E_long_old_fp"})
            ok=bounded(*w.m_fields.get(name,Dir{c},0),source_envelope,1) && ok;
    }
    return ok;
}
}

VacuumJouleErrorBound DeriveVacuumJouleErrorBound(
    amrex::Geometry const& g,Real dt,Real theta,int old_step,
    Real rt,Real at,Real eta) {
    VacuumJouleErrorBound b;
#if defined(WARPX_DIM_RZ)
    // Deliberately bounded proof capability. Other grids/BCs need their own
    // verified operator norms and barrier, not an interpolated empirical limit.
    if(g.Domain().smallEnd()!=amrex::IntVect(0) ||
       g.Domain().length(0)!=12 || g.Domain().length(1)!=6 ||
       g.ProbLo(0)!=0. || g.ProbHi(0)!=.5 ||
       g.ProbLo(1)!=-.5 || g.ProbHi(1)!=.5 ||
       g.isPeriodic(0) || !g.isPeriodic(1) || theta!=.5 ||
       old_step<0 || old_step>8 || !std::isfinite(dt) || dt<=0. ||
       !std::isfinite(rt) || !std::isfinite(at) || rt<0. || at<0. ||
       rt>1.e-15 || at>1.e-15 || !std::isfinite(eta) || eta<0.)return b;
    Real const dr=g.CellSize(0),dz=g.CellSize(1),R=g.ProbHi(0);
    Real const G=2./std::min(dr,dz),D=4./dr+2./dz;
    Real const L=8./(dr*dr)+4./(dz*dz),inverse=G*R*R/4.;
    // The contract permits tighter solves and restart tolerance changes.
    // Use its fixed maximum, so an older accepted projection is never bounded
    // using only a newly tightened solver tolerance.
    Real const tau=std::max(Real(1.e-15),Real(1.e-15)*D*source_envelope);
    b.projection=inverse*(tau+gamma(32)*D*source_envelope+
        gamma(32)*(L*potential_envelope+D*source_envelope))+
        gamma(3)*G*potential_envelope;
    // Old endpoint EL is a theta extrapolation. At theta=.5 its absolute
    // projection error budget grows by two stage budgets per accepted step;
    // add the extrapolation's own six-operation rounding budget. This is not
    // an assumption of bitwise checkpoint/history equality or a physical error
    // estimate of the nonlinear solution. Actual empty-edge current is checked.
    Real const electric=(2.*old_step+2.)*b.projection+
        Real(old_step)*gamma(6)*3.*source_envelope+gamma(3)*2.*source_envelope;
    b.current=PhysConst::epsilon_0/(theta*dt)*electric+
        gamma(32)*D*magnetic_envelope/PhysConst::mu0;
    b.eta=eta;b.dt=dt;
    b.summation_gamma=gamma(16.*3.*13.*7.+64.);
    b.valid=std::isfinite(b.current)&&b.current>0.;
#else
    amrex::ignore_unused(g,dt,theta,old_step,rt,at,eta);
#endif
    return b;
}

VacuumJouleErrorReceipt CheckVacuumJouleError(
    amrex::Geometry const& g,VacuumJouleErrorBound const& b,
    MF const& rho,MF const& mask,Real threshold,VacuumWorkVector const& jp,
    VacuumWorkVector const& ji,VacuumWorkVector const& er,
    VacuumWorkVector const& work) {
    VacuumJouleErrorReceipt out;out.current_bound=b.current;
#if defined(WARPX_DIM_RZ)
    if(!b.valid || !std::isfinite(threshold) || threshold<=0.)return out;
    Real invalid=0.,populated=0.,not_recovery=0.,current_bad=0.,work_bad=0.;
    for(int c=0;c<3;++c) {
        AMREX_ALWAYS_ASSERT(work[c] && jp[c] && ji[c] && er[c]);
        AMREX_ALWAYS_ASSERT(work[c]->nComp()>=4 &&
            jp[c]->boxArray()==work[c]->boxArray() &&
            ji[c]->boxArray()==work[c]->boxArray() &&
            er[c]->boxArray()==work[c]->boxArray() &&
            rho.nGrowVect().allGE(amrex::IntVect(1)) &&
            mask.nGrowVect().allGE(amrex::IntVect(1)));
        AMREX_ALWAYS_ASSERT(jp[c]->nComp()==1 && ji[c]->nComp()==1 && er[c]->nComp()==1 &&
            rho.is_nodal() && mask.is_nodal() && rho.nComp()==1 && mask.nComp()==1 &&
            rho.boxArray()==mask.boxArray() &&
            rho.boxArray()==amrex::convert(work[c]->boxArray(),amrex::IntVect(1)) &&
            rho.DistributionMap()==mask.DistributionMap() &&
            rho.DistributionMap()==work[c]->DistributionMap() &&
            jp[c]->DistributionMap()==work[c]->DistributionMap() &&
            ji[c]->DistributionMap()==work[c]->DistributionMap() &&
            er[c]->DistributionMap()==work[c]->DistributionMap());
        auto owner=work[c]->OwnerMask(g.periodicity());
        auto const volume=MakeQdsmcVolumeElement(g,work[c]->ixType());
        int const di=c==0?1:0,dj=c==2?1:0;
        Real const eta=b.eta,jbound=b.current,local_gamma=gamma(12);
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
            amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,
            amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax> ops;
        amrex::ReduceData<Real,Real,Real,Real,Real,Real,Real,Real,Real> data(ops);
        using Tuple=decltype(data)::Type;
        for(amrex::MFIter mfi(*work[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
            auto q=work[c]->const_array(mfi),p=jp[c]->const_array(mfi);
            auto ion=ji[c]->const_array(mfi),electric=er[c]->const_array(mfi);
            auto r=rho.const_array(mfi),m=mask.const_array(mfi);
            auto owned=owner->const_array(mfi);
            ops.eval(mfi.tilebox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
                if(!owned(i,j,k))return {0.,0.,0.,0.,0.,0.,0.,0.,0.};
                Real const signed_q=q(i,j,k,2),abs_q=q(i,j,k,3);
                if(!std::isfinite(signed_q)||!std::isfinite(abs_q)||abs_q<0.||
                    abs_q!=std::abs(signed_q))return {0.,0.,0.,0.,1.,0.,0.,0.,0.};
                if(abs_q==0.)return {0.,0.,0.,0.,0.,0.,0.,0.,0.};
                Real const x=p(i,j,k),e=electric(i,j,k),v=volume(i,j,k);
                bool const finite=std::isfinite(x)&&std::isfinite(e)&&
                    std::isfinite(ion(i,j,k))&&std::isfinite(r(i,j,k))&&
                    std::isfinite(r(i+di,j+dj,k))&&std::isfinite(m(i,j,k))&&
                    std::isfinite(m(i+di,j+dj,k));
                bool const empty=r(i,j,k)==0.&&r(i+di,j+dj,k)==0.&&ion(i,j,k)==0.;
                bool const rec=m(i,j,k)<threshold && m(i+di,j+dj,k)<threshold;
                Real const bound=v*eta*jbound*jbound*(1.+local_gamma);
                bool const j_ok=std::abs(x)<=jbound;
                bool const work_ok=abs_q<=bound&&std::abs(e)<=eta*jbound*(1.+local_gamma);
                return {signed_q,abs_q,bound,std::abs(x),finite?0.:1.,
                        empty?0.:1.,rec?0.:1.,j_ok?0.:1.,work_ok?0.:1.};
            });
        }
        auto t=data.value();
        out.signed_orphan+=amrex::get<0>(t);out.absolute_orphan+=amrex::get<1>(t);
        out.allowance+=amrex::get<2>(t);
        out.max_current=std::max(out.max_current,amrex::get<3>(t));
        invalid=std::max(invalid,amrex::get<4>(t));populated=std::max(populated,amrex::get<5>(t));
        not_recovery=std::max(not_recovery,amrex::get<6>(t));
        current_bad=std::max(current_bad,amrex::get<7>(t));work_bad=std::max(work_bad,amrex::get<8>(t));
    }
    Real sums[3]={out.signed_orphan,out.absolute_orphan,out.allowance};
    Real maxima[6]={out.max_current,invalid,populated,not_recovery,current_bad,work_bad};
    amrex::ParallelDescriptor::ReduceRealSum(sums,3);
    amrex::ParallelDescriptor::ReduceRealMax(maxima,6);
    out.signed_orphan=b.dt*sums[0];out.absolute_orphan=b.dt*sums[1];
    out.allowance=b.dt*sums[2]*(1.+b.summation_gamma);out.max_current=maxima[0];
    out.status=maxima[1]?VacuumJouleErrorStatus::InvalidData:
        maxima[2]?VacuumJouleErrorStatus::PopulatedOrIonCurrent:
        maxima[3]?VacuumJouleErrorStatus::NotRecovery:
        maxima[4]?VacuumJouleErrorStatus::CurrentResidual:
        maxima[5]?VacuumJouleErrorStatus::WorkResidual:VacuumJouleErrorStatus::Ready;
    if(!std::isfinite(out.allowance)||!std::isfinite(out.absolute_orphan)||
       !std::isfinite(out.signed_orphan))out.status=VacuumJouleErrorStatus::InvalidData;
#else
    amrex::ignore_unused(g,rho,mask,threshold,jp,ji,er,work);
#endif
    return out;
}

bool NativeVacuumJouleErrorEnabled() {
    bool enabled=false;
    amrex::ParmParse("implicit_evolve.thermal").query("vacuum_joule_solver_error",enabled);
    return enabled;
}
void ValidateNativeVacuumJouleErrorScope(WarpX& w) {
    if(!NativeVacuumJouleErrorEnabled())return;
    auto const& m=*w.get_pointer_HybridPICModel();
    ValidateNativeJouleWork(w);
    Real theta=.5;amrex::ParmParse("implicit_evolve").query("theta",theta);
    auto parser=amrex::ParmParse().makeParser(m.m_eta_expression,{});
    auto b=DeriveVacuumJouleErrorBound(w.Geom(0),w.getdt(0),theta,w.getistep(0),
        m.m_darwin_poisson_rtol,m.m_darwin_poisson_atol,parser.compileHost<0>()());
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(b.valid&&NativeVacuumEndpointEnabled()&&
        NativeJouleWorkEnabled()&&!m.m_include_electron_viscosity&&
        !m.m_include_hyper_resistivity_term&&!m.m_has_external_current&&
        WarpX::field_boundary_lo[0]==FieldBoundaryType::None&&
        WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC,
        "Vacuum Joule numerical receipt supports only the derived 12x6 axis/PEC periodic-z, theta=.5, EL<=1e-15, first nine steps, native constant-eta work, no external current/viscosity/hyper scope");
}
void ValidateNativeVacuumJouleErrorEntry(WarpX& w) {
    if(!NativeVacuumJouleErrorEnabled())return;
    ValidateNativeVacuumJouleErrorScope(w);
    // This proof has no physical imposed current. Uniform cylindrical components
    // are insufficient: constant B_theta has curl_z=B_theta/r and constant B_r
    // is not regular at the axis. The certified RZ background is strictly axial,
    // with B_r=B_theta=0 and constant B_z; nonzero transverse fields need their
    // own discrete curl/divergence and current-work bounds, however small.
    for(int c=0;c<3;++c) {
        auto const& fixed=*w.m_fields.get("hybrid_B_static_fp",Dir{c},0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bounded(fixed,magnetic_envelope) &&
            (c==2 ? fixed.min(0,0)==fixed.max(0,0) : fixed.norminf(0,0)==0),
            "Vacuum Joule numerical receipt requires the current-free axial RZ static magnetic background (Br=Btheta=0, constant Bz)");
    }
    auto const& charge=*w.m_fields.get(FT::rho_fp,0);
    MF rho(charge,amrex::make_alias,0,1);
    std::array<MF,3> full,nores;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(CaptureEnvelope(w,rho,full,nores),
        "Vacuum Joule numerical receipt old-state source/potential/magnetic envelope failed");
}
VacuumJouleErrorReceipt PrepareNativeVacuumJouleError(
    WarpX& w,EulerianThermalSources const& sources,NativeJouleWork const& joule,
    ThermalSourceLedger const& ledger,Real dt,Real theta) {
    VacuumJouleErrorReceipt out;
    if(!NativeVacuumJouleErrorEnabled())return out;
    ValidateNativeVacuumJouleErrorScope(w);
    auto const& m=*w.get_pointer_HybridPICModel();
    auto parser=amrex::ParmParse().makeParser(m.m_eta_expression,{});
    auto const b=DeriveVacuumJouleErrorBound(w.Geom(0),dt,theta,w.getistep(0),
        m.m_darwin_poisson_rtol,m.m_darwin_poisson_atol,parser.compileHost<0>()());
    auto const& charge=*w.m_fields.get(FT::rho_fp,0);
    MF rho(charge,amrex::make_alias,charge.nComp()/2,1);
    auto const& mask=*w.m_fields.get("hybrid_rho_vacmask_fp",0);
    Real const threshold=PhysConst::q_e*m.m_n_floor*m.m_darwin_vacuum_recovery_density_fraction;
    auto p=w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0);
    auto ion=w.m_fields.get_alldirs(FT::current_fp,0);
    VacuumWorkVector jp{p[0],p[1],p[2]},ji{ion[0],ion[1],ion[2]};
    std::array<MF,3> full,nores;
    if(!CaptureEnvelope(w,rho,full,nores)) {out.status=VacuumJouleErrorStatus::Envelope;return out;}
    out=CheckVacuumJouleError(m.ElectronThermalGeometry(),b,rho,mask,threshold,
        jp,ji,joule.Electric(),sources.JouleWorkScratch());
    Real const q=ledger.energy[SourceComponent::UnassignableJouleWorkAbs];
    Real const qs=ledger.energy[SourceComponent::DeclinedJouleWork];
    Real const transfer_error=b.summation_gamma*std::max(out.absolute_orphan,std::abs(q));
    if(std::abs(q-out.absolute_orphan)>transfer_error||
       std::abs(qs-out.signed_orphan)>transfer_error)
        out.status=VacuumJouleErrorStatus::LedgerMismatch;
    // Separate physical/model account over ALL recovery rows, including
    // populated interface rows. This is not deposited as electron heat and is
    // never hidden under the numerical allowance. Use the actual unchanged
    // completion, including electric physical images, then restore all E bytes.
    auto E=w.m_fields.get_alldirs(FT::Efield_fp,0);
    std::array<MF,3> saved;
    for(int c=0;c<3;++c) {
        saved[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,E[c]->nGrowVect());
        MF::Copy(saved[c],*E[c],0,0,1,E[c]->nGrowVect());
        MF::Copy(*E[c],full[c],0,0,1,E[c]->nGrowVect());
    }
    CompleteNativeVacuumLongitudinalSource(w,theta*dt);
    // The actual longitudinal RHS uses the recovery-completed source. Its
    // bound is required independently of the raw full-Ohm envelope above.
    // Always complete restoration below, including on a failed envelope.
    for(int c=0;c<3;++c)if(!bounded(*E[c],source_envelope))
        out.status=VacuumJouleErrorStatus::Envelope;
    Real signed_work=0.,absolute_work=0.;
    for(int c=0;c<3;++c) {
        auto owner=E[c]->OwnerMask(w.Geom(0).periodicity());
        auto const volume=MakeQdsmcVolumeElement(m.ElectronThermalGeometry(),E[c]->ixType());
        auto const iv=E[c]->ixType().toIntVect();
        amrex::GpuArray<int,3> node{1,1,1},stagger{iv[0],iv[1],1},ratio{1,1,1};
#if defined(WARPX_DIM_3D)
        stagger[2]=iv[2];
#endif
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum> ops;
        amrex::ReduceData<Real,Real> data(ops);using Tuple=decltype(data)::Type;
        for(amrex::MFIter mfi(*E[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
            auto owned=owner->const_array(mfi);auto r=mask.const_array(mfi);
            auto raw=full[c].const_array(mfi),completed=E[c]->const_array(mfi);
            auto j=jp[c]->const_array(mfi),i=ji[c]->const_array(mfi);
            ops.eval(mfi.tilebox(),data,[=] AMREX_GPU_DEVICE(int x,int y,int z)->Tuple {
                if(!owned(x,y,z)||ablastr::coarsen::sample::Interp(r,node,stagger,ratio,x,y,z,0)>=threshold)
                    return {0.,0.};
                Real const work=volume(x,y,z)*(j(x,y,z)-i(x,y,z))*(raw(x,y,z)-completed(x,y,z));
                return {work,std::abs(work)};
            });
        }
        auto t=data.value();signed_work+=amrex::get<0>(t);absolute_work+=amrex::get<1>(t);
        MF::Copy(*E[c],saved[c],0,0,1,E[c]->nGrowVect());
    }
    amrex::ParallelDescriptor::ReduceRealSum(signed_work);
    amrex::ParallelDescriptor::ReduceRealSum(absolute_work);
    if(!std::isfinite(signed_work)||!std::isfinite(absolute_work))
        out.status=VacuumJouleErrorStatus::InvalidData;
    amrex::Print()<<"Vacuum Joule numerical receipt: status="<<static_cast<int>(out.status)
        <<" orphan_signed_J="<<out.signed_orphan<<" orphan_abs_J="<<out.absolute_orphan
        <<" bound_J="<<out.allowance<<" max_J_A_m2="<<out.max_current
        <<" bound_J_A_m2="<<out.current_bound<<" recovery_removed_full_signed_J="<<dt*signed_work
        <<" recovery_removed_full_abs_J="<<dt*absolute_work<<" raw_fields_unchanged=1\n";
    return out;
}
}
