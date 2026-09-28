/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_SOURCE_GATE_NATIVE_INERTIA_INVENTORY_H_
#define WARPX_SOURCE_GATE_NATIVE_INERTIA_INVENTORY_H_
#include "BoundaryConditions/WarpX_PEC.H"
#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include <ablastr/coarsen/sample.H>
#include <AMReX_Reduce.H>
#include <array>
#include <limits>

// Fixture observer only. The native accepted-stage hook precedes endpoint
// extrapolation/OU/history rotation. Empty regular intervals are essential:
// this observer must not request the velocity synchronization used by diags.
// Every copy, projection and reduction below acts on private storage.
class NativeInertiaInventory final : public ReducedDiags {
    using Real=amrex::Real;
    using FieldType=warpx::fields::FieldType;
    using Direction=ablastr::fields::Direction;
    WarpX& m_w;
    HybridPICModel& m_model;
    amrex::Geometry m_g;
    amrex::MultiFab m_j0,m_jtheta,m_js,m_ei,m_a0,m_atheta,m_a1;
    std::array<amrex::MultiFab,3> m_jp,m_ji,m_raw,m_projected,m_el0,m_el_origin;
    Real m_theta=.5,m_dt=0;
    bool m_enabled=false,m_open=false;
    int m_captures=0;

public: // NVCC extended lambdas require public enclosing methods.
    // Native geometric trapezoid quadrature (not charge-deposition volume)
    // and independent physical clipped dual volumes use their own ownership.
    struct Measure {
        QdsmcVolumeElement clipped;
        amrex::IntVect nodal,small,big;
        amrex::GpuArray<Real,AMREX_SPACEDIM> dx,lo;
        amrex::IntVect domain_lo;
        bool physical=false;
        AMREX_GPU_HOST_DEVICE Real operator()(int i,int j,int k,int owner) const {
            if(physical) return owner?clipped(i,j,k):0.;
            Real const r=lo[0]+(i-domain_lo[0]+.5*(1-nodal[0]))*dx[0];
            Real v=r==0.?MathConst::pi*dx[0]/4.:2*MathConst::pi*r;
            if(r!=0. && nodal[0] && i==small[0]) v*=.5;
            if(nodal[0] && i==big[0]) v*=.5;
            if(nodal[1] && j==small[1]) v*=.5;
            if(nodal[1] && j==big[1]) v*=.5;
            return v*dx[0]*dx[1];
        }
    };
    Measure Weights(amrex::MultiFab const& f,amrex::Box const& b,bool clipped) const {
        return {MakeQdsmcVolumeElement(m_g,f.ixType()),f.ixType().toIntVect(),
            b.smallEnd(),b.bigEnd(),m_g.CellSizeArray(),m_g.ProbLoArray(),
            m_g.Domain().smallEnd(),clipped};
    }
    static void Copy(amrex::MultiFab& to,amrex::MultiFab const& from,
                     int component=0,int count=-1) {
        if(count<0) count=from.nComp();
        if(!to.isDefined()) to.define(from.boxArray(),from.DistributionMap(),count,from.nGrowVect());
        AMREX_ALWAYS_ASSERT(to.nComp()==count && to.boxArray()==from.boxArray());
        amrex::MultiFab::Copy(to,from,component,0,count,to.nGrowVect());
    }
    // Read the already-resolved pedestal registry directly. DensityPedestal()
    // can lazily refill accepted state and is deliberately not called here.
    void Coefficient(amrex::MultiFab& out,int rho_component) {
        auto const& rho=*m_w.m_fields.get(FieldType::rho_fp,0);
        if(!out.isDefined()) out.define(rho.boxArray(),rho.DistributionMap(),1,0);
        bool const use_ped=m_model.m_density_pedestal;
        AMREX_ALWAYS_ASSERT(!use_ped || !m_model.m_density_pedestal_stale);
        auto const* pedestal=use_ped?m_w.m_fields.get("hybrid_rho_pedestal_fp",0):nullptr;
        Real const floor=PhysConst::q_e*m_model.m_n_floor;
        Real const width=m_model.m_n_floor_smooth_width*floor;
        Real const taper=m_model.m_electron_inertia_floor_taper*floor;
        Real const mass_over_e=m_model.m_electron_inertia_mass/PhysConst::q_e;
        for(amrex::MFIter it(out);it.isValid();++it) {
            auto const a=out.array(it);auto const r=rho.const_array(it);
            auto const p=use_ped?pedestal->const_array(it):amrex::Array4<Real const>{};
            amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real const raw=r(i,j,k,rho_component);
                Real const gate=taper>0?.5*(1+std::tanh((raw-floor)/taper)):1.;
                a(i,j,k)=raw>0?gate*mass_over_e/
                    HybridSmoothFloor(raw+(use_ped?p(i,j,k):0.),floor,width):0.;
            });
        }
    }
    explicit NativeInertiaInventory(WarpX& w):ReducedDiags("source_gate_inertia"),
        m_w(w),m_model(*w.get_pointer_HybridPICModel()),m_g(m_model.ElectronThermalGeometry()) {
        m_intervals=ablastr::utils::text::IntervalsParser{};
        m_enabled=m_model.m_include_electron_inertia;
        amrex::ParmParse("implicit_evolve").query("theta",m_theta);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_g.IsRZ() && w.finestLevel()==0 && !EB::enabled()
            && !WarpX::do_single_precision_comms,
            "Native inertia fixture requires single-level RZ without EB, double-precision communication");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_enabled ||
            (m_model.m_electron_inertia_djedt_only && !m_model.m_electron_inertia_bdf2 &&
             !m_model.m_esolve_curlcurl && !m_model.m_esolve_tensor &&
             !m_model.m_electron_inertia_extrapolated_history),
            "Native inertia fixture requires dJe-only two-point E-form with default measured Je history");
        AMREX_ALWAYS_ASSERT(m_theta>0 && m_theta<=1 && !DoDiags(0));
    }
    void ComputeDiags(int) override {}
    void WriteToFile(int) const override {}
    void Begin() {
        AMREX_ALWAYS_ASSERT(!m_open);
        m_open=true;m_captures=0;m_dt=m_w.getdt(0);
        if(!m_enabled) return;
        AMREX_ALWAYS_ASSERT(m_model.m_inertia_history_initialized && m_dt>0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_model.m_electron_inertia_mass==PhysConst::m_e,
            "Native inertia fixture uses the physical electron mass, no mass scaling");
        Copy(m_j0,*m_w.m_fields.get("hybrid_Je_n_nodal",0));
        Coefficient(m_a0,0);
        for(int d=0;d<3;++d) Copy(m_el0[d],*m_w.m_fields.get("hybrid_E_long_fp",Direction{d},0));
    }
    void ComputeDiagsMidStep(int) override {
        AMREX_ALWAYS_ASSERT(m_open && m_captures==0);++m_captures;
        if(!m_enabled) return;
        Copy(m_jtheta,*m_w.m_fields.get("hybrid_Je_theta_nodal",0));
        Copy(m_ei,*m_w.m_fields.get("hybrid_E_inertial_nodal",0));
        auto const& rho=*m_w.m_fields.get(FieldType::rho_fp,0);
        Coefficient(m_atheta,rho.nComp()/2);
        for(int d=0;d<3;++d) {
            Copy(m_jp[d],*m_w.m_fields.get(FieldType::hybrid_current_fp_plasma,Direction{d},0));
            Copy(m_ji[d],*m_w.m_fields.get(FieldType::current_fp,Direction{d},0));
            Copy(m_el_origin[d],*m_w.m_fields.get("hybrid_E_long_old_fp",Direction{d},0));
        }
    }
    void PreparePrivateProjection() {
        ablastr::fields::VectorField projected;
        for(int d=0;d<3;++d) {
            auto const& native=*m_w.m_fields.get(FieldType::Efield_fp,Direction{d},0);
            if(!m_raw[d].isDefined()) {
                m_raw[d].define(native.boxArray(),native.DistributionMap(),1,native.nGrowVect());
                m_projected[d].define(native.boxArray(),native.DistributionMap(),1,native.nGrowVect());
            }
            m_raw[d].setVal(0.);
            auto const iv=native.ixType().toIntVect();
            // The inactive RZ dimension must have the same (nodal) flag.
            amrex::GpuArray<int,3> const src{1,1,1},dst{iv[0],iv[1],1},ratio{1,1,1};
            for(amrex::MFIter it(m_raw[d]);it.isValid();++it) {
                auto const out=m_raw[d].array(it);auto const ei=m_ei.const_array(it);
                amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    out(i,j,k)=ablastr::coarsen::sample::Interp(ei,src,dst,ratio,i,j,k,d);
                });
            }
            Copy(m_projected[d],m_raw[d]);projected[d]=&m_projected[d];
        }
        auto sync=[&] {
            amrex::Vector<amrex::MultiFab*> fields(projected.begin(),projected.end());
            amrex::FillBoundaryAndSync_nowait(fields,m_g.periodicity());
            amrex::FillBoundaryAndSync_finish(fields);
        };
        sync(); // Populate tangential ghosts before physical-corner parity.
        PEC::ApplyPECtoEfield(projected,m_w.GetFieldBoundaryLo(),m_w.GetFieldBoundaryHi(),
            FieldBoundaryType::PEC,m_w.get_ng_fieldgather(),m_w.Geom(0),0,PatchType::fine,{});
        PEC::ApplyPECtoBfield(projected,m_w.GetFieldBoundaryLo(),m_w.GetFieldBoundaryHi(),
            FieldBoundaryType::PMC,m_w.get_ng_fieldgather(),m_w.Geom(0),0,PatchType::fine,{});
        m_w.ApplyFieldBoundaryOnAxis(projected[0],projected[1],projected[2],0);
        sync();
    }
    void Nodal(bool physical) const {
        using Op=amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
            amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
            amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
            amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum>;
        Op op;amrex::ReduceData<Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real> data(op);
        auto owner=m_j0.OwnerMask(m_g.periodicity());Real const dt=m_dt,theta=m_theta;
        for(amrex::MFIter it(m_j0);it.isValid();++it) {
            auto const j0=m_j0.const_array(it),jt=m_jtheta.const_array(it),js=m_js.const_array(it);
            auto const ei=m_ei.const_array(it),a0=m_a0.const_array(it),at=m_atheta.const_array(it),a1=m_a1.const_array(it);
            auto const own=owner->const_array(it);auto const weight=Weights(m_j0,it.validbox(),physical);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real work=0,delta=0,temporal=0,coefficient=0,history=0,k0=0,k1=0;
                Real absolute_residual=0,budget=0,law_error=0,law_scale=0,history_abs=0,coefficient_abs=0;
                for(int d=0;d<3;++d) {
                    Real const old=j0(i,j,k,d),stage=jt(i,j,k,d),stored=js(i,j,k,d);
                    Real const virtual_end=(stage-(1-theta)*old)/theta;
                    Real const A=at(i,j,k),A0=a0(i,j,k),A1=a1(i,j,k);
                    Real const W=dt*stage*ei(i,j,k,d);
                    Real const K0=.5*A0*old*old,K1=.5*A1*stored*stored;
                    Real const DK=.25*(A1+A0)*(stored-old)*(stored+old)
                                  +.25*(A1-A0)*(stored*stored+old*old);
                    Real const T=(theta-.5)*A*(virtual_end-old)*(virtual_end-old);
                    Real const C=.5*(A-A1)*stored*stored+.5*(A0-A)*old*old;
                    Real const H=.5*A*(virtual_end-stored)*(virtual_end+stored);
                    work+=W;delta+=DK;temporal+=T;coefficient+=C;history+=H;k0+=K0;k1+=K1;
                    absolute_residual+=std::abs(W-DK-T-C-H);
                    budget+=std::abs(W)+K0+K1+std::abs(T)+std::abs(C)+std::abs(H);
                    law_error+=dt*std::abs(stage)*std::abs(ei(i,j,k,d)-A*(virtual_end-old)/dt);
                    law_scale+=std::abs(W)+std::abs(stage*A*virtual_end)+std::abs(stage*A*old);
                    history_abs+=std::abs(H);coefficient_abs+=std::abs(C);
                }
                Real const v=weight(i,j,k,own(i,j,k));
                return amrex::GpuTuple<Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real>{
                    v*work,v*delta,v*temporal,v*coefficient,v*history,v*k0,v*k1,v*absolute_residual,
                    v*budget,v*law_error,v*law_scale,v*history_abs,v*coefficient_abs,v,1.};
            });
        }
        auto const v=data.value();
        std::array<Real,15> r{amrex::get<0>(v),amrex::get<1>(v),amrex::get<2>(v),amrex::get<3>(v),
            amrex::get<4>(v),amrex::get<5>(v),amrex::get<6>(v),amrex::get<7>(v),amrex::get<8>(v),
            amrex::get<9>(v),amrex::get<10>(v),amrex::get<11>(v),amrex::get<12>(v),amrex::get<13>(v),amrex::get<14>(v)};
        amrex::ParallelDescriptor::ReduceRealSum(r.data(),int(r.size()));
        amrex::Print()<<"SOURCE_GATE_INERTIA_NODAL measure="<<(physical?"clipped":"native")
            <<" step="<<m_w.getistep(0)<<" work="<<r[0]<<" storage_delta="<<r[1]
            <<" theta_term="<<r[2]<<" coefficient_term="<<r[3]<<" history_term="<<r[4]
            <<" storage_old="<<r[5]<<" storage_new="<<r[6]<<" identity_l1="<<r[7]
            <<" identity_scale="<<r[8]<<" field_law_l1="<<r[9]<<" field_law_scale="<<r[10]
            <<" history_abs="<<r[11]<<" coefficient_abs="<<r[12]
            <<" volume="<<r[13]<<" entries="<<r[14]<<"\n";
    }
    void Edge(bool physical) const {
        using Op=amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
            amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum>;
        std::array<Real,6> r{};Real const dt=m_dt;
        for(int d=0;d<3;++d) {
            Op op;amrex::ReduceData<Real,Real,Real,Real,Real,Real> data(op);
            auto owner=m_raw[d].OwnerMask(m_g.periodicity());
            for(amrex::MFIter it(m_raw[d]);it.isValid();++it) {
                auto const raw=m_raw[d].const_array(it),projected=m_projected[d].const_array(it);
                auto const jp=m_jp[d].const_array(it),ji=m_ji[d].const_array(it);
                auto const el0=m_el0[d].const_array(it),elo=m_el_origin[d].const_array(it);
                auto const own=owner->const_array(it);auto const weight=Weights(m_raw[d],it.validbox(),physical);
                op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    Real const v=weight(i,j,k,own(i,j,k));
                    Real const je=jp(i,j,k)-ji(i,j,k);
                    Real const wr=dt*je*raw(i,j,k),wp=dt*je*projected(i,j,k);
                    Real const de=.5*PhysConst::epsilon_0*(elo(i,j,k)-el0(i,j,k))*(elo(i,j,k)+el0(i,j,k));
                    return amrex::GpuTuple<Real,Real,Real,Real,Real,Real>{v*wr,v*wp,
                        v*dt*je*(projected(i,j,k)-raw(i,j,k)),v*std::abs(wr),v*std::abs(wp),v*de};
                });
            }
            auto const v=data.value();r[0]+=amrex::get<0>(v);r[1]+=amrex::get<1>(v);
            r[2]+=amrex::get<2>(v);r[3]+=amrex::get<3>(v);r[4]+=amrex::get<4>(v);r[5]+=amrex::get<5>(v);
        }
        amrex::ParallelDescriptor::ReduceRealSum(r.data(),int(r.size()));
        amrex::Print()<<"SOURCE_GATE_INERTIA_EDGE measure="<<(physical?"clipped":"native")
            <<" step="<<m_w.getistep(0)<<" raw_work="<<r[0]<<" projected_work="<<r[1]
            <<" projection_delta="<<r[2]<<" raw_abs="<<r[3]<<" projected_abs="<<r[4]
            <<" EL_entry_jump="<<r[5]<<"\n";
    }
    void CurrentMap() const {
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum> op;
        amrex::ReduceData<Real,Real> data(op);
        auto owner=m_jtheta.OwnerMask(m_g.periodicity());
        for(int d=0;d<3;++d) {
            auto const iv=m_jp[d].ixType().toIntVect();
            amrex::GpuArray<int,3> const src{iv[0],iv[1],1},dst{1,1,1},ratio{1,1,1};
            for(amrex::MFIter it(m_jtheta);it.isValid();++it) {
                auto const jp=m_jp[d].const_array(it),ji=m_ji[d].const_array(it);
                auto const jt=m_jtheta.const_array(it);auto const own=owner->const_array(it);
                auto const weight=Weights(m_jtheta,it.validbox(),true);
                op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    Real const expected=ablastr::coarsen::sample::Interp(jp,src,dst,ratio,i,j,k,0)
                        -ablastr::coarsen::sample::Interp(ji,src,dst,ratio,i,j,k,0);
                    Real const v=weight(i,j,k,own(i,j,k));
                    return amrex::GpuTuple<Real,Real>{v*std::abs(jt(i,j,k,d)-expected),
                        v*(std::abs(expected)+std::abs(jt(i,j,k,d)))};
                });
            }
        }
        auto const v=data.value();Real error=amrex::get<0>(v),scale=amrex::get<1>(v);
        amrex::ParallelDescriptor::ReduceRealSum({error,scale});
        amrex::Print()<<"SOURCE_GATE_INERTIA_CURRENT_MAP step="<<m_w.getistep(0)
            <<" error="<<error<<" scale="<<scale<<"\n";
    }
    void End() {
        AMREX_ALWAYS_ASSERT(m_open && m_captures==1 && !DoDiags(m_w.getistep(0)));
        amrex::Print()<<"SOURCE_GATE_INERTIA_CONTEXT step="<<m_w.getistep(0)
            <<" enabled="<<int(m_enabled)<<" captures="<<m_captures<<" theta="<<m_theta
            <<" dt="<<m_dt<<" mass="<<(m_enabled?m_model.m_electron_inertia_mass:0.)
            <<" regular_intervals_active=0 measured_history=1"
            <<" density_floor="<<m_model.m_n_floor<<" pedestal="<<int(m_model.m_density_pedestal)
            <<" floor_width="<<m_model.m_n_floor_smooth_width
            <<" inertia_taper="<<m_model.m_electron_inertia_floor_taper<<"\n";
        if(m_enabled) {
            Copy(m_js,*m_w.m_fields.get("hybrid_Je_n_nodal",0));Coefficient(m_a1,0);
            CurrentMap();PreparePrivateProjection();
            Nodal(false);Edge(false);Nodal(true);Edge(true);
        }
        m_open=false;
    }
};
#endif
