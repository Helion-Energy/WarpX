/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/NativeStoppingReactionTranspose.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "FieldSolver/ImplicitSolvers/ThermalRandomCheckpoint.H"
#include "Fields.H"
#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

using namespace warpx::thermal;
using amrex::Real;
using ablastr::fields::Direction;
using warpx::fields::FieldType;
namespace {
constexpr int N=16;
constexpr Real nref=1.e19;
constexpr Real velocity_scale=30000.;
constexpr Real eps=std::numeric_limits<Real>::epsilon();
// A priori operand bounds, including reciprocal/harmonic mass arithmetic,
// cubic polynomial and coordinate arithmetic, tensor accumulation and RZ
// rotations. Maximum stencil: 64 terms/component. No reduction tolerance.
constexpr Real mass_bound=128.*eps*velocity_scale;
constexpr Real gather_bound=2048.*eps*velocity_scale;

amrex::IntVect type (int c) {
    amrex::IntVect v(1);
#if defined(WARPX_DIM_RZ)
    if(c!=1) { v[c/2]=0; }
#else
    v[c]=0;
#endif
    return v;
}
AMREX_GPU_HOST_DEVICE int canonical (int i) { return (i%N+N)%N; }
AMREX_GPU_HOST_DEVICE Real density (int i,int j,int k,bool periodic) {
#if defined(WARPX_DIM_RZ)
    if(periodic) { j=canonical(j); }
#else
    if(periodic) { i=canonical(i); j=canonical(j); k=canonical(k); }
#endif
    return PhysConst::q_e*nref*(2.+.125*i/N+.0625*j/N+.03125*k/N);
}
AMREX_GPU_HOST_DEVICE Real ped (int i,int j,int k,bool periodic) {
#if defined(WARPX_DIM_RZ)
    if(periodic) { j=canonical(j); }
#else
    if(periodic) { i=canonical(i); j=canonical(j); k=canonical(k); }
#endif
    return PhysConst::q_e*nref*(.25+.03125*i/N+.015625*j/N+.0078125*k/N);
}
AMREX_GPU_HOST_DEVICE Real analytic (int c,int i,int j,int k,bool periodic) {
#if defined(WARPX_DIM_RZ)
    if(periodic) { j=canonical(j); }
#else
    if(periodic) { i=canonical(i); j=canonical(j); k=canonical(k); }
#endif
    Real const a=(i+.5)/N,b=(j+.5)/N,d=(k+.5)/N;
#if defined(WARPX_DIM_RZ)
    if(c==1) { return 4000.*i/N*(1.+.125*b); }
#endif
    return (c+1)*2500.*(1.+.25*a+.125*b+.0625*d);
}
std::vector<char> bytes (std::vector<amrex::MultiFab const*> const& fields) {
    amrex::Gpu::synchronize(); std::vector<char> out;
    for(auto const* mf:fields) {
        for(amrex::MFIter mfi(*mf);mfi.isValid();++mfi) {
            auto const& fab=(*mf)[mfi]; auto const n=fab.size()*sizeof(Real);
            auto const p=out.size();out.resize(p+n);
            amrex::Gpu::dtoh_memcpy(out.data()+p,fab.dataPtr(),n);
        }
    }
    return out;
}
void restore (std::vector<amrex::MultiFab const*> const& fields,std::vector<char> const& data) {
    std::size_t offset=0;
    for(auto const* mf:fields) {
        for(amrex::MFIter mfi(*mf);mfi.isValid();++mfi) {
            auto& fab=(*const_cast<amrex::MultiFab*>(mf))[mfi];auto const n=fab.size()*sizeof(Real);
            AMREX_ALWAYS_ASSERT(offset+n<=data.size());
            amrex::Gpu::htod_memcpy(fab.dataPtr(),data.data()+offset,n);offset+=n;
        }
    }
    AMREX_ALWAYS_ASSERT(offset==data.size());amrex::Gpu::synchronize();
}
std::vector<char> rng () {
    amrex::Gpu::synchronize();std::ostringstream stream;amrex::SaveRandomState(stream);
    auto const s=stream.str();std::vector<char> out(s.begin(),s.end());
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    auto const n=ThermalRandomDeviceBytes(),p=out.size();out.resize(p+n);
    amrex::Gpu::dtoh_memcpy(out.data()+p,amrex::getRandState(),n);
#endif
    return out;
}
struct Fixture {
    amrex::Geometry geometry;
    amrex::BoxArray cells;
    amrex::DistributionMapping dm;
    ablastr::fields::MultiFabRegister fields;
    std::vector<amrex::MultiFab const*> inputs;
    ablastr::fields::ConstVectorField magnetic{};
    AcceptedStoppingOptions options;
    bool periodic=false,pedestal=false;
    void Build (int grid,bool zperiodic,bool with_pedestal) {
        periodic=zperiodic;pedestal=with_pedestal;
        amrex::Box domain(amrex::IntVect(0),amrex::IntVect(N-1));
        amrex::RealBox real({AMREX_D_DECL(0.,0.,0.)},{AMREX_D_DECL(1.,1.,1.)});
        int periodicity[AMREX_SPACEDIM]={AMREX_D_DECL(0,0,0)};
        periodicity[AMREX_SPACEDIM-1]=int(periodic);
#if defined(WARPX_DIM_3D)
        periodicity[0]=periodicity[1]=int(periodic);
#endif
#if defined(WARPX_DIM_RZ)
        geometry.define(domain,&real,1,periodicity);
#else
        geometry.define(domain,&real,0,periodicity);
#endif
        cells=amrex::BoxArray(domain);cells.maxSize(grid);dm=amrex::DistributionMapping(cells);
        options.number_density_floor=.1*nref;options.reference_number_density=nref;
        for(int d=0;d<AMREX_SPACEDIM;++d) {
            options.field_lo[d]=options.field_hi[d]=FieldBoundaryType::PMC;
            options.particle_lo[d]=options.particle_hi[d]=ParticleBoundaryType::Absorbing;
        }
#if defined(WARPX_DIM_RZ)
        options.field_hi[0]=FieldBoundaryType::PEC;
        options.particle_lo[0]=ParticleBoundaryType::None;
        options.particle_hi[0]=ParticleBoundaryType::Reflecting;
#endif
        auto const nodes=amrex::convert(cells,amrex::IntVect(1));
        auto* rho=fields.alloc_init(FieldType::rho_fp,0,nodes,dm,2,amrex::IntVect(3),123.);
        auto* pedestal_field=fields.alloc_init("hybrid_rho_pedestal_fp",0,nodes,dm,1,amrex::IntVect(3),456.);
        inputs={rho,pedestal_field};
        bool const per=periodic;
        for(amrex::MFIter mfi(*rho);mfi.isValid();++mfi) {
            auto const r=rho->array(mfi),pe=pedestal_field->array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                r(i,j,k,0)=density(i,j,k,per);r(i,j,k,1)=3.*density(i,j,k,per);
                pe(i,j,k)=ped(i,j,k,per);
            });
        }
        for(int c=0;c<3;++c) {
            auto const ba=amrex::convert(cells,type(c));
            auto* current=fields.alloc_init(NativeAcceptedStoppingContext::AcceptedCurrentName,Direction{c},0,ba,dm,1,amrex::IntVect(3),12345.);
            auto* stale=fields.alloc_init(FieldType::hybrid_current_fp_plasma,Direction{c},0,ba,dm,1,amrex::IntVect(3),67890.);
            magnetic[c]=fields.alloc_init(FieldType::Bfield_fp,Direction{c},0,amrex::convert(cells,amrex::IntVect(1)-type(c)),dm,1,amrex::IntVect(3),0.);
            inputs.push_back(current);inputs.push_back(stale);inputs.push_back(magnetic[c]);
            auto const nd=type(c);bool const pp=pedestal;
            for(amrex::MFIter mfi(*current);mfi.isValid();++mfi) {
                auto const a=current->array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    Real den=density(i,j,k,per)+(pp?ped(i,j,k,per):0.);
                    int const di=1-nd[0],dj=1-nd[1];
#if defined(WARPX_DIM_3D)
                    int const dk=1-nd[2];
#else
                    int const dk=0;
#endif
                    if(di+dj+dk) { den=.5*(den+density(i+di,j+dj,k+dk,per)+(pp?ped(i+di,j+dj,k+dk,per):0.)); }
                    a(i,j,k)=-den*analytic(c,i,j,k,per);
                });
            }
        }
    }
};
// Independent long-double cardinal spline; the oracle does not call the native
// shape routine, shared image-index helper, transpose or native mass map.
long double spline (long double x) {
    x=std::abs(x);
    if(x<1) { return (4-6*x*x+3*x*x*x)/6; }
    if(x<2) { return (2-x)*(2-x)*(2-x)/6; }
    return 0;
}
constexpr int oracle_size=AMREX_D_TERM((N+1),*(N+1),*(N+1));
int flat (amrex::IntVect const& iv) {
    return AMREX_D_TERM(iv[0],+(N+1)*iv[1],+(N+1)*(N+1)*iv[2]);
}
std::vector<StoppingIonImpulse> particles (int mode) {
    std::vector<StoppingIonImpulse> result;
    Real const points[5]={.125,3.875,4.125,12.125,15.875};
    for(int n=0;n<15;++n) {
        Real const r=points[n%5]/N,z=points[(n+2)%5]/N;
        for(int ring=0;ring<4;++ring) {
            StoppingIonImpulse t;
            Real const co[4]={n<10?1.:.6,n<10?0.:-.8,n<10?-1.:-.6,n<10?0.:.8};
            Real const si[4]={n<10?0.:.8,n<10?1.:.6,n<10?0.:-.8,n<10?-1.:-.6};
#if defined(WARPX_DIM_RZ)
            t.position={r*co[ring],r*si[ring],z};
            t.impulse={(2+n*.125)*co[ring]-(1+n*.25)*si[ring],
                       (2+n*.125)*si[ring]+(1+n*.25)*co[ring],1+n*.125};
#else
            amrex::ignore_unused(co,si);
            t.position={r,points[(n+1)%5]/N,z};
            t.impulse={2+n*.125,-1-n*.25,1+n*.125};
#endif
            for(int c=0;c<3;++c) { t.impulse[c]*=1.e-22; }
            t.weight=1.e7*(1+n*.125);
            if(mode==1 && ring!=0) { continue; } // RZ missing transverse mode control
            if(mode==2) { t.impulse={0,0,0}; }
            result.push_back(t);
        }
    }
#if defined(WARPX_DIM_RZ)
    StoppingIonImpulse axis;
    axis.position={0,0,.125/N};axis.impulse={0,0,mode==2?0.:2.e-22};axis.weight=1.e7;
    result.push_back(axis);
#endif
    return result;
}
amrex::IntVect particle_cell (StoppingIonImpulse const& p) {
#if defined(WARPX_DIM_RZ)
    Real const r=std::sqrt(p.position[0]*p.position[0]+p.position[1]*p.position[1]);
    return {std::min(N-1,int(r*N)),std::min(N-1,int(p.position[2]*N))};
#else
    return {std::min(N-1,int(p.position[0]*N)),std::min(N-1,int(p.position[1]*N)),
            std::min(N-1,int(p.position[2]*N))};
#endif
}
using Oracle=std::array<std::vector<long double>,3>;
Oracle deposit_oracle (std::vector<StoppingIonImpulse> const& pp,bool periodic) {
    Oracle out;for(auto& a:out) { a.resize(oracle_size,0.); }
    for(auto const& p:pp) {
        std::array<long double,3> impulse{p.impulse[0],p.impulse[1],p.impulse[2]};
        amrex::GpuArray<long double,AMREX_SPACEDIM> q{};
#if defined(WARPX_DIM_RZ)
        long double const x=p.position[0],y=p.position[1],r=std::sqrt(x*x+y*y);
        q={N*r,N*static_cast<long double>(p.position[2])};
        long double const co=r>0?x/r:1,si=r>0?y/r:0;
        impulse={co*p.impulse[0]+si*p.impulse[1],co*p.impulse[1]-si*p.impulse[0],p.impulse[2]};
#else
        q={N*static_cast<long double>(p.position[0]),N*static_cast<long double>(p.position[1]),
           N*static_cast<long double>(p.position[2])};
#endif
        for(int c=0;c<3;++c) {
            auto const nd=type(c);amrex::IntVect lo,hi;
            for(int d=0;d<AMREX_SPACEDIM;++d) {
                lo[d]=int(std::floor(q[d]-.5L*(1-nd[d])))-1;hi[d]=lo[d]+3;
            }
            for(amrex::BoxIterator it(amrex::Box(lo,hi));it.ok();++it) {
                auto index=it();long double weight=p.weight;
                for(int d=0;d<AMREX_SPACEDIM;++d) {
                    weight*=spline(q[d]-index[d]-.5L*(1-nd[d]));
#if defined(WARPX_DIM_RZ)
                    bool const per=periodic && d==1,normal=c==2*d;
#else
                    bool const per=periodic,normal=c==d;
#endif
                    if(per) { index[d]=canonical(index[d]);continue; }
                    int const end=N-1+nd[d];
                    if(index[d]<0) {
                        index[d]=-index[d]-(1-nd[d]);
#if defined(WARPX_DIM_RZ)
                        weight*=d==0?(c==2?1:-1):(normal?-1:1);
#else
                        weight*=normal?-1:1;
#endif
                    } else if(index[d]>end) { index[d]=2*end-index[d]+1-nd[d];weight*=normal?-1:1; }
                }
                out[c][flat(index)]+=weight*impulse[c];
            }
        }
    }
    return out;
}
std::vector<amrex::MultiFab const*> outputs (NativeStoppingReactionTranspose const& t) {
    std::vector<amrex::MultiFab const*> result;
    for(auto const fields:{t.ConjugateMomentumDensity(),t.ElectronCurrentImpulse(),t.InertiaImpulse()}) {
        for(auto const* f:fields) { result.push_back(f); }
    }
    return result;
}
struct Metrics {
    Real particle_work=0.,velocity_work=0.,current_work=0.,coefficient_error=0.,volume_error=0.;
    amrex::GpuArray<Real,3> integrated_pi{};
};
Metrics check (Fixture const& f,NativeAcceptedStoppingContext const& context,
               NativeStoppingReactionTranspose const& transpose,Oracle const& oracle,
               Real impulse_bound,Real work_bound,int corrupt) {
    Metrics m;auto const pi=transpose.ConjugateMomentumDensity();auto const b=transpose.InertiaImpulse();
    auto const velocity=context.Velocity(),current=context.Binding().electron_current;
    for(int c=0;c<3;++c) {
        auto const volume=MakeQdsmcVolumeElement(f.geometry,pi[c]->ixType());
        auto owner=pi[c]->OwnerMask(f.geometry.periodicity());
        long double sum=0,vwork=0,jwork=0;
        for(amrex::MFIter mfi(*pi[c]);mfi.isValid();++mfi) {
            auto const& pf=(*pi[c])[mfi];amrex::FArrayBox ph(pf.box(),1,amrex::The_Pinned_Arena());
            amrex::FArrayBox vh(pf.box(),1,amrex::The_Pinned_Arena()),jh(pf.box(),1,amrex::The_Pinned_Arena()),bh(pf.box(),1,amrex::The_Pinned_Arena());
            amrex::Gpu::dtoh_memcpy(ph.dataPtr(),pf.dataPtr(),pf.size()*sizeof(Real));
            amrex::Gpu::dtoh_memcpy(vh.dataPtr(),(*velocity[c])[mfi].dataPtr(),pf.size()*sizeof(Real));
            amrex::Gpu::dtoh_memcpy(jh.dataPtr(),(*current[c])[mfi].dataPtr(),pf.size()*sizeof(Real));
            amrex::Gpu::dtoh_memcpy(bh.dataPtr(),(*b[c])[mfi].dataPtr(),pf.size()*sizeof(Real));
            auto const& of=(*owner)[mfi];amrex::IArrayBox oh(of.box(),1,amrex::The_Pinned_Arena());
            amrex::Gpu::dtoh_memcpy(oh.dataPtr(),of.dataPtr(),of.size()*sizeof(int));
            auto const p=ph.array();auto const v=vh.const_array(),j=jh.const_array(),bi=bh.const_array();auto const o=oh.const_array();
            for(amrex::BoxIterator it(mfi.validbox());it.ok();++it) {
                auto index=it(),canonical_index=index;
                for(int d=0;d<AMREX_SPACEDIM;++d) { if(f.geometry.isPeriodic(d)) { canonical_index[d]=canonical(index[d]); } }
                auto const xyz=amrex::lbound(amrex::Box(index,index));auto const vol=volume(xyz.x,xyz.y,xyz.z);
                long double expected=oracle[c][flat(canonical_index)];
                long double got=static_cast<long double>(p(index))*vol;
                // Faults are injected into the independent observation only.
                if(corrupt==2 && c==0 && expected==0 && index==amrex::IntVect(8)) { got+=2*impulse_bound; }
                if(corrupt==4 && c==2 && std::abs(expected)>impulse_bound) { got+=2*impulse_bound; }
                auto const error=static_cast<Real>(std::abs(got-expected));
                m.coefficient_error=std::max(m.coefficient_error,error);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error<=impulse_bound,"transpose independent coefficient bound");
                if(!o(index)) { continue; }
                sum+=got;vwork+=got*v(index);jwork+=static_cast<long double>(vol)*j(index)*bi(index);
                long double independent=1;
                for(int d=0;d<AMREX_SPACEDIM;++d) {
                    long double const pos=(index[d]+.5L*(1-type(c)[d]))/N;
                    long double const low=f.geometry.isPeriodic(d)?pos-.5L/N:std::max(0.L,pos-.5L/N);
                    long double const high=f.geometry.isPeriodic(d)?pos+.5L/N:std::min(1.L,pos+.5L/N);
#if defined(WARPX_DIM_RZ)
                    if(d==0) { independent*=static_cast<long double>(MathConst::pi)*(high*high-low*low); }
                    else
#endif
                    { independent*=high-low; }
                }
                m.volume_error=std::max(m.volume_error,static_cast<Real>(std::abs(vol-independent)));
                AMREX_ALWAYS_ASSERT(std::abs(vol-independent)<=16*eps*independent);
            }
        }
        m.integrated_pi[c]=static_cast<Real>(sum);m.velocity_work+=static_cast<Real>(vwork);m.current_work+=static_cast<Real>(jwork);
    }
    amrex::ParallelDescriptor::ReduceRealSum(m.integrated_pi.data(),3);
    amrex::ParallelDescriptor::ReduceRealSum(m.velocity_work);amrex::ParallelDescriptor::ReduceRealSum(m.current_work);
    amrex::ParallelDescriptor::ReduceRealMax(m.coefficient_error);amrex::ParallelDescriptor::ReduceRealMax(m.volume_error);
    if(corrupt==1) { m.velocity_work+=2*work_bound; }
    return m;
}
Real deposit (Fixture const& f,NativeAcceptedStoppingContext const& context,
              NativeStoppingReactionTranspose& transpose,std::vector<StoppingIonImpulse> const& points,
              int invalid=0,Real* collision_work=nullptr,bool corrupt_tuple=false) {
    AMREX_ALWAYS_ASSERT(transpose.Begin());long double particle_work=0,frame_work=0;
    for(amrex::MFIter mfi(*context.Velocity()[0]);mfi.isValid();++mfi) {
        std::vector<StoppingIonImpulse> local;
        for(auto const& p:points) { if(f.cells[mfi.index()].contains(particle_cell(p))) { local.push_back(p); } }
        if(local.empty()) { transpose.DepositTile(mfi,nullptr,0);continue; }
        if(invalid==1) { local[0].weight=-1; }
        if(invalid==2) { local[0].position[2]=42.; }
        if(invalid==3) { local[0].impulse[0]=std::numeric_limits<Real>::quiet_NaN(); }
        if(invalid==4) { local[0].weight=1.e300;local[0].impulse[0]=1.e300; }
        amrex::Gpu::DeviceVector<StoppingIonImpulse> device(local.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,local.begin(),local.end(),device.begin());
        auto const tuple_bytes=local;
        transpose.DepositTile(mfi,device.data(),device.size());
        amrex::Gpu::DeviceVector<StoppingGatherResult> values(local.size());
        auto* value=values.data();auto const* tuple=device.data();auto const view=context.GatherView(mfi,f.magnetic);
        amrex::ParallelFor(static_cast<int>(local.size()),[=] AMREX_GPU_DEVICE(int i) {
            value[i]=view(tuple[i].position[0],tuple[i].position[1],tuple[i].position[2]);
        });
        std::vector<StoppingGatherResult> hvalue(local.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,values.begin(),values.end(),hvalue.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,device.begin(),device.end(),local.begin());
        if(corrupt_tuple) { local[0].impulse[0]+=1.; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::memcmp(local.data(),tuple_bytes.data(),local.size()*sizeof(StoppingIonImpulse))==0,"transpose exact impulse tuple purity");
        if(!invalid) {
            for(std::size_t p=0;p<local.size();++p) {
                AMREX_ALWAYS_ASSERT(hvalue[p].valid);
                auto impulse=local[p].impulse;
#if defined(WARPX_DIM_RZ)
                auto const xp=local[p].position[0],yp=local[p].position[1];
                auto const radius=std::sqrt(xp*xp+yp*yp),co=radius>0.?xp/radius:1.,si=radius>0.?yp/radius:0.;
                impulse={co*local[p].impulse[0]+si*local[p].impulse[1],co*local[p].impulse[1]-si*local[p].impulse[0],local[p].impulse[2]};
#endif
                for(int c=0;c<3;++c) {
                    particle_work+=static_cast<long double>(local[p].weight)*local[p].impulse[c]*hvalue[p].cartesian[c];
                    frame_work+=static_cast<long double>(local[p].weight)*impulse[c]*hvalue[p].collision_frame[c];
                }
            }
        }
    }
    if(invalid) { AMREX_ALWAYS_ASSERT(!transpose.Finish() && !transpose.Valid());return 0; }
    AMREX_ALWAYS_ASSERT(transpose.Finish());
    Real result=static_cast<Real>(particle_work);amrex::ParallelDescriptor::ReduceRealSum(result);
    if(collision_work) { *collision_work=static_cast<Real>(frame_work);amrex::ParallelDescriptor::ReduceRealSum(*collision_work); }
    return result;
}
void Run () {
    amrex::ParmParse pp;int grid=16,periodic=1,pedestal=1,mode=0,corrupt=0;
    pp.query("grid",grid);pp.query("periodic",periodic);pp.query("pedestal",pedestal);pp.query("mode",mode);pp.query("corrupt",corrupt);
    Fixture f;f.Build(grid,periodic,pedestal);
    NativeAcceptedStoppingContext context(f.geometry,f.cells,f.dm,f.options);
    auto const binding=NativeAcceptedStoppingContext::Bind(f.fields,f.pedestal,1.e-9,17);
    AMREX_ALWAYS_ASSERT(context.Prepare(f.fields,binding));
    NativeStoppingReactionTranspose transpose(context);
    auto const before=bytes(f.inputs),random=rng();
    auto const prepared=bytes({context.Velocity()[0],context.Velocity()[1],context.Velocity()[2],&context.Kappa()});
    auto const points=particles(mode);auto const oracle=deposit_oracle(points,periodic);
    long double absolute=0;std::array<long double,3> total{};
    for(auto const& p:points) { for(int c=0;c<3;++c) { total[c]+=static_cast<long double>(p.weight)*p.impulse[c];absolute+=std::abs(static_cast<long double>(p.weight)*p.impulse[c]); } }
    // Count all possible scatter terms, all allocated FAB entries, MPI reduction
    // entries and <=1024 arithmetic/basis operations. Both independently rounded
    // products are covered. Using u=epsilon (twice unit roundoff) is conservative.
    std::uint64_t count=1024+2*points.size()*3*64+2*amrex::ParallelDescriptor::NProcs();
    for(int ibox=0;ibox<f.cells.size();++ibox) { auto const& box=f.cells[ibox]; for(int c=0;c<3;++c) { count+=2*amrex::grow(amrex::convert(box,type(c)),f.options.ghosts).numPts(); } }
    auto const infinity=std::numeric_limits<Real>::infinity();
    auto const upper=[] (long double x) {
        return x==0?Real(0):std::nextafter(static_cast<Real>(x),std::numeric_limits<Real>::infinity());
    };
    long double const depth=static_cast<long double>(count)*eps;
    Real const gamma=upper(depth/(1-depth));
    // Include the host positive-product/sum roundoff before the outward cast.
    long double const source_sum_error=6*points.size()*std::numeric_limits<long double>::epsilon();
    Real const positive_upper=upper(absolute/(1-source_sum_error));
    AMREX_ALWAYS_ASSERT(count*eps<.01 && std::isfinite(static_cast<Real>(absolute)));
    Real const impulse_bound=upper(2.L*gamma*positive_upper);
    Real const work_bound=upper(2.L*velocity_scale*impulse_bound);
    AMREX_ALWAYS_ASSERT(std::isfinite(work_bound) && work_bound<infinity &&
        (absolute==0 || impulse_bound>=std::numeric_limits<Real>::min()));
    for(auto const* v:context.Velocity()) { AMREX_ALWAYS_ASSERT(v->norm0(0,1,v->nGrowVect(),false)<=velocity_scale); }
    Real frame_work=0.;
    Real const particle_work=deposit(f,context,transpose,points,0,&frame_work,corrupt==3);
    AMREX_ALWAYS_ASSERT(std::abs(frame_work-particle_work)<=work_bound);
    auto metric=check(f,context,transpose,oracle,impulse_bound,work_bound,corrupt);
    // Diagnostic of the transpose RESTRICTED to regular m0 velocity DOFs.
    // This does not mutate the sealed generic transpose or any accepted input.
    auto const pi=transpose.ConjugateMomentumDensity();
    auto const mass_impulse=transpose.InertiaImpulse();
    Real removed_axis=0.,axis_work=0.,axis_current=0.;
    auto owner=pi[1]->OwnerMask(f.geometry.periodicity());
    auto const dual=MakeQdsmcVolumeElement(f.geometry,pi[1]->ixType());
    for(amrex::MFIter mfi(*pi[1]);mfi.isValid();++mfi){
        amrex::FArrayBox p((*pi[1])[mfi].box(),1,amrex::The_Pinned_Arena());
        amrex::FArrayBox v((*context.Velocity()[1])[mfi].box(),1,amrex::The_Pinned_Arena());
        amrex::FArrayBox r((*mass_impulse[1])[mfi].box(),1,amrex::The_Pinned_Arena());
        amrex::IArrayBox o((*owner)[mfi].box(),1,amrex::The_Pinned_Arena());
        amrex::Gpu::dtoh_memcpy(p.dataPtr(),(*pi[1])[mfi].dataPtr(),p.size()*sizeof(Real));
        amrex::Gpu::dtoh_memcpy(v.dataPtr(),(*context.Velocity()[1])[mfi].dataPtr(),v.size()*sizeof(Real));
        amrex::Gpu::dtoh_memcpy(r.dataPtr(),(*mass_impulse[1])[mfi].dataPtr(),r.size()*sizeof(Real));
        amrex::Gpu::dtoh_memcpy(o.dataPtr(),(*owner)[mfi].dataPtr(),o.size()*sizeof(int));
        auto const pp=p.const_array(),vv=v.const_array(),rr=r.const_array();auto const oo=o.const_array();
        auto const box=mfi.validbox();
        if(box.smallEnd(0)==0){for(int j=box.smallEnd(1);j<=box.bigEnd(1);++j){if(oo(0,j,0)){
            removed_axis+=pp(0,j,0)*dual(0,j,0);
            axis_work+=pp(0,j,0)*vv(0,j,0)*dual(0,j,0);
            axis_current=std::max(axis_current,std::abs(rr(0,j,0)));
            AMREX_ALWAYS_ASSERT(vv(0,j,0)==0.);
        }}}
    }
    amrex::ParallelDescriptor::ReduceRealSum(removed_axis);amrex::ParallelDescriptor::ReduceRealSum(axis_work);
    amrex::ParallelDescriptor::ReduceRealMax(axis_current);
    AMREX_ALWAYS_ASSERT(axis_work==0.);
    if(mode!=2){AMREX_ALWAYS_ASSERT(std::abs(removed_axis)>100*impulse_bound && axis_current>0.);}
    else{AMREX_ALWAYS_ASSERT(removed_axis==0. && axis_current==0.);}
    amrex::Print()<<std::setprecision(17)<<"RZ_AXIS_REACTION restricted_axis_work="<<axis_work
        <<" removed_conjugate_impulse="<<removed_axis<<" unrestricted_mass_impulse_axis="<<axis_current
        <<" unrestricted_regular_current_rejected="<<(mode!=2)<<" impulse_bound="<<impulse_bound<<"\n";
    auto const balance=transpose.Balance();auto const A=bytes(outputs(transpose));
    auto const verr=std::abs(metric.velocity_work-particle_work),jerr=std::abs(metric.current_work+particle_work);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(verr<=work_bound,"transpose native gather work bound");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(jerr<=work_bound,"transpose accepted current mass work bound");
    for(int c=0;c<3;++c) {
        AMREX_ALWAYS_ASSERT(std::abs(balance.particle_cartesian[c]-total[c])<=impulse_bound);
        AMREX_ALWAYS_ASSERT(std::abs(balance.conjugate_integral[c]-metric.integrated_pi[c])<=impulse_bound);
#if defined(WARPX_DIM_RZ)
        if(periodic && (c==2 || mode!=1)) {
#else
        if(periodic) {
#endif
            AMREX_ALWAYS_ASSERT(std::abs(balance.unrepresented_cartesian[c])<=impulse_bound);
        }
    }
#if defined(WARPX_DIM_RZ)
    if(mode==1) { AMREX_ALWAYS_ASSERT(std::abs(balance.unrepresented_cartesian[0])>100*impulse_bound); }
#endif
    if(!periodic && mode!=2) { AMREX_ALWAYS_ASSERT(std::abs(balance.unrepresented_cartesian[2])>100*impulse_bound); }
    auto changed=points;for(auto& p:changed) { for(int c=0;c<3;++c) { p.impulse[c]*=-2.; } }
    deposit(f,context,transpose,changed);deposit(f,context,transpose,points);
#ifndef AMREX_USE_GPU
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bytes(outputs(transpose))==A,"CPU exact transpose A/B/A");
#else
    auto const repeat=check(f,context,transpose,oracle,impulse_bound,work_bound,0);
    AMREX_ALWAYS_ASSERT(std::abs(repeat.velocity_work-particle_work)<=work_bound);
    AMREX_ALWAYS_ASSERT(std::abs(repeat.current_work+particle_work)<=work_bound);
#endif
    for(int invalid=1;invalid<=4;++invalid) { deposit(f,context,transpose,points,invalid); }
    AMREX_ALWAYS_ASSERT(transpose.Begin());context.Invalidate();AMREX_ALWAYS_ASSERT(!transpose.Finish());
    AMREX_ALWAYS_ASSERT(context.Prepare(f.fields,binding));
    AMREX_ALWAYS_ASSERT(transpose.Begin());AMREX_ALWAYS_ASSERT(context.Prepare(f.fields,binding));
    AMREX_ALWAYS_ASSERT(!transpose.Finish()); // same epoch but new generation
    deposit(f,context,transpose,particles(2));
    for(auto const* field:outputs(transpose)) { AMREX_ALWAYS_ASSERT(field->norm0(0,1,field->nGrowVect(),false)==0.); }
    if(corrupt==5) { f.fields.get(FieldType::rho_fp,0)->plus(1.,0,1,0); }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bytes(f.inputs)==before && rng()==random,"transpose exact input/RNG purity");
    AMREX_ALWAYS_ASSERT(bytes({context.Velocity()[0],context.Velocity()[1],context.Velocity()[2],&context.Kappa()})==prepared);
    amrex::Print()<<std::setprecision(17)<<"STOPPING_TRANSPOSE PASS grid="<<grid<<" periodic="<<periodic<<" pedestal="<<pedestal<<" mode="<<mode
        <<" ranks="<<amrex::ParallelDescriptor::NProcs()<<" tuples="<<points.size()<<" particle_work="<<particle_work
        <<" velocity_work="<<metric.velocity_work<<" current_work="<<metric.current_work<<" velocity_error="<<verr<<" current_error="<<jerr<<" collision_frame_work="<<frame_work<<" collision_frame_error="<<std::abs(frame_work-particle_work)
        <<" work_bound="<<work_bound<<" coefficient_error="<<metric.coefficient_error<<" impulse_bound="<<impulse_bound
        <<" operation_depth="<<count<<" positive_impulse="<<static_cast<Real>(absolute)<<" positive_impulse_upper="<<positive_upper<<" volume_error="<<metric.volume_error
        <<" image_impulse_x="<<balance.unrepresented_cartesian[0]<<" image_impulse_y="<<balance.unrepresented_cartesian[1]
        <<" image_impulse_z="<<balance.unrepresented_cartesian[2]<<" exact_input_rng=1 stale_rejected=1 malformed_rejected=4\n";
}
}
int main (int argc,char** argv) { amrex::Initialize(argc,argv);{Run();}amrex::Finalize(); }
