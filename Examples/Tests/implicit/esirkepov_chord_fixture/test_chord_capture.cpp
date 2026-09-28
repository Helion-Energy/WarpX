/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Include the actual native pusher TU to exercise its internal completed-orbit
// capture, including the maintained specular reflection. No model is mocked.
#include "Particles/Pusher/ImplicitPushPX.cpp"
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <iomanip>

// Minimal storage view consumed by the real SetParticlePosition constructor.
struct Buffer {double* p;double* dataPtr()const{return p;}};
struct Storage {
    double r=0.,z=0.,theta=0.;
    Buffer GetRealData(int c){return {c==PIdx::r?&r:c==PIdx::z?&z:&theta};}
};
struct TileView {Storage* p;Storage& GetStructOfArrays()const{return *p;}};
amrex::GpuArray<amrex::ParticleReal,3> Push(bool reflect,double dt) {
    double xp=reflect?9.9999:8.25,yp=0.,zp=8.25;
    double const xn=xp,yn=yp,zn=zp;
    double ux=reflect?2.:.3125,uy=0.,uz=reflect?0.:.625;
    double const uxn=ux,uyn=uy,uzn=uz;
    Storage storage{xp,zp,0.};SetParticlePosition<PIdx> set(TileView{&storage});
    double norm=0.,bx=0.,by=0.,bz=0.;ImplicitGatherResult result;
    amrex::GpuArray<amrex::ParticleReal,3> chord{};
    amrex::GpuArray<amrex::GpuArray<double,2>,AMREX_SPACEDIM> domain{};
    amrex::GpuArray<amrex::GpuArray<bool,2>,AMREX_SPACEDIM> crop{};
    amrex::Array4<double const> empty;amrex::IndexType cell;
    GetExternalEBField ext{};
    bool const ok=PushXPSingleStep<no_exteb,no_qed>(0,dt,set,false,
        xp,yp,zp,&ux,&uy,&uz,xn,yn,zn,uxn,uyn,uzn,norm,result,{},nullptr,chord,1.e-14,4,
        reflect,10.,nullptr,0.,0.,0.,0.,0.,0.,bx,by,bz,0,
        empty,empty,empty,empty,empty,empty,cell,cell,cell,cell,cell,cell,
        {1.,1.,1.},{0.,0.,0.},domain,crop,{0,0,0},1,3,
        CurrentDepositionAlgo::Esirkepov,ext,nullptr,1.,0.,ParticlePusherAlgo::Boris,false);
    AMREX_ALWAYS_ASSERT(ok && result);
    if(reflect) {
        double const endpoint=2.*10.-(xn+dt*uxn);
        AMREX_ALWAYS_ASSERT(std::abs(chord[0]-(endpoint-xn))<1.e-14);
        AMREX_ALWAYS_ASSERT(std::abs(chord[0]/dt-ux)>1.);
        double const uend=2*ux-uxn;
        AMREX_ALWAYS_ASSERT(std::abs(uend*uend-uxn*uxn)<1.e-14);
    } else {
        double const invgamma=GetImplicitGammaInverse(uxn,uyn,uzn,ux,uy,uz);
        AMREX_ALWAYS_ASSERT(chord[0]==dt*ux*invgamma);
        AMREX_ALWAYS_ASSERT(chord[2]==dt*uz*invgamma);
        if(dt<1.e-15)AMREX_ALWAYS_ASSERT(2*(xp-xn)==0.&&chord[0]!=0.);
    }
    return chord;
}
int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {for(double dt:{1.e-5,1.e-10,1.e-17}) {
        auto a=Push(false,dt);auto b=Push(true,.01);auto repeated=Push(false,dt);
        for(int d=0;d<3;++d)AMREX_ALWAYS_ASSERT(a[d]==repeated[d]);
        amrex::Print()<<std::setprecision(17)<<"chord h="<<dt<<" ballistic="<<a[0]<<","<<a[2]<<" reflected="<<b[0]<<"\n";
    }amrex::Print()<<"PASS CHORD_CAPTURE native pusher preaddition chord, specular endpoint energy and A/B/A\n";}
    amrex::Finalize();
}
