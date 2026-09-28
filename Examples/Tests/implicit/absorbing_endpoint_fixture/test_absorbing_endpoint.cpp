/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ImplicitAbsorbingEndpoint.H"
#include "Utils/WarpXConst.H"
#include "Particles/ParticleBoundaries_K.H"
#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Parser.H>
#include <AMReX_Random.H>
#include <limits>
#include <vector>
using namespace warpx::implicit;
using Status=AbsorbingEndpointStatus;
using Point=amrex::GpuArray<amrex::ParticleReal,3>;
struct Case { Point x,u; AbsorbingEndpointBoundary b; int expected; unsigned faces; bool compare_native; unsigned reflected=0; };
int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
        amrex::Parser zero("0"),fraction("0.5"),signed_cap("if(u>0,0,1)");
        zero.registerVariables({"u"}); fraction.registerVariables({"u"}); signed_cap.registerVariables({"u"});
        auto z=zero.compile<1>(),f=fraction.compile<1>(),signed_reflection=signed_cap.compile<1>();
        ParticleBoundaries::ParticleBoundariesData native{};
        native.xmin_bc=native.ymin_bc=ParticleBoundaryType::Reflecting;
        native.xmax_bc=native.ymax_bc=ParticleBoundaryType::Reflecting;
        native.zmin_bc=native.zmax_bc=ParticleBoundaryType::Absorbing;
        native.reflection_model_xlo=native.reflection_model_xhi=z;
        native.reflection_model_ylo=native.reflection_model_yhi=z;
        native.reflection_model_zlo=native.reflection_model_zhi=z;
        native.reflect_all_velocities=false;
        amrex::Box box(amrex::IntVect(0),amrex::IntVect(15));
        amrex::RealBox real(AMREX_D_DECL(0.,0.,0.),AMREX_D_DECL(1.,1.,1.));
        int periodic[AMREX_SPACEDIM]{};
        amrex::Geometry geom(box,&real,0,periodic);
        amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> low,high;
        low.fill(FieldBoundaryType::PEC); high.fill(FieldBoundaryType::PEC);
        low[AMREX_SPACEDIM-1]=high[AMREX_SPACEDIM-1]=FieldBoundaryType::PMC;
        auto b=MakeAbsorbingEndpointBoundary(geom,native,low,high);
        std::vector<Case> cases;
        auto add=[&](Point x,AbsorbingEndpointBoundary bc,Status expected,unsigned faces=0,bool native_check=false,unsigned reflected=0) {
            cases.push_back({x,{.2,-.3,.4},bc,int(expected),faces,native_check,reflected});
        };
        unsigned const lower=1u<<(2*(AMREX_SPACEDIM-1)),upper=lower<<1;
        add({.2,.13,.5},b,Status::Interior,0,true);
        add({.2,.13,-.01},b,Status::Absorbed,lower,true);
        add({.2,.13,1.01},b,Status::Absorbed,upper,true);
        add({.2,.13,0.},b,Status::Interior,0,true);
        add({.2,.13,1.},b,Status::Unsupported);
        add({.2,.13,std::nextafter(1.,0.)},b,Status::Interior,0,true);
        add({.2,.13,std::nextafter(1.,2.)},b,Status::Absorbed,upper,true);
        auto probabilistic=b; probabilistic.native.reflection_model_zhi=f;
        add({.2,.13,1.01},probabilistic,Status::Unsupported);
        auto thermal=b; thermal.native.zmax_bc=ParticleBoundaryType::Thermal;
        add({.2,.13,1.01},thermal,Status::Unsupported);
        auto reflecting=b; reflecting.native.zmax_bc=ParticleBoundaryType::Reflecting;
        add({.2,.13,1.01},reflecting,Status::Unsupported);
        auto open=probabilistic; open.native.zmax_bc=ParticleBoundaryType::Open;
        add({.2,.13,1.01},open,Status::Absorbed,upper,true);
        auto disallowed=b; disallowed.qualified_faces=0;
        add({.2,.13,1.01},disallowed,Status::Unsupported);
        auto per=b; per.native.zmin_bc=per.native.zmax_bc=ParticleBoundaryType::Periodic;
        add({.2,.13,1.01},per,Status::Interior,0,true);
        add({1.1,.0,.5},b,Status::Unsupported);
        add({1.1,.0,1.01},b,Status::Unsupported);
        add({.2,.13,std::numeric_limits<double>::quiet_NaN()},b,Status::Nonfinite);
        auto bad=cases.front();bad.u[1]=std::numeric_limits<double>::infinity();bad.expected=int(Status::Nonfinite);bad.compare_native=false;cases.push_back(bad);
#ifdef WARPX_DIM_RZ
        auto radial=MakeAbsorbingEndpointBoundary(geom,native,low,high,true);
        Point const outside{1.1*std::cos(.7),1.1*std::sin(.7),.5};
        add(outside,radial,Status::Interior,0,true,2);
        auto corner=outside;corner[2]=1.01;
        add(corner,radial,Status::Absorbed,upper,true,2);
        auto all=radial;all.native.reflect_all_velocities=true;
        add(outside,all,Status::Interior,0,true,2);
        add(corner,all,Status::Absorbed,upper,true,2);
        // Native loss classification precedes reflect_all_velocities. Reversing
        // uz first would change this deterministic absorbing cap into reflection.
        auto signed_corner=all;signed_corner.native.reflection_model_zhi=signed_reflection;
        add(corner,signed_corner,Status::Absorbed,upper,true,2);
        add({2.1,0,.5},radial,Status::Unsupported);
        add({1.,0,.5},radial,Status::Unsupported);
        auto wrong_field=high;wrong_field[0]=FieldBoundaryType::PMC;
        auto unsupported_field=MakeAbsorbingEndpointBoundary(geom,native,low,wrong_field,true);
        add(outside,unsupported_field,Status::Unsupported);
        auto thermal_radial=radial;thermal_radial.native.xmax_bc=ParticleBoundaryType::Thermal;
        add(outside,thermal_radial,Status::Unsupported);
        auto stochastic_corner=radial;stochastic_corner.native.reflection_model_zhi=f;
        add(corner,stochastic_corner,Status::Unsupported);
#endif
        amrex::Gpu::DeviceVector<Case> input(cases.size());
        amrex::Gpu::DeviceVector<int> errors(cases.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,cases.begin(),cases.end(),input.begin());
        auto const* in=input.data();auto* out=errors.data();
        amrex::ParallelFor(cases.size(),[=] AMREX_GPU_DEVICE(amrex::Long i) {
            auto const& c=in[i];auto const r=InspectAbsorbingEndpoint(c.x,c.u,c.b);
            out[i]=(int(r.status)!=c.expected || r.faces!=c.faces || r.reflected_faces!=c.reflected);
#ifdef WARPX_DIM_RZ
            if(c.compare_native && c.reflected) {
                auto const image=MapRadialEndpoint(c.x,c.u,c.b.low[0],c.b.high[0],c.b.native,true);
                auto x=c.x,u=c.u;bool lost=false;
                x={std::sqrt(x[0]*x[0]+x[1]*x[1]),std::atan2(x[1],x[0]),x[2]};
#ifdef AMREX_USE_GPU
                amrex::RandomEngine const unused_engine{nullptr};
#else
                amrex::RandomEngine const unused_engine{};
#endif
                ApplyParticleBoundaries::apply_boundaries(x[0],x[1],x[2],{0.,0.,0.},{1.,1.,1.},
                    u[0],u[1],u[2],lost,c.b.native,unused_engine);
                out[i]|=!image.valid || !image.reflected;
                for(int d=0;d<3;++d)out[i]|=image.stored[d]!=x[d] || image.momentum[d]!=u[d];
                auto const again=MapRadialEndpoint(c.x,c.u,c.b.low[0],c.b.high[0],c.b.native,true);
                for(int d=0;d<3;++d)out[i]|=again.stored[d]!=image.stored[d] || again.momentum[d]!=image.momentum[d];
            }
#endif
        });
        std::vector<int> failures(cases.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,errors.begin(),errors.end(),failures.begin());
        for (std::size_t i=0;i<failures.size();++i) {
            if(failures[i]) { amrex::Print()<<"classifier case failed "<<i<<"\n"; }
            AMREX_ALWAYS_ASSERT(failures[i]==0);
        }
#ifndef AMREX_USE_GPU
        // Independent native BC comparator, including strict endpoint equality.
        // Qualified zero-reflection/open cases consume no RNG.
        amrex::ResetRandomSeed(711);auto expected_random=amrex::Random();amrex::ResetRandomSeed(711);
        for(auto const& c:cases) if(c.compare_native) {
            auto x=c.x,u=c.u;bool lost=false;
#ifdef WARPX_DIM_RZ
            x={std::sqrt(x[0]*x[0]+x[1]*x[1]),std::atan2(x[1],x[0]),x[2]};
#endif
            ApplyParticleBoundaries::apply_boundaries(x[0],x[1],x[2],{0.,0.,0.},{1.,1.,1.},u[0],u[1],u[2],lost,c.b.native,amrex::RandomEngine{});
            AMREX_ALWAYS_ASSERT(lost==(c.expected==int(Status::Absorbed)));
            if(!c.reflected) {
                AMREX_ALWAYS_ASSERT(u[0]==c.u[0] && u[1]==c.u[1] && u[2]==c.u[2]);
            } else {
                auto const image=MapRadialEndpoint(c.x,c.u,c.b.low[0],c.b.high[0],c.b.native,true);
                for(int d=0;d<3;++d)AMREX_ALWAYS_ASSERT(image.stored[d]==x[d] && image.momentum[d]==u[d]);
                double before=0,after=0;
                for(int d=0;d<3;++d){before+=c.u[d]*c.u[d];after+=u[d]*u[d];}
                AMREX_ALWAYS_ASSERT(std::abs(after-before)<=8*std::numeric_limits<double>::epsilon()*before);
            }
        }
        AMREX_ALWAYS_ASSERT(amrex::Random()==expected_random);
#endif
        amrex::Print()<<"PASS deterministic absorption classifier cases="<<cases.size()<<"; native strict-boundary convention, unsupported actions and finite-state guards\n";
    }
    amrex::Finalize();
}
