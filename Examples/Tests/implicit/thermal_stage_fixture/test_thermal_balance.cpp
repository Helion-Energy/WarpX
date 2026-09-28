/* Copyright 2026 The WarpX Community; License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ThermalEnergyBalance.H"
#include "Utils/WarpXConst.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MFIter.H>
#include <cmath>
using namespace warpx::thermal;
using amrex::Real;
void near(Real a,Real b) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(a) && std::abs(a-b)<1.e-11,
                                    "Physical energy receipt differs from analytic balance");
}
int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
        int max_grid=4; amrex::ParmParse("test").query("max_grid_size",max_grid);
        amrex::Box domain({0,0},{15,31});amrex::BoxArray cells(domain);cells.maxSize(max_grid);
        amrex::DistributionMapping dm(cells);amrex::RealBox box({0.,0.},{1.,2.});
        int periodic[2]={0,0};amrex::Geometry geom(domain,&box,1,periodic);
        amrex::MultiFab old(cells,dm,1,0),n(cells,dm,1,0),b(cells,dm,3,0),q(cells,dm,1,0),u(cells,dm,1,0),r(cells,dm,1,0);
        old.setVal(10);n.setVal(1.e18);b.setVal(0);b.setVal(1,2,1);q.setVal(3);u.setVal(10.15);
        ThermalStageOptions o;o.dt=.1;o.order=2;
        int callbacks=0;
        EulerianThermalStage uniform(geom,old,n,b,q,o);
        uniform.SetSourceEvaluator([&](auto const&,auto const&,auto const&,auto& source,auto& derivative) {
            ++callbacks;source.setVal(3);derivative.setVal(0);return true;
        });
        AMREX_ALWAYS_ASSERT(uniform.Residual(u,r));
        int calls=callbacks;
        Real const volume=2*MathConst::pi;
        auto a=MeasureThermalEnergyBalance(uniform,u);
        near(a.initial,10*volume);near(a.final,10.3*volume);near(a.source,.3*volume);
        near(a.defect,0);near(a.absolute_defect,0);
        auto repeated=MeasureThermalEnergyBalance(uniform,u);
        near(repeated.final,a.final);AMREX_ALWAYS_ASSERT(callbacks==calls);
        // Known outward flux through r=1 and both caps. A deliberately
        // unconverged unchanged U must report its nonzero missing heat.
        u.setVal(10);q.setVal(0);
        o.boundary[0][1]={BoundaryKind::PrescribedFlux,2};
        o.boundary[1][0]={BoundaryKind::PrescribedFlux,1};
        o.boundary[1][1]={BoundaryKind::PrescribedFlux,3};
        EulerianThermalStage wall(geom,old,n,b,q,o);
        AMREX_ALWAYS_ASSERT(wall.Residual(u,r));a=MeasureThermalEnergyBalance(wall,u);
        near(a.conduction,-.1*(2*4*MathConst::pi+4*MathConst::pi));
        near(a.defect,-a.conduction);near(a.absolute_defect,a.defect);
        // Linear outward velocity, constant U. Independent analytic surface
        // enthalpy and pressure work exactly balance the supplied heating.
        o.boundary={};Real const gradient=.25;
        q.setVal(o.gamma*10*gradient);
        std::array<amrex::MultiFab,2> velocity;
        ThermalFaceContext faces;
        for(int d=0;d<2;++d) {
            velocity[d].define(amrex::convert(cells,amrex::IntVect::TheDimensionVector(d)),dm,1,0);
            velocity[d].setVal(0);faces.velocity[d]=&velocity[d];
        }
        for(amrex::MFIter mfi(velocity[1]);mfi.isValid();++mfi) {
            auto out=velocity[1].array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){out(i,j,k)=gradient*j/16.;});
        }
        EulerianThermalStage flow(geom,old,n,n,n,b,q,faces,o);
        AMREX_ALWAYS_ASSERT(flow.Residual(u,r));a=MeasureThermalEnergyBalance(flow,u);
        near(a.advection,-.1*10*gradient*volume);
        near(a.compression,-.1*(o.gamma-1)*10*gradient*volume);
        near(a.source,.1*o.gamma*10*gradient*volume);near(a.defect,0);near(a.absolute_defect,0);
        amrex::Print()<<"PASS physical energy receipt, source purity, wall surface and compression balance\n";
    }
    amrex::Finalize();
}
