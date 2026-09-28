/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/ThermalFixedPointAcceleration.H"
#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Print.H>

int main (int argc, char** argv)
{
    amrex::Initialize(argc,argv);
    {
        amrex::BoxArray boxes(amrex::Box(amrex::IntVect(0),amrex::IntVect(15)));
        boxes.maxSize(8);
        amrex::DistributionMapping ranks(boxes);
        amrex::MultiFab temperature(boxes,ranks,1,1), pressure(boxes,ranks,1,1);
        temperature.setVal(200.); pressure.setVal(2000.);
        ThermalFixedPointAcceleration inverse;
        for (int iteration = 0; iteration < 3; ++iteration) {
            inverse.Capture(temperature);
            for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi) {
                auto t = temperature.array(mfi), p = pressure.array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    t(i,j,k) = 100.-.25*(t(i,j,k)-100.);
                    p(i,j,k) = 10.*t(i,j,k);
                });
            }
            auto const omega = inverse.Apply(temperature,pressure);
            AMREX_ALWAYS_ASSERT(std::isfinite(omega) && omega >= .1 && omega <= 2.);
        }
        temperature.plus(-100.,0,1,0); pressure.plus(-1000.,0,1,0);
        AMREX_ALWAYS_ASSERT(temperature.norminf() < 1.e-11);
        AMREX_ALWAYS_ASSERT(pressure.norminf() < 1.e-10);
        amrex::Print() << "THERMAL_FIXED_POINT PASS secant_solution constitutive_pressure\n";
    }
    amrex::Finalize();
}
