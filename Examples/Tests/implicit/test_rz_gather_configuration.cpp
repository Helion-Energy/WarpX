/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Constructor-only validation: no mesh or particles are allocated with InitData.
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_Print.H>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& warpx = WarpX::GetInstance();
        amrex::ignore_unused(warpx);
        amrex::Print() << "GATHER_CONFIGURATION_ACCEPTED_BEFORE_INITDATA\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
