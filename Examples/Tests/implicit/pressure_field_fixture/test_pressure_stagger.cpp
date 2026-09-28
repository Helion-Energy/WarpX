/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "test_pressure_stagger.H"
#include <AMReX.H>
int
main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    {
        std::array<amrex::IntVect, 3> types;
        for (int c = 0; c < 3; ++c) {
            types[c] = amrex::IntVect::TheNodeVector();
#if AMREX_SPACEDIM == 2
            if (c != 1) { types[c][c/2] = 0; }
#else
            types[c][c] = 0;
#endif
        }
        TestPressureResponseStagger(types);
    }
    amrex::Finalize();
}
