/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_RZ_YEE_GREEN_SOLVER_H_
#define WARPX_DARWIN_RZ_YEE_GREEN_SOLVER_H_

#include <array>
#include <memory>

namespace amrex {
class Geometry;
class MultiFab;
}

/** Cached positive inverse of the full, unmasked Yee Hodge completion in RZ.
 *
 * Solves (curl^* curl + grad grad^*) E = rhs on the native staggered electric
 * grid, with a regular axis, PEC outer radial wall and PMC axial caps. The
 * gradient here includes every free scalar node; it is not the collapsed
 * plasma/vacuum gradient. A masked Hodge inverse needs a separate interface
 * correction. This object alone must not be used as that exact masked inverse.
 *
 * Radial factors use the native Yee curl/divergence metrics, including the
 * axis and outer normal-electric row. Axial even nodal and odd cell-centered
 * extensions impose the physical PMC parities without physical periodicity.
 * All Nyquist modes are retained. No node/edge averaging is used.
 *
 * Requires single-level no-EB RZ (level/EB checked by the caller), AMReX FFT
 * and GPU-aware MPI for multiple GPU ranks. Recreate when geometry changes.
 * No runtime parameter or production preconditioner selects this helper yet.
 */
class DarwinRZYeeGreenSolver
{
public:
    // Public type permits NVCC extended device lambdas; implementation stays private.
    struct Impl;

    DarwinRZYeeGreenSolver (amrex::Geometry const& geometry, bool pmc_zlo, bool pmc_zhi);
    ~DarwinRZYeeGreenSolver ();
    DarwinRZYeeGreenSolver (DarwinRZYeeGreenSolver const&) = delete;
    DarwinRZYeeGreenSolver& operator= (DarwinRZYeeGreenSolver const&) = delete;

    void SolveComponent (amrex::MultiFab& solution, amrex::MultiFab const& rhs, int component);

private:
    std::array<std::unique_ptr<Impl>, 3> m_components;
};
#endif
