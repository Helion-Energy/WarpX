/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_RZ_GREEN_SOLVER_H_
#define WARPX_DARWIN_RZ_GREEN_SOLVER_H_

#include <memory>

namespace amrex {
class Geometry;
class MultiFab;
} // namespace amrex

/** Cached inverse of the bounded nodal RZ vector-Poisson correction.
 *
 * The r/theta inverse solves (L_r + D_zz - 1/r^2) u = rhs, with zero
 * correction at the axis and outer radius. Each z end can be Dirichlet
 * (PEC) or Neumann (PMC tangential response). Optional all-component setup
 * also caches the z inverse, L_r + D_zz: regular Neumann axis, Dirichlet
 * outer radius and both z ends, including at a PMC normal face.
 * Odd/even extensions implement the corresponding finite-interval transform;
 * they do not impose periodic physical boundaries.
 * The full radial coupling is retained in cached per-mode factors.
 * Source and replacement masks belong to the caller, outside this inverse.
 * Recreate this object when geometry changes. Input/output box layouts may
 * vary: ParallelCopy redistributes through fixed device-resident pencils.
 * Requires RZ, no EB (checked by the caller), and AMReX FFT support.
 */
class DarwinRZGreenSolver
{
  public:
    // NVCC requires the type containing extended device lambdas to be public.
    // Its definition and all cached state remain in the implementation file.
    struct Impl;

    /** The auxiliary electric inverse uses the native Hodge PC parity:
     * radial Neumann at the outer PEC wall; z is Neumann at PEC caps
     * and Dirichlet at PMC caps. Recovery keeps its established A parity. */
    enum class BoundaryFamily { VectorPotentialRecovery, ElectricAuxiliary };

    explicit DarwinRZGreenSolver (amrex::Geometry const &geometry, bool pmc_zlo = false,
                                  bool pmc_zhi = false);
    DarwinRZGreenSolver (amrex::Geometry const &geometry, bool pmc_zlo, bool pmc_zhi,
                         bool all_components);
    DarwinRZGreenSolver (amrex::Geometry const &geometry, bool pmc_zlo, bool pmc_zhi,
                         bool all_components, BoundaryFamily boundary_family);
    ~DarwinRZGreenSolver ();
    DarwinRZGreenSolver (DarwinRZGreenSolver const&) = delete;
    DarwinRZGreenSolver& operator=(DarwinRZGreenSolver const&) = delete;

    // Established flux-only interface; identical to component 1.
    void Solve (amrex::MultiFab& solution, amrex::MultiFab const& rhs);
    void SolveComponent (amrex::MultiFab& solution, amrex::MultiFab const& rhs, int component);

  private:
    std::unique_ptr<Impl> m_impl;
};
#endif
