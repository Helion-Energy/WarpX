/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ElectronInertiaElliptic.H"

#include "BoundaryConditions/WarpX_PEC.H"
#include "Fields.H"
// The finite-difference algorithm headers stub out the operators that a given
// build's geometry does not have, so only the matching one may be included.
#if defined(WARPX_DIM_RZ)
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include <ablastr/profiler/ProfilerWrapper.H>
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <cmath>

using namespace amrex;
using warpx::fields::FieldType;

namespace
{
    /** Number of ghost cells carried by the Krylov iterate.
     *
     * One layer is exactly what the operator reads. The intermediate curl
     * is evaluated on the valid box grown by one cell, and because a Yee
     * curl component that is nodal along a direction never differences
     * along that direction, none of those points reaches more than one
     * ghost cell of the iterate. Carrying more would leave layers that are
     * filled but never read, which is how uninitialised ghost data quietly
     * becomes a NaN source.
     */
    constexpr int ng_iterate = 1;

    /** y[d] <- y[d] + a * x[d] on the valid region, all three components. */
    void Saxpy3 (ElectronInertiaElliptic::EVec& y, Real a,
                 ElectronInertiaElliptic::EVec const& x)
    {
        for (int d = 0; d < 3; ++d) {
            MultiFab::Saxpy(y[d], a, x[d], 0, 0, 1, 0);
        }
    }

    /** dst[d] <- a * u[d] + b * v[d] on the valid region. */
    void LinComb3 (ElectronInertiaElliptic::EVec& dst,
                   Real a, ElectronInertiaElliptic::EVec const& u,
                   Real b, ElectronInertiaElliptic::EVec const& v)
    {
        for (int d = 0; d < 3; ++d) {
            MultiFab::LinComb(dst[d], a, u[d], 0, b, v[d], 0, 0, 1, 0);
        }
    }

    void Copy3 (ElectronInertiaElliptic::EVec& dst,
                ElectronInertiaElliptic::EVec const& src, int ng)
    {
        for (int d = 0; d < 3; ++d) {
            MultiFab::Copy(dst[d], src[d], 0, 0, 1, ng);
        }
    }

    void SetVal3 (ElectronInertiaElliptic::EVec& x, Real v)
    {
        for (int d = 0; d < 3; ++d) { x[d].setVal(v); }
    }

    /** Componentwise product dst[d] <- u[d] * w[d] (the Jacobi solve). */
    void Precond3 (ElectronInertiaElliptic::EVec& dst,
                   ElectronInertiaElliptic::EVec const& u,
                   ElectronInertiaElliptic::EVec const& w)
    {
        for (int d = 0; d < 3; ++d) {
            MultiFab::Copy(dst[d], u[d], 0, 0, 1, 0);
            MultiFab::Multiply(dst[d], w[d], 0, 0, 1, 0);
        }
    }

    /** True if any domain face carries the given field boundary type. */
    bool AnyFaceIs (FieldBoundaryType t)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (WarpX::field_boundary_lo[idim] == t) { return true; }
            if (WarpX::field_boundary_hi[idim] == t) { return true; }
        }
        return false;
    }

    // ----------------------------------------------------------------------
    // Batched reductions.
    //
    // Every scalar the Krylov recurrence needs is a reduction over all three
    // field components. Taking them one component at a time costs three
    // collectives where one would do, and on a GPU it also costs three
    // device-to-host synchronisations, because each device reduction ends in
    // one. That is invisible on a CPU, where the arithmetic dominates, and
    // it is most of the cost on a GPU, where the arithmetic nearly vanishes
    // and the per-reduction latency is left standing alone.
    //
    // Every helper below therefore computes per-component LOCAL values and
    // issues exactly ONE collective for the group. How the local values are
    // obtained is the one thing that differs by backend, and it differs
    // because the two backends are limited by different things:
    //
    //   CPU: MultiFab::Dot / norm0 with local = true. These are the tuned
    //        AMReX kernels and there is no synchronisation to save, so
    //        anything else is strictly slower -- measured at +8% per solve
    //        when the fused path below was used on CPU.
    //   GPU: one fused ReduceOps pass over all three components, so the
    //        group costs a single synchronisation instead of three.
    //
    // Both paths accumulate the group total on the host in component order,
    // so they agree to the last bit with each other; only the within-
    // component summation order differs from the pre-batching code.
    //
    // Nodes shared between neighbouring boxes are still counted once per
    // box, so the dot products remain the same weighted inner product as
    // before -- symmetric and positive definite, which is all the recurrence
    // requires, and used consistently for every scalar in the iteration.
    // ----------------------------------------------------------------------

#ifdef AMREX_USE_GPU

    // GPU: one fused device reduction per group, so the group costs a single
    // device-to-host synchronisation instead of one per component. The
    // MFIter runs on component 0 and asks for the other components'
    // tileboxes, which is the same per-box coverage MultiFab::Dot visits.

    Real Dot3 (ElectronInertiaElliptic::EVec const& x,
               ElectronInertiaElliptic::EVec const& y)
    {
        amrex::ReduceOps<amrex::ReduceOpSum> ops;
        amrex::ReduceData<Real> data(ops);
        for (MFIter mfi(x[0]); mfi.isValid(); ++mfi) {
            for (int d = 0; d < 3; ++d) {
                Array4<Real const> const& xa = x[d].const_array(mfi);
                Array4<Real const> const& ya = y[d].const_array(mfi);
                ops.eval(mfi.tilebox(x[d].ixType().toIntVect()), data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        -> amrex::GpuTuple<Real>
                    { return { xa(i, j, k) * ya(i, j, k) }; });
            }
        }
        Real s = amrex::get<0>(data.value(ops));
        ParallelDescriptor::ReduceRealSum(s);
        return s;
    }

    void Dot3Pair (ElectronInertiaElliptic::EVec const& a,
                   ElectronInertiaElliptic::EVec const& b,
                   ElectronInertiaElliptic::EVec const& c,
                   ElectronInertiaElliptic::EVec const& d_,
                   Real& s_ab, Real& s_cd)
    {
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
        amrex::ReduceData<Real, Real> data(ops);
        for (MFIter mfi(a[0]); mfi.isValid(); ++mfi) {
            for (int d = 0; d < 3; ++d) {
                Array4<Real const> const& aa = a[d].const_array(mfi);
                Array4<Real const> const& ba = b[d].const_array(mfi);
                Array4<Real const> const& ca = c[d].const_array(mfi);
                Array4<Real const> const& da = d_[d].const_array(mfi);
                ops.eval(mfi.tilebox(a[d].ixType().toIntVect()), data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        -> amrex::GpuTuple<Real, Real>
                    { return { aa(i, j, k) * ba(i, j, k),
                               ca(i, j, k) * da(i, j, k) }; });
            }
        }
        auto const hv = data.value(ops);
        Real v[2] = {amrex::get<0>(hv), amrex::get<1>(hv)};
        ParallelDescriptor::ReduceRealSum(v, 2);
        s_ab = v[0];
        s_cd = v[1];
    }

    Real NormInf3 (ElectronInertiaElliptic::EVec const& x)
    {
        amrex::ReduceOps<amrex::ReduceOpMax> ops;
        amrex::ReduceData<Real> data(ops);
        for (MFIter mfi(x[0]); mfi.isValid(); ++mfi) {
            for (int d = 0; d < 3; ++d) {
                Array4<Real const> const& xa = x[d].const_array(mfi);
                ops.eval(mfi.tilebox(x[d].ixType().toIntVect()), data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        -> amrex::GpuTuple<Real>
                    { return { std::abs(xa(i, j, k)) }; });
            }
        }
        Real m = amrex::get<0>(data.value(ops));
        ParallelDescriptor::ReduceRealMax(m);
        return m;
    }

    void NormInfPair3 (std::array<MultiFab const*, 3> const& x,
                       std::array<MultiFab const*, 3> const& y,
                       Real& nx, Real& ny)
    {
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
        amrex::ReduceData<Real, Real> data(ops);
        for (MFIter mfi(*x[0]); mfi.isValid(); ++mfi) {
            for (int d = 0; d < 3; ++d) {
                Array4<Real const> const& xa = x[d]->const_array(mfi);
                Array4<Real const> const& ya = y[d]->const_array(mfi);
                ops.eval(mfi.tilebox(x[d]->ixType().toIntVect()), data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        -> amrex::GpuTuple<Real, Real>
                    { return { std::abs(xa(i, j, k)),
                               std::abs(ya(i, j, k)) }; });
            }
        }
        auto const hv = data.value(ops);
        Real v[2] = {amrex::get<0>(hv), amrex::get<1>(hv)};
        ParallelDescriptor::ReduceRealMax(v, 2);
        nx = v[0];
        ny = v[1];
    }

#else

    // CPU: the tuned AMReX kernels for the local part -- there is no
    // synchronisation to save here, and replacing them with the fused path
    // above measured 8% slower per solve -- with the collective still
    // batched to one per group.

    Real Dot3 (ElectronInertiaElliptic::EVec const& x,
               ElectronInertiaElliptic::EVec const& y)
    {
        Real v[3];
        for (int d = 0; d < 3; ++d) {
            v[d] = MultiFab::Dot(x[d], 0, y[d], 0, 1, 0, /*local=*/true);
        }
        ParallelDescriptor::ReduceRealSum(v, 3);
        return v[0] + v[1] + v[2];
    }

    void Dot3Pair (ElectronInertiaElliptic::EVec const& a,
                   ElectronInertiaElliptic::EVec const& b,
                   ElectronInertiaElliptic::EVec const& c,
                   ElectronInertiaElliptic::EVec const& d_,
                   Real& s_ab, Real& s_cd)
    {
        Real v[6];
        for (int d = 0; d < 3; ++d) {
            v[d]     = MultiFab::Dot(a[d], 0, b[d], 0, 1, 0, /*local=*/true);
            v[3 + d] = MultiFab::Dot(c[d], 0, d_[d], 0, 1, 0, /*local=*/true);
        }
        ParallelDescriptor::ReduceRealSum(v, 6);
        s_ab = v[0] + v[1] + v[2];
        s_cd = v[3] + v[4] + v[5];
    }

    Real NormInf3 (ElectronInertiaElliptic::EVec const& x)
    {
        Real v[3];
        for (int d = 0; d < 3; ++d) {
            v[d] = x[d].norm0(0, 0, /*local=*/true);
        }
        ParallelDescriptor::ReduceRealMax(v, 3);
        return std::max(v[0], std::max(v[1], v[2]));
    }

    void NormInfPair3 (std::array<MultiFab const*, 3> const& x,
                       std::array<MultiFab const*, 3> const& y,
                       Real& nx, Real& ny)
    {
        Real v[6];
        for (int d = 0; d < 3; ++d) {
            v[d]     = x[d]->norm0(0, 0, /*local=*/true);
            v[3 + d] = y[d]->norm0(0, 0, /*local=*/true);
        }
        ParallelDescriptor::ReduceRealMax(v, 6);
        nx = std::max(v[0], std::max(v[1], v[2]));
        ny = std::max(v[3], std::max(v[4], v[5]));
    }

#endif
}

void
ElectronInertiaElliptic::CheckBoundarySupport ()
{
    auto const& warpx = WarpX::GetInstance();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (int side = 0; side < 2; ++side) {
            auto const bc = (side == 0) ? warpx.field_boundary_lo[idim]
                                        : warpx.field_boundary_hi[idim];
            const bool supported =
                   (bc == FieldBoundaryType::Periodic)
                || (bc == FieldBoundaryType::PEC)
                || (bc == FieldBoundaryType::PMC)   // == Neumann
                || (bc == FieldBoundaryType::None);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                supported,
                "hybrid_pic_model.include_electron_inertia_elliptic requires "
                "field boundaries that have a homogeneous linear form, so "
                "that they can be imposed on the correction inside the "
                "elliptic solve. Supported: periodic, pec, pmc/neumann, none "
                "(the r=0 axis). The boundary condition set on dimension "
                + std::to_string(idim) + " is not one of these; it is either "
                "affine in E or carries its own auxiliary state, and no "
                "homogeneous counterpart exists that could legitimately be "
                "applied to a Krylov iterate.");
        }
    }
}

void
ElectronInertiaElliptic::Define (ablastr::fields::VectorField const& Efield, int lev)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0,
        "The elliptic electron-inertia solve supports a single level only, "
        "matching the rest of the hybrid-PIC solver.");

#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) \
    && !defined(WARPX_DIM_1D_Z)
    WARPX_ABORT_WITH_MESSAGE(
        "hybrid_pic_model.include_electron_inertia_elliptic is implemented "
        "for RZ and for the 1D, 2D and 3D Cartesian staggered grids only.");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::grid_type == GridType::Staggered,
        "The elliptic electron-inertia solve requires the staggered (Yee) "
        "grid: it is built from the same upward/downward curl pair that the "
        "Faraday and Ampere updates use.");

    CheckBoundarySupport();

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField Bfield =
        warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);

    EVec* const evecs[] = {&m_de2, &m_jac, &m_x, &m_r, &m_rhat, &m_p,
                           &m_v, &m_s, &m_t, &m_b, &m_ax};
    for (auto* e : evecs) {
        for (int d = 0; d < 3; ++d) {
            (*e)[d].define(Efield[d]->boxArray(), Efield[d]->DistributionMap(),
                           1, ng_iterate);
            (*e)[d].setVal(0._rt);
        }
    }
    // The intermediate curl lives at the B-field staggering, taken from the
    // registered B fields rather than derived, so that the cylindrical
    // index types are right by construction.
    for (int d = 0; d < 3; ++d) {
        m_curl[d].define(Bfield[d]->boxArray(), Bfield[d]->DistributionMap(),
                         1, ng_iterate);
        m_curl[d].setVal(0._rt);
    }

    m_defined = true;
    m_coefs_valid = false;
}

void
ElectronInertiaElliptic::PrepareCoefficients (
    amrex::MultiFab const& rhofield, amrex::Real rho_floor, int lev)
{
    ABLASTR_PROFILE("ElectronInertiaElliptic::PrepareCoefficients()");

    // The coefficients depend only on the density, and the density is held
    // fixed across the magnetic substeps. Skip the rebuild when rho has not
    // moved; the signature is a pair of global reductions over rho, which is
    // one grid pass against roughly a hundred for the solve it guards.
    const Real rho_sum = rhofield.sum(0);
    const Real rho_norm0 = rhofield.norm0(0, 0);
    if (m_coefs_valid && rho_sum == m_rho_sum && rho_norm0 == m_rho_norm0) {
        return;
    }

    const double t0 = amrex::second();

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField Efield =
        warpx.m_fields.get_alldirs(FieldType::Efield_fp, lev);

    // Staggering of the E components, with unused dimensions forced nodal:
    // the coarsen/sample interpolator indexes all three dimensions and reads
    // out of bounds if a collapsed dimension is left cell-centred.
    GpuArray<int, 3> stag[3];
    for (int d = 0; d < 3; ++d) {
        const IntVect it = Efield[d]->ixType().toIntVect();
        stag[d] = GpuArray<int, 3>{1, 1, 1};
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            stag[d][idim] = it[idim];
        }
    }
    const GpuArray<int, 3> nodal = {1, 1, 1};
    const GpuArray<int, 3> coarsen = {1, 1, 1};

    const auto dx = warpx.Geom(lev).CellSizeArray();
#if defined(WARPX_DIM_RZ)
    const Real inv_dr2 = 1._rt / (dx[0] * dx[0]);
    const Real inv_dz2 = 1._rt / (dx[1] * dx[1]);
    const Real dr = dx[0];
    const Real rmin = warpx.Geom(lev).ProbLo(0);
    // Diagonal of the discrete curl-curl, component by component, obtained
    // by differentiating the composed stencils. Radial: 2/dz^2. Axial:
    // 2/dr^2 off the axis, 4/dr^2 on it. Azimuthal: 2/dz^2 plus an
    // r-dependent radial part that tends to 2/dr^2 away from the axis.
    // These enter only the Jacobi preconditioner, so an inexact entry costs
    // iterations and never changes the converged field.
    const Real diag_r = 2._rt * inv_dz2;
    const Real diag_z = 2._rt * inv_dr2;
#elif defined(WARPX_DIM_3D)
    const Real inv_dx2 = 1._rt / (dx[0] * dx[0]);
    const Real inv_dy2 = 1._rt / (dx[1] * dx[1]);
    const Real inv_dz2 = 1._rt / (dx[2] * dx[2]);
    const Real diag[3] = {2._rt * (inv_dy2 + inv_dz2),
                          2._rt * (inv_dz2 + inv_dx2),
                          2._rt * (inv_dx2 + inv_dy2)};
#elif defined(WARPX_DIM_XZ)
    const Real inv_dx2 = 1._rt / (dx[0] * dx[0]);
    const Real inv_dz2 = 1._rt / (dx[1] * dx[1]);
    const Real diag[3] = {2._rt * inv_dz2,
                          2._rt * (inv_dz2 + inv_dx2),
                          2._rt * inv_dx2};
#else // WARPX_DIM_1D_Z
    const Real inv_dz2 = 1._rt / (dx[0] * dx[0]);
    // Only the two transverse components feel the curl-curl for k along z;
    // the parallel component is annihilated by it, which is the correct
    // physics -- Ampere's law returns no parallel current for k parallel to
    // the field, so the inertia term has nothing to act on there.
    const Real diag[3] = {2._rt * inv_dz2, 2._rt * inv_dz2, 0._rt};
#endif

    // d_e^2 = m_e / (mu0 e^2 n) and rho = e n, hence m_e / (mu0 e rho).
    const Real de2_num = PhysConst::m_e / (PhysConst::mu0 * PhysConst::q_e);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(m_de2[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        for (int d = 0; d < 3; ++d)
        {
            Array4<Real> const& de2 = m_de2[d].array(mfi);
            Array4<Real> const& jac = m_jac[d].array(mfi);
            const Box tb = mfi.tilebox(m_de2[d].ixType().toIntVect());
            const GpuArray<int, 3> st = stag[d];
#if defined(WARPX_DIM_RZ)
            // The azimuthal diagonal needs the nodal radius; the radial and
            // axial ones are uniform.
            const int comp = d;
            const Real dg_r = diag_r;
            const Real dg_z = diag_z;
            const Real idr2 = inv_dr2;
            const Real idz2 = inv_dz2;
            const Real drl = dr;
            const Real rmn = rmin;
#else
            const Real dg = diag[d];
#endif
            amrex::ParallelFor(tb,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const Real rho_val = amrex::max(
                        ablastr::coarsen::sample::Interp(
                            rho, nodal, st, coarsen, i, j, k, 0),
                        rho_floor);
                    const Real d2 = de2_num / rho_val;
                    de2(i, j, k) = d2;
#if defined(WARPX_DIM_RZ)
                    Real dgc;
                    if (comp == 0) {
                        dgc = dg_r;
                    } else if (comp == 2) {
                        // Axial: 2/dr^2, except on the axis where the
                        // regularised (1/r) d(r W_theta)/dr stencil is
                        // 4 W_theta / dr and the diagonal doubles.
                        dgc = (rmn + i * drl > 0.5_rt * drl)
                            ? dg_z : 2._rt * dg_z;
                    } else {
                        // Azimuthal: 2/dz^2 + [x/(x+1/2) + x/(x-1/2)]/dr^2
                        // with x = r/dr, and no curl-curl coupling on the
                        // axis row because E_theta vanishes there for m = 0.
                        const Real rn = rmn + i * drl;
                        if (rn > 0.5_rt * drl) {
                            const Real x = rn / drl;
                            dgc = 2._rt * idz2
                                + (x / (x + 0.5_rt) + x / (x - 0.5_rt)) * idr2;
                        } else {
                            dgc = 0._rt;
                        }
                    }
#else
                    const Real dgc = dg;
#endif
                    jac(i, j, k) = 1._rt / (1._rt + d2 * dgc);
                });
        }
    }

    amrex::Gpu::streamSynchronize();
    m_setup_time += amrex::second() - t0;
    ++m_n_setups;
    m_rho_sum = rho_sum;
    m_rho_norm0 = rho_norm0;
    m_coefs_valid = true;
    // The first solve after a rebuild is the one whose warm-start guess is
    // furthest from the answer; Solve accounts it separately.
    m_fresh_coefs = true;
}

void
ElectronInertiaElliptic::ApplyHomogeneousBC (EVec& X, int lev)
{
    auto& warpx = WarpX::GetInstance();
    std::array<MultiFab*, 3> ptrs = {&X[0], &X[1], &X[2]};

    // Only the first guard layer is ever read by the operator, so the PEC
    // routines are asked for exactly that -- passing the production
    // ng_fieldgather here would index past the iterate's single ghost.
    const IntVect ng(ng_iterate);
    if (AnyFaceIs(FieldBoundaryType::PEC)) {
        PEC::ApplyPECtoEfield(
            ptrs, warpx.field_boundary_lo, warpx.field_boundary_hi,
            FieldBoundaryType::PEC, ng, warpx.Geom(lev), lev,
            PatchType::fine, warpx.refRatio());
    }
    if (AnyFaceIs(FieldBoundaryType::PMC)) {
        PEC::ApplyPECtoBfield(
            ptrs, warpx.field_boundary_lo, warpx.field_boundary_hi,
            FieldBoundaryType::PMC, ng, warpx.Geom(lev), lev,
            PatchType::fine, warpx.refRatio());
    }
}

void
ElectronInertiaElliptic::FillIterateBoundary (EVec& X, int lev)
{
    auto& warpx = WarpX::GetInstance();

    ablastr::utils::communication::FillBoundary(
        {&X[0], &X[1], &X[2]}, WarpX::do_single_precision_comms,
        warpx.Geom(lev).periodicity(), /*nodal_sync=*/true);
    ApplyHomogeneousBC(X, lev);
}

void
ElectronInertiaElliptic::ApplyOperator (EVec& X, EVec& out, int lev)
{
    FillIterateBoundary(X, lev);

    auto& warpx = WarpX::GetInstance();
    const auto dx = warpx.Geom(lev).CellSizeArray();

#if defined(WARPX_DIM_RZ)
    const GpuArray<Real, 1> cr = {1._rt / dx[0]};
    const GpuArray<Real, 1> cz = {1._rt / dx[1]};
    const Real dr = dx[0];
    const Real rmin = warpx.Geom(lev).ProbLo(0);
    using T_Algo = CylindricalYeeAlgorithm;
#else
#   if defined(WARPX_DIM_3D)
    const GpuArray<Real, 1> cx = {1._rt / dx[0]};
    const GpuArray<Real, 1> cy = {1._rt / dx[1]};
    const GpuArray<Real, 1> cz = {1._rt / dx[2]};
#   elif defined(WARPX_DIM_XZ)
    const GpuArray<Real, 1> cx = {1._rt / dx[0]};
    const GpuArray<Real, 1> cy = {0._rt};
    const GpuArray<Real, 1> cz = {1._rt / dx[1]};
#   else // WARPX_DIM_1D_Z
    const GpuArray<Real, 1> cx = {0._rt};
    const GpuArray<Real, 1> cy = {0._rt};
    const GpuArray<Real, 1> cz = {1._rt / dx[0]};
#   endif
    using T_Algo = CartesianYeeAlgorithm;
#endif

    // ---- first curl: E staggering -> B staggering (upward differences,
    // exactly the operator inside Faraday's law) ----
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(m_curl[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real const> const& Ex = X[0].const_array(mfi);
        Array4<Real const> const& Ey = X[1].const_array(mfi);
        Array4<Real const> const& Ez = X[2].const_array(mfi);
        Array4<Real> const& Wx = m_curl[0].array(mfi);
        Array4<Real> const& Wy = m_curl[1].array(mfi);
        Array4<Real> const& Wz = m_curl[2].array(mfi);

        // Grow by one cell on both sides. The low halo feeds the downward
        // differences of the second curl; the high halo is needed because a
        // component of the outgoing field that is NODAL in some direction
        // has a valid point sitting on the upper domain face, and its
        // downward difference reaches the cell beyond the last valid one.
        // One layer is enough in both directions: a Yee curl component that
        // is nodal along a direction never differences along that direction,
        // so no evaluation here reads more than one ghost cell of X.
        Box tbx = mfi.tilebox(m_curl[0].ixType().toIntVect(), IntVect(1));
        Box tby = mfi.tilebox(m_curl[1].ixType().toIntVect(), IntVect(1));
        Box tbz = mfi.tilebox(m_curl[2].ixType().toIntVect(), IntVect(1));

#if defined(WARPX_DIM_RZ)
        Real const* const AMREX_RESTRICT pcr = cr.data();
        Real const* const AMREX_RESTRICT pcz = cz.data();
        const Real drl = dr;
        const Real rmn = rmin;
        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // (curl E)_r = -dE_theta/dz, zero on the axis for m = 0.
                const Real r = rmn + i * drl;
                Wx(i, j, k) = (r == 0._rt)
                    ? 0._rt
                    : -T_Algo::UpwardDz(Ey, pcz, 1, i, j, k, 0);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // (curl E)_theta = dE_r/dz - dE_z/dr
                Wy(i, j, k) = T_Algo::UpwardDz(Ex, pcz, 1, i, j, k, 0)
                            - T_Algo::UpwardDr(Ez, pcr, 1, i, j, k, 0);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // (curl E)_z = (1/r) d(r E_theta)/dr, on a radius that is
                // cell-centred and therefore never zero.
                const Real r = rmn + (i + 0.5_rt) * drl;
                Wz(i, j, k) =
                    T_Algo::UpwardDrr_over_r(Ey, r, drl, pcr, 1, i, j, k, 0);
            });
#else
        Real const* const AMREX_RESTRICT pcx = cx.data();
        Real const* const AMREX_RESTRICT pcy = cy.data();
        Real const* const AMREX_RESTRICT pcz = cz.data();
        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Wx(i, j, k) = T_Algo::UpwardDy(Ez, pcy, 1, i, j, k)
                            - T_Algo::UpwardDz(Ey, pcz, 1, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Wy(i, j, k) = T_Algo::UpwardDz(Ex, pcz, 1, i, j, k)
                            - T_Algo::UpwardDx(Ez, pcx, 1, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Wz(i, j, k) = T_Algo::UpwardDx(Ey, pcx, 1, i, j, k)
                            - T_Algo::UpwardDy(Ex, pcy, 1, i, j, k);
            });
#endif
    }

    // ---- second curl: B staggering -> E staggering (downward differences,
    // exactly the operator inside Ampere's law) plus the identity term ----
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real const> const& Wx = m_curl[0].const_array(mfi);
        Array4<Real const> const& Wy = m_curl[1].const_array(mfi);
        Array4<Real const> const& Wz = m_curl[2].const_array(mfi);
        Array4<Real const> const& Ex = X[0].const_array(mfi);
        Array4<Real const> const& Ey = X[1].const_array(mfi);
        Array4<Real const> const& Ez = X[2].const_array(mfi);
        Array4<Real const> const& d2x = m_de2[0].const_array(mfi);
        Array4<Real const> const& d2y = m_de2[1].const_array(mfi);
        Array4<Real const> const& d2z = m_de2[2].const_array(mfi);
        Array4<Real> const& Ox = out[0].array(mfi);
        Array4<Real> const& Oy = out[1].array(mfi);
        Array4<Real> const& Oz = out[2].array(mfi);

        const Box& tox = mfi.tilebox(out[0].ixType().toIntVect());
        const Box& toy = mfi.tilebox(out[1].ixType().toIntVect());
        const Box& toz = mfi.tilebox(out[2].ixType().toIntVect());

#if defined(WARPX_DIM_RZ)
        Real const* const AMREX_RESTRICT pcr = cr.data();
        Real const* const AMREX_RESTRICT pcz = cz.data();
        const Real drl = dr;
        const Real rmn = rmin;
        amrex::ParallelFor(tox, toy, toz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // (curl W)_r = -dW_theta/dz
                const Real c = -T_Algo::DownwardDz(Wy, pcz, 1, i, j, k, 0);
                Ox(i, j, k) = Ex(i, j, k) + d2x(i, j, k) * c;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // (curl W)_theta = dW_r/dz - dW_z/dr, zero on the axis for
                // m = 0, mirroring Ampere's on-axis treatment.
                const Real r = rmn + i * drl;
                Real c = 0._rt;
                if (r > 0.5_rt * drl) {
                    c = T_Algo::DownwardDz(Wx, pcz, 1, i, j, k, 0)
                      - T_Algo::DownwardDr(Wz, pcr, 1, i, j, k, 0);
                }
                Oy(i, j, k) = Ey(i, j, k) + d2y(i, j, k) * c;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // (curl W)_z = (1/r) d(r W_theta)/dr, regularised on axis
                // where W_theta is linear in r for m = 0.
                const Real r = rmn + i * drl;
                const Real c = (r > 0.5_rt * drl)
                    ? T_Algo::DownwardDrr_over_r(Wy, r, drl, pcr, 1, i, j, k, 0)
                    : 4._rt * Wy(i, j, k, 0) / drl;
                Oz(i, j, k) = Ez(i, j, k) + d2z(i, j, k) * c;
            });
#else
        Real const* const AMREX_RESTRICT pcx = cx.data();
        Real const* const AMREX_RESTRICT pcy = cy.data();
        Real const* const AMREX_RESTRICT pcz = cz.data();
        amrex::ParallelFor(tox, toy, toz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const Real c = T_Algo::DownwardDy(Wz, pcy, 1, i, j, k)
                             - T_Algo::DownwardDz(Wy, pcz, 1, i, j, k);
                Ox(i, j, k) = Ex(i, j, k) + d2x(i, j, k) * c;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const Real c = T_Algo::DownwardDz(Wx, pcz, 1, i, j, k)
                             - T_Algo::DownwardDx(Wz, pcx, 1, i, j, k);
                Oy(i, j, k) = Ey(i, j, k) + d2y(i, j, k) * c;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const Real c = T_Algo::DownwardDx(Wy, pcx, 1, i, j, k)
                             - T_Algo::DownwardDy(Wx, pcy, 1, i, j, k);
                Oz(i, j, k) = Ez(i, j, k) + d2z(i, j, k) * c;
            });
#endif
    }

    // A boundary condition that FORCES a field value -- tangential E at a
    // PEC wall, normal E at a PMC cap -- removes that value from the set of
    // unknowns: FillIterateBoundary above overwrote it, so it does not
    // appear in its own equation and no Krylov update can move it. Those
    // rows must therefore be projected out of the system on BOTH sides.
    // Applying the same forcing to the operator output (and, in Solve, to
    // the right-hand side) zeroes them consistently. Without this the
    // system is inconsistent and the iteration stalls at the residual
    // carried by the boundary rows instead of converging.
    ApplyHomogeneousBC(out, lev);
}

void
ElectronInertiaElliptic::Solve (ablastr::fields::VectorField const& Efield, int lev)
{
    ABLASTR_PROFILE("ElectronInertiaElliptic::Solve()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_defined && m_coefs_valid,
        "ElectronInertiaElliptic::Solve called before the operator was "
        "defined and its coefficients built.");

    const double t0 = amrex::second();

    // Build the right-hand side for the correction. With E0 the inertialess
    // solution, E = E0 + dE and A = I + d_e^2 C give
    //     A dE = E0 - A E0 = -d_e^2 C E0.
    // Solving for the correction rather than for E is what makes the
    // boundary conditions homogeneous: E0 already satisfies the physical
    // ones, so dE must satisfy their homogeneous form.
    for (int d = 0; d < 3; ++d) {
        MultiFab::Copy(m_p[d], *Efield[d], 0, 0, 1, 0);
    }
    ApplyOperator(m_p, m_ax, lev);
    LinComb3(m_b, 1._rt, m_p, -1._rt, m_ax);
    // Same projection as in ApplyOperator: the forced boundary values are
    // not unknowns, so their rows carry no residual.
    ApplyHomogeneousBC(m_b, lev);

    Real bnorm = 0._rt;
    Real e0norm = 0._rt;
    NormInfPair3({&m_b[0], &m_b[1], &m_b[2]},
                 {Efield[0], Efield[1], Efield[2]}, bnorm, e0norm);

    // Nothing to correct: the inertia term is below the tolerance relative
    // to E itself. Chasing it further would only iterate on round-off.
    if (bnorm <= m_rtol * e0norm || bnorm == 0._rt) {
        SetVal3(m_x, 0._rt);
        ++m_n_solves;
        amrex::Gpu::streamSynchronize();
        m_solve_time += amrex::second() - t0;
        return;
    }

    const bool cold = !m_warm_start || (m_n_solves == 0) || m_fresh_coefs;
    m_fresh_coefs = false;
    if (cold) { SetVal3(m_x, 0._rt); }

    // r = b - A x
    if (cold) {
        Copy3(m_r, m_b, 0);
    } else {
        ApplyOperator(m_x, m_ax, lev);
        LinComb3(m_r, 1._rt, m_b, -1._rt, m_ax);
    }
    Copy3(m_rhat, m_r, 0);
    SetVal3(m_p, 0._rt);
    SetVal3(m_v, 0._rt);

    Real rho_old = 1._rt, alpha = 1._rt, omega = 1._rt;
    const Real tol = m_rtol * bnorm;
    int iter = 0;
    bool converged = (NormInf3(m_r) <= tol);

    for (; iter < m_max_iter && !converged; ++iter)
    {
        const Real rho_new = Dot3(m_rhat, m_r);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(rho_new) && rho_new != 0._rt,
            "The elliptic electron-inertia solve broke down (the BiCGStab "
            "shadow residual became orthogonal or non-finite). This "
            "indicates a corrupted E field or density entering the solve.");

        const Real beta = (rho_new / rho_old) * (alpha / omega);
        // p = r + beta * (p - omega * v)
        Saxpy3(m_p, -omega, m_v);
        for (int d = 0; d < 3; ++d) { m_p[d].mult(beta, 0, 1, 0); }
        Saxpy3(m_p, 1._rt, m_r);

        Precond3(m_s, m_p, m_jac);            // y = M^-1 p, held in m_s
        ApplyOperator(m_s, m_v, lev);         // v = A y

        const Real rhat_v = Dot3(m_rhat, m_v);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(rhat_v) && rhat_v != 0._rt,
            "The elliptic electron-inertia solve broke down (zero curvature "
            "in the BiCGStab search direction).");
        alpha = rho_new / rhat_v;

        Saxpy3(m_x, alpha, m_s);              // x = x + alpha * y
        Saxpy3(m_r, -alpha, m_v);             // s = r - alpha * v (in place)

        if (NormInf3(m_r) <= tol) { converged = true; ++iter; break; }

        Precond3(m_s, m_r, m_jac);            // z = M^-1 s
        ApplyOperator(m_s, m_t, lev);         // t = A z

        Real tt = 0._rt, tr = 0._rt;
        Dot3Pair(m_t, m_t, m_t, m_r, tt, tr);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(tt) && tt > 0._rt,
            "The elliptic electron-inertia solve broke down (null stabiliser "
            "direction in BiCGStab).");
        omega = tr / tt;

        Saxpy3(m_x, omega, m_s);              // x = x + omega * z
        Saxpy3(m_r, -omega, m_t);             // r = s - omega * t

        const Real rnorm = NormInf3(m_r);
        if (m_verbose > 2) {
            amrex::Print() << "    it " << iter << "  |r|/|b| "
                           << rnorm / bnorm << "  rho " << rho_new
                           << "  alpha " << alpha << "  omega " << omega
                           << "\n";
        }

        if (rnorm <= tol) { converged = true; ++iter; break; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            omega != 0._rt,
            "The elliptic electron-inertia solve stalled (omega vanished).");
        rho_old = rho_new;
    }

    const Real final_res = NormInf3(m_r);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        converged,
        "The elliptic electron-inertia solve did not converge within "
        + std::to_string(m_max_iter) + " iterations (relative residual "
        + std::to_string(static_cast<double>(final_res / bnorm)) + " against "
        "a tolerance of "
        + std::to_string(static_cast<double>(m_rtol)) + "). A partially "
        "converged E is not accepted: it would silently restore the "
        "unbounded high-wavenumber branch that this term exists to remove. "
        "Raise hybrid_pic_model.electron_inertia_max_iterations or relax "
        "hybrid_pic_model.electron_inertia_relative_tolerance deliberately.");

    // E <- E0 + dE
    for (int d = 0; d < 3; ++d) {
        MultiFab::Add(*Efield[d], m_x[d], 0, 0, 1, 0);
    }

    ++m_n_solves;
    m_n_iters += iter;
    m_max_iters_seen = std::max(m_max_iters_seen, iter);
    if (cold) { ++m_n_cold_solves; m_n_cold_iters += iter; }
    amrex::Gpu::streamSynchronize();
    m_solve_time += amrex::second() - t0;

    if (m_verbose > 1) {
        amrex::Print() << "  electron inertia: " << iter << " iterations, "
                       << "relative residual " << final_res / bnorm
                       << ", |dE|/|E| " << NormInf3(m_x) / std::max(e0norm, 1.e-30_rt)
                       << "\n";
    }
}

void
ElectronInertiaElliptic::ReportAndResetStats ()
{
    if (m_verbose < 1 || m_n_solves == 0) { return; }
    const double mean = static_cast<double>(m_n_iters)
                      / static_cast<double>(std::max(m_n_solves, 1L));
    const double cold_mean = (m_n_cold_solves > 0)
        ? static_cast<double>(m_n_cold_iters)
          / static_cast<double>(m_n_cold_solves)
        : 0.0;
    amrex::Print() << "Electron inertia (elliptic): " << m_n_solves
                   << " solves, mean " << mean << " iterations (max "
                   << m_max_iters_seen << "), first-after-rebuild mean "
                   << cold_mean << "; setup " << m_setup_time * 1.e3
                   << " ms over " << m_n_setups << " rebuilds, solve "
                   << m_solve_time * 1.e3 << " ms, "
                   << (m_solve_time * 1.e3
                       / static_cast<double>(std::max(m_n_solves, 1L)))
                   << " ms/solve\n";
    m_n_solves = 0; m_n_iters = 0; m_max_iters_seen = 0;
    m_n_setups = 0; m_setup_time = 0.0; m_solve_time = 0.0;
    m_n_cold_solves = 0; m_n_cold_iters = 0;
}
