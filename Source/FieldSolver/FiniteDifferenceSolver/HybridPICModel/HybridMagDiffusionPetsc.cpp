/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Bowen Zhu
 *
 * License: BSD-3-Clause-LBNL
 *
 * PETSc KSP/PC implementation for the optional matrix-free hybrid
 * magnetic-diffusion curl-curl solve. This translation unit includes `petsc*.h`
 * and therefore MUST NOT include WarpX/ablastr headers (WarpX's own global
 * `ReductionType` enum, WarpXAlgorithmSelection.H, collides with PETSc's global
 * `ReductionType`, petscvec.h). Only pure AMReX headers are used here, mirroring
 * the WarpX_PETSc.cpp isolation precedent.
 *
 * The matrix-free operator (matvec) and the scaled-Jacobi preconditioner are
 * supplied as callbacks by HybridMagDiffusion.cpp, which owns the Yee/RZ
 * curl-curl apply() and the field MultiFabs. This file owns the PETSc objects
 * (Vec, MatShell, assembled Pmat, KSP, PC), the DOF layout, and the assembled
 * frozen-eta Laplacian Pmat.
 */
#include "HybridMagDiffusionPetsc.H"

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_Loop.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#ifdef AMREX_USE_PETSC
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscvec.h>
#endif

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#ifdef AMREX_USE_PETSC
// PetscCall (PETSc's macro) can only be used inside functions returning
// PetscErrorCode (the static callbacks and assembleFrozenLaplacian). In other
// functions (ctor, make, solve) we abort on a non-zero PETSc error.
#define MAGDIFF_PETSC_CHK(call) do { \
    PetscErrorCode const _magdiff_ierr = (call); \
    if (_magdiff_ierr != 0) { \
        amrex::Abort("HybridMagDiffusion PETSc call failed"); } \
} while (0)
#endif


#ifdef AMREX_USE_PETSC

namespace {

/**
 * \brief PETSc KSP driver for the matrix-free mag-diff operator.
 *
 * DOF layout: component-major (B0, B1, B2), per-rank MFIter order, interior
 * (nghost=0) cells only. A per-component global-index iMultiFab (nghost=1,
 * -1 = exterior) maps each interior cell to its PETSc row/column; the matvec
 * and PC callbacks (in HybridMagDiffusion.cpp) read this same index to
 * scatter/gather their field MultiFabs, so the Vec and the assembled Pmat stay
 * self-consistent regardless of tiling. Norms use the nghost=0 interior only
 * (the FPE-trap invariant).
 */
class MagDiffPetscSolverImpl
{
public:
    MagDiffPetscSolverImpl (
        amrex::Array<amrex::MultiFab const*,3> const& B_proto,
        amrex::Array<amrex::MultiFab const*,3> const& eta_pc,
        amrex::Geometry const& geom, amrex::Real theta_dt, amrex::Real mu0,
        MagDiffPetscPC pc_choice,
        amrex::Real rtol, amrex::Real atol, int max_iter, int verbose,
        MagDiffMatvecFn matvec, MagDiffPCFn pcapply, void* opctx,
        amrex::Array<amrex::iMultiFab const*,3> const* eb_update_B)
        : m_geom(geom), m_theta_dt(theta_dt), m_mu0(mu0),
          m_pc_choice(pc_choice), m_rtol(rtol), m_atol(atol),
          m_max_iter(max_iter), m_verbose(verbose),
          m_matvec(matvec), m_pcapply(pcapply), m_opctx(opctx)
    {
        // EB masks live on the device under CUDA. Copy to pinned host once so
        // all LoopOnCpu DOF counting / indexing is host-safe (raw device
        // Array4 from eb_update_B in LoopOnCpu SEGVs on GPU builds).
        if (eb_update_B) {
            copyEbMasksToHost(*eb_update_B);
        }

        // Per-component r-staggering (1 = NODE in r, 0 = CELL) for the
        // cylindrical 1/r metric in assembleFrozenLaplacian (Br is NODE in r;
        // Bt/Bz are CELL in r). Read once from B_proto's index type.
        for (int idim = 0; idim < 3; ++idim) {
            m_r_nodal[idim] = B_proto[idim]->ixType().toIntVect()[0];
        }

        // Local DOF count over interior (nghost=0) cells, component-major,
        // skipping covered B DOFs when the EB mask is provided.
        for (int idim = 0; idim < 3; ++idim) {
            for (amrex::MFIter mfi(*B_proto[idim]); mfi.isValid(); ++mfi) {
                if (m_eb_mask_host[0]) {
                    auto const& mask_arr = m_eb_mask_host[idim]->const_array(mfi);
                    amrex::Box const& tb = mfi.tilebox();
                    amrex::LoopOnCpu(amrex::lbound(tb), amrex::ubound(tb),
                        [&] (int i, int j, int k) {
                            if (mask_arr(i, j, k) != 0) { ++m_n_local; }
                        });
                } else {
                    m_n_local += mfi.tilebox().numPts();
                }
            }
        }

        MAGDIFF_PETSC_CHK(VecCreateMPI(
            PETSC_COMM_WORLD, m_n_local, PETSC_DETERMINE, &m_x));
        MAGDIFF_PETSC_CHK(VecDuplicate(m_x, &m_b));
        PetscInt rstart = 0;
        MAGDIFF_PETSC_CHK(VecGetOwnershipRange(m_x, &rstart, nullptr));
        m_rstart = rstart;
        PetscInt ng = 0;
        MAGDIFF_PETSC_CHK(VecGetSize(m_x, &ng));
        m_n_global = ng;

        buildGlobalIndex(B_proto);
        buildEtaGhosts(eta_pc);

        // MatShell operator: matvec = caller's apply (homogeneous A_lin).
        MAGDIFF_PETSC_CHK(MatCreateShell(
            PETSC_COMM_WORLD, m_n_local, m_n_local, m_n_global, m_n_global,
            this, &m_A));
        MAGDIFF_PETSC_CHK(MatShellSetOperation(
            m_A, MATOP_MULT, reinterpret_cast<void(*)(void)>(applyMatOp)));
        MAGDIFF_PETSC_CHK(MatSetUp(m_A));

        MAGDIFF_PETSC_CHK(KSPCreate(PETSC_COMM_WORLD, &m_ksp));
        MAGDIFF_PETSC_CHK(KSPSetOperators(m_ksp, m_A, m_A));

        PC pc = nullptr;
        MAGDIFF_PETSC_CHK(KSPGetPC(m_ksp, &pc));
        if (m_pc_choice == MagDiffPetscPC::frozen_laplacian) {
            // Assembled frozen-eta Laplacian Pmat (see assembleFrozenLaplacian).
            // Lets runtime -pc_type asm/ilu/hypre factor a real matrix; default
            // PC for an assembled Pmat is block-Jacobi ILU(0).
            PetscInt const nz = 1 + 2 * AMREX_SPACEDIM;
            MAGDIFF_PETSC_CHK(MatCreateAIJ(
                PETSC_COMM_WORLD, m_n_local, m_n_local, m_n_global, m_n_global,
                nz, nullptr, nz, nullptr, &m_P));
            assembleFrozenLaplacian();
            MAGDIFF_PETSC_CHK(KSPSetOperators(m_ksp, m_A, m_P));
        } else {
            // Shell PC reusing the caller's scaled Jacobi (precond callback).
            MAGDIFF_PETSC_CHK(PCSetType(pc, PCSHELL));
            MAGDIFF_PETSC_CHK(PCShellSetApply(pc, applyShellPC));
            MAGDIFF_PETSC_CHK(PCShellSetContext(pc, this));
        }
        MAGDIFF_PETSC_CHK(KSPSetPCSide(m_ksp, PC_RIGHT));
        MAGDIFF_PETSC_CHK(KSPSetType(m_ksp, KSPGMRES));
        MAGDIFF_PETSC_CHK(KSPSetTolerances(
            m_ksp, m_rtol, m_atol, PETSC_DEFAULT, m_max_iter));
        MAGDIFF_PETSC_CHK(KSPSetNormType(m_ksp, KSP_NORM_UNPRECONDITIONED));
        MAGDIFF_PETSC_CHK(KSPGMRESSetRestart(
            m_ksp, std::min(m_max_iter, 50)));
        if (m_verbose > 0) {
            MAGDIFF_PETSC_CHK(KSPMonitorSet(
                m_ksp, monitorResidual, nullptr, nullptr));
        }
        // Let -ksp_type / -pc_type / -ksp_rtol ... from argv or PETSC_OPTIONS win.
        MAGDIFF_PETSC_CHK(KSPSetFromOptions(m_ksp));
    }

    ~MagDiffPetscSolverImpl () {
        if (m_x)   { VecDestroy(&m_x); }
        if (m_b)   { VecDestroy(&m_b); }
        if (m_A)   { MatDestroy(&m_A); }
        if (m_P)   { MatDestroy(&m_P); }
        if (m_ksp) { KSPDestroy(&m_ksp); }
    }

    MagDiffPetscSolverImpl (MagDiffPetscSolverImpl const&) = delete;
    MagDiffPetscSolverImpl& operator= (MagDiffPetscSolverImpl const&) = delete;

    [[nodiscard]] amrex::Long nlocal () const { return m_n_local; }
    [[nodiscard]] amrex::Long rstart () const { return m_rstart; }

    amrex::Array<amrex::iMultiFab const*,3> gindexView () const {
        return {m_gindex[0].get(), m_gindex[1].get(), m_gindex[2].get()};
    }

    int solve (amrex::Real const* rhs, amrex::Real* sol, amrex::Real& rnorm_out) {
        PetscScalar* barr = nullptr;
        MAGDIFF_PETSC_CHK(VecGetArray(m_b, &barr));
        for (amrex::Long i = 0; i < m_n_local; ++i) { barr[i] = rhs[i]; }
        MAGDIFF_PETSC_CHK(VecRestoreArray(m_b, &barr));

        PetscScalar* xarr = nullptr;
        MAGDIFF_PETSC_CHK(VecGetArray(m_x, &xarr));
        for (amrex::Long i = 0; i < m_n_local; ++i) { xarr[i] = sol[i]; }
        MAGDIFF_PETSC_CHK(VecRestoreArray(m_x, &xarr));

        MAGDIFF_PETSC_CHK(KSPSolve(m_ksp, m_b, m_x));

        MAGDIFF_PETSC_CHK(VecGetArray(m_x, &xarr));
        for (amrex::Long i = 0; i < m_n_local; ++i) { sol[i] = static_cast<amrex::Real>(xarr[i]); }
        MAGDIFF_PETSC_CHK(VecRestoreArray(m_x, &xarr));

        PetscInt iters = 0;
        KSPConvergedReason reason = static_cast<KSPConvergedReason>(0);
        PetscReal rnorm = 0.0;
        MAGDIFF_PETSC_CHK(KSPGetIterationNumber(m_ksp, &iters));
        MAGDIFF_PETSC_CHK(KSPGetConvergedReason(m_ksp, &reason));
        MAGDIFF_PETSC_CHK(KSPGetResidualNorm(m_ksp, &rnorm));
        rnorm_out = static_cast<amrex::Real>(rnorm);

        if (reason <= 0) {
            amrex::Print() << "### WARNING: HybridMagDiffusion PETSc KSP did not"
                           << " converge (iters=" << iters
                           << " residual=" << rnorm
                           << " reason=" << reason
                           << "). Keeping last iterate.\n";
        } else if (m_verbose > 0) {
            amrex::Print() << "HybridMagDiffusion PETSc KSP iterations=" << iters
                           << " residual=" << rnorm
                           << " reason=" << reason << "\n";
        }
        return static_cast<int>(reason);
    }

private:
    static PetscErrorCode applyMatOp (Mat A, Vec in, Vec out) {
        PetscFunctionBeginUser;
        MagDiffPetscSolverImpl* self = nullptr;
        PetscCall(MatShellGetContext(A, reinterpret_cast<void**>(&self)));
        const PetscScalar* x = nullptr; PetscScalar* y = nullptr;
        PetscCall(VecGetArrayRead(in, &x));
        PetscCall(VecGetArray(out, &y));
        self->m_matvec(self->m_opctx,
                       reinterpret_cast<amrex::Real const*>(x),
                       reinterpret_cast<amrex::Real*>(y),
                       self->gindexView(), self->m_rstart);
        PetscCall(VecRestoreArrayRead(in, &x));
        PetscCall(VecRestoreArray(out, &y));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode applyShellPC (PC pc, Vec in, Vec out) {
        PetscFunctionBeginUser;
        MagDiffPetscSolverImpl* self = nullptr;
        PetscCall(PCShellGetContext(pc, reinterpret_cast<void**>(&self)));
        const PetscScalar* x = nullptr; PetscScalar* y = nullptr;
        PetscCall(VecGetArrayRead(in, &x));
        PetscCall(VecGetArray(out, &y));
        self->m_pcapply(self->m_opctx,
                        reinterpret_cast<amrex::Real const*>(x),
                        reinterpret_cast<amrex::Real*>(y),
                        self->gindexView(), self->m_rstart);
        PetscCall(VecRestoreArrayRead(in, &x));
        PetscCall(VecRestoreArray(out, &y));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode monitorResidual (KSP ksp, PetscInt it,
                                           PetscReal rnorm, void* /*ctx*/) {
        PetscFunctionBeginUser;
        amrex::ignore_unused(ksp);
        amrex::Print() << "  HybridMagDiffusion PETSc KSP: it="
                       << it << " residual=" << rnorm << "\n";
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Copy device (or host) EB B-masks onto pinned host iMultiFabs for safe
    // LoopOnCpu access. Called once from the ctor when eb_update_B is non-null.
    void copyEbMasksToHost (
        amrex::Array<amrex::iMultiFab const*,3> const& eb_update_B)
    {
        amrex::MFInfo const host_info =
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            m_eb_mask_host[idim] = std::make_unique<amrex::iMultiFab>(
                eb_update_B[idim]->boxArray(),
                eb_update_B[idim]->DistributionMap(), 1, 0, host_info);
            // iMultiFab::Copy handles device -> pinned host under CUDA.
            amrex::iMultiFab::Copy(*m_eb_mask_host[idim], *eb_update_B[idim],
                                   0, 0, 1, 0);
        }
#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
    }

    // Per-component global DOF index for interior cells (component-major).
    // nghost=1 so neighbor columns resolve; -1 marks exterior/unknown or covered
    // B DOFs (when an EB mask is provided). Pinned host arena: PETSc scatter/gather
    // and Mat assembly use LoopOnCpu host access; device-arena iMultiFab would SEGV
    // under CUDA WarpX.
    void buildGlobalIndex (amrex::Array<amrex::MultiFab const*,3> const& B_proto) {
        amrex::MFInfo const host_info =
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            m_gindex[idim] = std::make_unique<amrex::iMultiFab>(
                B_proto[idim]->boxArray(),
                B_proto[idim]->DistributionMap(), 1, amrex::IntVect::Unit,
                host_info);
            m_gindex[idim]->setVal(-1);
        }
        // amrex::Box is BoxND<AMREX_SPACEDIM>, so iterate with the dim-agnostic
        // LoopOnCpu(lbound,ubound,(i,j,k)) (k pads to 0 in 2D/1D), not explicit
        // smallEnd(2)/bigEnd(2) which is out of range for BoxND<2>.
        PetscInt run = static_cast<PetscInt>(m_rstart);
        bool const skip_covered = (m_eb_mask_host[0] != nullptr);
        for (int idim = 0; idim < 3; ++idim) {
            for (amrex::MFIter mfi(*B_proto[idim]); mfi.isValid(); ++mfi) {
                auto gix = m_gindex[idim]->array(mfi);
                amrex::Box const& tb = mfi.tilebox();
                if (skip_covered) {
                    auto const& mask_arr = m_eb_mask_host[idim]->const_array(mfi);
                    amrex::LoopOnCpu(amrex::lbound(tb), amrex::ubound(tb),
                        [&] (int i, int j, int k) {
                            if (mask_arr(i, j, k) == 0) { return; }
                            gix(i, j, k) = static_cast<int>(run);
                            ++run;
                        });
                } else {
                    amrex::LoopOnCpu(amrex::lbound(tb), amrex::ubound(tb),
                        [&] (int i, int j, int k) {
                            gix(i, j, k) = static_cast<int>(run);
                            ++run;
                        });
                }
            }
        }
        for (int idim = 0; idim < 3; ++idim) {
            m_gindex[idim]->FillBoundary(m_geom.periodicity());
        }
    }

    // nghost=1 copy of B-centered eta for neighbor reads in the assembled Pmat.
    // Pinned host: assembleFrozenLaplacian uses LoopOnCpu host access.
    void buildEtaGhosts (amrex::Array<amrex::MultiFab const*,3> const& eta_pc) {
        amrex::MFInfo const host_info =
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        for (int idim = 0; idim < 3; ++idim) {
            m_eta_g[idim].define(eta_pc[idim]->boxArray(),
                                 eta_pc[idim]->DistributionMap(), 1,
                                 amrex::IntVect::Unit, host_info);
            m_eta_g[idim].setVal(amrex::Real(0.0));
            // eta_pc may be device memory under CUDA; Copy handles H<->D.
            amrex::MultiFab::Copy(m_eta_g[idim], *eta_pc[idim], 0, 0, 1, 0);
            m_eta_g[idim].FillBoundary(m_geom.periodicity());
        }
    }

    // Assemble P = I + theta*dt*(eta/mu0)*L_geom per component, block-diagonal
    // (no Br-Bt / Br-Bz coupling -- those live only in the matrix-free matvec;
    // P is a PC proxy, P != A). Exterior neighbors (global index -1) are dropped,
    // matching the homogeneous-operator BC the matvec imposes.
    //
    // Cartesian dims (XZ/3D/1D_Z): symmetric face-averaged scalar Laplacian,
    // identical for all three components.
    //
    // Cylindrical dims (RZ / RCYLINDER): the radial direction (AMReX dir 0 = r)
    // uses the flux-form cylindrical scalar Laplacian (1/r) d/dr (r chi d/dr f),
    // so the +/- radial off-diagonals carry r_face/r_dof weights (asymmetric,
    // one-sided at the axis); for uniform dr the radial diagonal is still
    // 2/dr^2. B_t (idim 1) additionally gets -1/r^2 on the diagonal (the
    // cylindrical vector-Laplacian azimuthal term; the Bt~r null mode:
    // nabla^2(r) - r/r^2 = 1/r - 1/r = 0), applied only off-axis where it keeps
    // the Laplacian diagonal positive -- near the axis 1/r^2 would dominate and
    // drive the assembled-P diagonal negative (ILU-unstable; the prompt forbids
    // zero/negative diags), so the near-axis Bt cell stays on the scalar flux
    // form (v1 axis compromise). The axis r=0 (only Br, NODE in r) skips the
    // radial direction entirely (1/0 + matvec pins Br(0,*) = 0). The z
    // direction (RZ, dir 1) is Cartesian (no metric). See
    // notes/2026-07-20_cylindrical_frozen_laplacian_pc.md.
    PetscErrorCode assembleFrozenLaplacian () {
        PetscFunctionBeginUser;
        auto const* const dx = m_geom.CellSize();
        amrex::Real dxinv2[3] = {amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            dxinv2[d] = amrex::Real(1.0) / (dx[d] * dx[d]);
        }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        amrex::Real const rmin = m_geom.ProbLo(0);
        amrex::Real const dr = dx[0];
#endif

        std::vector<PetscInt> cols;
        std::vector<PetscScalar> vals;
        cols.reserve(1 + 2 * AMREX_SPACEDIM);
        vals.reserve(1 + 2 * AMREX_SPACEDIM);

        for (int idim = 0; idim < 3; ++idim) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
            // r at the DOF: rmin + (i + ishift)*dr; ishift = 0 for Br (NODE in
            // r), 0.5 for Bt/Bz (CELL in r). is_bt -> apply -1/r^2 (Bt only).
            amrex::Real const ishift =
                m_r_nodal[idim] ? amrex::Real(0.0) : amrex::Real(0.5);
            bool const is_bt = (idim == 1);
#endif
            for (amrex::MFIter mfi(*m_gindex[idim]); mfi.isValid(); ++mfi) {
                auto const& gix = m_gindex[idim]->const_array(mfi);
                auto const& et = m_eta_g[idim].const_array(mfi);
                amrex::Box const& tb = mfi.tilebox();
                amrex::LoopOnCpu(amrex::lbound(tb), amrex::ubound(tb),
                [&] (int i, int j, int k) {
                    PetscInt const row = gix(i, j, k);
                    if (row < 0) { return; }  // covered B DOF (EB), skip
                    amrex::Real const chi_c =
                        std::max(et(i, j, k), amrex::Real(0.0)) / m_mu0;
                    PetscScalar diag = 1.0;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    amrex::Real const r_dof = rmin + (i + ishift) * dr;
                    // chi-free Laplacian diagonal (radial metric + z), used only
                    // to guard the Bt -1/r^2 term so the diagonal stays positive.
                    amrex::Real lap_diag_scalar = amrex::Real(0.0);
#endif
                    cols.clear();
                    vals.clear();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        int ip = i, jp = j, kp = k;
                        int im = i, jm = j, km = k;
                        if (d == 0) { ip += 1; im -= 1; }
                        else if (d == 1) { jp += 1; jm -= 1; }
                        else { kp += 1; km -= 1; }
                        PetscInt const cp = gix(ip, jp, kp);
                        PetscInt const cm = gix(im, jm, km);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                        if (d == 0) {
                            // Radial direction: cylindrical flux-form metric.
                            // Skip at the axis (r_dof <= 0, only Br at r=0) to
                            // avoid 1/0; the matvec pins Br(0,*) = 0 there.
                            if (r_dof > amrex::Real(0.0)) {
                                if (cp >= 0) {
                                    amrex::Real const r_face =
                                        r_dof + amrex::Real(0.5) * dr;
                                    amrex::Real const chi_n =
                                        std::max(et(ip, jp, kp), amrex::Real(0.0)) / m_mu0;
                                    amrex::Real const chi_f = amrex::Real(0.5) * (chi_c + chi_n);
                                    PetscScalar const v = -m_theta_dt * chi_f
                                        * dxinv2[d] * (r_face / r_dof);
                                    diag -= v;
                                    lap_diag_scalar += (r_face / r_dof) * dxinv2[d];
                                    cols.push_back(cp);
                                    vals.push_back(v);
                                }
                                if (cm >= 0) {
                                    amrex::Real const r_face =
                                        r_dof - amrex::Real(0.5) * dr;
                                    amrex::Real const chi_n =
                                        std::max(et(im, jm, km), amrex::Real(0.0)) / m_mu0;
                                    amrex::Real const chi_f = amrex::Real(0.5) * (chi_c + chi_n);
                                    PetscScalar const v = -m_theta_dt * chi_f
                                        * dxinv2[d] * (r_face / r_dof);
                                    diag -= v;
                                    lap_diag_scalar += (r_face / r_dof) * dxinv2[d];
                                    cols.push_back(cm);
                                    vals.push_back(v);
                                }
                            }
                            continue;  // radial handled; z (RZ d=1) is Cartesian below
                        }
#endif
                        // Cartesian direction (z in RZ; any dir in XZ/3D/1D_Z):
                        // unchanged face-averaged Laplacian.
                        if (cp >= 0) {
                            amrex::Real const chi_n =
                                std::max(et(ip, jp, kp), amrex::Real(0.0)) / m_mu0;
                            amrex::Real const chi_f = amrex::Real(0.5) * (chi_c + chi_n);
                            PetscScalar const v =
                                -m_theta_dt * chi_f * dxinv2[d];
                            diag -= v;
                            cols.push_back(cp);
                            vals.push_back(v);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                            lap_diag_scalar += dxinv2[d];
#endif
                        }
                        if (cm >= 0) {
                            amrex::Real const chi_n =
                                std::max(et(im, jm, km), amrex::Real(0.0)) / m_mu0;
                            amrex::Real const chi_f = amrex::Real(0.5) * (chi_c + chi_n);
                            PetscScalar const v =
                                -m_theta_dt * chi_f * dxinv2[d];
                            diag -= v;
                            cols.push_back(cm);
                            vals.push_back(v);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                            lap_diag_scalar += dxinv2[d];
#endif
                        }
                    }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                    // B_t (idim 1): cylindrical vector Laplacian adds -1/r^2.
                    // Apply only off-axis (lap_diag_scalar - 1/r^2 > 0) so the
                    // assembled-P diagonal stays positive (no zero/negative
                    // diags); near the axis the Bt cell keeps the scalar flux
                    // form (the v1 axis compromise documented above).
                    if (is_bt && r_dof > amrex::Real(0.0)) {
                        amrex::Real const inv_r2 =
                            amrex::Real(1.0) / (r_dof * r_dof);
                        if (lap_diag_scalar - inv_r2 > amrex::Real(0.0)) {
                            diag -= m_theta_dt * chi_c * inv_r2;
                        }
                    }
#endif
                    cols.push_back(row);
                    vals.push_back(diag);
                    PetscInt const ncol = static_cast<PetscInt>(cols.size());
                    MatSetValues(m_P, 1, &row, ncol,
                                 cols.data(), vals.data(), INSERT_VALUES);
                });
            }
        }
        PetscCall(MatAssemblyBegin(m_P, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(m_P, MAT_FINAL_ASSEMBLY));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    amrex::Geometry m_geom;
    amrex::Real m_theta_dt = amrex::Real(0.0);
    amrex::Real m_mu0 = amrex::Real(0.0);
    MagDiffPetscPC m_pc_choice = MagDiffPetscPC::shell_jacobi;
    amrex::Real m_rtol = amrex::Real(0.0);
    amrex::Real m_atol = amrex::Real(0.0);
    int m_max_iter = 0;
    int m_verbose = 0;
    MagDiffMatvecFn m_matvec = nullptr;
    MagDiffPCFn m_pcapply = nullptr;
    void* m_opctx = nullptr;

    PetscInt m_n_local = 0;
    PetscInt m_n_global = 0;
    amrex::Long m_rstart = 0;
    Vec m_x = nullptr;
    Vec m_b = nullptr;
    Mat m_A = nullptr;
    Mat m_P = nullptr;
    KSP m_ksp = nullptr;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_gindex;
    // r-direction staggering per B component (1 = NODE in r, 0 = CELL in r),
    // read from B_proto's ixType. Used by assembleFrozenLaplacian to place the
    // cylindrical 1/r metric at the correct face (Br is NODE in r; Bt/Bz are
    // CELL in r -- see ComputeCurlACylindrical). Only read under the
    // WARPX_DIM_RZ / WARPX_DIM_RCYLINDER guard; harmless otherwise.
    std::array<int,3> m_r_nodal{{0, 0, 0}};
    amrex::Array<amrex::MultiFab,3> m_eta_g;
    // Host (pinned) copy of eb_update_B when EB is on. Empty when EB off.
    // Never LoopOnCpu into the live device eb_update_B MultiFabs.
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_eb_mask_host;
};

} // namespace

struct MagDiffPetscSolver { std::unique_ptr<MagDiffPetscSolverImpl> impl; };

MagDiffPetscSolver* magdiff_petsc_make (
    amrex::Array<amrex::MultiFab const*,3> const& B_proto,
    amrex::Array<amrex::MultiFab const*,3> const& eta_pc,
    amrex::Geometry const& geom, amrex::Real theta_dt, amrex::Real mu0,
    MagDiffPetscPC pc_choice, amrex::Real rtol, amrex::Real atol,
    int max_iter, int verbose,
    MagDiffMatvecFn matvec, MagDiffPCFn pcapply, void* opctx,
    amrex::Array<amrex::iMultiFab const*,3> const* eb_update_B)
{
    auto* s = new MagDiffPetscSolver;
    s->impl = std::make_unique<MagDiffPetscSolverImpl>(
        B_proto, eta_pc, geom, theta_dt, mu0, pc_choice,
        rtol, atol, max_iter, verbose, matvec, pcapply, opctx, eb_update_B);
    return s;
}

amrex::Long magdiff_petsc_nlocal (MagDiffPetscSolver const* s) {
    return s->impl->nlocal();
}
amrex::Long magdiff_petsc_rstart (MagDiffPetscSolver const* s) {
    return s->impl->rstart();
}
amrex::Array<amrex::iMultiFab const*,3>
magdiff_petsc_gindex (MagDiffPetscSolver const* s) {
    return s->impl->gindexView();
}

int magdiff_petsc_solve (MagDiffPetscSolver* s, amrex::Real const* rhs,
                         amrex::Real* sol, amrex::Real& residual_norm)
{
    return s->impl->solve(rhs, sol, residual_norm);
}

void magdiff_petsc_destroy (MagDiffPetscSolver* s) {
    delete s;
}

#else // !AMREX_USE_PETSC

// Stubs so the translation unit links without PETSc. These are never called:
// HybridMagDiffusion.cpp only reaches them inside its own AMREX_USE_PETSC guard.

MagDiffPetscSolver* magdiff_petsc_make (
    amrex::Array<amrex::MultiFab const*,3> const&,
    amrex::Array<amrex::MultiFab const*,3> const&,
    amrex::Geometry const&, amrex::Real, amrex::Real,
    MagDiffPetscPC, amrex::Real, amrex::Real, int, int,
    MagDiffMatvecFn, MagDiffPCFn, void*,
    amrex::Array<amrex::iMultiFab const*,3> const*)
{
    amrex::Abort("magdiff_petsc_make: WarpX was not built with PETSc "
                 "(AMREX_USE_PETSC undefined).");
    return nullptr;
}

amrex::Long magdiff_petsc_nlocal (MagDiffPetscSolver const*) { return 0; }
amrex::Long magdiff_petsc_rstart (MagDiffPetscSolver const*) { return 0; }
amrex::Array<amrex::iMultiFab const*,3>
magdiff_petsc_gindex (MagDiffPetscSolver const*) { return {nullptr, nullptr, nullptr}; }

int magdiff_petsc_solve (MagDiffPetscSolver*, amrex::Real const*, amrex::Real*,
                         amrex::Real&) { return -1; }

void magdiff_petsc_destroy (MagDiffPetscSolver*) {}

#endif // AMREX_USE_PETSC
