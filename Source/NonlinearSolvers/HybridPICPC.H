/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * Cartesian and collocated implementation imported from Prabhat Kumar,
 * impl_ext_vec_potential@c53f7d915b46386d2009889c5f5449171c99c23a.
 * Native staggered RZ is implemented separately in HybridPICRZPC.H.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_PIC_PC_H_
#define WARPX_HYBRID_PIC_PC_H_

#include "Preconditioner.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/utils/Enums.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

/**
 * \brief Physics-based EMHD preconditioner for the theta-implicit hybrid-PIC
 *  solver, after L. Chacon, JCP 530 (2025) 113910.
 *
 *  With particles, density and electron pressure frozen, the only dependence
 *  of E_Ohm on the Newton iterate E is through Faraday's law,
 *  dB = -theta*dt curl(dE), giving the stiff electron (EMHD) block
 *
 *    J_pc dE = dE + U (curl curl dE),
 *    U v     = c_w (v x bhat) + a_e v - a_H del2 v,
 *    c_w     = theta*dt |B0| / (mu0 rho_c),  a_e = theta*dt eta / mu0,
 *    a_H     = theta*dt eta_H / mu0,
 *  with eta frozen at the linearization state: eta(rho, |J0|),
 *  J0 = curl B0 / mu0 (freezing at |J| = 0 would underestimate anomalous,
 *  J-dependent resistivity precisely in current sheets), plus the frozen
 *  electron-convection term -theta*dt ue0 x curl(dE) of the E-row Jacobian,
 *  ue0 = (J_i - J_plasma)0 / rho_c.
 *
 *  Primal pair. The auxiliary W = curl curl dE is kept as a second unknown
 *  (the parabolization of Chacon 2025 Sec. 3 applied globally) and the pair
 *
 *    E-row: xE + (I+S)^-1 [ U xW - theta*dt ue0 x (curl xE) ] = b~
 *    W-row: xW - curlcurl xE = 0
 *
 *  is smoothed with a per-node joint 2x2 block damped-Jacobi sweep (Chacon
 *  2025 Sec. 4.1): the defect correction builds the block with curlcurl ->
 *  l2 while all residuals use the exact discrete operators, so the fixed
 *  point is exact,
 *
 *    P dxE = rE - U_loc rW / s_W,   dxW = rW + s_W l2 dxE,
 *    P     = alpha I + beta (. x bhat),
 *    alpha = 1 + theta*dt |ue0| l1 + l2 (a_e + 2 l2 a_H),  beta = wdf l2 c_w,
 *    P^-1 v = [alpha v - beta (v x bhat) + (beta^2/alpha)(v.bhat) bhat]
 *             / (alpha^2 + beta^2),
 *
 *  where l2 is the Nyquist symbol of -del2 (sum_d 2/dx_d^2 for the wide
 *  central curl composition of the collocated Cartesian grid, sum_d 4/dx_d^2
 *  for the compact composition of the staggered and RZ grids), l1 = sum_d
 *  2/dx_d that of the first derivative, the hyper-resistive defect uses the
 *  compact-Laplacian Nyquist symbol 2 l2, and wdf = pair_whistler_defect
 *  places the whistler coupling defect inside the curl-curl symbol (the wide
 *  central composition peaks at l2/2). The W-half is stored scaled by the
 *  local weight s_W = max(c_w, 1/l2) (coefficient comp 14), so the
 *  W-residual of each node is weighted by its own E-row coupling ||U_loc||:
 *  this makes the block norm E-space accurate on strong density-contrast
 *  profiles. The curl-free (electrostatic) content of b -- physical and
 *  large in hybrid PIC -- lives on the E-row identity with an O(1) diagonal,
 *  and the whistler stiffness is inverted analytically inside the block.
 *  The m_xF/bF/rF containers hold the E-half, m_xW/bW/rW the scaled W-half.
 *
 *  Multigrid: geometric V(3,3) cycles (factor-2 coarsening while the boxes
 *  allow, same DistributionMapping; coefficient injection, l2 rebuilt per
 *  level; nodal full-weighting restriction, (tri)linear prolongation; 8
 *  bottom sweeps). The V-cycle preconditions an inner right-preconditioned
 *  GMRES (CGS2 orthogonalization, <= inner_max iterations to inner_rtol in
 *  the s_W-weighted block norm, which is E-space accurate): the Krylov
 *  space absorbs the per-level Nyquist rings of the central-difference
 *  curl that smoothing alone cannot, cheaply relative to the outer
 *  iterations (each an evaluation of the particle residual). The inner
 *  solve stops at an RHS-dependent iteration count, so the preconditioner
 *  is NONLINEAR and requires a flexible outer Krylov method
 *  (newton.linear_solver = amrex_fgmres, or petsc_ksp with
 *  PETSC_OPTIONS="-ksp_type fgmres"). With inner_max = 0 one application is instead vcycles
 *  V-cycles: a fixed linear operator valid under any outer solver
 *  (amrex_gmres included), at the price of more outer iterations on
 *  problems where the V-cycle stalls.
 *
 *  Physical boundaries: PEC(_Insulator) faces are modeled exactly (identity
 *  tangential-E rows with dead columns and parity-mirrored delta-B/delta-J
 *  ghosts, matching the residual's boundary application; see ClosureClass);
 *  other non-periodic BCs fall back to a homogeneous zero-extension closure.
 *  With embedded boundaries, EB-covered nodes are masked off the same nodal
 *  eb_update_E flag the residual's Ohm solves consume and act as identity
 *  rows.
 *
 *  With implicit_evolve.use_mass_matrices_pc = 1 the deposited reduced
 *  mass-matrix diagonal Sigma = dJ/dE (the ion response, as consumed by the
 *  EM-PIC preconditioners) is folded through the Hall term as
 *  J_pc = (I + S) + U K, S v = (Sigma v) x B0 / rho_c, with the exact
 *  reduction preserved via the pointwise closed-form (I+S)^-1.
 *
 *  Geometries: the collocated (nodal) grid in Cartesian 1D/2D/3D and in RZ
 *  restricted to the m = 0 azimuthal mode (the residual's CylindricalYee
 *  up/down stencil compositions, on-axis branches and axis mirror parities
 *  are reproduced exactly; the r = 0 axis and the PEC outer wall are part of
 *  the boundary closure), and the staggered (Yee) Cartesian grid with
 *  periodic BCs. On the staggered grid the internal objects keep the
 *  single-MultiFab layout on a CELL-indexed container with per-component
 *  staggered interpretation (E/J components cell-centered along their own
 *  direction, B components nodal along theirs; every component's index
 *  range fits the cell box, and periodic FillBoundary produces the correct
 *  images for both interpretations). Index-space the Yee curls are the
 *  compact up/down differences, the Hall rotation carries the residual's
 *  node-interpolation chain (HybridPICPCUStag), and the smoother is the
 *  node-sandwich block of SmoothLevel (every component-coupling term goes
 *  through the operator's own down-average/cross/up-average chain).
 *
 *  Input parameters (prefix pc_hybrid_pic.):
 *    sigma                 (float, 1.3)    Jacobi damping of the E rows, in (0, 2)
 *    sigma_w               (float, 1.2)    Jacobi damping of the W rows, in (0, 2)
 *    pair_whistler_defect  (float, 0.4)    whistler defect wdf, in (0, 1]
 *    pair_sw_min           (float, 0.0)    floor of the local W-hat scale as a
 *                                          fraction of the global one, in [0, 1]
 *    pe_row_sweeps         (int, 0)        upwind/conduction sweeps on the pe
 *                                          row after its pointwise solve
 *    mg_floor              (int, 4)        coarsest multigrid box width
 *    inner_max             (int, 8)        inner-GMRES iteration cap; 0 = no
 *                                          inner Krylov (linear V-cycle PC)
 *    inner_rtol            (float, 0.3)    inner-GMRES relative tolerance
 *                                          (1.0 = one V-cycle + one Krylov
 *                                          step per application)
 *    vcycles               (int, 1)        V-cycles per application when
 *                                          inner_max = 0
 *    verbose               (bool, false)
 *  and implicit_evolve.pe_ue_cap_fac (float, 1.0): the residual's marker-CFL
 *  cap of |ue| in the pe advance, mirrored by the pe-row leg.
 */

#include "HybridPICPCKernels.H"

template <class T, class Ops>
class HybridPICPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;
        using MF = amrex::MultiFab;

        /** V-cycle smoothing depths (Chacon 2025 Sec. 4: V(3,3), 8 bottom
         *  sweeps). The pre-smooth carries the x-is-zero initialization, so
         *  kNuPre >= 1 is required. A grid that cannot be coarsened (see
         *  mg_floor) is smoothed kNuSingle times per cycle instead. */
        static constexpr int kNuPre = 3;
        static constexpr int kNuPost = 3;
        static constexpr int kNuBottom = 8;
        static constexpr int kNuSingle = 48;
        /** Components of a Krylov block: the W-half (0..2) and the E-half
         *  (3..5) of the pair. */
        static constexpr int kNKB = 6;
        /** Largest number of Krylov coefficients passed by value to the
         *  block kernels (GpuArray kernel argument); bounds inner_max. */
        static constexpr int kMaxKry = 64;
        /** Thread-block size of the batched dot-product kernel. */
        static constexpr int kDotThreads = 256;
        /** Iterate ghost layers: one exchange feeds U (1) + curl (1) + curl
         *  (1) per sweep. MG coarsening keeps boxes >= mg_floor >= 4 cells
         *  so periodic exchanges never wrap more than one period. */
        static constexpr int kNGrow = 3;
        /** Number of frozen coefficient components (see m_coef). */
        static constexpr int kNCoef = 15;

        HybridPICPC () = default;
        ~HybridPICPC () override = default;

        HybridPICPC(const HybridPICPC&) = delete;
        HybridPICPC& operator=(const HybridPICPC&) = delete;
        HybridPICPC(HybridPICPC&&) noexcept = delete;
        HybridPICPC& operator=(HybridPICPC&&) noexcept = delete;

        void Define (const T&, Ops*) override;

        /** Refreeze the coefficients (bhat, c_w, eta terms) from the current
         *  Newton linearization state and inject them down the hierarchy. */
        void Update (const T& a_U) override;

        /** Approximately solve J_pc x = b on the primal pair: the inner
         *  GMRES preconditioned by multigrid V-cycles, or vcycles V-cycles
         *  when inner_max = 0. */
        void Apply (T& a_x, const T& a_b) override;

        void printParameters () const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

        /* Implementation details; public because CUDA requires the enclosing
         * function of an extended device lambda to have public access. */

        /** Pair staging: m_tmp = F = curl xE on growntilebox(2) from the
         *  ghost-filled E-half m_xF[a_lev]; tc and kk are then evaluated
         *  inline per node (HybridPICPCPairTcKk) by every consumer. */
        void StagePair (int a_lev);

        /** Damped joint block-Jacobi sweeps on level a_lev: per sweep, one
         *  ghost exchange of the two iterates, the staging, then the update. */
        void SmoothLevel (int a_lev, int a_nsweeps, bool a_x_is_zero);

        /** rE = bF - xE - tc, rW = bW - xW + kk on the valid region of a_lev. */
        void ResidualLevel (int a_lev);

        /** Inner-GMRES matvec on the fine level, in block form (see m_kryB):
         *  a_blk (ghost-filled, kNKB comps) -> a_out (kNKB comps, valid),
         *  out_W = W - kk, out_E = xE + tc. */
        void ApplyReducedOp (const MF& a_blk, MF& a_out);

        /** Block helpers of the inner GMRES (see m_kryB). SplitBlock reads
         *  the kNKB comps of a_blk starting at a_scomp into m_bW[0] (comps
         *  0..2) and m_bF[0] (comps 3..5) on the valid region; MergeBlock
         *  writes m_xW[0]/m_xF[0] into comps 0..kNKB-1 of a_blk. */
        void SplitBlock (const MF& a_blk, int a_scomp);
        void MergeBlock (MF& a_blk) const;

        /** Batched masked inner products of the block a_w with the first
         *  a_nv Arnoldi basis vectors of m_kryB: a_out[n] = <a_w, v_n> for
         *  n < a_nv and, when a_with_ww, a_out[a_nv] = <a_w, a_w>. One
         *  kernel per box (per-block partial sums, no atomics), ONE
         *  device->host copy, fixed-order host summation (deterministic),
         *  one vector MPI reduction. */
        void BatchDot (const MF& a_w, int a_nv, bool a_with_ww, RT* a_out);

        /** Fused block linear combination on the valid region:
         *  dst(comps a_dcomp..) = a_scale * ( [a_src] - sum_{n<a_nv} a_c[n] v_n ),
         *  with the a_src term present iff a_src != nullptr. One kernel. */
        void BlockAxpy (MF& a_dst, int a_dcomp, const MF* a_src, int a_nv,
                        const amrex::GpuArray<amrex::Real, kMaxKry>& a_c,
                        amrex::Real a_scale) const;

        /** Full-weighting restriction of rW[a_lev] and rF[a_lev] into
         *  bW[a_lev+1] and bF[a_lev+1]. */
        void RestrictLevel (int a_lev);

        /** (Tri)linear prolongation of xW[a_lev+1] and xF[a_lev+1] added
         *  into xW[a_lev] and xF[a_lev]. */
        void ProlongLevel (int a_lev);

        /** Coefficient injection from level a_lev-1 to a_lev. */
        void RestrictCoef (int a_lev);

        /** Which discrete curl composition stage this is: the Faraday curl
         *  (E-like -> B-like, upward stencils in RZ) or the Ampere curl
         *  (B-like -> J-like, downward stencils with the on-axis branches
         *  in RZ). Cartesian uses the same central stencil for both. */
        enum class CurlKind { EtoB, BtoJ };

        /** out = curl a on the valid region grown by a_grow (a must have
         *  ghosts), reading a's components a_scomp..a_scomp+2. */
        void ComputeCurl (const MF& a, MF& out, int a_lev, int a_grow,
                          CurlKind a_kind, int a_scomp = 0) const;

        /** Pointwise (I+S)^-1 on the valid region of the fine-level a. */
        void RotateValid (MF& a) const;

        /** Index-space copy of one solver-vector component into component
         *  a_comp of a 3-component container (MultiFab::Copy requires
         *  matching ixTypes; the cell-indexed container's valid region is a
         *  subset of every staggered component's, so the reads are safe). */
        void CopyCompIn (const MF& a_src, MF& a_dst, int a_comp, bool a_add) const;

        /** Staggered output: x_n = src_n on each component's own staggered
         *  valid box (top nodal planes read the container's ghost ring,
         *  which holds the periodic images). */
        void WriteXOut (const MF& a_src, T& a_x) const;

        /** Physical-boundary closure classes for the vector objects in the
         *  PC pipeline, mirroring the residual's PEC handling exactly
         *  (see ApplyDomainClosure): at a PEC face with normal component n,
         *   EField : linearized E-like objects (the PC input b and the
         *            E-half of the pair): tangential components zeroed ON
         *            the wall plane (ApplyEfieldBoundary is applied to the
         *            iterate AND to the Ohm output in the residual), ghosts
         *            mirror-filled tangential-odd / normal-even;
         *   BField : first-curl (delta-B-like) intermediates: normal
         *            component zeroed on the wall, ghosts tangential-even /
         *            normal-odd (ApplyBfieldBoundary after the Faraday
         *            stage);
         *   JField : delta-J-like objects (the W-half, W = curlcurl dE =
         *            -mu0 dJ_plasma / (theta dt)): ghosts tangential-odd /
         *            normal-even, NO wall write (CalculatePlasmaCurrent
         *            output receives no BC; its wall/ghost values follow
         *            from the BC-filled B);
         *   BCoef  : the coefficient-state total B (mirror ghosts with
         *            B parities, no wall write -- the copied field already
         *            satisfies its BCs);
         *   Coef   : the frozen coefficients (comps 0-2 = bhat with B-parity
         *            signs, the rest even; mirror ghosts, no wall write). */
        enum class ClosureClass { EField, BField, JField, BCoef, Coef };

        /** Apply the physical-boundary closure of class a_cls to a on level
         *  a_lev (PEC/PEC_Insulator faces only; other non-periodic BCs keep
         *  the zero-extension), then zero EB-covered nodes on the fine level
         *  when an EB mask is present. The closure indexes components
         *  RELATIVE to a_scomp (a_ncomp < 0: all remaining components). */
        void ApplyDomainClosure (MF& a, int a_lev, ClosureClass a_cls,
                                 int a_scomp = 0, int a_ncomp = -1) const;

        /** Ghost exchange of a preconditioner container on level a_lev.
         *  When the level consists of a single box spanning the domain the
         *  AMReX FillBoundary reduces to a periodic wrap of the ghost region
         *  onto the valid box, done here as ONE kernel with no host-device
         *  synchronization. Equals FillBoundary for periodically consistent
         *  data (the only case in which FillBoundary is itself well defined
         *  on a nodal periodic box): only ghost cells that overlap a
         *  periodic image of the valid box are written -- nothing at
         *  non-periodic faces (the closures handle those), the valid region
         *  including the duplicated nodal plane is untouched, and the
         *  source of a ghost is its wrap into the primary [lo, lo+N) range.
         *  Falls back to FillBoundary for multi-box layouts. */
        void FillGhosts (MF& a, int a_lev) const;

        /** Enforce the exact identity rows of the true Jacobian on a_out:
         *  wall-tangential components at PEC faces get a_out = a_in, and
         *  EB-covered nodes get a_out = a_in (the PC acts as the identity
         *  there; the covered residual rows are structurally zero). */
        void OverrideIdentityRows (T& a_out, const T& a_in) const;

    private:

        /** Recursive V(kNuPre, kNuPost) cycle for the correction on level
         *  a_lev; a non-coarsenable grid smooths kNuSingle times instead. */
        void Vcycle (int a_lev, bool a_x_is_zero);

        /** One preconditioner application for the inner GMRES (one V-cycle
         *  on the block RHS at a_scomp of a_blk); result in m_xW[0]/m_xF[0]. */
        void ApplyMToW (const MF& a_blk, int a_scomp);

        /** Inner GMRES on the pair (<= inner_max its or inner_rtol); result
         *  in m_xW[0]/m_xF[0]. Classical Gram-Schmidt with
         *  reorthogonalization (CGS2): two batched projection passes per
         *  iteration -- two host syncs -- instead of the j+2 sequential
         *  reductions of modified GS. */
        void InnerGMRES ();

        /** Inner-GMRES iteration cap (0: no inner Krylov, m_vcycles
         *  V-cycles per application) and relative tolerance. */
        int m_inner_max = 8;
        RT m_inner_rtol = 0.3;
        int m_vcycles = 1;
        /** Jacobi damping of the E rows. */
        RT m_sigma = 1.3;
        /** Damping of the W-row update. The W eigenvalue of P^-1 A is 1 on
         *  curl-null content, so sigma_w > 1 under-damps W junk while
         *  sigma > 1 speeds the whistler band. */
        RT m_sigma_w = 1.2;
        /** Whistler coupling defect wdf: d_perp = wdf*l2 in the block
         *  (beta = d_perp*c_w) and in the perp-b part of the W increment
         *  (par-b keeps l2). Collocated grid only. */
        RT m_pair_wdf = RT(0.4);
        /** Floor of the local W-hat scale s_W (coef comp 14) as a fraction
         *  of the global scale max(beta, 1)/l2, beta = l2 max(c_w): 0 keeps
         *  the purely local weight (strong density-contrast profiles such
         *  as the FRC), 1 the global one (the GEM setting). */
        RT m_pair_sw_min = RT(0.0);

        /** Ion-response leg active: the deposited reduced mass-matrix
         *  diagonal (raw dJ/dE) is available via
         *  implicit_evolve.use_mass_matrices_pc = 1 (set in Update()). */
        bool m_have_mm = false;

        /** pe-row leg: when the solver vector carries the pe scalar row
         *  (implicit_evolve.pe_newton_unknown), precondition it with a
         *  block-upper-triangular leg instead of the bare identity:
         *  x_p = D_p^-1 b_p with the pointwise diagonal D_p = 1 + coef(10),
         *  coef(10) = theta*dt*[gamma*sum_d |ue_d|/dx_d + conduction], then
         *  b_E <- b_E - grad(x_p/kappa)/rho_e before the E-system solve (the
         *  Ohm gradient coupling dF_E/dpe the identity leg left entirely to
         *  GMRES). |ue| is clamped like the residual's pe advance, ue_c =
         *  c tanh(|ue|/c), c = implicit_evolve.pe_ue_cap_fac * dx_min /
         *  (theta dt). Collocated Cartesian, in-loop pe closure only;
         *  inert for E-only vectors. */
        bool m_pe_row_active = false;
        std::unique_ptr<MF> m_pe_row_scr;   //!< ghosted dpe scratch (1 comp)
        amrex::Real m_pe_ue_cap_fac = 1.0;
        /** pc_hybrid_pic.pe_row_sweeps = n: follow the pointwise D_p^-1
         *  pe-row solve with n damped-Jacobi sweeps carrying the signed
         *  UPWIND advection stencil theta*dt*gamma ue0.grad(.) and the
         *  conduction diffusion stencil -c_k del2(.) (c_k sized so its
         *  Nyquist diagonal equals the conduction part of D_p): the
         *  off-diagonal content the pointwise diagonal drops for both legs,
         *  damped by the same D_p. Collocated Cartesian only. */
        int m_pe_row_sweeps = 0;
        std::unique_ptr<MF> m_pe_row_scr2;  //!< sweep iterate scratch (1 comp)

        /** Pair E-half: per-level E unknown/RHS/residual. */
        std::vector<std::unique_ptr<MF>> m_xF;
        std::vector<std::unique_ptr<MF>> m_bF;
        std::vector<std::unique_ptr<MF>> m_rF;
        /** Ping-pong partners of m_xW/m_xF for the collocated smoother: its
         *  fused update reads the xW neighbours (hyper-resistive Laplacian
         *  inside HybridPICPCPairTcKk) of the iterate it writes, so an
         *  in-place update is a scheduling-dependent Jacobi/Gauss-Seidel
         *  mix. Each sweep writes the partner and the unique_ptrs are
         *  swapped (no copy), making the sweep a deterministic Jacobi
         *  iteration. */
        std::vector<std::unique_ptr<MF>> m_xW2;
        std::vector<std::unique_ptr<MF>> m_xF2;

        bool m_is_defined = false;
        int m_mg_floor = 4;
        bool m_verbose = false;
        /** Per level: single-kernel ghost fill eligible (see FillGhosts). */
        std::vector<bool> m_fastfill_ok;
        /** True when the level solves need no pre-zeroed x: the collocated
         *  sweep writes x = 0 + sigma dE on the first sweep of a zero
         *  initial guess and never reads x before (ghosts are refilled from
         *  the written valid region), so the setVal kernels per level per
         *  V-cycle are skipped. */
        bool m_zero_free = false;

        Ops* m_ops = nullptr;

        int m_nlevels = 1;
        amrex::Vector<amrex::Geometry> m_geom;

        /** Per level: l2, the Nyquist symbol of -del2 (see the class docs). */
        std::vector<RT> m_l2;
        /** Per level: l1 = sum_d 2/dx_d (Nyquist symbol of the first
         *  derivative; sizes the block's convection diagonal). */
        std::vector<RT> m_l1;
        /** Per level: the stencil kernels' grid constants. */
        std::vector<HybridPICPCGrid> m_grid;

        /** Per-node frozen coefficients (per level, kNCoef comps):
         *  comp 0-2  : bhat (unit total B),
         *  comp 3    : c_w = theta*dt |B| / (mu0 rho_e),
         *  comp 4    : a_e = theta*dt eta(rho, |J0|) / mu0,
         *  comp 5    : a_H = theta*dt eta_H / mu0,
         *  comp 6-8  : w_c = Sigma_cc |B| / rho_e (deposited-mass-matrix
         *              ion response through the Hall term; zero unless
         *              implicit_evolve.use_mass_matrices_pc = 1),
         *  comp 9    : rho_e (the floored charge density; used by the
         *              staggered Hall chain, which -- like the residual --
         *              crosses at the nodes WITHOUT the density and divides
         *              by rho interpolated to the component location),
         *  comp 10   : pe-row diagonal D_p - 1 (level 0 only),
         *  comp 11-13: theta*dt ue0 = theta*dt (J_i - J_plasma)/rho_e
         *              (pair convection),
         *  comp 14   : local W-hat scale s_W = max(c_w, 1/l2) (pair). */
        std::vector<std::unique_ptr<MF>> m_coef;

        /** Per level (all 3-component): scaled W-half iterate (kNGrow
         *  ghosts), RHS (0), residual (1, for restriction), the staged F =
         *  curl xE, and the staggered node-sandwich scratch. */
        std::vector<std::unique_ptr<MF>> m_xW;
        std::vector<std::unique_ptr<MF>> m_bW;
        std::vector<std::unique_ptr<MF>> m_rW;
        std::vector<std::unique_ptr<MF>> m_tmp;
        std::vector<std::unique_ptr<MF>> m_tmp2;

        /** Fine level: ghosted copy of the PC input b (2 ghosts; doubles as
         *  the B scratch of the coefficient build). */
        std::unique_ptr<MF> m_bE;

        /** Inner-GMRES storage in BLOCK form: every Krylov object carries
         *  kNKB components -- the W-half in comps 0..2, the E-half in comps
         *  3..5 -- so each vector operation is ONE kernel, and the whole
         *  Arnoldi basis lives in a single MultiFab (basis vector j = comps
         *  [j*kNKB, (j+1)*kNKB)), so the Gram-Schmidt projections onto all
         *  j+1 basis vectors are a single batched kernel. */
        std::unique_ptr<MF> m_kryB;   //!< (inner_max+1)*kNKB comps, 0 ghosts
        std::unique_ptr<MF> m_bKB;    //!< RHS block, 0 ghosts
        std::unique_ptr<MF> m_wB;     //!< A M v_j, 0 ghosts
        std::unique_ptr<MF> m_zB;     //!< M v_j, kNGrow ghosts (matvec input)
        /** Batched dot-product scratch: per-block partial sums, slot-major
         *  [slot][global block], one device->host copy per pass. */
        amrex::Gpu::DeviceVector<amrex::Real> m_dot_dev;
        amrex::Gpu::PinnedVector<amrex::Real> m_dot_host;
        std::vector<int> m_dot_nblocks;      //!< per local box
        std::vector<int> m_dot_box_offset;   //!< prefix sum of m_dot_nblocks
        int m_dot_nblocks_total = 0;

        /** Physical-BC closure state: per index dimension and side, whether
         *  the face is a PEC(_Insulator) wall the closure must model; the
         *  index-dim -> vector-component map; and the EB mask (the nodal
         *  eb_update_E flag on the collocated grid; nullptr without EB). */
        bool m_has_pec_wall = false;
        /** RZ: the r-lo face is the symmetry axis (rmin == 0). */
        bool m_has_axis = false;
        /** Staggered (Yee) grid: solver-vector components carry their
         *  native staggerings; internally everything is stored on the
         *  nodal-superset layout with per-component staggered
         *  interpretation. E_c is cell-centered along the index dimension
         *  that maps to component c (m_dim2comp) and nodal elsewhere. */
        bool m_staggered = false;
        bool m_wall_pec[AMREX_SPACEDIM][2] = {};
        /** Non-periodic faces under the zero-extension fallback: on the
         *  staggered grid the hi-side wall plane of components nodal along
         *  the face normal lives in the container's ghost region and is not
         *  modeled -- those rows are made exact identity rows instead. */
        bool m_wall_open[AMREX_SPACEDIM][2] = {};
        int m_dim2comp[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 1, 2)};
        const amrex::iMultiFab* m_eb_flag = nullptr;

};

#include "HybridPICPCSetup.H"
#include "HybridPICPCMultigrid.H"
#include "HybridPICPCAlgebra.H"
#include "HybridPICPCBoundary.H"
#include "HybridPICPCSolve.H"

#endif
