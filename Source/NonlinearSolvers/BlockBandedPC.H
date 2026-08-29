/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef BLOCK_BANDED_PC_H_
#define BLOCK_BANDED_PC_H_

#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/TextMsg.H"
#include "Preconditioner.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Loop.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

#ifdef WARPX_USE_CUDSS
#include <cudss.h>
#define WARPX_CUDSS_CHECK(call)                                              \
    do {                                                                     \
        const cudssStatus_t _cudss_status = (call);                         \
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(                                    \
            _cudss_status == CUDSS_STATUS_SUCCESS,                           \
            "pc_block_banded cuDSS call failed: " #call);                    \
    } while (0)
#endif

/**
 * \brief Block-banded direct preconditioner for the theta-implicit hybrid
 *  (generalized Ohm's law) solver in RZ geometry.
 *
 *  Chacon-family (JCP 526 (2025) 113789) treatment of the whistler-stiff
 *  Ohm Jacobian, taken all the way to a direct solve: the frozen-coefficient
 *  linearization of F(E) = E - E_ohm(B(E), J(E), rho, Pe) is assembled
 *  EXACTLY (by probing, see below) into a matrix over the E unknowns,
 *  permuted into one dense block per radial index -- each block holding all
 *  three E components over the full z-line of the box, so every z-wavelength
 *  of the whistler/Hall inter-component coupling (and the periodic-z wrap)
 *  lives INSIDE the dense blocks -- and factored by block-banded LU across
 *  the radial block index. A point-block PC (pc_hall_jacobi) only damps the
 *  grid-scale whistler cluster; this one removes the entire branch.
 *
 *  THE LINEARIZED OPERATOR (frozen coefficients at the current iterate;
 *  ion deposits are frozen during FD-Jacobian probes, so this is the same
 *  operator GMRES sees up to the deliberately omitted legs listed below):
 *
 *      delta_B = theta*dt * FaradayCurl(delta_E)         (EvolveBCylindrical
 *                                                         expressions, incl.
 *                                                         axis rules)
 *      delta_J = AmpereCurl(delta_B)/mu0                 (CalculateCurrent-
 *                                                         AmpereCylindrical,
 *                                                         incl. axis rules)
 *      delta_enE_nodal = delta_J_nodal x B0_nodal        (whistler leg)
 *                      + Je0_nodal x delta_B_nodal       (drift/motional leg,
 *                                                         Je0 = (J0 if Hall
 *                                                         else 0) - Ji0)
 *      J*delta_E = delta_E - [ Interp(delta_enE)/rho_lim (Holmstrom-masked)
 *                              + eta(rho,|J0|,t) * delta_J
 *                              - eta_H(rho,|B0|) * Lap(delta_J)
 *                              + Interp(ci * delta_J_nodal)  (electron
 *                                inertia: ci = (m_e_eff/e) s_dt / rho_lim,
 *                                zero in true vacuum) ]
 *
 *  mirroring HybridPICSolveECylindrical's nodal assembly and per-component
 *  stagger interpolations, with m=0 axis parity folds (B_theta, B_r, J_r
 *  odd; B_z, J_z even) and the kernel's axis special cases (E_theta and
 *  J_theta identically zero on axis; J_z axis regularization 4*B_theta/dr).
 *
 *  Omitted from the PC by choice (frozen-coefficient block-PC evidence,
 *  Chacon JCP 526 (2025): unnecessary for convergence-in-first-restart):
 *  the mass-matrix ion response, the continuity/density response, the
 *  grad-Pe closure response (frozen deposits => frozen Pe during probes),
 *  Darwin E_L rows, and the delta-legs of the inertia term's advective
 *  (Je.grad)u and u*drho/dt pieces (electron-drift-CFL small; the exact
 *  dJe/dt chain above is the stiff piece and is carried in full).
 *
 *  ASSEMBLY BY PROBING (no hand-derived composite stencils): the chain
 *  above is implemented as a host-side "frozen apply" that mirrors the
 *  runtime kernels' difference formulas; the banded matrix entries are
 *  extracted by applying it to colored unit vectors (spacing wider than
 *  twice the operator reach), and the extracted matrix is self-validated
 *  against the frozen apply on a random vector at every rebuild (abort on
 *  disagreement -- an underestimated band would silently degrade GMRES).
 *  The radial block bandwidth is MEASURED at build time (expected 2).
 *
 *  PARALLEL/MULTI-BOX: each box gets its own block system with couplings
 *  truncated at box boundaries (additive-Schwarz closure; exact for a
 *  single box). Long-z coupling is only captured within a box, so decks
 *  should decompose in r if possible (z-lines stay intact per rank).
 *  Duplicated nodal DOFs at box seams are reconciled by OverrideSync.
 *
 *  COST: rebuild = colored probing (cheap) + block-banded LU
 *  (~nb * (3*nz)^3 flops, host, OpenMP-threaded); apply = banded block
 *  forward/back substitution. Rebuild cadence is controlled by
 *  precond.bb_update_interval (default: once per time step). On GPU builds
 *  coefficients and RHS vectors are staged through pinned host memory
 *  (first implementation; a cuSOLVER path is the follow-up if the host
 *  solve shows up in profiles).
 *
 *  Knobs:
 *    pc_block_banded.verbose        (bool, default false)
 *    precond.bb_update_interval     (int, default 1: rebuild on the first
 *                                    Update() of every N-th step; 0 =
 *                                    rebuild every Newton iteration)
 *    precond.bb_include_drift       (bool, default true: the Je0 x delta_B
 *                                    motional leg)
 *    precond.bb_include_hyper       (bool, default true when the model runs
 *                                    hyper-resistivity)
 *    precond.bb_verify              (bool, default false: FD-JVP vs
 *                                    assembled-operator agreement gate +
 *                                    factorization round-trip check)
 *    precond.bb_max_mem_gb          (Real, default 8: abort if the block
 *                                    storage estimate exceeds this)
 */

template <class T, class Ops>
class BlockBandedPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        BlockBandedPC () = default;
        ~BlockBandedPC () override = default;

        BlockBandedPC(const BlockBandedPC&) = delete;
        BlockBandedPC& operator=(const BlockBandedPC&) = delete;
        BlockBandedPC(BlockBandedPC&&) noexcept = delete;
        BlockBandedPC& operator=(BlockBandedPC&&) noexcept = delete;

        void Define (const T&, Ops*) override;
        void Update (const T& a_U) override;
        void Apply (T&, const T&) override;
        void printParameters() const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

        /** pc_curlcurl_banded entry: follow hybrid_pic_model.esolve. With
         *  e_form the assembly is exactly pc_block_banded's; with
         *  curlcurl_form the outer Jacobian is J = A^{-1} (A - N) -- A the
         *  multiplied-through inner elliptic operator (beta + volume-scaled
         *  curl-curl + gated vacuum-Gauss closure), N the numerator
         *  response (the delta-B -> delta-J -> delta[(J-Ji)xB] chain
         *  without the 1/rho division, plus rho-weighted eta/eta_H legs) --
         *  so the PC applies P^{-1} r = (A - N)^{-1} (A r): one banded
         *  matvec (A, probed unfactored) + one factored banded solve.
         *  Must be called before Define(). */
        inline void SetFormAware (bool a_v) { m_form_aware = a_v; }

    protected:

        bool m_is_defined = false;
        bool m_verbose = false;
        /** Follow the configured esolve form (pc_curlcurl_banded). */
        bool m_form_aware = false;
        /** Resolved in Define(): assemble the curlcurl_form composition. */
        bool m_form_curlcurl = false;
        /** Hall term active (cached per Update): gates the whistler
         *  delta-J x B0 leg of the curlcurl_form numerator response (the
         *  kernel's delta-Je is delta-J only when Hall is on; the ion
         *  deposit is frozen). */
        bool m_hall_on = true;
        /** Electron inertia active in the frozen operator (set at each
         *  coefficient fill from the hybrid model; routes the exact nodal
         *  inertia chain in the frozen apply). */
        bool m_inertia_exact = false;
        RT m_theta = 0.5;
        int m_update_interval = 1;
        bool m_include_drift = true;
        bool m_include_hyper = true;
        /** Follows hybrid_pic_model.hyper_resistivity_curlcurl (cached per
         *  Update): eta_H rows use the composed curl(eta_H curl dJ) instead
         *  of the truncated -eta_H lap(dJ), matching the configured
         *  residual operator either way (same +-2 band reach). */
        bool m_hyperres_cc = false;
        bool m_verify = false;
        RT m_max_mem_gb = 8.0;
        /** PEC-wall treatment: -1 auto (from the upper-r field BC),
         *  0 off (plain truncation), 1 force on. */
        int m_wall_identity = -1;
        /** Restricted-additive-Schwarz overlap depth (cells) in r at box
         *  seams (Schwarz mode only). Dirichlet-truncated Schwarz -- any
         *  overlap -- stalls GMRES at large whistler CFL: the wave operator
         *  does not localize. Measured on 8/16-cell r-boxes at dt x8. */
        int m_overlap = 8;
        /** Device direct solve via NVIDIA cuDSS (WarpX_CUDSS=ON builds):
         *  the extracted band system is handed to cuDSS as CSR --
         *  nested-dissection reordering cuts the factorization cost ~100x
         *  vs the band ordering, and factor+solve run on the IOProcessor's
         *  GPU with the solution Bcast (the redundant per-rank host
         *  factorization disappears). precond.bb_device_solve, default on
         *  when compiled in; requires a single system per rank (single box
         *  or replicated-global mode). */
#ifdef WARPX_USE_CUDSS
        bool m_device_solve = true;
        bool m_cudss_ready = false;
        cudssHandle_t m_cudss_handle{};
        cudssConfig_t m_cudss_config{};
        cudssData_t m_cudss_data{};
        cudssMatrix_t m_cudss_matrix{};
        cudssMatrix_t m_cudss_rhs{};
        cudssMatrix_t m_cudss_solution{};
        std::vector<int> m_csr_rowptr, m_csr_colind;
        std::vector<RT> m_csr_vals;
        amrex::Gpu::DeviceVector<int> m_d_rowptr, m_d_colind;
        amrex::Gpu::DeviceVector<RT> m_d_vals, m_d_rhs, m_d_sol;
#endif

        /** Multi-box mode: -1 auto (replicated-global whenever the E
         *  BoxArray has more than one box), 0 per-box Schwarz (research
         *  knob), 1 force replicated-global. In global mode every rank
         *  gathers the domain-wide coefficients (MB-scale), assembles and
         *  factors the SAME global band system redundantly, solves the
         *  gathered RHS redundantly per apply, and writes its owned boxes
         *  -- no seams, identical counts to a single-box run. */
        int m_global = -1;
        bool m_global_on = false;

        Ops* m_ops = nullptr;
        int m_num_amr_levels = 0;
        bool m_work_defined = false;
        RT m_last_build_time = std::numeric_limits<RT>::lowest();
        int m_steps_since_build = 0;
        bool m_have_built = false;

        void readParameters ();

#if defined(WARPX_DIM_RZ)

        /** Host array over an (r, z) index window with axis-parity mirror
         *  reads below the window and zero reads elsewhere; z is either
         *  periodic-wrapped (box spans the full periodic extent) or
         *  zero-truncated. */
        struct PArr
        {
            std::vector<RT> v;
            int ilo=0, ihi=-1, jlo=0, jhi=-1, nj=0;
            bool zwrap=false; int zn=1;
            int mirror=0;   // 0 none, 1 nodal (image -i), 2 cell (image -1-i)
            RT msign=RT(1.0);
            bool axis=false;
            // upper (PEC-wall) mirror about node uwall: nodal image
            // 2*uwall - i, cell image 2*uwall - 1 - i
            bool uactive=false;
            int uwall=0, umirror=0;
            RT usign=RT(1.0);

            void define (int a_ilo, int a_ihi, int a_jlo, int a_jhi,
                         bool a_zwrap, int a_zn,
                         int a_mirror, RT a_msign, bool a_axis)
            {
                ilo=a_ilo; ihi=a_ihi; jlo=a_jlo; jhi=a_jhi;
                zwrap=a_zwrap; zn=std::max(a_zn,1);
                mirror=a_mirror; msign=a_msign; axis=a_axis;
                uactive=false; uwall=0; umirror=0; usign=RT(1.0);
                nj = jhi - jlo + 1;
                v.assign(std::size_t(std::max(0, ihi-ilo+1))*std::size_t(std::max(nj,0)), RT(0.0));
            }
            void setUpperMirror (int a_wall, int a_umirror, RT a_usign)
            {
                uactive=true; uwall=a_wall; umirror=a_umirror; usign=a_usign;
            }
            void setzero () { std::fill(v.begin(), v.end(), RT(0.0)); }
            [[nodiscard]] int jmap (int j) const
            {
                if (zwrap) { int t=(j-jlo)%zn; if (t<0) { t+=zn; } return t; }
                return j - jlo;
            }
            [[nodiscard]] bool jvalid (int j) const
            {
                return zwrap || (j >= jlo && j <= jhi);
            }
            [[nodiscard]] RT get (int i, int j) const
            {
                if (!jvalid(j)) { return RT(0.0); }
                if (i < ilo || i > ihi) {
                    if (axis && (mirror != 0) && i < ilo) {
                        const int im = (mirror==1) ? -i : -1-i;
                        if (im >= ilo && im <= ihi) {
                            return msign*v[std::size_t(im-ilo)*nj + jmap(j)];
                        }
                    }
                    if (uactive && i > ihi) {
                        const int im = (umirror==1) ? 2*uwall - i : 2*uwall - 1 - i;
                        if (im >= ilo && im <= ihi) {
                            return usign*v[std::size_t(im-ilo)*nj + jmap(j)];
                        }
                    }
                    return RT(0.0);
                }
                return v[std::size_t(i-ilo)*nj + jmap(j)];
            }
            RT& ref (int i, int j)
            {
                return v[std::size_t(i-ilo)*nj + jmap(j)];
            }
        };

        /** Per-box block-banded system + frozen coefficients + scratch. */
        struct BoxSys
        {
            int rlo=0, rhi=0, zlo=0, zhi=0;   // owned cell extents
            int rlo_s=0, rhi_s=0;              // system cell extents (owned
                                               // + restricted-additive-Schwarz
                                               // overlap into r neighbors)
            int nrc=0, nzc=0, nzn=0;
            bool zwrap=false, has_axis=false;
            bool wall_pec=false;   // this system reaches a PEC upper-r wall
            int wall_node=0;       // global r-node index of that wall
            int nb=0;
            int bw=2;    // allocated half-bandwidth (probing budget)
            int bwf=2;   // effective half-bandwidth (trimmed after harvest)
            std::vector<int> bsz;
            std::vector<long> boff;
            long ntot=0;
            // blk[b][d]: row-major bsz[b] x bsz[b+d-bw] block, col block b+d-bw
            std::vector<std::vector<std::vector<RT>>> blk;
            std::vector<std::vector<int>> piv;
            // curlcurl_form: the UNfactored inner elliptic operator A
            // (same block layout), applied as a matvec ahead of the
            // factored (A - N) solve
            std::vector<std::vector<std::vector<RT>>> blkA;

            // frozen coefficients
            PArr B0rn, B0tn, B0zn;          // nodal B0
            PArr Je0rn, Je0tn, Je0zn;       // nodal frozen electron current
            PArr etaEr, etaEt, etaEz;       // eta at row points
            PArr etahEr, etahEt, etahEz;    // eta_H at row points
            PArr etahBr, etahBt, etahBz;    // eta_H at face (curl J) points
                                            //   (hyper_resistivity_curlcurl)
            PArr irhoEr, irhoEt, irhoEz;    // Holmstrom-masked 1/rho_lim
            PArr adEr, adEt, adEz;          // extra diagonal (reserved; zero)
            PArr ciN;                       // nodal inertia coefficient
                                            //   (m_e_eff/e) s_dt / rho_lim,
                                            //   zero in true vacuum

            // curlcurl_form coefficient windows (allocated in that mode
            // only). The theta sector reads the LIVE midpoint density with
            // the kernels' cc-convention averages of the nodal deposit
            // (mirrored, including the known half-cell misregistration);
            // the poloidal sector reads the per-step-frozen rho^n snapshot
            // exactly as SolveEPolCurlCurlRZ does.
            PArr acTh, foldTh, rhswTh;      // theta acoef, fold rho, RHS wt
            PArr acPr, foldPr, rhswPr;      // Er rows
            PArr acPz, foldPz, rhswPz;      // Ez rows
            PArr gamN;                      // nodal vacuum-closure gate

            // scratch for the frozen apply
            PArr inEr, inEt, inEz;
            PArr dBr, dBt, dBz;
            PArr dJr, dJt, dJz;
            PArr cWr, cWt, cWz;             // eta_H * curl(delta-J) at faces
                                            //   (hyper_resistivity_curlcurl)
            PArr enR, enT, enZ;
            PArr eiR, eiT, eiZ;             // nodal inertia response ci*dJe
            PArr oEr, oEt, oEz;
            std::vector<RT> x;
        };

        std::vector<BoxSys> m_sys;

        // ghost-padded copies for the RAS overlap: coefficients (filled per
        // rebuild) and the per-apply RHS staging (filled per Apply)
        amrex::Array<amrex::MultiFab,3> m_Bpad;
        amrex::MultiFab m_rhopad;
        amrex::Array<amrex::MultiFab,3> m_Jipad;
        amrex::Array<amrex::MultiFab,3> m_bpad;
        bool m_pads_defined = false;

        // geometry facts (level 0)
        RT m_dr = 1.0, m_dz = 1.0, m_rmin = 0.0;

        /** Bounds-checked host read of a (possibly device) fab copied to
         *  pinned memory; small helper built in GatherCoefficients. */
        struct HostFab
        {
            std::vector<RT> v;
            amrex::Box bx;
            int ncomp=1;
            [[nodiscard]] RT get (int i, int j, int n=0) const
            {
                const amrex::IntVect iv(i,j);
                if (!bx.contains(iv)) { return RT(0.0); }
                const auto lo = bx.smallEnd();
                const auto len = bx.size();
                const long p = long(i-lo[0]) + long(len[0])*(long(j-lo[1])
                              + long(len[1])*long(n));
                return v[p];
            }
        };
        static void CopyFabToHost (const amrex::MultiFab& mf,
                                   const amrex::MFIter& mfi, HostFab& h);

        void BuildSystems (const T& a_U);
        void GatherCoefficients (const T& a_U);
        void FillCoeffs (BoxSys& S,
            const HostFab& hBr, const HostFab& hBt, const HostFab& hBz,
            const HostFab& hRho, const HostFab& hJir, const HostFab& hJit,
            const HostFab& hJiz, bool have_ji,
            const HostFab& hRhoPol, bool have_rho_pol);
        void GatherGlobal (const amrex::MultiFab& src, int scomp,
                           HostFab& out, int ng) const;
        void ApplyFrozenBox (BoxSys& S, bool a_A_only = false) const;
        /** curlcurl_form stage 4: rows of (A - N) (or A alone). wallEt/
         *  wallEz carry the saved wall-column inputs for the identity
         *  rows (the PEC column kill of ApplyFrozenBox). */
        void ApplyFrozenBoxCurlCurl (BoxSys& S, bool a_A_only,
            const std::vector<RT>& wallEt, const std::vector<RT>& wallEz) const;
        void ProbeAndFill (BoxSys& S, bool a_A_only = false);
        void FactorBox (BoxSys& S) const;
        void SolveBox (BoxSys& S) const;

        /** Poloidal closure node volume (without the common dz), the
         *  PolNodeVol mirror: interior r_n dr; axis half cell dr^2/8 at
         *  rmin = 0; wall half cell at the domain-top node. */
        [[nodiscard]] RT PolVol (int i) const
        {
            if (i == 0) { return (m_rmin + RT(0.25)*m_dr)*RT(0.5)*m_dr; }
            if (i == m_ihi_nd) {
                return (m_rmin + i*m_dr - RT(0.25)*m_dr)*RT(0.5)*m_dr;
            }
            return (m_rmin + i*m_dr)*m_dr;
        }
        // domain index facts (set in BuildSystems): top r node, top r cell,
        // nodal-z domain ends (Dirichlet-cap tests)
        int m_ihi_nd = 0, m_ircc_hi = 0, m_zlo_d = 0, m_zhi_nd = 0;
#ifdef WARPX_USE_CUDSS
        bool BuildCSR (const BoxSys& S);
        void CuDSSFactor (BoxSys& S);
        void CuDSSSolve (BoxSys& S);
#endif
        [[nodiscard]] bool UseDeviceSolve () const
        {
#ifdef WARPX_USE_CUDSS
            return m_device_solve && (m_sys.size() == 1)
                && (m_global_on || amrex::ParallelDescriptor::NProcs() == 1);
#else
            return false;
#endif
        }
        void SolveDispatch (BoxSys& S)
        {
#ifdef WARPX_USE_CUDSS
            if (UseDeviceSolve()) { CuDSSSolve(S); return; }
#endif
            SolveBox(S);
        }
        void SelfCheck (BoxSys& S, int isys, bool a_A_only = false) const;

    public:
        // public: contains an extended __device__ lambda (ParallelFor);
        // nvcc 13 requires such member functions be public (2fc85ed9b rule)
        void VerifyFD (const T& a_U);

    protected:

        // packed-index helpers -----------------------------------------
        // block b of box S holds [Er(rlo+b, :) (if b<nb-1)][Et(rlo+b, :)][Ez(rlo+b, :)]
        [[nodiscard]] static bool blockHasEr (const BoxSys& S, int b) { return b <= S.nb - 2; }
        [[nodiscard]] static int locEr (const BoxSys& S, int /*b*/, int j) { return S.oEr.jmap(j); }
        [[nodiscard]] static int locEt (const BoxSys& S, int b, int j)
        { return (blockHasEr(S,b) ? S.nzn : 0) + S.oEt.jmap(j); }
        [[nodiscard]] static int locEz (const BoxSys& S, int b, int j)
        { return (blockHasEr(S,b) ? S.nzn : 0) + S.nzn + S.oEz.jmap(j); }

        // dense kernels (row-major) ------------------------------------
        static void lu_factor (int n, RT* A, int* piv);
        static void lu_solve (int n, const RT* A, const int* piv, RT* b);
        static void lu_solve_T (int n, const RT* A, const int* piv, RT* b);
        static void solve_right (int m, int n, RT* X, const RT* Df, const int* piv);
        static void gemm_minus (int m, int k, int n, const RT* A, const RT* B, RT* C);
        static void gemv_minus (int m, int n, const RT* A, const RT* x, RT* y);
        static void ApplyBlocks (const BoxSys& S,
            const std::vector<std::vector<std::vector<RT>>>& blk,
            const std::vector<RT>& v, std::vector<RT>& w);

        /** Host worker count for the dense kernels on builds without
         *  OpenMP (GPU builds): WARPX_PC_BB_THREADS, default 16. */
        static int PCThreads ()
        {
            static const int nt = [] {
                int v = 16;
                if (const char* e = std::getenv("WARPX_PC_BB_THREADS")) {
                    v = std::atoi(e);
                }
                return std::max(1, std::min(64, v));
            }();
            return nt;
        }

        /** Parallel loop over [0, n): OpenMP when available, std::thread
         *  chunks otherwise (GPU builds compile host code without OMP).
         *  grain = minimum n before threads engage. */
        template <typename F>
        static void ParFor (int n, int grain, F&& f)
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel for schedule(static) if (n > grain)
            for (int i = 0; i < n; i++) { f(i); }
#else
            const int nt = (n > grain)
                ? std::min(PCThreads(), std::max(1, n/std::max(grain,1))) : 1;
            if (nt <= 1) {
                for (int i = 0; i < n; i++) { f(i); }
                return;
            }
            std::vector<std::thread> ts;
            ts.reserve(nt - 1);
            const int chunk = (n + nt - 1)/nt;
            for (int t = 1; t < nt; t++) {
                const int lo = t*chunk;
                const int hi = std::min(n, lo + chunk);
                if (lo >= hi) { break; }
                ts.emplace_back([lo, hi, &f] () {
                    for (int i = lo; i < hi; i++) { f(i); }
                });
            }
            for (int i = 0; i < std::min(chunk, n); i++) { f(i); }
            for (auto& th : ts) { th.join(); }
#endif
        }


#endif
};

template <class T, class Ops>
void BlockBandedPC<T,Ops>::printParameters () const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_block_banded);
    Print() << pc_name << " verbose:          " << (m_verbose?"true":"false") << "\n";
    Print() << pc_name << " theta:            " << m_theta << "\n";
    Print() << pc_name << " update_interval:  " << m_update_interval << "\n";
    Print() << pc_name << " include_drift:    " << (m_include_drift?"true":"false") << "\n";
    Print() << pc_name << " include_hyper:    " << (m_include_hyper?"true":"false") << "\n";
    Print() << pc_name << " verify:           " << (m_verify?"true":"false") << "\n";
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::readParameters ()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_block_banded));
    pp.query("verbose", m_verbose);

    const amrex::ParmParse pp_precond("precond");
    pp_precond.query("bb_update_interval", m_update_interval);
    pp_precond.query("bb_include_drift", m_include_drift);
    pp_precond.query("bb_include_hyper", m_include_hyper);
    pp_precond.query("bb_verify", m_verify);
    pp_precond.query("bb_max_mem_gb", m_max_mem_gb);
    pp_precond.query("bb_wall_identity", m_wall_identity);
    pp_precond.query("bb_overlap", m_overlap);
    pp_precond.query("bb_global", m_global);
#ifdef WARPX_USE_CUDSS
    pp_precond.query("bb_device_solve", m_device_solve);
#endif

    // The hybrid solver's GetThetaForPC() returns 0 (no implicit light
    // waves); read the actual time-centering directly.
    const amrex::ParmParse pp_theta("implicit_evolve");
    pp_theta.query("theta", m_theta);
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::Define ( const T& a_U, Ops* const a_ops )
{
    BL_PROFILE("BlockBandedPC::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(), "BlockBandedPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr), "BlockBandedPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "BlockBandedPC::Define() must be called with Efield_fp type");

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(a_U, a_ops);
    WARPX_ABORT_WITH_MESSAGE(
        "pc_block_banded is currently implemented for RZ geometry only "
        "(use pc_hall_jacobi as the dimension-agnostic tier)");
#else
    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_num_amr_levels == 1,
        "pc_block_banded supports a single AMR level");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_ops->GetHybridPICModel() != nullptr,
        "pc_block_banded requires the theta-implicit hybrid solver");

    readParameters();

    // pc_curlcurl_banded: resolve the configured esolve form. e_form keeps
    // the pc_block_banded assembly bit-for-bit; curlcurl_form switches the
    // frozen apply to the A/(A - N) composition (class doc at SetFormAware)
    // and forces the replicated-global system (one global coefficient
    // gather covers the frozen-rho snapshot too).
    if (m_form_aware) {
        const HybridPICModel* hybrid = m_ops->GetHybridPICModel();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!hybrid->m_esolve_tensor,
            "pc_curlcurl_banded: no outer preconditioner is implemented for "
            "esolve = tensor_form (the point-implicit gyration internalizes "
            "the whistler response; run jacobian.pc_type = none with "
            "hybrid_pic_model.esolve_pc = block_banded for the inner solve)");
        m_form_curlcurl = hybrid->m_esolve_curlcurl;
        if (m_form_curlcurl) { m_global = 1; }
    }

    const auto& geom = m_ops->GetGeometry(0);
    m_dr = geom.CellSize(0);
    m_dz = geom.CellSize(1);
    m_rmin = geom.ProbLo(0);
#endif

    m_is_defined = true;
}

#if defined(WARPX_DIM_RZ)

template <class T, class Ops>
void BlockBandedPC<T,Ops>::CopyFabToHost (const amrex::MultiFab& mf,
                                          const amrex::MFIter& mfi, HostFab& h)
{
    const auto& fab = mf[mfi];
    h.bx = fab.box();
    h.ncomp = mf.nComp();
    const std::size_t npts = std::size_t(h.bx.numPts())*h.ncomp;
    h.v.resize(npts);
#ifdef AMREX_USE_GPU
    amrex::Gpu::dtoh_memcpy(h.v.data(), fab.dataPtr(), npts*sizeof(amrex::Real));
#else
    std::copy_n(fab.dataPtr(), npts, h.v.data());
#endif
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::BuildSystems (const T& a_U)
{
    using namespace amrex;

    const auto& geom = m_ops->GetGeometry(0);
    const Box dom = geom.Domain();
    const bool zper = geom.isPeriodic(1);
    const int znD = dom.length(1);
    // curlcurl_form domain index facts (Dirichlet-row/axis/wall tests of
    // the mirrored inner solves; global nodal indices as in the kernels)
    m_ircc_hi = dom.bigEnd(0);
    m_ihi_nd = dom.bigEnd(0) + 1;
    m_zlo_d = dom.smallEnd(1);
    m_zhi_nd = dom.bigEnd(1) + 1;
    // The kernels' axis special cases trigger on r < dr/2; a domain is
    // axis-bearing when its lower r edge sits below that (annulus decks
    // with rmin >= dr get plain truncation instead of parity mirrors).
    const bool axis_in_domain = (m_rmin < RT(0.5)*m_dr);

    auto& u_mfarrvec = a_U.getArrayVec();
    const MultiFab& Ermf = *(u_mfarrvec[0][0]);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Ermf.nComp() == 1,
        "pc_block_banded supports the m=0 azimuthal mode only");

    m_sys.clear();
    double mem_gb = 0.0;

    // multi-box mode resolution: replicated-global by default whenever the
    // decomposition splits the domain (Dirichlet-truncated Schwarz stalls
    // GMRES on the whistler operator; see the class doc)
    const int nboxes_global = int(Ermf.boxArray().size());
    m_global_on = (m_global > 0) || (m_global < 0 && nboxes_global > 1);

    amrex::Vector<Box> sysboxes;
    if (m_global_on) {
        sysboxes.push_back(dom);
    } else {
        for (MFIter mfi(Ermf); mfi.isValid(); ++mfi) {
            sysboxes.push_back(amrex::enclosedCells(mfi.validbox()));
        }
    }

    for (const Box& cbx : sysboxes) {

        BoxSys S;
        S.rlo = cbx.smallEnd(0); S.rhi = cbx.bigEnd(0);
        S.zlo = cbx.smallEnd(1); S.zhi = cbx.bigEnd(1);
        // restricted-additive-Schwarz overlap: the assembled system extends
        // into r neighbors; only owned rows are written back in Apply
        S.rlo_s = std::max(S.rlo - m_overlap, dom.smallEnd(0));
        S.rhi_s = std::min(S.rhi + m_overlap, dom.bigEnd(0));
        S.nrc = S.rhi_s - S.rlo_s + 1;
        S.nzc = S.zhi - S.zlo + 1;
        S.zwrap = zper && (S.nzc == znD);
        S.nzn = S.zwrap ? S.nzc : S.nzc + 1;
        // domain-level flag: parity mirrors and the r < dr/2 row tests are
        // position-based, so boxes away from the axis never trigger them
        S.has_axis = axis_in_domain;
        // PEC upper-r wall: tangential-E columns are dead (the BC zeroes
        // them before the Faraday advance) and delta-B ghosts beyond the
        // wall are PEC images (tangential even, normal odd). Auto-detected
        // from the field BC; override with precond.bb_wall_identity.
        {
            const auto& bc_hi = m_ops->GetFieldBoundaryHi();
            const bool bc_is_pec = (bc_hi[0] == FieldBoundaryType::PEC) ||
                                   (bc_hi[0] == FieldBoundaryType::PEC_Insulator);
            const bool wall_on = (m_wall_identity < 0) ? bc_is_pec
                                                       : (m_wall_identity > 0);
            S.wall_node = dom.bigEnd(0) + 1;
            S.wall_pec = wall_on && (S.rhi_s == dom.bigEnd(0));
        }
        S.nb = S.nrc + 1;

        const bool zw = S.zwrap; const int zn = S.nzc;
        const int zpad = 3;
        const int zclo = zw ? S.zlo : S.zlo - zpad;
        const int zchi = zw ? S.zhi : S.zhi + zpad;      // cell-type z range
        const int znlo = zw ? S.zlo : S.zlo - zpad;
        const int znhi = zw ? S.zhi : S.zhi + 1 + zpad;  // node-type z range

        const bool ax = S.has_axis;
        const int rpad = 3;
        const int rNodeLo = std::max(S.rlo_s - rpad, 0);
        const int rCellLo = std::max(S.rlo_s - rpad, 0);
        // r index clipping above is only physical at the domain top; box-level
        // truncation is fine for coefficient windows too (zero reads).
        const int rCellHi = std::min(S.rhi_s + rpad, dom.bigEnd(0));
        const int rNodeHi = std::min(S.rhi_s + 1 + rpad, dom.bigEnd(0) + 1);

        // nodal coefficient windows (nodes in r AND z)
        S.B0rn.define (S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.B0tn.define (S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.B0zn.define (S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.Je0rn.define(S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.Je0tn.define(S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.Je0zn.define(S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.ciN.define  (S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);

        // row coefficient windows (per E-component stagger)
        S.etaEr.define (S.rlo_s, S.rhi_s,   znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.etaEt.define (S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.etaEz.define (S.rlo_s, S.rhi_s+1, zclo, zchi, zw, zn, 0, RT(1.0), false);
        S.etahEr = S.etaEr; S.etahEt = S.etaEt; S.etahEz = S.etaEz;
        S.irhoEr = S.etaEr; S.irhoEt = S.etaEt; S.irhoEz = S.etaEz;
        S.adEr = S.etaEr;   S.adEt = S.etaEt;   S.adEz = S.etaEz;
        if (m_form_curlcurl) {
            S.acTh = S.etaEt; S.foldTh = S.etaEt; S.rhswTh = S.etaEt;
            S.acPr = S.etaEr; S.foldPr = S.etaEr; S.rhswPr = S.etaEr;
            S.acPz = S.etaEz; S.foldPz = S.etaEz; S.rhswPz = S.etaEz;
        }

        // face-point eta_H windows + eta_H*curl(dJ) scratch for the
        // composed hyper-resistivity rows (hyper_resistivity_curlcurl).
        // No parity mirrors and z clipped to the domain in non-wrap runs:
        // out-of-band reads return zero, matching the Ohm wrapper's
        // zero-extension of curl J past non-periodic boundaries.
        {
            const int zcloD = zw ? zclo : std::max(zclo, dom.smallEnd(1));
            const int zchiD = zw ? zchi : std::min(zchi, dom.bigEnd(1));
            const int znloD = zw ? znlo : std::max(znlo, dom.smallEnd(1));
            const int znhiD = zw ? znhi : std::min(znhi, dom.bigEnd(1)+1);
            // Wr (node r, cell z); Wt (cell r, cell z); Wz (cell r, node z)
            S.cWr.define(rNodeLo, rNodeHi, zcloD, zchiD, zw, zn, 0, RT(1.0), false);
            S.cWt.define(rCellLo, rCellHi, zcloD, zchiD, zw, zn, 0, RT(1.0), false);
            S.cWz.define(rCellLo, rCellHi, znloD, znhiD, zw, zn, 0, RT(1.0), false);
            S.etahBr = S.cWr; S.etahBt = S.cWt; S.etahBz = S.cWz;
        }

        // scratch: inputs (valid DOF windows), stages (padded windows)
        S.inEr.define(S.rlo_s, S.rhi_s,   znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.inEt.define(S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.inEz.define(S.rlo_s, S.rhi_s+1, zclo, zchi, zw, zn, 0, RT(1.0), false);
        // delta-B: Br (node r, cell z) odd; Bt (cell, cell) odd; Bz (cell, node z) even
        S.dBr.define(rNodeLo, rNodeHi, zclo, zchi, zw, zn, 1, RT(-1.0), ax);
        S.dBt.define(rCellLo, rCellHi, zclo, zchi, zw, zn, 2, RT(-1.0), ax);
        S.dBz.define(rCellLo, rCellHi, znlo, znhi, zw, zn, 2, RT(+1.0), ax);
        // delta-J: Jr (cell r, node z) odd; Jt (node, node) odd; Jz (node r, cell z) even
        S.dJr.define(rCellLo, rCellHi, znlo, znhi, zw, zn, 2, RT(-1.0), ax);
        S.dJt.define(rNodeLo, rNodeHi, znlo, znhi, zw, zn, 1, RT(-1.0), ax);
        S.dJz.define(rNodeLo, rNodeHi, zclo, zchi, zw, zn, 1, RT(+1.0), ax);
        if (S.wall_pec) {
            // PEC images beyond the wall for the delta-B stage reads
            // (tangential even, normal odd -- the BC fill the real Ampere
            // curl consumes through its ghost ring)
            S.dBr.setUpperMirror(S.wall_node, 1, RT(-1.0));
            S.dBt.setUpperMirror(S.wall_node, 2, RT(+1.0));
            S.dBz.setUpperMirror(S.wall_node, 2, RT(+1.0));
            // one delta-J ghost cell beyond the wall: the real kernels
            // compute J on tileboxes grown by 1 from the BC-filled B ghosts,
            // and the nodal interp at the wall node consumes it
            S.dJr.define(rCellLo, S.wall_node, znlo, znhi, zw, zn, 2, RT(-1.0), ax);
        }
        // nodal enE over row-reachable nodes
        S.enR.define(S.rlo_s, S.rhi_s+1, znlo, znhi, zw, zn, 0, RT(1.0), false);
        S.enT = S.enR; S.enZ = S.enR;
        S.eiR = S.enR; S.eiT = S.enR; S.eiZ = S.enR;
        if (m_form_curlcurl) { S.gamN = S.enR; }
        // outputs = row windows
        S.oEr.define(S.rlo_s, S.rhi_s,   zw ? S.zlo : S.zlo, zw ? S.zhi : S.zhi+1, zw, zn, 0, RT(1.0), false);
        S.oEt.define(S.rlo_s, S.rhi_s+1, zw ? S.zlo : S.zlo, zw ? S.zhi : S.zhi+1, zw, zn, 0, RT(1.0), false);
        S.oEz.define(S.rlo_s, S.rhi_s+1, S.zlo, S.zhi, zw, zn, 0, RT(1.0), false);

        S.bsz.resize(S.nb);
        S.boff.resize(S.nb);
        S.ntot = 0;
        for (int b = 0; b < S.nb; b++) {
            S.bsz[b] = (blockHasEr(S,b) ? S.nzn : 0) + S.nzn + S.nzc;
            S.boff[b] = S.ntot;
            S.ntot += S.bsz[b];
        }
        S.x.assign(S.ntot, RT(0.0));

        // block storage estimate at the max bandwidth (3); curlcurl mode
        // additionally stores the unfactored A blocks
        double bytes = 0.0;
        for (int b = 0; b < S.nb; b++) {
            for (int d = -3; d <= 3; d++) {
                const int bc = b + d;
                if (bc < 0 || bc >= S.nb) { continue; }
                bytes += double(S.bsz[b])*double(S.bsz[bc])*sizeof(RT);
            }
        }
        mem_gb += (m_form_curlcurl ? 2.0 : 1.0)*bytes/1.0e9;

        m_sys.push_back(std::move(S));
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mem_gb <= double(m_max_mem_gb),
        "pc_block_banded: block storage estimate exceeds precond.bb_max_mem_gb"
        " -- this grid's z-lines are too long for dense z-blocks on this rank"
        " (decompose in r, or raise the limit)");

    // ghost-padded staging for the RAS overlap: coefficient copies deep
    // enough to cover the extended system windows, and a per-apply RHS copy
    // (Schwarz mode only; global mode gathers instead)
    if (!m_pads_defined && !m_global_on) {
        const amrex::IntVect ngc(m_overlap + 4);
        const amrex::IntVect ngb(m_overlap + 2);
        const auto B_mfs = m_ops->GetBfieldThetaForPC(0);
        const MultiFab* rho_mf = m_ops->GetRhoMidForPC(0);
        const auto Ji_mfs = m_ops->GetIonCurrentForPC(0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (B_mfs[0] != nullptr) && (rho_mf != nullptr),
            "pc_block_banded: theta-stage B / midpoint rho not available");
        for (int d = 0; d < 3; d++) {
            m_Bpad[d].define(B_mfs[d]->boxArray(),
                             B_mfs[d]->DistributionMap(), 1, ngc);
            if (Ji_mfs[d] != nullptr) {
                m_Jipad[d].define(Ji_mfs[d]->boxArray(),
                                  Ji_mfs[d]->DistributionMap(), 1, ngc);
            }
            m_bpad[d].define(u_mfarrvec[0][d]->boxArray(),
                             u_mfarrvec[0][d]->DistributionMap(), 1, ngb);
        }
        m_rhopad.define(rho_mf->boxArray(), rho_mf->DistributionMap(), 1, ngc);
        m_pads_defined = true;
    }

    if (m_verbose) {
        amrex::AllPrint() << "pc_block_banded: rank " << amrex::ParallelDescriptor::MyProc()
            << " boxes " << m_sys.size()
            << " block-mem(bw3-est) " << mem_gb << " GB\n";
    }

    m_work_defined = true;
}

/** Frozen linearized operator: inputs in S.inE*, outputs into S.oE*.
 *  e_form: rows of I - delta-E_ohm. curlcurl_form: rows of (A - N) --
 *  the mirrored inner elliptic operator minus the scaled numerator
 *  response -- or of A alone when a_A_only is set (the matvec leg of
 *  P^{-1} = (A - N)^{-1} A). */
template <class T, class Ops>
void BlockBandedPC<T,Ops>::ApplyFrozenBox (BoxSys& S, bool a_A_only) const
{
    const RT tdt = m_theta * this->m_dt;
    const RT dr = m_dr, dz = m_dz, rmin = m_rmin;
    const RT mu0 = static_cast<RT>(PhysConst::mu0);

    // PEC wall: the BC zeroes tangential E on the iterate before the
    // Faraday advance, so those columns are dead in the true Jacobian.
    // Zero the stage-visible values; the rows keep their identity diagonal
    // through the saved values (the wall E_ohm still responds to interior
    // E via the PEC-imaged delta-B).
    std::vector<RT> wallEt, wallEz;
    if (S.wall_pec && S.wall_node <= S.inEt.ihi) {
        const int W = S.wall_node;
        wallEt.assign(std::size_t(std::max(S.inEt.nj,0)), RT(0.0));
        wallEz.assign(std::size_t(std::max(S.inEz.nj,0)), RT(0.0));
        for (int j = S.inEt.jlo; j <= S.inEt.jhi; j++) {
            wallEt[S.inEt.jmap(j)] = S.inEt.get(W,j);
            S.inEt.ref(W,j) = RT(0.0);
        }
        for (int j = S.inEz.jlo; j <= S.inEz.jhi; j++) {
            wallEz[S.inEz.jmap(j)] = S.inEz.get(W,j);
            S.inEz.ref(W,j) = RT(0.0);
        }
    }

    // Stages 1-3 build the numerator-response chain; the A-only apply
    // (curlcurl_form matvec leg) acts on the raw inputs and skips them.
    if (!a_A_only) {
    // ---- stage 1: delta-B = theta*dt * (Faraday expressions) ----
    // (EvolveBCylindrical: Br += dt*UpDz(Et); Bt += dt*(UpDr(Ez) - UpDz(Er));
    //  Bz += -dt*UpDrr_over_r(Et); axis: Br(0)=0)
    for (int i = S.dBr.ilo; i <= S.dBr.ihi; i++) {
        const RT r = rmin + i*dr;
        for (int j = S.dBr.jlo; j <= S.dBr.jhi; j++) {
            S.dBr.ref(i,j) = (S.has_axis && i == 0 && std::abs(r) < RT(0.5)*dr)
                ? RT(0.0)
                : tdt*(S.inEt.get(i,j+1) - S.inEt.get(i,j))/dz;
        }
    }
    for (int i = S.dBt.ilo; i <= S.dBt.ihi; i++) {
        for (int j = S.dBt.jlo; j <= S.dBt.jhi; j++) {
            S.dBt.ref(i,j) = tdt*( (S.inEz.get(i+1,j) - S.inEz.get(i,j))/dr
                                 - (S.inEr.get(i,j+1) - S.inEr.get(i,j))/dz );
        }
    }
    for (int i = S.dBz.ilo; i <= S.dBz.ihi; i++) {
        const RT rc = rmin + (i + RT(0.5))*dr;
        for (int j = S.dBz.jlo; j <= S.dBz.jhi; j++) {
            S.dBz.ref(i,j) = -tdt*( (rc + RT(0.5)*dr)*S.inEt.get(i+1,j)
                                  - (rc - RT(0.5)*dr)*S.inEt.get(i,j) )/(rc*dr);
        }
    }

    // ---- stage 2: delta-J = AmpereCurl(delta-B)/mu0 ----
    for (int i = S.dJr.ilo; i <= S.dJr.ihi; i++) {
        for (int j = S.dJr.jlo; j <= S.dJr.jhi; j++) {
            S.dJr.ref(i,j) = -(S.dBt.get(i,j) - S.dBt.get(i,j-1))/(mu0*dz);
        }
    }
    for (int i = S.dJt.ilo; i <= S.dJt.ihi; i++) {
        const RT r = rmin + i*dr;
        const bool on_axis = S.has_axis && (r < RT(0.5)*dr) && (r > -RT(0.5)*dr);
        for (int j = S.dJt.jlo; j <= S.dJt.jhi; j++) {
            S.dJt.ref(i,j) = on_axis ? RT(0.0)
                : ( -(S.dBz.get(i,j) - S.dBz.get(i-1,j))/dr
                    +(S.dBr.get(i,j) - S.dBr.get(i,j-1))/dz )/mu0;
        }
    }
    for (int i = S.dJz.ilo; i <= S.dJz.ihi; i++) {
        const RT r = rmin + i*dr;
        const bool on_axis = S.has_axis && (r < RT(0.5)*dr) && (r > -RT(0.5)*dr);
        for (int j = S.dJz.jlo; j <= S.dJz.jhi; j++) {
            if (on_axis) {
                S.dJz.ref(i,j) = RT(4.0)*S.dBt.get(i,j)/(mu0*dr);
            } else {
                S.dJz.ref(i,j) = ( (r + RT(0.5)*dr)*S.dBt.get(i,j)
                                 - (r - RT(0.5)*dr)*S.dBt.get(i-1,j) )/(mu0*r*dr);
            }
        }
    }

    // ---- stage 2.5: cW = eta_H * curl(delta-J) at faces ----
    // (hyper_resistivity_curlcurl rows; out-of-band reads of cW return
    // zero = the wrapper's zero-extension convention)
    if (m_hyperres_cc) {
        for (int i = S.cWr.ilo; i <= S.cWr.ihi; i++) {
            for (int j = S.cWr.jlo; j <= S.cWr.jhi; j++) {
                S.cWr.ref(i,j) = -S.etahBr.get(i,j)
                    *(S.dJt.get(i,j+1) - S.dJt.get(i,j))/dz;
            }
        }
        for (int i = S.cWt.ilo; i <= S.cWt.ihi; i++) {
            for (int j = S.cWt.jlo; j <= S.cWt.jhi; j++) {
                S.cWt.ref(i,j) = S.etahBt.get(i,j)
                    *( (S.dJr.get(i,j+1) - S.dJr.get(i,j))/dz
                     - (S.dJz.get(i+1,j) - S.dJz.get(i,j))/dr );
            }
        }
        for (int i = S.cWz.ilo; i <= S.cWz.ihi; i++) {
            const RT r = rmin + (i + RT(0.5))*dr;
            for (int j = S.cWz.jlo; j <= S.cWz.jhi; j++) {
                S.cWz.ref(i,j) = S.etahBz.get(i,j)
                    *( (rmin + (i+1)*dr)*S.dJt.get(i+1,j)
                     - (rmin + i*dr)*S.dJt.get(i,j) )/(r*dr);
            }
        }
    }

    // ---- stage 3: nodal delta-enE ----
    for (int i = S.enR.ilo; i <= S.enR.ihi; i++) {
        for (int j = S.enR.jlo; j <= S.enR.jhi; j++) {
            // delta-J to nodes
            const RT jr = RT(0.5)*(S.dJr.get(i-1,j) + S.dJr.get(i,j));
            const RT jt = S.dJt.get(i,j);
            const RT jz = RT(0.5)*(S.dJz.get(i,j-1) + S.dJz.get(i,j));
            const RT b0r = S.B0rn.get(i,j);
            const RT b0t = S.B0tn.get(i,j);
            const RT b0z = S.B0zn.get(i,j);
            // curlcurl_form: the kernel's delta-Je is (hall ? delta-J : 0)
            // -- the ion deposit is frozen -- so the whistler leg gates on
            // the Hall term. (e_form keeps its landed convention.)
            const bool whl = !m_form_curlcurl || m_hall_on;
            RT er = whl ? (jt*b0z - jz*b0t) : RT(0.0);
            RT et = whl ? (jz*b0r - jr*b0z) : RT(0.0);
            RT ez = whl ? (jr*b0t - jt*b0r) : RT(0.0);
            if (m_include_drift) {
                // delta-B to nodes
                const RT br = RT(0.5)*(S.dBr.get(i,j-1) + S.dBr.get(i,j));
                const RT bt = RT(0.25)*( S.dBt.get(i-1,j-1) + S.dBt.get(i-1,j)
                                       + S.dBt.get(i,j-1)   + S.dBt.get(i,j) );
                const RT bz = RT(0.5)*(S.dBz.get(i-1,j) + S.dBz.get(i,j));
                const RT jer = S.Je0rn.get(i,j);
                const RT jet = S.Je0tn.get(i,j);
                const RT jez = S.Je0zn.get(i,j);
                er += jet*bz - jez*bt;
                et += jez*br - jer*bz;
                ez += jer*bt - jet*br;
            }
            S.enR.ref(i,j) = er;
            S.enT.ref(i,j) = et;
            S.enZ.ref(i,j) = ez;
            if (m_inertia_exact) {
                // exact nodal inertia response: delta-E_inertial =
                // ci * delta-Je(nodal), delta-Je = the same nodal delta-J
                // (frozen ion current drops out of the delta)
                const RT ci = S.ciN.get(i,j);
                S.eiR.ref(i,j) = ci*jr;
                S.eiT.ref(i,j) = ci*jt;
                S.eiZ.ref(i,j) = ci*jz;
            }
        }
    }
    } // end !a_A_only (stages 1-3)

    if (m_form_curlcurl) {
        ApplyFrozenBoxCurlCurl(S, a_A_only, wallEt, wallEz);
        return;
    }

    // ---- stage 4: rows out = in - delta_E_ohm ----
    // Er rows (cell r, node z)
    for (int i = S.oEr.ilo; i <= S.oEr.ihi; i++) {
        const RT rc = rmin + (i + RT(0.5))*dr;
        for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) {
            const RT enE = RT(0.5)*(S.enR.get(i,j) + S.enR.get(i+1,j));
            RT ohm = enE*S.irhoEr.get(i,j) + S.etaEr.get(i,j)*S.dJr.get(i,j);
            if (m_inertia_exact) {
                ohm += RT(0.5)*(S.eiR.get(i,j) + S.eiR.get(i+1,j));
            }
            if (m_hyperres_cc) {
                // composed row: +(curl cW)_r = -d_z cWt
                ohm += -(S.cWt.get(i,j) - S.cWt.get(i,j-1))/dz;
            } else {
            const RT etah = S.etahEr.get(i,j);
            if (etah != RT(0.0)) {
                const RT lap =
                    ( (rc + RT(0.5)*dr)*(S.dJr.get(i+1,j) - S.dJr.get(i,j))
                    - (rc - RT(0.5)*dr)*(S.dJr.get(i,j) - S.dJr.get(i-1,j)) )/(rc*dr*dr)
                    + (S.dJr.get(i,j-1) - RT(2.0)*S.dJr.get(i,j) + S.dJr.get(i,j+1))/(dz*dz)
                    - S.dJr.get(i,j)/(rc*rc);
                ohm -= etah*lap;
            }
            }
            S.oEr.ref(i,j) = (RT(1.0) + S.adEr.get(i,j))*S.inEr.get(i,j) - ohm;
        }
    }
    // Etheta rows (node r, node z)
    for (int i = S.oEt.ilo; i <= S.oEt.ihi; i++) {
        const RT r = rmin + i*dr;
        const bool on_axis = S.has_axis && (r < RT(0.5)*dr) && (r > -RT(0.5)*dr);
        for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) {
            if (on_axis) {
                S.oEt.ref(i,j) = S.inEt.get(i,j);   // identity row (kernel: Etheta(axis)=0)
                continue;
            }
            RT ohm = S.enT.get(i,j)*S.irhoEt.get(i,j)
                   + S.etaEt.get(i,j)*S.dJt.get(i,j);
            if (m_inertia_exact) {
                ohm += S.eiT.get(i,j);
            }
            if (m_hyperres_cc) {
                // composed row: +(curl cW)_t = d_z cWr - d_r cWz
                ohm += (S.cWr.get(i,j) - S.cWr.get(i,j-1))/dz
                     - (S.cWz.get(i,j) - S.cWz.get(i-1,j))/dr;
            } else {
            const RT etah = S.etahEt.get(i,j);
            if (etah != RT(0.0)) {
                const RT lap =
                    ( (r + RT(0.5)*dr)*(S.dJt.get(i+1,j) - S.dJt.get(i,j))
                    - (r - RT(0.5)*dr)*(S.dJt.get(i,j) - S.dJt.get(i-1,j)) )/(r*dr*dr)
                    + (S.dJt.get(i,j-1) - RT(2.0)*S.dJt.get(i,j) + S.dJt.get(i,j+1))/(dz*dz)
                    - S.dJt.get(i,j)/(r*r);
                ohm -= etah*lap;
            }
            }
            const RT et_in = (S.wall_pec && i == S.wall_node)
                ? wallEt[S.inEt.jmap(j)] : S.inEt.get(i,j);
            S.oEt.ref(i,j) = (RT(1.0) + S.adEt.get(i,j))*et_in - ohm;
        }
    }
    // Ez rows (node r, cell z)
    for (int i = S.oEz.ilo; i <= S.oEz.ihi; i++) {
        const RT r = rmin + i*dr;
        const bool on_axis = S.has_axis && (r < RT(0.5)*dr) && (r > -RT(0.5)*dr);
        for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) {
            const RT enE = RT(0.5)*(S.enZ.get(i,j) + S.enZ.get(i,j+1));
            RT ohm = enE*S.irhoEz.get(i,j) + S.etaEz.get(i,j)*S.dJz.get(i,j);
            if (m_inertia_exact) {
                ohm += RT(0.5)*(S.eiZ.get(i,j) + S.eiZ.get(i,j+1));
            }
            if (m_hyperres_cc) {
                // composed row: +(curl cW)_z = (1/r) d_r (r cWt); the
                // Ampere-closure axis limit 4 cWt/dr is exact (no factor-2
                // soft spot, unlike the truncated fallback below).
                ohm += on_axis
                    ? RT(4.0)*S.cWt.get(i,j)/dr
                    : ( (r + RT(0.5)*dr)*S.cWt.get(i,j)
                      - (r - RT(0.5)*dr)*S.cWt.get(i-1,j) )/(r*dr);
            } else {
            const RT etah = S.etahEz.get(i,j);
            if (etah != RT(0.0)) {
                RT lap = (S.dJz.get(i,j-1) - RT(2.0)*S.dJz.get(i,j) + S.dJz.get(i,j+1))/(dz*dz);
                if (on_axis) {
                    // m = 0 regularity: the axis limit of (1/r)d_r(r d_r)
                    // is 2 d_rr (matches the corrected kernel row).
                    lap += RT(2.0)*
                        (S.dJz.get(i-1,j) - RT(2.0)*S.dJz.get(i,j) + S.dJz.get(i+1,j))/(dr*dr);
                } else {
                    lap += ( (r + RT(0.5)*dr)*(S.dJz.get(i+1,j) - S.dJz.get(i,j))
                           - (r - RT(0.5)*dr)*(S.dJz.get(i,j) - S.dJz.get(i-1,j)) )/(r*dr*dr);
                }
                ohm -= etah*lap;
            }
            }
            const RT ez_in = (S.wall_pec && i == S.wall_node)
                ? wallEz[S.inEz.jmap(j)] : S.inEz.get(i,j);
            S.oEz.ref(i,j) = (RT(1.0) + S.adEz.get(i,j))*ez_in - ohm;
        }
    }
}

/** curlcurl_form stage 4: mirrored rows of the inner solves.
 *  theta rows = the MLNodeABecLaplacian-form operator of
 *  SolveEThetaCurlCurlRZ; Er/Ez rows = the fused energy-form apply of
 *  SolveEPolCurlCurlRZ (V-scaled composed curl-curl + gated
 *  -grad(gamma div .) closure with the PolNodeVol half cells); minus the
 *  RHS-weighted numerator response unless a_A_only. Dirichlet rows
 *  (theta axis/wall + z caps, Ez wall, Er z caps) are identity on the
 *  saved input; their RHS weights are zeroed in FillCoeffs, matching the
 *  solves' own row zeroing. Non-periodic z is mirrored structurally but
 *  cap COLUMNS are not killed (v1; the gate decks run periodic z). */
template <class T, class Ops>
void BlockBandedPC<T,Ops>::ApplyFrozenBoxCurlCurl (BoxSys& S, bool a_A_only,
    const std::vector<RT>& wallEt, const std::vector<RT>& wallEz) const
{
    const RT dr = m_dr, dz = m_dz, rmin = m_rmin;
    const bool zw = S.zwrap;
    const bool wall_saved_t = S.wall_pec && !wallEt.empty();
    const bool wall_saved_z = S.wall_pec && !wallEz.empty();

    // ---- Er rows (cell r, node z) ----
    for (int i = S.oEr.ilo; i <= S.oEr.ihi; i++) {
        const RT rc = rmin + (i + RT(0.5))*dr;
        const RT rcm = rmin + (i - RT(0.5))*dr;
        for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) {
            if (!zw && (j <= m_zlo_d || j >= m_zhi_nd)) {
                S.oEr.ref(i,j) = S.inEr.get(i,j);   // Dirichlet cap row
                continue;
            }
            const RT wp = (S.inEr.get(i,j+1) - S.inEr.get(i,j))/dz
                        - (S.inEz.get(i+1,j) - S.inEz.get(i,j))/dr;
            const RT wm = (S.inEr.get(i,j) - S.inEr.get(i,j-1))/dz
                        - (S.inEz.get(i+1,j-1) - S.inEz.get(i,j-1))/dr;
            RT dd = RT(0.0);
            if (S.gamN.get(i,j) > RT(0.0)) {
                RT f = rc*S.inEr.get(i,j);
                if (i > 0) { f -= rcm*S.inEr.get(i-1,j); }
                dd += f/PolVol(i)
                    + (S.inEz.get(i,j) - S.inEz.get(i,j-1))/dz;
            }
            if (S.gamN.get(i+1,j) > RT(0.0)) {
                RT f = -rc*S.inEr.get(i,j);
                if (i + 1 <= m_ircc_hi) {
                    f += (rmin + (i + RT(1.5))*dr)*S.inEr.get(i+1,j);
                }
                dd -= f/PolVol(i+1)
                    + (S.inEz.get(i+1,j) - S.inEz.get(i+1,j-1))/dz;
            }
            RT out = S.acPr.get(i,j)*S.inEr.get(i,j)
                   - rc*dr*(wp - wm)/dz + rc*dd;
            if (!a_A_only) {
                RT dnum = RT(0.5)*(S.enR.get(i,j) + S.enR.get(i+1,j));
                RT res = S.etaEr.get(i,j)*S.dJr.get(i,j);
                if (m_hyperres_cc) {
                    res += -(S.cWt.get(i,j) - S.cWt.get(i,j-1))/dz;
                } else {
                    const RT etah = S.etahEr.get(i,j);
                    if (etah != RT(0.0)) {
                        const RT lap =
                            ( (rc + RT(0.5)*dr)*(S.dJr.get(i+1,j) - S.dJr.get(i,j))
                            - (rc - RT(0.5)*dr)*(S.dJr.get(i,j) - S.dJr.get(i-1,j)) )/(rc*dr*dr)
                            + (S.dJr.get(i,j-1) - RT(2.0)*S.dJr.get(i,j) + S.dJr.get(i,j+1))/(dz*dz)
                            - S.dJr.get(i,j)/(rc*rc);
                        res -= etah*lap;
                    }
                }
                dnum += S.foldPr.get(i,j)*res;
                out -= S.rhswPr.get(i,j)*dnum;
            }
            S.oEr.ref(i,j) = out;
        }
    }

    // ---- Etheta rows (node r, node z) ----
    for (int i = S.oEt.ilo; i <= S.oEt.ihi; i++) {
        const RT r = rmin + i*dr;
        const RT rcp = rmin + (i + RT(0.5))*dr;
        const RT rcm = rmin + (i - RT(0.5))*dr;
        for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) {
            if (i <= 0 || i >= m_ihi_nd
                || (!zw && (j <= m_zlo_d || j >= m_zhi_nd))) {
                // Dirichlet identity row (axis regularity / PEC wall /
                // z caps); the wall input was column-killed above, so the
                // row diagonal reads the saved value
                S.oEt.ref(i,j) = (wall_saved_t && i == S.wall_node)
                    ? wallEt[S.inEt.jmap(j)] : S.inEt.get(i,j);
                continue;
            }
            // axis column is Dirichlet in the solve: reads of the i = 0
            // DOF return zero (the wall column is killed on the input)
            const RT et_im = (i - 1 == 0) ? RT(0.0) : S.inEt.get(i-1,j);
            RT out = S.acTh.get(i,j)*S.inEt.get(i,j)
                - ( rcp*(S.inEt.get(i+1,j) - S.inEt.get(i,j))
                  - rcm*(S.inEt.get(i,j) - et_im) )/(dr*dr)
                - RT(0.5)*(rcp + rcm)
                  *(S.inEt.get(i,j+1) - RT(2.0)*S.inEt.get(i,j) + S.inEt.get(i,j-1))
                  /(dz*dz);
            if (!a_A_only) {
                RT res = S.etaEt.get(i,j)*S.dJt.get(i,j);
                if (m_hyperres_cc) {
                    res += (S.cWr.get(i,j) - S.cWr.get(i,j-1))/dz
                         - (S.cWz.get(i,j) - S.cWz.get(i-1,j))/dr;
                } else {
                    const RT etah = S.etahEt.get(i,j);
                    if (etah != RT(0.0)) {
                        const RT lap =
                            ( (r + RT(0.5)*dr)*(S.dJt.get(i+1,j) - S.dJt.get(i,j))
                            - (r - RT(0.5)*dr)*(S.dJt.get(i,j) - S.dJt.get(i-1,j)) )/(r*dr*dr)
                            + (S.dJt.get(i,j-1) - RT(2.0)*S.dJt.get(i,j) + S.dJt.get(i,j+1))/(dz*dz)
                            - S.dJt.get(i,j)/(r*r);
                        res -= etah*lap;
                    }
                }
                const RT dnum = S.enT.get(i,j) + S.foldTh.get(i,j)*res;
                out -= S.rhswTh.get(i,j)*dnum;
            }
            S.oEt.ref(i,j) = out;
        }
    }

    // ---- Ez rows (node r, cell z) ----
    for (int i = S.oEz.ilo; i <= S.oEz.ihi; i++) {
        const RT r = rmin + i*dr;
        const RT rcp = rmin + (i + RT(0.5))*dr;
        const RT rcm = rmin + (i - RT(0.5))*dr;
        const bool on_axis = S.has_axis && (r < RT(0.5)*dr) && (r > -RT(0.5)*dr);
        for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) {
            if (i >= m_ihi_nd) {
                S.oEz.ref(i,j) = (wall_saved_z && i == S.wall_node)
                    ? wallEz[S.inEz.jmap(j)] : S.inEz.get(i,j);
                continue;
            }
            const RT wp = (S.inEr.get(i,j+1) - S.inEr.get(i,j))/dz
                        - (S.inEz.get(i+1,j) - S.inEz.get(i,j))/dr;
            RT cc = rcp*wp;
            if (i > 0) {
                const RT wm = (S.inEr.get(i-1,j+1) - S.inEr.get(i-1,j))/dz
                            - (S.inEz.get(i,j) - S.inEz.get(i-1,j))/dr;
                cc -= rcm*wm;
            }
            if (S.gamN.get(i,j) > RT(0.0) || S.gamN.get(i,j+1) > RT(0.0)) {
                const RT Vn = PolVol(i);
                RT dd = RT(0.0);
                if (S.gamN.get(i,j) > RT(0.0)) {
                    RT f = rcp*S.inEr.get(i,j);
                    if (i > 0) { f -= rcm*S.inEr.get(i-1,j); }
                    dd += f/Vn + (S.inEz.get(i,j) - S.inEz.get(i,j-1))/dz;
                }
                if (S.gamN.get(i,j+1) > RT(0.0)) {
                    RT f = rcp*S.inEr.get(i,j+1);
                    if (i > 0) { f -= rcm*S.inEr.get(i-1,j+1); }
                    dd -= f/Vn + (S.inEz.get(i,j+1) - S.inEz.get(i,j))/dz;
                }
                cc += Vn*dd/dz;
            }
            RT out = S.acPz.get(i,j)*S.inEz.get(i,j) + cc;
            if (!a_A_only) {
                RT dnum = RT(0.5)*(S.enZ.get(i,j) + S.enZ.get(i,j+1));
                RT res = S.etaEz.get(i,j)*S.dJz.get(i,j);
                if (m_hyperres_cc) {
                    res += on_axis
                        ? RT(4.0)*S.cWt.get(i,j)/dr
                        : ( (r + RT(0.5)*dr)*S.cWt.get(i,j)
                          - (r - RT(0.5)*dr)*S.cWt.get(i-1,j) )/(r*dr);
                } else {
                    const RT etah = S.etahEz.get(i,j);
                    if (etah != RT(0.0)) {
                        RT lap = (S.dJz.get(i,j-1) - RT(2.0)*S.dJz.get(i,j)
                                  + S.dJz.get(i,j+1))/(dz*dz);
                        if (on_axis) {
                            lap += RT(2.0)*
                                (S.dJz.get(i-1,j) - RT(2.0)*S.dJz.get(i,j)
                                 + S.dJz.get(i+1,j))/(dr*dr);
                        } else {
                            lap += ( (r + RT(0.5)*dr)*(S.dJz.get(i+1,j) - S.dJz.get(i,j))
                                   - (r - RT(0.5)*dr)*(S.dJz.get(i,j) - S.dJz.get(i-1,j)) )
                                   /(r*dr*dr);
                        }
                        res -= etah*lap;
                    }
                }
                dnum += S.foldPz.get(i,j)*res;
                out -= S.rhswPz.get(i,j)*dnum;
            }
            S.oEz.ref(i,j) = out;
        }
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::GatherGlobal (const amrex::MultiFab& src, int scomp,
                                         HostFab& out, int ng) const
{
    using namespace amrex;
    const auto& geom = m_ops->GetGeometry(0);
    const Box dombx = amrex::convert(geom.Domain(), src.ixType());
    BoxArray ba(dombx);
    DistributionMapping dm(Vector<int>{0});
    MultiFab g(ba, dm, 1, ng);
    if (ParallelDescriptor::MyProc() == 0) { g.setVal(RT(0.0)); }
    g.ParallelCopy(src, scomp, 0, 1, src.nGrowVect(), IntVect(ng),
                   geom.periodicity());
    g.FillBoundary(geom.periodicity());
    out.bx = amrex::grow(dombx, ng);
    out.ncomp = 1;
    const std::size_t n = std::size_t(out.bx.numPts());
    out.v.assign(n, RT(0.0));
    for (MFIter mfi(g); mfi.isValid(); ++mfi) {
        const auto& fab = g[mfi];
#ifdef AMREX_USE_GPU
        amrex::Gpu::dtoh_memcpy(out.v.data(), fab.dataPtr(),
                                n*sizeof(amrex::Real));
#else
        std::copy_n(fab.dataPtr(), n, out.v.data());
#endif
    }
    ParallelDescriptor::Bcast(out.v.data(), n,
        ParallelDescriptor::IOProcessorNumber());
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::GatherCoefficients (const T& a_U)
{
    using namespace amrex;
    using warpx::fields::FieldType;

    const MultiFab* rho_check = m_ops->GetRhoMidForPC(0);
    amrex::ignore_unused(rho_check);
    const auto B_mfs = m_ops->GetBfieldThetaForPC(0);
    const MultiFab* rho_mf = m_ops->GetRhoMidForPC(0);
    const auto Ji_mfs = m_ops->GetIonCurrentForPC(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (B_mfs[0] != nullptr) && (rho_mf != nullptr),
        "pc_block_banded: theta-stage B / midpoint rho not available");
    const int rho_mid_comp = rho_mf->nComp()/2;

    // stage the frozen coefficients into the overlap-deep padded copies
    // (imports the sources' physically-filled ghosts, then refreshes the
    // interior/periodic ghost rings)
    const auto& geom0 = m_ops->GetGeometry(0);
    const auto& period0 = geom0.periodicity();
    const bool have_ji_glob = (Ji_mfs[0] != nullptr);
    if (!m_global_on) {
    for (int d = 0; d < 3; d++) {
        m_Bpad[d].setVal(RT(0.0));
        m_Bpad[d].ParallelCopy(*B_mfs[d], 0, 0, 1,
            B_mfs[d]->nGrowVect(), m_Bpad[d].nGrowVect(), period0);
        m_Bpad[d].FillBoundary(period0);
        if (have_ji_glob) {
            m_Jipad[d].setVal(RT(0.0));
            m_Jipad[d].ParallelCopy(*Ji_mfs[d], 0, 0, 1,
                Ji_mfs[d]->nGrowVect(), m_Jipad[d].nGrowVect(), period0);
            m_Jipad[d].FillBoundary(period0);
        }
    }
    m_rhopad.setVal(RT(0.0));
    m_rhopad.ParallelCopy(*rho_mf, rho_mid_comp, 0, 1,
        rho_mf->nGrowVect(), m_rhopad.nGrowVect(), period0);
    m_rhopad.FillBoundary(period0);
    }

    auto& u_mfarrvec = a_U.getArrayVec();
    const MultiFab& Ermf = *(u_mfarrvec[0][0]);

    if (m_global) {
        // replicated global system: gather each coefficient field onto a
        // domain-spanning host buffer on every rank
        HostFab hBr, hBt, hBz, hRho, hJir, hJit, hJiz, hRhoPol;
        GatherGlobal(*B_mfs[0], 0, hBr, 3);
        GatherGlobal(*B_mfs[1], 0, hBt, 3);
        GatherGlobal(*B_mfs[2], 0, hBz, 3);
        GatherGlobal(*rho_mf, rho_mid_comp, hRho, 3);
        if (have_ji_glob) {
            GatherGlobal(*Ji_mfs[0], 0, hJir, 3);
            GatherGlobal(*Ji_mfs[1], 0, hJit, 3);
            GatherGlobal(*Ji_mfs[2], 0, hJiz, 3);
        }
        bool have_rho_pol = false;
        if (m_form_curlcurl) {
            // the poloidal stage's own density source: the per-step-frozen
            // rho^n snapshot when in play, the midpoint deposit otherwise
            const MultiFab* rho_pol_mf = m_ops->GetRhoPolFrozenForPC(0);
            if (rho_pol_mf != nullptr) {
                GatherGlobal(*rho_pol_mf, 0, hRhoPol, 3);
                have_rho_pol = true;
            }
        }
        FillCoeffs(m_sys[0], hBr, hBt, hBz, hRho, hJir, hJit, hJiz,
                   have_ji_glob, hRhoPol, have_rho_pol);
        return;
    }

    int isys = 0;
    for (MFIter mfi(Ermf); mfi.isValid(); ++mfi, ++isys) {
        BoxSys& S = m_sys[isys];
        HostFab hBr, hBt, hBz, hRho, hJir, hJit, hJiz;
        CopyFabToHost(m_Bpad[0], mfi, hBr);
        CopyFabToHost(m_Bpad[1], mfi, hBt);
        CopyFabToHost(m_Bpad[2], mfi, hBz);
        CopyFabToHost(m_rhopad, mfi, hRho);
        const bool have_ji = have_ji_glob;
        if (have_ji) {
            CopyFabToHost(m_Jipad[0], mfi, hJir);
            CopyFabToHost(m_Jipad[1], mfi, hJit);
            CopyFabToHost(m_Jipad[2], mfi, hJiz);
        }
        // Schwarz path never runs in curlcurl mode (Define forces the
        // replicated-global system there): no frozen-rho staging needed
        const HostFab hRhoPolEmpty{};
        FillCoeffs(S, hBr, hBt, hBz, hRho, hJir, hJit, hJiz, have_ji,
                   hRhoPolEmpty, false);
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::FillCoeffs (BoxSys& S,
    const HostFab& hBr, const HostFab& hBt, const HostFab& hBz,
    const HostFab& hRho, const HostFab& hJir, const HostFab& hJit,
    const HostFab& hJiz, bool have_ji,
    const HostFab& hRhoPol, bool have_rho_pol)
{
    using namespace amrex;

    const HybridPICModel* hybrid = m_ops->GetHybridPICModel();
    const RT rho_floor = static_cast<RT>(hybrid->m_n_floor * PhysConst::q_e);
    const RT floor_w = static_cast<RT>(hybrid->m_n_floor_smooth_width)*rho_floor;
    const RT taper_w =
        static_cast<RT>(hybrid->m_electron_inertia_floor_taper)*rho_floor;
    const bool hall_on = hybrid->m_include_hall_term;
    const bool has_eta = (hybrid->m_eta_expression != "0.0");
    const bool has_J_dep = hybrid->m_resistivity_has_J_dependence;
    const bool hyper_on = m_include_hyper && hybrid->m_include_hyper_resistivity_term;
    const bool holm = hybrid->m_holmstrom_vacuum_region;
    const RT holm_iw = (hybrid->m_holmstrom_transition_width > RT(0.0))
        ? RT(1.0)/(static_cast<RT>(hybrid->m_holmstrom_transition_width)*rho_floor) : RT(0.0);
    const bool holm_smooth = holm && (holm_iw > RT(0.0));
    const bool inertia_on = hybrid->m_include_electron_inertia;
    const RT me_eff = inertia_on ? static_cast<RT>(hybrid->m_electron_inertia_mass) : RT(0.0);
    // Stage-centered time-derivative factor of the linearized inertia leg:
    // dJe/dt responds to the theta-stage iterate through
    // c_p1/theta = (2 theta + 1)/(2 theta dt) (three-point stencil) or
    // 1/(theta dt) (two-point form / first step after seeding); mirrors
    // ComputeElectronInertiaNodal. The theta*dt of the Faraday stage
    // cancels it back to the dimensionless (2 theta + 1)/2 class.
    // curlcurl_form: the dJe/dt E-response IS the elliptic operator A (and
    // the measured numerator is frozen), so the exact nodal inertia chain
    // is not part of the numerator response
    m_inertia_exact = inertia_on && !m_form_curlcurl;
    m_hall_on = hall_on;
    RT s_dt = RT(0.0);
    if (m_inertia_exact) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(this->m_dt > RT(0.0),
            "pc_block_banded: inertia coefficients need the time step "
            "(CurTimeStep not called before Update)");
        const bool bdf2 = hybrid->m_electron_inertia_bdf2
            && (hybrid->m_inertia_history_levels >= 2);
        s_dt = bdf2
            ? (RT(2.0)*m_theta + RT(1.0))/(RT(2.0)*m_theta*this->m_dt)
            : RT(1.0)/(m_theta*this->m_dt);
    }
    ParserExecutor<3> eta_exec{};
    ParserExecutor<2> etah_exec{};
    if (has_eta) { eta_exec = hybrid->m_eta; }
    if (hyper_on) { etah_exec = hybrid->m_eta_h; }
    const RT time = this->m_time;
    const RT mu0 = static_cast<RT>(PhysConst::mu0);
    const RT dr = m_dr, dz = m_dz, rmin = m_rmin;

    {
        const bool zw = S.zwrap; const int zn = S.nzc;
        const bool ax = S.has_axis;

        // staggered B0 windows with parity mirrors (for J0 and nodal B0)
        PArr B0r, B0t, B0z;
        const int rNodeLo = std::max(S.rlo_s-3, 0), rCellLo = std::max(S.rlo_s-3, 0);
        const int rCellHi = S.rhi_s+3, rNodeHi = S.rhi_s+4;
        const int zclo = S.dBt.jlo, zchi = S.dBt.jhi;
        const int znlo = S.dBz.jlo, znhi = S.dBz.jhi;
        B0r.define(rNodeLo, rNodeHi, zclo, zchi, zw, zn, 1, RT(-1.0), ax);
        B0t.define(rCellLo, rCellHi, zclo, zchi, zw, zn, 2, RT(-1.0), ax);
        B0z.define(rCellLo, rCellHi, znlo, znhi, zw, zn, 2, RT(+1.0), ax);
        for (int i = B0r.ilo; i <= B0r.ihi; i++) {
            for (int j = B0r.jlo; j <= B0r.jhi; j++) { B0r.ref(i,j) = hBr.get(i,j); }
        }
        for (int i = B0t.ilo; i <= B0t.ihi; i++) {
            for (int j = B0t.jlo; j <= B0t.jhi; j++) { B0t.ref(i,j) = hBt.get(i,j); }
        }
        for (int i = B0z.ilo; i <= B0z.ihi; i++) {
            for (int j = B0z.jlo; j <= B0z.jhi; j++) { B0z.ref(i,j) = hBz.get(i,j); }
        }

        // frozen J0 = AmpereCurl(B0)/mu0 on J staggers (same formulas as
        // stage 2 of the frozen apply)
        PArr J0r, J0t, J0z;
        J0r.define(rCellLo, rCellHi, znlo, znhi, zw, zn, 2, RT(-1.0), ax);
        J0t.define(rNodeLo, rNodeHi, znlo, znhi, zw, zn, 1, RT(-1.0), ax);
        J0z.define(rNodeLo, rNodeHi, zclo, zchi, zw, zn, 1, RT(+1.0), ax);
        for (int i = J0r.ilo; i <= J0r.ihi; i++) {
            for (int j = J0r.jlo; j <= J0r.jhi; j++) {
                J0r.ref(i,j) = -(B0t.get(i,j) - B0t.get(i,j-1))/(mu0*dz);
            }
        }
        for (int i = J0t.ilo; i <= J0t.ihi; i++) {
            const RT r = rmin + i*dr;
            const bool on_axis = ax && (r < RT(0.5)*dr) && (r > -RT(0.5)*dr);
            for (int j = J0t.jlo; j <= J0t.jhi; j++) {
                J0t.ref(i,j) = on_axis ? RT(0.0)
                    : ( -(B0z.get(i,j) - B0z.get(i-1,j))/dr
                        +(B0r.get(i,j) - B0r.get(i,j-1))/dz )/mu0;
            }
        }
        for (int i = J0z.ilo; i <= J0z.ihi; i++) {
            const RT r = rmin + i*dr;
            const bool on_axis = ax && (r < RT(0.5)*dr) && (r > -RT(0.5)*dr);
            for (int j = J0z.jlo; j <= J0z.jhi; j++) {
                if (on_axis) {
                    J0z.ref(i,j) = RT(4.0)*B0t.get(i,j)/(mu0*dr);
                } else {
                    J0z.ref(i,j) = ( (r + RT(0.5)*dr)*B0t.get(i,j)
                                   - (r - RT(0.5)*dr)*B0t.get(i-1,j) )/(mu0*r*dr);
                }
            }
        }

        // frozen ion current windows (read actual fab data incl. ghosts)
        PArr Jir, Jit, Jiz;
        Jir.define(rCellLo, rCellHi, znlo, znhi, zw, zn, 2, RT(-1.0), ax);
        Jit.define(rNodeLo, rNodeHi, znlo, znhi, zw, zn, 1, RT(-1.0), ax);
        Jiz.define(rNodeLo, rNodeHi, zclo, zchi, zw, zn, 1, RT(+1.0), ax);
        if (have_ji) {
            for (int i = Jir.ilo; i <= Jir.ihi; i++) {
                for (int j = Jir.jlo; j <= Jir.jhi; j++) { Jir.ref(i,j) = hJir.get(i,j); }
            }
            for (int i = Jit.ilo; i <= Jit.ihi; i++) {
                for (int j = Jit.jlo; j <= Jit.jhi; j++) { Jit.ref(i,j) = hJit.get(i,j); }
            }
            for (int i = Jiz.ilo; i <= Jiz.ihi; i++) {
                for (int j = Jiz.jlo; j <= Jiz.jhi; j++) { Jiz.ref(i,j) = hJiz.get(i,j); }
            }
        }

        // nodal coefficient assembly
        for (int i = S.B0rn.ilo; i <= S.B0rn.ihi; i++) {
            for (int j = S.B0rn.jlo; j <= S.B0rn.jhi; j++) {
                S.B0rn.ref(i,j) = RT(0.5)*(B0r.get(i,j-1) + B0r.get(i,j));
                S.B0tn.ref(i,j) = RT(0.25)*( B0t.get(i-1,j-1) + B0t.get(i-1,j)
                                           + B0t.get(i,j-1)   + B0t.get(i,j) );
                S.B0zn.ref(i,j) = RT(0.5)*(B0z.get(i-1,j) + B0z.get(i,j));
                const RT j0r = RT(0.5)*(J0r.get(i-1,j) + J0r.get(i,j));
                const RT j0t = J0t.get(i,j);
                const RT j0z = RT(0.5)*(J0z.get(i,j-1) + J0z.get(i,j));
                const RT jir = RT(0.5)*(Jir.get(i-1,j) + Jir.get(i,j));
                const RT jit = Jit.get(i,j);
                const RT jiz = RT(0.5)*(Jiz.get(i,j-1) + Jiz.get(i,j));
                S.Je0rn.ref(i,j) = (hall_on ? j0r : RT(0.0)) - jir;
                S.Je0tn.ref(i,j) = (hall_on ? j0t : RT(0.0)) - jit;
                S.Je0zn.ref(i,j) = (hall_on ? j0z : RT(0.0)) - jiz;
                // nodal inertia coefficient: (m_e_eff/e) s_dt / rho_lim on
                // the RAW midpoint deposit, zeroed in true vacuum (the
                // kernel zeroes E_inertial where rho_mid <= 0; floored
                // cells keep their inertia). The delta-legs of the
                // advective (Je.grad)u term and of u drho/dt are omitted
                // (electron-drift-CFL small; priced by the FD gate).
                RT ci = RT(0.0);
                if (m_inertia_exact) {
                    const RT rho_mid_n = hRho.get(i, j, 0);
                    if (rho_mid_n > RT(0.0)) {
                        const RT rho_lim_n =
                            HybridSmoothFloor(rho_mid_n, rho_floor, floor_w);
                        ci = me_eff*s_dt
                            / (static_cast<RT>(PhysConst::q_e)*rho_lim_n);
                        if (taper_w > RT(0.0)) {
                            // fluid-validity taper (mirrors the kernel)
                            ci *= RT(0.5)*(RT(1.0) + std::tanh(
                                (rho_mid_n - rho_floor)/taper_w));
                        }
                    }
                }
                S.ciN.ref(i,j) = ci;
            }
        }

        // row coefficients per E-component stagger. rho is nodal (single
        // valid slot pair; midpoint deposit is component rho_mid_comp).
        auto rho_nodal = [&](int i, int j) -> RT {
            return hRho.get(i, j, 0);
        };
        auto row_coeffs = [&](int comp, int i, int j, PArr& eta_a, PArr& etah_a,
                              PArr& irho_a, PArr& ad_a)
        {
            // rho interpolated to the row stagger (matches the kernels)
            RT rho_val;
            if (comp == 0) {        // Er (cell r, node z)
                rho_val = RT(0.5)*(rho_nodal(i,j) + rho_nodal(i+1,j));
            } else if (comp == 1) { // Etheta (node, node)
                rho_val = rho_nodal(i,j);
            } else {                // Ez (node r, cell z)
                rho_val = RT(0.5)*(rho_nodal(i,j) + rho_nodal(i,j+1));
            }
            const RT rho_lim = HybridSmoothFloor(rho_val,
                std::max(rho_floor, RT(1.0e-300)), floor_w);
            RT irho = RT(1.0)/rho_lim;
            if (holm && !holm_smooth && rho_val < rho_floor) { irho = RT(0.0); }
            if (holm_smooth) {
                irho *= RT(0.5)*(RT(1.0) + std::tanh((rho_val - rho_floor)*holm_iw));
            }
            irho_a.ref(i,j) = irho;

            RT jtot = RT(0.0);
            if (has_eta && has_J_dep) {
                RT jr, jt, jjz;
                if (comp == 0) {
                    jr = J0r.get(i,j);
                    jt = RT(0.5)*(J0t.get(i,j) + J0t.get(i+1,j));
                    jjz = RT(0.25)*( J0z.get(i,j-1) + J0z.get(i,j)
                                   + J0z.get(i+1,j-1) + J0z.get(i+1,j) );
                } else if (comp == 1) {
                    jr = RT(0.5)*(J0r.get(i-1,j) + J0r.get(i,j));
                    jt = J0t.get(i,j);
                    jjz = RT(0.5)*(J0z.get(i,j-1) + J0z.get(i,j));
                } else {
                    jr = RT(0.25)*( J0r.get(i-1,j) + J0r.get(i,j)
                                  + J0r.get(i-1,j+1) + J0r.get(i,j+1) );
                    jt = RT(0.5)*(J0t.get(i,j) + J0t.get(i,j+1));
                    jjz = J0z.get(i,j);
                }
                jtot = std::sqrt(jr*jr + jt*jt + jjz*jjz);
            }
            eta_a.ref(i,j) = has_eta
                ? static_cast<RT>(eta_exec(rho_val, jtot, time)) : RT(0.0);

            if (hyper_on) {
                RT btot = RT(0.0);
                if (hybrid->m_hyper_resistivity_has_B_dependence) {
                    // 2-point interps of staggered B0 to the row point
                    RT br, bt, bz;
                    if (comp == 0) {
                        br = RT(0.5)*(B0r.get(i,j) + B0r.get(i+1,j));  // approx
                        bt = RT(0.5)*(B0t.get(i,j-1) + B0t.get(i,j));
                        bz = B0z.get(i,j);
                    } else if (comp == 1) {
                        br = RT(0.5)*(B0r.get(i,j-1) + B0r.get(i,j));
                        bt = RT(0.25)*( B0t.get(i-1,j-1) + B0t.get(i-1,j)
                                      + B0t.get(i,j-1) + B0t.get(i,j) );
                        bz = RT(0.5)*(B0z.get(i-1,j) + B0z.get(i,j));
                    } else {
                        br = B0r.get(i,j);
                        bt = RT(0.5)*(B0t.get(i-1,j) + B0t.get(i,j));
                        bz = RT(0.25)*( B0z.get(i-1,j) + B0z.get(i,j)
                                      + B0z.get(i-1,j+1) + B0z.get(i,j+1) );
                    }
                    btot = std::sqrt(br*br + bt*bt + bz*bz);
                }
                etah_a.ref(i,j) = static_cast<RT>(etah_exec(rho_val, btot));
            } else {
                etah_a.ref(i,j) = RT(0.0);
            }

            // (the electron-inertia response is carried exactly by the
            // nodal ciN chain in the frozen apply; no diagonal surrogate)
            ad_a.ref(i,j) = RT(0.0);
        };

        for (int i = S.oEr.ilo; i <= S.oEr.ihi; i++) {
            for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) {
                row_coeffs(0, i, j, S.etaEr, S.etahEr, S.irhoEr, S.adEr);
            }
        }
        for (int i = S.oEt.ilo; i <= S.oEt.ihi; i++) {
            for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) {
                row_coeffs(1, i, j, S.etaEt, S.etahEt, S.irhoEt, S.adEt);
            }
        }
        for (int i = S.oEz.ilo; i <= S.oEz.ihi; i++) {
            for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) {
                row_coeffs(2, i, j, S.etaEz, S.etahEz, S.irhoEz, S.adEz);
            }
        }

        // Face-point eta_H for the composed hyper-resistivity rows
        // (mirrors the Ohm wrapper's face-averaged coefficient inside the
        // outer curl). Cached knob: rows must match the configured
        // residual operator.
        m_hyperres_cc = hyper_on && hybrid->m_hyper_resistivity_curlcurl;
        if (m_hyperres_cc) {
            const bool b_dep = hybrid->m_hyper_resistivity_has_B_dependence;
            auto etah_face = [&](int comp, int i, int j) -> RT {
                RT rho_f, br, bt, bz;
                if (comp == 0) {        // Wr point (node r, cell z)
                    rho_f = RT(0.5)*(rho_nodal(i,j) + rho_nodal(i,j+1));
                    br = B0r.get(i,j);
                    bt = RT(0.5)*(B0t.get(i-1,j) + B0t.get(i,j));
                    bz = RT(0.25)*( B0z.get(i-1,j) + B0z.get(i,j)
                                  + B0z.get(i-1,j+1) + B0z.get(i,j+1) );
                } else if (comp == 1) { // Wt point (cell r, cell z)
                    rho_f = RT(0.25)*( rho_nodal(i,j) + rho_nodal(i+1,j)
                                     + rho_nodal(i,j+1) + rho_nodal(i+1,j+1) );
                    br = RT(0.5)*(B0r.get(i,j) + B0r.get(i+1,j));
                    bt = B0t.get(i,j);
                    bz = RT(0.5)*(B0z.get(i,j) + B0z.get(i,j+1));
                } else {                // Wz point (cell r, node z)
                    rho_f = RT(0.5)*(rho_nodal(i,j) + rho_nodal(i+1,j));
                    br = RT(0.25)*( B0r.get(i,j-1) + B0r.get(i,j)
                                  + B0r.get(i+1,j-1) + B0r.get(i+1,j) );
                    bt = RT(0.5)*(B0t.get(i,j-1) + B0t.get(i,j));
                    bz = B0z.get(i,j);
                }
                const RT btot = b_dep
                    ? std::sqrt(br*br + bt*bt + bz*bz) : RT(0.0);
                return static_cast<RT>(etah_exec(rho_f, btot));
            };
            for (int i = S.etahBr.ilo; i <= S.etahBr.ihi; i++) {
                for (int j = S.etahBr.jlo; j <= S.etahBr.jhi; j++) {
                    S.etahBr.ref(i,j) = etah_face(0, i, j);
                }
            }
            for (int i = S.etahBt.ilo; i <= S.etahBt.ihi; i++) {
                for (int j = S.etahBt.jlo; j <= S.etahBt.jhi; j++) {
                    S.etahBt.ref(i,j) = etah_face(1, i, j);
                }
            }
            for (int i = S.etahBz.ilo; i <= S.etahBz.ihi; i++) {
                for (int j = S.etahBz.jlo; j <= S.etahBz.jhi; j++) {
                    S.etahBz.ref(i,j) = etah_face(2, i, j);
                }
            }
        }

        // ---- curlcurl_form coefficient windows ----
        // theta rows read the LIVE midpoint density; poloidal rows read
        // the per-step-frozen rho^n snapshot when in play (hRhoPol), the
        // midpoint deposit otherwise -- exactly the solves' own selection.
        // Both use the kernels' cc-convention index arithmetic on the
        // nodal deposit (mirrored, including the flagged half-cell
        // misregistration: the PC targets the operator as built).
        if (m_form_curlcurl) {
            const RT q_e = static_cast<RT>(PhysConst::q_e);
            const RT me_eff_cc =
                static_cast<RT>(hybrid->m_electron_inertia_mass);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(me_eff_cc > RT(0.0),
                "pc_curlcurl_banded: electron inertia mass not resolved "
                "before the first preconditioner update");
            const RT beta_fac = mu0*q_e/me_eff_cc;
            const RT rhs_fac = beta_fac;
            // one-count membership anchor (mirrors the solves' gates;
            // 0 = the legacy strictly-zero gating)
            const RT rho_1c =
                q_e*static_cast<RT>(hybrid->m_curlcurl_n_min);
            auto rhoPol = [&](int i, int j) -> RT {
                return have_rho_pol ? hRhoPol.get(i,j,0) : hRho.get(i,j,0);
            };
            for (int i = S.acTh.ilo; i <= S.acTh.ihi; i++) {
                const RT rr = std::max(rmin + i*dr, RT(0.125)*dr);
                for (int j = S.acTh.jlo; j <= S.acTh.jhi; j++) {
                    const RT rho_n4 = std::max(RT(0.0),
                        RT(0.25)*(hRho.get(i,j) + hRho.get(i-1,j)
                                + hRho.get(i,j-1) + hRho.get(i-1,j-1)));
                    S.acTh.ref(i,j) = rr*beta_fac*rho_n4 + RT(1.0)/rr;
                    S.foldTh.ref(i,j) = rho_n4;
                    const bool dir = (i <= 0) || (i >= m_ihi_nd)
                        || (!S.zwrap && (j <= m_zlo_d || j >= m_zhi_nd))
                        || (rho_1c > RT(0.0) && !(rho_n4 > rho_1c));
                    S.rhswTh.ref(i,j) = dir ? RT(0.0) : rr*rhs_fac;
                }
            }
            for (int i = S.acPr.ilo; i <= S.acPr.ihi; i++) {
                const RT V = (rmin + (i + RT(0.5))*dr)*dr;
                for (int j = S.acPr.jlo; j <= S.acPr.jhi; j++) {
                    const RT rho_p = std::max(RT(0.0),
                        RT(0.5)*(rhoPol(i,j) + rhoPol(i,j-1)));
                    S.acPr.ref(i,j) = beta_fac*rho_p*V;
                    S.foldPr.ref(i,j) = rho_p;
                    const bool dir =
                        (!S.zwrap && (j <= m_zlo_d || j >= m_zhi_nd));
                    S.rhswPr.ref(i,j) = (dir || !(rho_p > rho_1c))
                        ? RT(0.0) : V*rhs_fac;
                }
            }
            for (int i = S.acPz.ilo; i <= S.acPz.ihi; i++) {
                const RT V = (i == 0) ? RT(0.125)*dr*dr : (rmin + i*dr)*dr;
                for (int j = S.acPz.jlo; j <= S.acPz.jhi; j++) {
                    const RT rho_p = std::max(RT(0.0),
                        RT(0.5)*(rhoPol(i,j) + rhoPol(std::max(i-1,0),j)));
                    S.acPz.ref(i,j) = beta_fac*rho_p*V;
                    S.foldPz.ref(i,j) = rho_p;
                    const bool dir = (i >= m_ihi_nd);
                    S.rhswPz.ref(i,j) = (dir || !(rho_p > rho_1c))
                        ? RT(0.0) : V*rhs_fac;
                }
            }
            // nodal vacuum-closure gate (one-sided at the axis/wall nodes,
            // zero on non-periodic caps; out-of-window reads return zero =
            // the solve's setBndry(0) convention)
            for (int i = S.gamN.ilo; i <= S.gamN.ihi; i++) {
                for (int j = S.gamN.jlo; j <= S.gamN.jhi; j++) {
                    if (!S.zwrap && (j <= m_zlo_d || j >= m_zhi_nd)) {
                        S.gamN.ref(i,j) = RT(0.0);
                        continue;
                    }
                    RT rho_n;
                    if (i <= 0) {
                        rho_n = RT(0.5)*(rhoPol(i,j) + rhoPol(i,j-1));
                    } else if (i >= m_ihi_nd) {
                        rho_n = RT(0.5)*(rhoPol(i-1,j) + rhoPol(i-1,j-1));
                    } else {
                        rho_n = RT(0.25)*(rhoPol(i,j) + rhoPol(i-1,j)
                                        + rhoPol(i,j-1) + rhoPol(i-1,j-1));
                    }
                    S.gamN.ref(i,j) = (rho_n > rho_1c) ? RT(0.0) : RT(1.0);
                }
            }
        }
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::ProbeAndFill (BoxSys& S, bool a_A_only)
{
    // DOF (column) windows per component: lit points may only sit on
    // actual unknowns, not on the padded input windows
    const PArr* colw[3] = { &S.oEr, &S.oEt, &S.oEz };
    PArr* inw[3] = { &S.inEr, &S.inEt, &S.inEz };

    // Allocate at the probing bandwidth budget (3); the effective band is
    // trimmed after harvest from what actually landed. (A pre-measurement
    // pass is unreliable on narrow boxes, where the only couplings at a
    // given reach can run through killed wall columns.)
    const int SP = 8;
    S.bw = 3;

    // ---- allocate blocks (a_A_only extracts the unfactored A blocks) ----
    auto& tblk = a_A_only ? S.blkA : S.blk;
    tblk.assign(S.nb, {});
    if (!a_A_only) { S.piv.assign(S.nb, {}); }
    for (int b = 0; b < S.nb; b++) {
        tblk[b].resize(2*S.bw + 1);
        for (int d = 0; d <= 2*S.bw; d++) {
            const int bc = b + d - S.bw;
            if (bc < 0 || bc >= S.nb) { continue; }
            tblk[b][d].assign(std::size_t(S.bsz[b])*S.bsz[bc], RT(0.0));
        }
        if (!a_A_only) { S.piv[b].assign(S.bsz[b], 0); }
    }

    // ---- pass 2: extraction ----
    // z spacing: for wrapped boxes it must divide the period so the lit
    // pattern stays uniform across the seam (>= 7 keeps a 3-reach window
    // unambiguous; if no divisor exists, fall back to one z per probe --
    // the extraction self-check backstops any residual aliasing)
    int SZ = SP;
    if (S.zwrap) {
        SZ = 0;
        for (int s = 8; s <= std::min(S.nzc, 64); s++) {
            if (S.nzc % s == 0) { SZ = s; break; }
        }
        if (SZ == 0) { SZ = S.nzc; }
    }
    const int zPeriod = S.nzc;

    for (int pc = 0; pc < 3; pc++) {
        const PArr& cw = *colw[pc];
        PArr& in = *inw[pc];
        for (int pr = 0; pr < SP; pr++) {
            for (int pz = 0; pz < SZ; pz++) {
                S.inEr.setzero(); S.inEt.setzero(); S.inEz.setzero();
                bool any = false;
                for (int i = cw.ilo + pr; i <= cw.ihi; i += SP) {
                    for (int j = cw.jlo + pz; j <= cw.jhi; j += SZ) {
                        in.ref(i,j) = RT(1.0);
                        any = true;
                    }
                }
                if (!any) { continue; }
                ApplyFrozenBox(S, a_A_only);

                // for every lit column, harvest the rows in its window
                for (int ci = cw.ilo + pr; ci <= cw.ihi; ci += SP) {
                    const int bcol = ci - S.rlo_s;
                    const int ncol = S.bsz[bcol];
                    for (int cj = cw.jlo + pz; cj <= cw.jhi; cj += SZ) {
                        int lcol;
                        if (pc == 0) { lcol = locEr(S, bcol, cj); }
                        else if (pc == 1) { lcol = locEt(S, bcol, cj); }
                        else { lcol = locEz(S, bcol, cj); }

                        auto harvest = [&](const PArr& out, int rcomp) {
                            for (int io = std::max(out.ilo, ci - S.bw);
                                 io <= std::min(out.ihi, ci + S.bw); io++) {
                                const int brow = io - S.rlo_s;
                                const int d = bcol - brow + S.bw;
                                if (d < 0 || d > 2*S.bw) { continue; }
                                auto& blkv = tblk[brow][d];
                                if (blkv.empty()) { continue; }
                                for (int dj = -3; dj <= 3; dj++) {
                                    int jo = cj + dj;
                                    if (out.zwrap) {
                                        jo = out.jlo + ((jo - out.jlo) % zPeriod + zPeriod) % zPeriod;
                                    } else if (jo < out.jlo || jo > out.jhi) { continue; }
                                    const RT val = out.get(io, jo);
                                    if (val == RT(0.0)) { continue; }
                                    int lrow;
                                    if (rcomp == 0) { lrow = locEr(S, brow, jo); }
                                    else if (rcomp == 1) { lrow = locEt(S, brow, jo); }
                                    else { lrow = locEz(S, brow, jo); }
                                    blkv[std::size_t(lrow)*ncol + lcol] = val;
                                }
                            }
                        };
                        harvest(S.oEr, 0);
                        harvest(S.oEt, 1);
                        harvest(S.oEz, 2);
                    }
                }
            }
        }
    }

    // trim the effective bandwidth: drop outer block diagonals that are
    // entirely zero (FactorBox/SolveBox/ApplyBlocks loop to S.bwf).
    // Skipped for the A-only pass: A's reach is a subset of the full
    // operator's and blkA is only ever used through the full-band matvec.
    if (a_A_only) { return; }
    int bwf = 1;
    for (int b = 0; b < S.nb; b++) {
        for (int d = 0; d <= 2*S.bw; d++) {
            const int reach = std::abs(d - S.bw);
            if (reach <= bwf) { continue; }
            const auto& A = S.blk[b][d];
            for (const RT v : A) {
                if (v != RT(0.0)) { bwf = reach; break; }
            }
        }
    }
    S.bwf = bwf;
}

// ---------------- dense kernels (row-major) ----------------

template <class T, class Ops>
void BlockBandedPC<T,Ops>::lu_factor (int n, RT* A, int* piv)
{
    // Right-looking blocked LU with partial pivoting: the O(n^3) work is
    // in the trailing GEMM update, threaded via ParFor; the O(n^2 * NB)
    // panel factorization stays serial.
    constexpr int NB = 64;
    for (int kp = 0; kp < n; kp += NB) {
        const int ke = std::min(kp + NB, n);
        // panel columns kp..ke-1 (full-row pivot swaps)
        for (int k = kp; k < ke; k++) {
            int p = k;
            RT amax = std::abs(A[std::size_t(k)*n + k]);
            for (int i = k+1; i < n; i++) {
                const RT v = std::abs(A[std::size_t(i)*n + k]);
                if (v > amax) { amax = v; p = i; }
            }
            piv[k] = p;
            if (p != k) {
                for (int j = 0; j < n; j++) {
                    std::swap(A[std::size_t(k)*n + j], A[std::size_t(p)*n + j]);
                }
            }
            const RT dkk = A[std::size_t(k)*n + k];
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(dkk != RT(0.0),
                "pc_block_banded: singular diagonal block in LU");
            const RT inv = RT(1.0)/dkk;
            const RT* rowk = &A[std::size_t(k)*n];
            for (int i = k+1; i < n; i++) {
                RT* rowi = &A[std::size_t(i)*n];
                rowi[k] *= inv;
                const RT lik = rowi[k];
                if (lik == RT(0.0)) { continue; }
                for (int j = k+1; j < ke; j++) {
                    rowi[j] -= lik*rowk[j];
                }
            }
        }
        if (ke >= n) { break; }
        // U-panel: A[kp:ke, ke:n] := L(kp:ke, kp:ke)^{-1} * A[kp:ke, ke:n]
        // (unit-lower solve, parallel over trailing columns)
        ParFor(n - ke, 256, [&](int jj) {
            const int j = ke + jj;
            for (int i = kp+1; i < ke; i++) {
                const RT* rowi = &A[std::size_t(i)*n];
                RT s = rowi[j];
                for (int k2 = kp; k2 < i; k2++) {
                    s -= rowi[k2]*A[std::size_t(k2)*n + j];
                }
                A[std::size_t(i)*n + j] = s;
            }
        });
        // trailing update: A[ke:n, ke:n] -= A[ke:n, kp:ke] * A[kp:ke, ke:n]
        ParFor(n - ke, 16, [&](int ii) {
            const int i = ke + ii;
            RT* ci = &A[std::size_t(i)*n];
            for (int k2 = kp; k2 < ke; k2++) {
                const RT a = ci[k2];
                if (a == RT(0.0)) { continue; }
                const RT* bk = &A[std::size_t(k2)*n];
                for (int j = ke; j < n; j++) {
                    ci[j] -= a*bk[j];
                }
            }
        });
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::lu_solve (int n, const RT* A, const int* piv, RT* b)
{
    for (int k = 0; k < n; k++) {
        if (piv[k] != k) { std::swap(b[k], b[piv[k]]); }
    }
    for (int i = 1; i < n; i++) {
        const RT* rowi = &A[std::size_t(i)*n];
        RT s = b[i];
        for (int j = 0; j < i; j++) { s -= rowi[j]*b[j]; }
        b[i] = s;
    }
    for (int i = n-1; i >= 0; i--) {
        const RT* rowi = &A[std::size_t(i)*n];
        RT s = b[i];
        for (int j = i+1; j < n; j++) { s -= rowi[j]*b[j]; }
        b[i] = s/rowi[i];
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::lu_solve_T (int n, const RT* A, const int* piv, RT* b)
{
    // Solve A^T y = b with P A = L U  =>  A^T = U^T L^T P.
    // U^T w = b (forward, non-unit diag); L^T v = w (backward, unit diag);
    // y = P^T v (undo row swaps in reverse order).
    for (int i = 0; i < n; i++) {
        RT s = b[i];
        for (int j = 0; j < i; j++) { s -= A[std::size_t(j)*n + i]*b[j]; }
        b[i] = s/A[std::size_t(i)*n + i];
    }
    for (int i = n-1; i >= 0; i--) {
        RT s = b[i];
        for (int j = i+1; j < n; j++) { s -= A[std::size_t(j)*n + i]*b[j]; }
        b[i] = s;
    }
    for (int k = n-1; k >= 0; k--) {
        if (piv[k] != k) { std::swap(b[k], b[piv[k]]); }
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::solve_right (int m, int n, RT* X, const RT* Df, const int* piv)
{
    // X (m x n, row-major) := X * D^{-1}: each row x of X solves D^T y = x.
    ParFor(m, 4, [&](int i) {
        lu_solve_T(n, Df, piv, &X[std::size_t(i)*n]);
    });
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::gemm_minus (int m, int k, int n, const RT* A, const RT* B, RT* C)
{
    // C (m x n) -= A (m x k) * B (k x n), all row-major
    ParFor(m, 8, [&](int i) {
        RT* ci = &C[std::size_t(i)*n];
        const RT* ai = &A[std::size_t(i)*k];
        for (int p = 0; p < k; p++) {
            const RT a = ai[p];
            if (a == RT(0.0)) { continue; }
            const RT* bp = &B[std::size_t(p)*n];
            for (int j = 0; j < n; j++) { ci[j] -= a*bp[j]; }
        }
    });
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::gemv_minus (int m, int n, const RT* A, const RT* x, RT* y)
{
    ParFor(m, 128, [&](int i) {
        const RT* ai = &A[std::size_t(i)*n];
        RT s = RT(0.0);
        for (int j = 0; j < n; j++) { s += ai[j]*x[j]; }
        y[i] -= s;
    });
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::ApplyBlocks (const BoxSys& S,
    const std::vector<std::vector<std::vector<RT>>>& blk,
    const std::vector<RT>& v, std::vector<RT>& w)
{
    w.assign(S.ntot, RT(0.0));
    for (int b = 0; b < S.nb; b++) {
        for (int d = 0; d <= 2*S.bw; d++) {
            const int bc = b + d - S.bw;
            if (bc < 0 || bc >= S.nb) { continue; }
            const auto& A = blk[b][d];
            if (A.empty()) { continue; }
            const int m = S.bsz[b], n = S.bsz[bc];
            for (int i = 0; i < m; i++) {
                RT s = RT(0.0);
                const RT* ai = &A[std::size_t(i)*n];
                const RT* xv = &v[S.boff[bc]];
                for (int j = 0; j < n; j++) { s += ai[j]*xv[j]; }
                w[S.boff[b] + i] += s;
            }
        }
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::FactorBox (BoxSys& S) const
{
    for (int b = 0; b < S.nb; b++) {
        auto& D = S.blk[b][S.bw];
        lu_factor(S.bsz[b], D.data(), S.piv[b].data());
        for (int l = 1; l <= S.bwf && b + l < S.nb; l++) {
            auto& L = S.blk[b+l][S.bw - l];   // row b+l, col b
            if (L.empty()) { continue; }
            solve_right(S.bsz[b+l], S.bsz[b], L.data(), D.data(), S.piv[b].data());
            for (int m = 1; m <= S.bwf && b + m < S.nb; m++) {
                auto& U = S.blk[b][S.bw + m];     // row b, col b+m
                if (U.empty()) { continue; }
                auto& C = S.blk[b+l][S.bw - l + m]; // row b+l, col b+m
                if (C.empty()) { continue; }
                gemm_minus(S.bsz[b+l], S.bsz[b], S.bsz[b+m],
                           L.data(), U.data(), C.data());
            }
        }
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::SolveBox (BoxSys& S) const
{
    RT* x = S.x.data();
    // forward (unit block lower)
    for (int b = 0; b < S.nb; b++) {
        for (int l = 1; l <= S.bwf && b - l >= 0; l++) {
            const auto& L = S.blk[b][S.bw - l];
            if (L.empty()) { continue; }
            gemv_minus(S.bsz[b], S.bsz[b-l], L.data(), &x[S.boff[b-l]], &x[S.boff[b]]);
        }
    }
    // backward
    for (int b = S.nb - 1; b >= 0; b--) {
        for (int m = 1; m <= S.bwf && b + m < S.nb; m++) {
            const auto& U = S.blk[b][S.bw + m];
            if (U.empty()) { continue; }
            gemv_minus(S.bsz[b], S.bsz[b+m], U.data(), &x[S.boff[b+m]], &x[S.boff[b]]);
        }
        lu_solve(S.bsz[b], S.blk[b][S.bw].data(), S.piv[b].data(), &x[S.boff[b]]);
    }
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::SelfCheck (BoxSys& S, int isys, bool a_A_only) const
{
    // extracted matrix must reproduce the frozen apply on a random vector
    // (exact modulo roundoff; disagreement means the band was aliased)
    auto hashv = [](int a, int b2, int c) {
        std::uint32_t h = std::uint32_t(a*73856093) ^ std::uint32_t(b2*19349663)
                        ^ std::uint32_t(c*83492791);
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        return RT(h % 20011)/RT(20011.0) - RT(0.5);
    };
    S.inEr.setzero(); S.inEt.setzero(); S.inEz.setzero();
    std::vector<RT> v(S.ntot, RT(0.0));
    for (int b = 0; b < S.nb; b++) {
        const int i = S.rlo_s + b;
        if (blockHasEr(S,b)) {
            for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) {
                const RT val = hashv(i,j,0);
                S.inEr.ref(i,j) = val;
                v[S.boff[b] + locEr(S,b,j)] = val;
            }
        }
        for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) {
            const RT val = hashv(i,j,1);
            S.inEt.ref(i,j) = val;
            v[S.boff[b] + locEt(S,b,j)] = val;
        }
        for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) {
            const RT val = hashv(i,j,2);
            S.inEz.ref(i,j) = val;
            v[S.boff[b] + locEz(S,b,j)] = val;
        }
    }
    ApplyFrozenBox(S, a_A_only);

    // matrix apply
    std::vector<RT> w;
    ApplyBlocks(S, a_A_only ? S.blkA : S.blk, v, w);

    RT num = RT(0.0), den = RT(0.0);
    for (int b = 0; b < S.nb; b++) {
        const int i = S.rlo_s + b;
        auto acc = [&](const PArr& out, int lidx, int j, int comp) {
            amrex::ignore_unused(comp);
            const RT d = out.get(i,j) - w[S.boff[b] + lidx];
            num += d*d;
            den += out.get(i,j)*out.get(i,j);
        };
        if (blockHasEr(S,b)) {
            for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) { acc(S.oEr, locEr(S,b,j), j, 0); }
        }
        for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) { acc(S.oEt, locEt(S,b,j), j, 1); }
        for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) { acc(S.oEz, locEz(S,b,j), j, 2); }
    }
    const RT rel = (den > RT(0.0)) ? std::sqrt(num/den) : std::sqrt(num);
    if (m_verbose || rel > RT(1.0e-11)) {
        amrex::AllPrint() << "pc_block_banded: box " << isys
            << (a_A_only ? " A-extraction" : " extraction")
            << " self-check rel err = " << rel
            << " (bw " << S.bw << ")\n";
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rel <= RT(1.0e-10),
        "pc_block_banded: extracted matrix disagrees with the frozen apply "
        "(band aliasing or extraction bug)");
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::VerifyFD (const T& a_U)
{
    // FD-JVP of the true residual vs the assembled operator on a
    // deterministic pseudo-random direction. Expected agreement ~1e-5
    // (FD noise + deliberately omitted legs).
    T dU, R0, R1, Z, Jd;
    dU.Define(a_U); R0.Define(a_U); R1.Define(a_U); Z.Define(a_U); Jd.Define(a_U);
    dU.zero();

    // wrap the z index before hashing: duplicated periodic z-nodes must
    // carry the SAME direction value, or the solver's nodal sync makes the
    // effective FD direction differ from the one the PC matvec reads
    const auto& geomv = m_ops->GetGeometry(0);
    const int zdlo = geomv.Domain().smallEnd(1);
    const int zdn = geomv.Domain().length(1);
    const int zwrapv = geomv.isPeriodic(1) ? 1 : 0;

    auto& du_vec = dU.getArrayVec();
    for (int dim = 0; dim < 3; dim++) {
        amrex::MultiFab& mf = *(du_vec[0][dim]);
        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.validbox();
            auto arr = mf.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                int jw = j;
                if (zwrapv) { jw = zdlo + (((j - zdlo) % zdn) + zdn) % zdn; }
                unsigned int h = static_cast<unsigned int>(i*73856093)
                               ^ static_cast<unsigned int>(jw*19349663)
                               ^ static_cast<unsigned int>(dim*83492791);
                h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
                arr(i,j,k) = amrex::Real(h % 20011u)/amrex::Real(20011.0) - amrex::Real(0.5);
            });
        }
    }

    const RT normU = a_U.norm2();
    const RT normdU = dU.norm2();
    const RT eps = (normU > RT(0.0)) ? RT(1.0e-6)*normU/normdU : RT(1.0e-6)/normdU;

    m_ops->ComputeRHS(R0, a_U, this->m_time, -1, true);
    {
        // Determinism probe: evaluate the residual twice at the IDENTICAL
        // state and report the difference. On a deterministic backend this
        // is exactly zero; any nonzero value is the per-evaluation
        // stochasticity of the residual (e.g. deposit-order effects) and
        // sets the floor below which FD Jacobian probes lose coherence.
        T R0b;
        R0b.Define(a_U);
        m_ops->ComputeRHS(R0b, a_U, this->m_time, -1, true);
        R0b.increment(R0, -RT(1.0));
        const RT n0 = R0.norm2();
        const RT dd = R0b.norm2();
        amrex::Print() << "pc_block_banded DETERMINISM: ||R(U)-R(U)|| = "
            << dd << "  rel = " << (n0 > RT(0.0) ? dd/n0 : dd) << "\n";
        // localize: worst row per component (residual-map address of the
        // statefulness/stochasticity)
        for (int dim = 0; dim < 3; dim++) {
            const amrex::MultiFab& d_mf = *R0b.getArrayVec()[0][dim];
            RT dmax = RT(0.0);
            int imax = -1, jmax = -1;
            for (amrex::MFIter mfi(d_mf); mfi.isValid(); ++mfi) {
                HostFab hd;
                CopyFabToHost(d_mf, mfi, hd);
                const amrex::Box vb = mfi.validbox();
                const auto lo = vb.smallEnd(); const auto hi = vb.bigEnd();
                for (int i = lo[0]; i <= hi[0]; i++) {
                    for (int j = lo[1]; j <= hi[1]; j++) {
                        const RT v = std::abs(hd.get(i,j));
                        if (v > dmax) { dmax = v; imax = i; jmax = j; }
                    }
                }
            }
            amrex::AllPrint() << "  det comp " << dim << ": max |dR| = "
                << dmax << " at (r,z idx) = (" << imax << "," << jmax
                << ")\n";
        }
    }
    Z.Copy(a_U);
    Z.increment(dU, eps);
    m_ops->ComputeRHS(R1, Z, this->m_time, -1, true);

    // FD Jacobian action: J*dU = dU - (R1 - R0)/eps
    Jd.Copy(dU);
    Jd.increment(R1, -RT(1.0)/eps);
    Jd.increment(R0,  RT(1.0)/eps);

    // assembled-operator action on the same direction, per box
    T Jm;
    Jm.Define(a_U);
    Jm.zero();
    auto& jm_vec = Jm.getArrayVec();
    auto& du_vec2 = dU.getArrayVec();
    const amrex::MultiFab& Ermf = *(du_vec2[0][0]);

    if (m_global_on) {
        // gather the direction once (collective), apply the single global
        // system, then write the owned boxes
        BoxSys& S = m_sys[0];
        auto apply_model = [&](T& a_out, T& a_in, bool a_A_only)
        {
            auto& in_vec = a_in.getArrayVec();
            auto& out_vec = a_out.getArrayVec();
            HostFab hEr, hEt, hEz;
            GatherGlobal(*(in_vec[0][0]), 0, hEr, 1);
            GatherGlobal(*(in_vec[0][1]), 0, hEt, 1);
            GatherGlobal(*(in_vec[0][2]), 0, hEz, 1);
            S.inEr.setzero(); S.inEt.setzero(); S.inEz.setzero();
            for (int i = S.inEr.ilo; i <= S.inEr.ihi; i++) {
                for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) { S.inEr.ref(i,j) = hEr.get(i,j); }
            }
            for (int i = S.inEt.ilo; i <= S.inEt.ihi; i++) {
                for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) { S.inEt.ref(i,j) = hEt.get(i,j); }
            }
            for (int i = S.inEz.ilo; i <= S.inEz.ihi; i++) {
                for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) { S.inEz.ref(i,j) = hEz.get(i,j); }
            }
            ApplyFrozenBox(S, a_A_only);
            for (int dim = 0; dim < 3; dim++) {
                amrex::MultiFab& mf = *(out_vec[0][dim]);
                const PArr& out = (dim==0) ? S.oEr : ((dim==1) ? S.oEt : S.oEz);
                for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
                    const amrex::Box tb = amrex::convert(mfi.validbox(), mf.ixType());
#ifdef AMREX_USE_GPU
                    amrex::FArrayBox hout(tb, 1, amrex::The_Pinned_Arena());
                    auto ha = hout.array();
                    amrex::LoopOnCpu(tb, [&](int i, int j, int k) { ha(i,j,k) = out.get(i,j); });
                    amrex::FArrayBox& dfab = mf[mfi];
                    amrex::Gpu::htod_memcpy(dfab.dataPtr(), hout.dataPtr(),
                                            hout.size()*sizeof(amrex::Real));
#else
                    auto darr = mf.array(mfi);
                    amrex::LoopOnCpu(tb, [&](int i, int j, int k) { darr(i,j,k) = out.get(i,j); });
#endif
                }
            }
        };
        apply_model(Jm, dU, false);
        if (m_form_curlcurl) {
            // A-composed gate: the model J = A^{-1}(A - N) cannot be
            // applied without factoring A, so compare pure applies instead:
            // A (J_fd du) must equal (A - N) du. Replace Jd by A*Jd; Jm
            // already holds (A - N) du.
            T AJd;
            AJd.Define(a_U);
            AJd.zero();
            apply_model(AJd, Jd, true);
            Jd.Copy(AJd);
        }
    } else {

    int isys = 0;
    for (amrex::MFIter mfi(Ermf); mfi.isValid(); ++mfi, ++isys) {
        BoxSys& S = m_sys[isys];
        HostFab hEr, hEt, hEz;
        CopyFabToHost(*(du_vec2[0][0]), mfi, hEr);
        CopyFabToHost(*(du_vec2[0][1]), mfi, hEt);
        CopyFabToHost(*(du_vec2[0][2]), mfi, hEz);
        S.inEr.setzero(); S.inEt.setzero(); S.inEz.setzero();
        for (int i = S.inEr.ilo; i <= S.inEr.ihi; i++) {
            for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) { S.inEr.ref(i,j) = hEr.get(i,j); }
        }
        for (int i = S.inEt.ilo; i <= S.inEt.ihi; i++) {
            for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) { S.inEt.ref(i,j) = hEt.get(i,j); }
        }
        for (int i = S.inEz.ilo; i <= S.inEz.ihi; i++) {
            for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) { S.inEz.ref(i,j) = hEz.get(i,j); }
        }
        ApplyFrozenBox(S);

        // write outputs into Jm on the valid box (host staging)
        for (int dim = 0; dim < 3; dim++) {
            amrex::MultiFab& mf = *(jm_vec[0][dim]);
            const amrex::Box tb = amrex::convert(mfi.validbox(), mf.ixType());
            const PArr& out = (dim==0) ? S.oEr : ((dim==1) ? S.oEt : S.oEz);
#ifdef AMREX_USE_GPU
            amrex::FArrayBox hout(tb, 1, amrex::The_Pinned_Arena());
            auto ha = hout.array();
            amrex::LoopOnCpu(tb, [&](int i, int j, int k) {
                ha(i,j,k) = out.get(i, j);
            });
            amrex::FArrayBox& dfab = mf[mfi];
            amrex::Gpu::htod_memcpy(dfab.dataPtr(), hout.dataPtr(),
                                    hout.size()*sizeof(amrex::Real));
#else
            auto darr = mf.array(mfi);
            amrex::LoopOnCpu(tb, [&](int i, int j, int k) {
                darr(i,j,k) = out.get(i, j);
            });
#endif
        }
    }
    }

    // compare
    T diff;
    diff.Define(a_U);
    diff.Copy(Jd);
    diff.increment(Jm, -RT(1.0));
    const RT nJd = Jd.norm2();
    const RT ndiff = diff.norm2();
    if (m_form_curlcurl) {
        amrex::Print() << "pc_curlcurl_banded VERIFY (A-composed): "
            << "||A J_fd du|| = " << nJd
            << ", ||A J_fd du - (A-N) du||/||A J_fd du|| = "
            << (nJd > RT(0.0) ? ndiff/nJd : ndiff) << "\n";
    } else {
    amrex::Print() << "pc_block_banded VERIFY: ||J_fd*du|| = " << nJd
        << ", ||J_fd*du - J_pc*du||/||J_fd*du|| = "
        << (nJd > RT(0.0) ? ndiff/nJd : ndiff) << "\n";
    }

    // per-component breakdown + worst-row locator (localizes wall/axis
    // ring truncation vs distributed stencil error)
    for (int dim = 0; dim < 3; dim++) {
        const amrex::MultiFab& d_mf = *(diff.getArrayVec()[0][dim]);
        const amrex::MultiFab& j_mf = *(Jd.getArrayVec()[0][dim]);
        const RT dn = d_mf.norm2(0);
        const RT jn = j_mf.norm2(0);
        RT dmax = RT(0.0);
        int imax = -1, jmax = -1;
        for (amrex::MFIter mfi(d_mf); mfi.isValid(); ++mfi) {
            HostFab hd;
            CopyFabToHost(d_mf, mfi, hd);
            const amrex::Box vb = mfi.validbox();
            const auto lo = vb.smallEnd(); const auto hi = vb.bigEnd();
            for (int i = lo[0]; i <= hi[0]; i++) {
                for (int j = lo[1]; j <= hi[1]; j++) {
                    const RT v = std::abs(hd.get(i,j));
                    if (v > dmax) { dmax = v; imax = i; jmax = j; }
                }
            }
        }
        amrex::Print() << "  comp " << dim << ": rel = "
            << (jn > RT(0.0) ? dn/jn : dn)
            << ", max |diff| = " << dmax << " at (r,z idx) = ("
            << imax << "," << jmax << ")\n";
    }
}

#endif  // WARPX_DIM_RZ

#if defined(WARPX_DIM_RZ) && defined(WARPX_USE_CUDSS)

template <class T, class Ops>
bool BlockBandedPC<T,Ops>::BuildCSR (const BoxSys& S)
{
    // rows in the packed order (block-major); columns ascend because the
    // block diagonals ascend and columns ascend within each block
    std::vector<int> rp(S.ntot + 1);
    std::vector<int> ci;
    std::vector<RT> v;
    ci.reserve(m_csr_colind.empty() ? std::size_t(S.ntot)*40 : m_csr_colind.size());
    v.reserve(ci.capacity());
    long row = 0;
    for (int b = 0; b < S.nb; b++) {
        for (int i = 0; i < S.bsz[b]; i++, row++) {
            rp[row] = int(ci.size());
            for (int d = 0; d <= 2*S.bw; d++) {
                const int bc = b + d - S.bw;
                if (bc < 0 || bc >= S.nb) { continue; }
                const auto& A = S.blk[b][d];
                if (A.empty()) { continue; }
                const int ncol = S.bsz[bc];
                const RT* ai = &A[std::size_t(i)*ncol];
                const long cbase = S.boff[bc];
                for (int j = 0; j < ncol; j++) {
                    if (ai[j] != RT(0.0)) {
                        ci.push_back(int(cbase + j));
                        v.push_back(ai[j]);
                    }
                }
            }
        }
    }
    rp[S.ntot] = int(ci.size());
    const bool pattern_changed =
        (rp != m_csr_rowptr) || (ci != m_csr_colind);
    m_csr_rowptr = std::move(rp);
    m_csr_colind = std::move(ci);
    m_csr_vals = std::move(v);
    return pattern_changed;
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::CuDSSFactor (BoxSys& S)
{
    BL_PROFILE("BlockBandedPC::CuDSSFactor()");
    static_assert(std::is_same_v<amrex::Real, double>,
        "the cuDSS pc_block_banded backend requires double-precision amrex::Real");
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    const bool pattern_changed = BuildCSR(S);
    const auto nrows = static_cast<std::int64_t>(S.ntot);
    const auto nnz = static_cast<std::int64_t>(m_csr_vals.size());

    if (!m_cudss_ready || pattern_changed) {
        if (m_cudss_ready) {
            WARPX_CUDSS_CHECK(cudssMatrixDestroy(m_cudss_matrix));
            WARPX_CUDSS_CHECK(cudssMatrixDestroy(m_cudss_rhs));
            WARPX_CUDSS_CHECK(cudssMatrixDestroy(m_cudss_solution));
            WARPX_CUDSS_CHECK(cudssDataDestroy(m_cudss_handle, m_cudss_data));
            WARPX_CUDSS_CHECK(cudssDataCreate(m_cudss_handle, &m_cudss_data));
        } else {
            WARPX_CUDSS_CHECK(cudssCreate(&m_cudss_handle));
            WARPX_CUDSS_CHECK(cudssSetStream(m_cudss_handle, amrex::Gpu::gpuStream()));
            WARPX_CUDSS_CHECK(cudssConfigCreate(&m_cudss_config));
            WARPX_CUDSS_CHECK(cudssDataCreate(m_cudss_handle, &m_cudss_data));
        }
        m_d_rowptr.resize(m_csr_rowptr.size());
        m_d_colind.resize(m_csr_colind.size());
        m_d_vals.resize(m_csr_vals.size());
        m_d_rhs.resize(S.ntot);
        m_d_sol.resize(S.ntot);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_csr_rowptr.begin(),
                              m_csr_rowptr.end(), m_d_rowptr.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_csr_colind.begin(),
                              m_csr_colind.end(), m_d_colind.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_csr_vals.begin(),
                              m_csr_vals.end(), m_d_vals.begin());
        amrex::Gpu::streamSynchronize();
        WARPX_CUDSS_CHECK(cudssMatrixCreateCsr(
            &m_cudss_matrix, nrows, nrows, nnz,
            m_d_rowptr.data(), nullptr, m_d_colind.data(), m_d_vals.data(),
            CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        WARPX_CUDSS_CHECK(cudssMatrixCreateDn(&m_cudss_rhs, nrows, 1, nrows,
            m_d_rhs.data(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK(cudssMatrixCreateDn(&m_cudss_solution, nrows, 1, nrows,
            m_d_sol.data(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_ANALYSIS,
            m_cudss_config, m_cudss_data, m_cudss_matrix,
            m_cudss_solution, m_cudss_rhs));
        m_cudss_ready = true;
        if (m_verbose) {
            amrex::Print() << "pc_block_banded cuDSS: analysis done (n=" << nrows
                << ", nnz=" << nnz << ")\n";
        }
    } else {
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_csr_vals.begin(),
                              m_csr_vals.end(), m_d_vals.begin());
        amrex::Gpu::streamSynchronize();
        WARPX_CUDSS_CHECK(cudssMatrixSetValues(m_cudss_matrix, m_d_vals.data()));
    }
    WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_FACTORIZATION,
        m_cudss_config, m_cudss_data, m_cudss_matrix,
        m_cudss_solution, m_cudss_rhs));
    amrex::Gpu::streamSynchronize();
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::CuDSSSolve (BoxSys& S)
{
    BL_PROFILE("BlockBandedPC::CuDSSSolve()");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_cudss_ready,
            "pc_block_banded: CuDSSSolve() called before CuDSSFactor()");
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, S.x.begin(), S.x.end(),
                              m_d_rhs.begin());
        WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_SOLVE,
            m_cudss_config, m_cudss_data, m_cudss_matrix,
            m_cudss_solution, m_cudss_rhs));
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, m_d_sol.begin(),
                              m_d_sol.end(), S.x.begin());
        amrex::Gpu::streamSynchronize();
    }
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        amrex::ParallelDescriptor::Bcast(S.x.data(), std::size_t(S.ntot),
            amrex::ParallelDescriptor::IOProcessorNumber());
    }
}

#endif  // WARPX_DIM_RZ && WARPX_USE_CUDSS

template <class T, class Ops>
void BlockBandedPC<T,Ops>::Update (const T& a_U)
{
    BL_PROFILE("BlockBandedPC::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "BlockBandedPC::Update() called on undefined object" );

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(a_U);
#else
    // rebuild cadence: interval 0 = every Newton iteration; N >= 1 = the
    // first Update() of every N-th time step
    bool rebuild = (m_update_interval == 0) || !m_have_built;
    if (!rebuild && this->m_time != m_last_build_time) {
        m_steps_since_build++;
        m_last_build_time = this->m_time;   // count each new step once
        if (m_steps_since_build >= m_update_interval) { rebuild = true; }
    }
    if (!rebuild) { return; }

    if (!m_work_defined) { BuildSystems(a_U); }

    const auto t0 = amrex::second();
    GatherCoefficients(a_U);
    for (std::size_t s = 0; s < m_sys.size(); s++) {
        ProbeAndFill(m_sys[s]);
        SelfCheck(m_sys[s], int(s));
        if (m_form_curlcurl) {
            // second pass: the unfactored A blocks for the matvec leg of
            // P^{-1} = (A - N)^{-1} A
            ProbeAndFill(m_sys[s], true);
            SelfCheck(m_sys[s], int(s), true);
        }
    }
    const auto t1 = amrex::second();
    // under verify, keep pre-factor copies for an LU round-trip check
    std::vector<std::vector<std::vector<std::vector<RT>>>> blk_save;
    if (m_verify) {
        blk_save.reserve(m_sys.size());
        for (auto& S : m_sys) { blk_save.push_back(S.blk); }
    }
    for (auto& S : m_sys) {
#ifdef WARPX_USE_CUDSS
        if (UseDeviceSolve()) { CuDSSFactor(S); continue; }
#endif
        FactorBox(S);
    }
    const auto t2 = amrex::second();
    if (m_verify) {
        for (std::size_t s = 0; s < m_sys.size(); s++) {
            BoxSys& S = m_sys[s];
            std::vector<RT> r(S.ntot);
            for (long i = 0; i < S.ntot; i++) {
                unsigned int h = static_cast<unsigned int>(i*2654435761u) + 12345u;
                h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
                r[i] = RT(h % 20011u)/RT(20011.0) - RT(0.5);
            }
            std::copy(r.begin(), r.end(), S.x.begin());
            SolveDispatch(S);
            std::vector<RT> w;
            ApplyBlocks(S, blk_save[s], S.x, w);
            RT num = RT(0.0), den = RT(0.0);
            for (long i = 0; i < S.ntot; i++) {
                const RT d = w[i] - r[i];
                num += d*d; den += r[i]*r[i];
            }
            amrex::AllPrint() << "pc_block_banded: box " << s
                << " LU round-trip rel err = " << std::sqrt(num/den) << "\n";
        }
    }

    m_have_built = true;
    m_last_build_time = this->m_time;
    m_steps_since_build = 0;

    if (m_verbose) {
        amrex::Print() << "pc_block_banded: rebuild (probe+extract "
            << (t1-t0) << " s, factor " << (t2-t1) << " s)\n";
    }

    if (m_verify) { VerifyFD(a_U); }

    // end-to-end Apply round-trip (global mode): r -> x = P^-1 r via the
    // REAL Apply (gather/pack/solve/unpack through the T vectors), then
    // A*pack(x) must reproduce pack(r). Exercises exactly the paths the
    // FD gate cannot (multi-fab gather and owned-box write-back).
    if (m_verify && m_global_on) {
        T r, x;
        r.Define(a_U); x.Define(a_U);
        r.zero(); x.zero();
        auto& r_vec = r.getArrayVec();
        const auto& geomv = m_ops->GetGeometry(0);
        const int zdlo = geomv.Domain().smallEnd(1);
        const int zdn = geomv.Domain().length(1);
        const int zwrapv = geomv.isPeriodic(1) ? 1 : 0;
        for (int dim = 0; dim < 3; dim++) {
            amrex::MultiFab& mf = *(r_vec[0][dim]);
            for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
                auto arr = mf.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    int jw = j;
                    if (zwrapv) { jw = zdlo + (((j - zdlo) % zdn) + zdn) % zdn; }
                    unsigned int h = static_cast<unsigned int>(i*73856093)
                                   ^ static_cast<unsigned int>(jw*19349663)
                                   ^ static_cast<unsigned int>(dim*83492791);
                    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
                    arr(i,j,k) = amrex::Real(h % 20011u)/amrex::Real(20011.0)
                               - amrex::Real(0.5);
                });
            }
        }
        Apply(x, r);
        BoxSys& S = m_sys[0];
        HostFab hEr, hEt, hEz, hRr, hRt, hRz;
        auto& x_vec2 = x.getArrayVec();
        GatherGlobal(*(x_vec2[0][0]), 0, hEr, 1);
        GatherGlobal(*(x_vec2[0][1]), 0, hEt, 1);
        GatherGlobal(*(x_vec2[0][2]), 0, hEz, 1);
        GatherGlobal(*(r_vec[0][0]), 0, hRr, 1);
        GatherGlobal(*(r_vec[0][1]), 0, hRt, 1);
        GatherGlobal(*(r_vec[0][2]), 0, hRz, 1);
        auto fill_in = [&](const HostFab& fEr, const HostFab& fEt,
                           const HostFab& fEz)
        {
            S.inEr.setzero(); S.inEt.setzero(); S.inEz.setzero();
            for (int i = S.inEr.ilo; i <= S.inEr.ihi; i++) {
                for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) { S.inEr.ref(i,j) = fEr.get(i,j); }
            }
            for (int i = S.inEt.ilo; i <= S.inEt.ihi; i++) {
                for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) { S.inEt.ref(i,j) = fEt.get(i,j); }
            }
            for (int i = S.inEz.ilo; i <= S.inEz.ihi; i++) {
                for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) { S.inEz.ref(i,j) = fEz.get(i,j); }
            }
        };
        // curlcurl mode: x = (A-N)^{-1} A r, so the round-trip reference is
        // A r (captured by an A-only apply); e_form reference is r itself
        std::vector<RT> refR, refT, refZ;
        if (m_form_curlcurl) {
            fill_in(hRr, hRt, hRz);
            ApplyFrozenBox(S, true);
            auto save = [&](const PArr& out, std::vector<RT>& dst) {
                dst.assign(std::size_t(std::max(0, out.ihi-out.ilo+1))
                           *std::size_t(std::max(out.nj,0)), RT(0.0));
                for (int i = out.ilo; i <= out.ihi; i++) {
                    for (int j = out.jlo; j <= out.jhi; j++) {
                        dst[std::size_t(i-out.ilo)*out.nj + out.jmap(j)]
                            = out.get(i,j);
                    }
                }
            };
            save(S.oEr, refR); save(S.oEt, refT); save(S.oEz, refZ);
        }
        fill_in(hEr, hEt, hEz);
        ApplyFrozenBox(S);
        RT num = RT(0.0), den = RT(0.0), dmax = RT(0.0);
        int imax = -1, jmax = -1, cmax = -1;
        auto acc = [&](const PArr& out, const HostFab& hr,
                       const std::vector<RT>& ref, int comp) {
            for (int i = out.ilo; i <= out.ihi; i++) {
                for (int j = out.jlo; j <= out.jhi; j++) {
                    const RT rr = m_form_curlcurl
                        ? ref[std::size_t(i-out.ilo)*out.nj + out.jmap(j)]
                        : hr.get(i,j);
                    const RT d = out.get(i,j) - rr;
                    num += d*d; den += rr*rr;
                    if (std::abs(d) > dmax) { dmax = std::abs(d); imax = i; jmax = j; cmax = comp; }
                }
            }
        };
        acc(S.oEr, hRr, refR, 0); acc(S.oEt, hRt, refT, 1);
        acc(S.oEz, hRz, refZ, 2);
        amrex::Print() << "pc_block_banded APPLY round-trip rel err = "
            << (den > RT(0.0) ? std::sqrt(num/den) : std::sqrt(num))
            << ", max |diff| = " << dmax << " at comp " << cmax
            << " (r,z) = (" << imax << "," << jmax << ")\n";
    }
#endif
}

template <class T, class Ops>
void BlockBandedPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("BlockBandedPC::Apply()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "BlockBandedPC::Apply() called on undefined object" );

#if !defined(WARPX_DIM_RZ)
    a_x.Copy(a_b);
#else
    if (!m_have_built) { a_x.Copy(a_b); return; }

    auto& b_vec = a_b.getArrayVec();
    auto& x_vec = a_x.getArrayVec();

    if (m_global_on) {
        BoxSys& S = m_sys[0];
        HostFab hEr, hEt, hEz;
        GatherGlobal(*(b_vec[0][0]), 0, hEr, 1);
        GatherGlobal(*(b_vec[0][1]), 0, hEt, 1);
        GatherGlobal(*(b_vec[0][2]), 0, hEz, 1);
        for (int b = 0; b < S.nb; b++) {
            const int i = S.rlo_s + b;
            if (blockHasEr(S,b)) {
                for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) {
                    S.x[S.boff[b] + locEr(S,b,j)] = hEr.get(i,j);
                }
            }
            for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) {
                S.x[S.boff[b] + locEt(S,b,j)] = hEt.get(i,j);
            }
            for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) {
                S.x[S.boff[b] + locEz(S,b,j)] = hEz.get(i,j);
            }
        }
        if (m_form_curlcurl) {
            // P^{-1} r = (A - N)^{-1} (A r): banded matvec ahead of the
            // factored solve
            std::vector<RT> wv;
            ApplyBlocks(S, S.blkA, S.x, wv);
            std::copy(wv.begin(), wv.end(), S.x.begin());
        }
        SolveDispatch(S);
        // Every rank holds the identical global solution; write owned boxes.
        // The (b_local - b_gathered) pass-through keeps P nonsingular on the
        // duplicated-seam-DOF difference subspace: the true Jacobian acts as
        // ~identity there (the state set syncs duplicates before the fields
        // advance), and a P that collapses all copies onto the owner value
        // makes J*P^-1 singular on that subspace -- measured as a hard GMRES
        // stall (0.1-0.2 inner residual) on any multi-box layout. The term
        // is exactly zero at owner points (bitwise-copied), so single-box
        // behavior is unchanged.
        for (int dim = 0; dim < 3; dim++) {
            amrex::MultiFab& mf = *(x_vec[0][dim]);
            const amrex::MultiFab& bmf = *(b_vec[0][dim]);
            const HostFab& hg = (dim==0) ? hEr : ((dim==1) ? hEt : hEz);
            for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
                const amrex::Box tb = amrex::convert(mfi.validbox(), mf.ixType());
                const int zPeriod = S.nzc;
                HostFab hb;
                CopyFabToHost(bmf, mfi, hb);
                auto fill = [&](auto& arr, int i, int j, int k) {
                    int jj = j;
                    if (S.zwrap) {
                        jj = S.zlo + ((j - S.zlo) % zPeriod + zPeriod) % zPeriod;
                    }
                    const int b = i - S.rlo_s;
                    int l;
                    if (dim == 0) { l = locEr(S,b,jj); }
                    else if (dim == 1) { l = locEt(S,b,jj); }
                    else { l = locEz(S,b,jj); }
                    arr(i,j,k) = S.x[S.boff[b] + l]
                               + (hb.get(i,j) - hg.get(i,j));
                };
#ifdef AMREX_USE_GPU
                amrex::FArrayBox hout(tb, 1, amrex::The_Pinned_Arena());
                auto ha = hout.array();
                amrex::LoopOnCpu(tb, [&](int i, int j, int k) { fill(ha, i, j, k); });
                amrex::FArrayBox& dfab = mf[mfi];
                amrex::Gpu::htod_memcpy(dfab.dataPtr(), hout.dataPtr(),
                                        hout.size()*sizeof(amrex::Real));
#else
                auto darr = mf.array(mfi);
                amrex::LoopOnCpu(tb, [&](int i, int j, int k) { fill(darr, i, j, k); });
#endif
            }
        }
        return;
    }

    // Schwarz communication: stage the RHS into the overlap-padded copies
    const auto& geom0 = m_ops->GetGeometry(0);
    const auto& period0 = geom0.periodicity();
    for (int d = 0; d < 3; d++) {
        m_bpad[d].setVal(RT(0.0));
        m_bpad[d].ParallelCopy(*(b_vec[0][d]), 0, 0, 1,
            amrex::IntVect(0), m_bpad[d].nGrowVect(), period0);
        m_bpad[d].FillBoundary(period0);
    }

    const amrex::MultiFab& Ermf = *(b_vec[0][0]);
    int isys = 0;
    for (amrex::MFIter mfi(Ermf); mfi.isValid(); ++mfi, ++isys) {
        BoxSys& S = m_sys[isys];

        // pack RHS (unique z points, overlap included)
        HostFab hEr, hEt, hEz;
        CopyFabToHost(m_bpad[0], mfi, hEr);
        CopyFabToHost(m_bpad[1], mfi, hEt);
        CopyFabToHost(m_bpad[2], mfi, hEz);
        for (int b = 0; b < S.nb; b++) {
            const int i = S.rlo_s + b;
            if (blockHasEr(S,b)) {
                for (int j = S.oEr.jlo; j <= S.oEr.jhi; j++) {
                    S.x[S.boff[b] + locEr(S,b,j)] = hEr.get(i,j);
                }
            }
            for (int j = S.oEt.jlo; j <= S.oEt.jhi; j++) {
                S.x[S.boff[b] + locEt(S,b,j)] = hEt.get(i,j);
            }
            for (int j = S.oEz.jlo; j <= S.oEz.jhi; j++) {
                S.x[S.boff[b] + locEz(S,b,j)] = hEz.get(i,j);
            }
        }

        if (m_form_curlcurl) {
            std::vector<RT> wv;
            ApplyBlocks(S, S.blkA, S.x, wv);
            std::copy(wv.begin(), wv.end(), S.x.begin());
        }
        SolveDispatch(S);

        // unpack solution over the FULL valid boxes (duplicated periodic
        // z-nodes get the wrapped value)
        for (int dim = 0; dim < 3; dim++) {
            amrex::MultiFab& mf = *(x_vec[0][dim]);
            const amrex::Box tb = amrex::convert(mfi.validbox(), mf.ixType());
            const int zPeriod = S.nzc;
            auto fill = [&](auto& arr, int i, int j, int k) {
                int jj = j;
                if (S.zwrap) {
                    jj = S.zlo + ((j - S.zlo) % zPeriod + zPeriod) % zPeriod;
                }
                const int b = i - S.rlo_s;
                int l;
                if (dim == 0) { l = locEr(S,b,jj); }
                else if (dim == 1) { l = locEt(S,b,jj); }
                else { l = locEz(S,b,jj); }
                arr(i,j,k) = S.x[S.boff[b] + l];
            };
#ifdef AMREX_USE_GPU
            amrex::FArrayBox hout(tb, 1, amrex::The_Pinned_Arena());
            auto ha = hout.array();
            amrex::LoopOnCpu(tb, [&](int i, int j, int k) { fill(ha, i, j, k); });
            amrex::FArrayBox& dfab = mf[mfi];
            amrex::Gpu::htod_memcpy(dfab.dataPtr(), hout.dataPtr(),
                                    hout.size()*sizeof(amrex::Real));
#else
            auto darr = mf.array(mfi);
            amrex::LoopOnCpu(tb, [&](int i, int j, int k) { fill(darr, i, j, k); });
#endif
        }
    }

    // reconcile duplicated nodal DOFs at box seams (multi-box only)
    if (Ermf.size() > 1 || amrex::ParallelDescriptor::NProcs() > 1) {
        const auto& geom = m_ops->GetGeometry(0);
        for (int dim = 0; dim < 3; dim++) {
            x_vec[0][dim]->OverrideSync(geom.periodicity());
        }
    }
#endif
}

#endif
