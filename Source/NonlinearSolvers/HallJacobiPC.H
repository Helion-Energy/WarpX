/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef HALL_JACOBI_PC_H_
#define HALL_JACOBI_PC_H_

#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXConst.H"
#include "Preconditioner.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>

#include <cmath>

/**
 * \brief Point-block analytic Hall/whistler preconditioner for the
 *  theta-implicit hybrid (generalized Ohm's law) solver.
 *
 *  Chacon's EMHD block preconditioner (JCP 526 (2025) 113789, Appendix C)
 *  collapsed to a per-cell 3x3 right PC on the delta-E system, with the
 *  curl-curl operator replaced by its grid-scale diagonal surrogate
 *  (defect-correction Laplacian). See HALL_PC_DESIGN.md (Phase 1).
 *
 *  SIGN DERIVATION (traceable to the Ohm kernels):
 *  The hybrid residual is F(E) = E - E_ohm(B(E), J(E), rho, Pe). The Ohm
 *  kernels (HybridPICSolveE.cpp:694-711) build, nodally,
 *      enE = (J - J_i) x B,          E_ohm = enE / rho + eta*J - eta_H*Lap(J) + ...
 *  with rho the (floored) charge density e*n and J = curl(B)/mu0 the total
 *  Ampere current. During FD-Jacobian probes the ion deposits are frozen
 *  (delta J_i = 0), and the drift branch (J0 - J_i0) x delta_B / (e n0) is
 *  proportional to curl(B0) -- long-wavelength, not grid-stiff -- so the
 *  stiff (whistler) leg is delta[(J x B)] = (delta_J x B0)/(e n0).
 *  The Darwin/Faraday advance gives B(E) = B_static + curl(A^n - theta*dt*E_T),
 *  so  delta_B = -theta*dt * curl(delta_E)
 *      delta_J = curl(delta_B)/mu0 = -(theta*dt/mu0) * curlcurl(delta_E)
 *      delta_E_hall = (delta_J x B0)/(e n0)
 *                   = -(theta*dt/(mu0 e n0)) * (curlcurl delta_E) x B0.
 *  The Jacobian action is J*delta_E = delta_E - delta_E_hall, i.e.
 *      J*delta_E = delta_E + (theta*dt/(mu0 e n0)) * (curlcurl delta_E) x B0.
 *  The two minus signs (residual subtraction, and delta_B = -theta*dt curl
 *  delta_E) cancel: the code's Hall convention yields the cross-product
 *  matrix X with the PLUS sign,  X r = + r x bhat0.
 *
 *  Under the surrogate curlcurl -> kg2 * I with kg2 = sum_d 4/dx_d^2 (the
 *  largest-|k| diagonal of the periodic curl-curl symbol |k|^2 I - k k^T
 *  for the transverse row; the other rows max out lower, e.g. 4/dz^2 for
 *  E_r, so this uniformly over-damps them -- acceptable for a smoother
 *  diagonal; in RZ kg2 = 4/dr^2 + 4/dz^2, uniform-cell approximation --
 *  the RZ axis metric correction is deferred, see HALL_PC_DESIGN.md
 *  sec. 4.4), the per-cell 3x3 block is
 *      P = a*I + beta*X(bhat0),
 *      beta = theta*dt*kg2*|B0| / (mu0 * e * n0),  e*n0 = max(rho0, rho_floor).
 *
 *  Resistive/hyper-resistive diagonal legs (same delta chase):
 *    +eta*J leg:    delta(eta J) = eta*delta_J = -(theta*dt*eta/mu0) curlcurl delta_E
 *                   -> -(theta*dt*eta*kg2/mu0) delta_E; enters F with a minus
 *                   -> +theta*dt*eta*kg2/mu0 on the diagonal.
 *    -eta_H*Lap(J): delta = +(theta*dt*eta_H/mu0) Lap(curlcurl delta_E)
 *                   -> (Lap -> -kg2) -(theta*dt*eta_H*kg2^2/mu0) delta_E
 *                   -> +theta*dt*eta_H*kg2^2/mu0 on the diagonal.
 *  Hence the scalar diagonal
 *      a = 1 + MM_diag + (theta*dt*kg2/mu0) * (eta0 + etaH0*kg2),
 *  with MM_diag the mass-matrix PC diagonal when available (zero if the
 *  mass-matrix PC is off).
 *
 *  CLOSED-FORM INVERSE (Chacon Eq. C.4 generalized) and its verification:
 *  For unit bhat: X bhat = bhat x bhat = 0, and
 *      X^2 r = (r x bhat) x bhat = bhat (bhat . r) - r,  i.e. X^2 = bb^T - I,
 *  so X^3 = X(bb^T - I) = -X. Claim:
 *      P^{-1} = [ a*I - beta*X + (beta^2/a) * b b^T ] / (a^2 + beta^2).
 *  Check P * P^{-1} * (a^2 + beta^2) =
 *      (a I + beta X)(a I - beta X + (beta^2/a) b b^T)
 *    = a^2 I - a*beta X + beta^2 b b^T + a*beta X - beta^2 X^2
 *      + (beta^3/a) X b b^T
 *    = a^2 I + beta^2 b b^T - beta^2 (b b^T - I) + 0
 *    = (a^2 + beta^2) I.   QED
 *  Limits: |B0| -> 0 gives beta -> 0 and P^{-1} -> I/a (pure diagonal);
 *  as beta -> infinity, P^{-1} -> b b^T / a - X/beta: the plane
 *  PERPENDICULAR to bhat is damped like 1/beta (the stiff whistler
 *  rotation plane), while the b-parallel component keeps its 1/a
 *  diagonal response (X has a null space along b, P b = a b). The
 *  formula is finite for all finite beta; beta itself is clamped at
 *  build time against a user-disabled density floor. NOTE the
 *  coefficients ignore any Holmstrom vacuum masking of E_ohm: where
 *  that masking makes the true Jacobian rows ~identity, this PC still
 *  applies whistler damping -- harmless for correctness (right PC),
 *  possibly suboptimal for GMRES conditioning in masked bands.
 *
 *  PC APPROXIMATIONS (exactness is not required of a preconditioner):
 *  - Coefficients (a, beta, bhat) are built per E-stagger by 2-point
 *    sample-averaging B (face-centered) and rho (nodal) to each E
 *    component's stagger (ablastr::coarsen::sample::Interp, cr = 1).
 *  - The three E components live on different staggers (RZ/Yee). In the
 *    coupled 3x3 apply, the other two components' residuals are
 *    sample-averaged to the local stagger from ghost-padded copies whose
 *    out-of-domain ghosts are zero (no BC queries: Open-BC safe).
 *  - eta0 is evaluated from the resistivity parser at (rho0, |J|=0, t) and
 *    etaH0 from the hyper-resistivity parser at (rho0, |B0|); the user can
 *    override either with the constant knobs precond.hall_eta0 /
 *    precond.hall_etah0.
 *
 *  Knobs:
 *    pc_hall_jacobi.verbose      (bool, default false)
 *    precond.hall_eta0           (Real; constant resistivity override)
 *    precond.hall_etah0          (Real; constant hyper-resistivity override)
 *    precond.hall_jacobi_sweeps  (int, default 0; reserved for future
 *                                 defect-correction sweeps, currently unused)
 *  The solver theta is read from implicit_evolve.theta (default 0.5);
 *  this->m_dt holds the full time step, so theta*dt = m_theta * m_dt.
 */

template <class T, class Ops>
class HallJacobiPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        HallJacobiPC () = default;

        ~HallJacobiPC () override = default;

        HallJacobiPC(const HallJacobiPC&) = delete;
        HallJacobiPC& operator=(const HallJacobiPC&) = delete;
        HallJacobiPC(HallJacobiPC&&) noexcept = delete;
        HallJacobiPC& operator=(HallJacobiPC&&) noexcept = delete;

        void Define (const T&, Ops*) override;

        void Update (const T& a_U) override;

        /**
         * \brief Apply x = P^{-1} b per cell via the closed-form rotation
         *        inverse (see class doc).
         */
        void Apply (T&, const T&) override;

        void printParameters() const override;

        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

    protected:

        bool m_is_defined = false;
        bool m_verbose = false;

        /** Time-centering parameter of the implicit solve (implicit_evolve.theta).
         *  GetThetaForPC() returns 0 for the hybrid solver, so it is parsed here. */
        RT m_theta = 0.5;

        /** Constant-eta overrides; parser evaluation is used when unset. */
        RT m_eta0 = 0.0;
        bool m_eta0_set = false;
        RT m_etah0 = 0.0;
        bool m_etah0_set = false;

        /** Reserved: number of damped block-Jacobi defect-correction sweeps
         *  with the banded true operator (HALL_PC_DESIGN.md Phase 1).
         *  Parsed and stored; not yet used. */
        int m_sweeps = 0;

        Ops* m_ops = nullptr;

        int m_num_amr_levels = 0;

        const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* m_bcoefs = nullptr;

        bool m_work_defined = false;

        /** Per-level, per-E-stagger coefficients: comp 0 = a, comp 1 = beta,
         *  comps 2-4 = bhat. */
        amrex::Vector<amrex::Array<amrex::MultiFab,3>> m_coefs;

        /** Ghost-padded (1 ghost) copies of the RHS components for the
         *  cross-stagger averaging in Apply. WarpXSolverVec MultiFabs carry
         *  zero ghost cells; out-of-domain ghosts stay zero (no BC query). */
        amrex::Vector<amrex::Array<amrex::MultiFab,3>> m_r_ghost;

        void readParameters();

        /** Stagger of a MultiFab as {0,1} per direction (1 = nodal), with
         *  unused dimensions set to nodal -- the convention required by
         *  ablastr::coarsen::sample::Interp (see HybridPICModel::SetIndexTypes). */
        static amrex::GpuArray<int,3> Stag (const amrex::MultiFab& a_mf)
        {
            amrex::GpuArray<int,3> s = {1, 1, 1};
            for (int d = 0; d < AMREX_SPACEDIM; d++) {
                s[d] = a_mf.ixType().nodeCentered(d) ? 1 : 0;
            }
            return s;
        }

    private:

};

template <class T, class Ops>
void HallJacobiPC<T,Ops>::printParameters() const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_hall_jacobi);
    Print() << pc_name << " verbose:              " << (m_verbose?"true":"false") << "\n";
    Print() << pc_name << " theta:                " << m_theta << "\n";
    Print() << pc_name << " eta0:                 "
            << (m_eta0_set ? std::to_string(m_eta0) : std::string("(from parser)")) << "\n";
    Print() << pc_name << " etah0:                "
            << (m_etah0_set ? std::to_string(m_etah0) : std::string("(from parser)")) << "\n";
    Print() << pc_name << " sweeps (reserved):    " << m_sweeps << "\n";
}

template <class T, class Ops>
void HallJacobiPC<T,Ops>::readParameters()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_hall_jacobi));
    pp.query("verbose", m_verbose);

    const amrex::ParmParse pp_precond("precond");
    m_eta0_set = pp_precond.query("hall_eta0", m_eta0);
    m_etah0_set = pp_precond.query("hall_etah0", m_etah0);
    pp_precond.query("hall_jacobi_sweeps", m_sweeps);

    // The hybrid solver's GetThetaForPC() returns 0 (no implicit light
    // waves); read the actual time-centering directly.
    const amrex::ParmParse pp_theta("implicit_evolve");
    pp_theta.query("theta", m_theta);
}

template <class T, class Ops>
void HallJacobiPC<T,Ops>::Define ( const T& a_U,
                                   Ops* const a_ops )
{
    BL_PROFILE("HallJacobiPC::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "HallJacobiPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "HallJacobiPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "HallJacobiPC::Define() must be called with Efield_fp type");

    m_ops = a_ops;
    m_num_amr_levels = m_ops->numAMRLevels();
    m_bcoefs = m_ops->GetMassMatricesCoeff();

    readParameters();

    m_is_defined = true;
}

template <class T, class Ops>
void HallJacobiPC<T,Ops>::Update (const T& a_U)
{
    BL_PROFILE("HallJacobiPC::Update()");
    using namespace amrex;
    using ablastr::coarsen::sample::Interp;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "HallJacobiPC::Update() called on undefined object" );

    auto& u_mfarrvec = a_U.getArrayVec();

    if (!m_work_defined) {
        m_coefs.resize(m_num_amr_levels);
        m_r_ghost.resize(m_num_amr_levels);
        for (int n = 0; n < m_num_amr_levels; n++) {
            for (int dim = 0; dim < 3; dim++) {
                m_coefs[n][dim].define(
                    u_mfarrvec[n][dim]->boxArray(),
                    u_mfarrvec[n][dim]->DistributionMap(),
                    5, 0);
                m_r_ghost[n][dim].define(
                    u_mfarrvec[n][dim]->boxArray(),
                    u_mfarrvec[n][dim]->DistributionMap(),
                    1, IntVect(1));
            }
        }
        m_work_defined = true;
    }

    const HybridPICModel* hybrid = m_ops->GetHybridPICModel();

    // Frozen-coefficient parameters shared by all levels
    const RT theta_dt = m_theta * this->m_dt;
    const RT time = this->m_time;
    const RT rho_floor = (hybrid != nullptr)
        ? static_cast<RT>(hybrid->m_n_floor * PhysConst::q_e) : RT(0.0);
    const bool hall_on = (hybrid != nullptr) && hybrid->m_include_hall_term;
    const bool eta_from_parser =
        !m_eta0_set && (hybrid != nullptr) && (hybrid->m_eta_expression != "0.0");
    const bool etah_from_parser =
        !m_etah0_set && (hybrid != nullptr) && hybrid->m_include_hyper_resistivity_term;
    ParserExecutor<3> eta_exec{};
    ParserExecutor<2> etah_exec{};
    if (eta_from_parser)  { eta_exec  = hybrid->m_eta; }
    if (etah_from_parser) { etah_exec = hybrid->m_eta_h; }
    const RT eta0_const = m_eta0;
    const RT etah0_const = m_etah0;
    const bool eta0_set = m_eta0_set;
    const bool etah0_set = m_etah0_set;

    for (int n = 0; n < m_num_amr_levels; n++) {

        // Grid-scale curl-curl surrogate diagonal kg2 = sum_d 4/dx_d^2
        // (uniform-cell approximation; RZ axis metric correction deferred).
        const auto dx = m_ops->GetGeometry(n).CellSizeArray();
        RT kg2 = RT(0.0);
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            kg2 += RT(4.0)/(dx[d]*dx[d]);
        }

        const auto B_mfs = m_ops->GetBfieldThetaForPC(n);
        const MultiFab* rho_mf = m_ops->GetRhoMidForPC(n);

        const bool have_B = (B_mfs[0] != nullptr) && (B_mfs[1] != nullptr)
            && (B_mfs[2] != nullptr);
        const bool have_rho = (rho_mf != nullptr);
        const bool have_hall = hall_on && have_B && have_rho;

        // rho_fp carries two time slots; component nComp()/2 is the
        // midpoint-position deposit rho^{n+1/2} of the current iterate
        // (see ThetaImplicitHybrid::GetRhoMidForPC).
        const int rho_mid_comp = have_rho ? rho_mf->nComp()/2 : 0;

        GpuArray<int,3> rho_stag = {1, 1, 1};
        GpuArray<int,3> Bx_stag = {1, 1, 1};
        GpuArray<int,3> By_stag = {1, 1, 1};
        GpuArray<int,3> Bz_stag = {1, 1, 1};
        if (have_rho) { rho_stag = Stag(*rho_mf); }
        if (have_B) {
            Bx_stag = Stag(*B_mfs[0]);
            By_stag = Stag(*B_mfs[1]);
            Bz_stag = Stag(*B_mfs[2]);
        }
        const GpuArray<int,3> coarsen = {1, 1, 1};

        for (int dim = 0; dim < 3; dim++) {

            auto& c_mf = m_coefs[n][dim];
            const GpuArray<int,3> dst_stag = Stag(c_mf);

            const MultiFab* mm_mf = (m_bcoefs != nullptr) ? (*m_bcoefs)[n][dim] : nullptr;
            const int mm_diag_comp = (mm_mf != nullptr) ? (mm_mf->nComp() - 1)/2 : 0;

            for (MFIter mfi(c_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box bx = mfi.tilebox();
                auto c_arr = c_mf.array(mfi);

                Array4<const Real> rho_arr, Bx_arr, By_arr, Bz_arr, mm_arr;
                if (have_rho) { rho_arr = rho_mf->const_array(mfi); }
                if (have_B) {
                    Bx_arr = B_mfs[0]->const_array(mfi);
                    By_arr = B_mfs[1]->const_array(mfi);
                    Bz_arr = B_mfs[2]->const_array(mfi);
                }
                if (mm_mf != nullptr) { mm_arr = mm_mf->const_array(mfi); }
                const bool have_mm = (mm_mf != nullptr);

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    // rho^{n+1/2} averaged to this E stagger, floored
                    RT rho0 = RT(0.0);
                    if (have_rho) {
                        rho0 = Interp(rho_arr, rho_stag, dst_stag, coarsen,
                                      i, j, k, rho_mid_comp);
                    }
                    const RT rho_lim =
                        std::max(std::max(rho0, rho_floor), RT(1.0e-300));

                    // B^{n+theta} averaged to this E stagger
                    RT B0x = RT(0.0), B0y = RT(0.0), B0z = RT(0.0);
                    if (have_B) {
                        B0x = Interp(Bx_arr, Bx_stag, dst_stag, coarsen, i, j, k, 0);
                        B0y = Interp(By_arr, By_stag, dst_stag, coarsen, i, j, k, 0);
                        B0z = Interp(Bz_arr, Bz_stag, dst_stag, coarsen, i, j, k, 0);
                    }
                    const RT B0mag = std::sqrt(B0x*B0x + B0y*B0y + B0z*B0z);

                    RT beta = RT(0.0);
                    RT bhx = RT(0.0), bhy = RT(0.0), bhz = RT(0.0);
                    if (have_hall && (B0mag > RT(0.0))) {
                        // min() guards beta*beta overflow when the user
                        // disables the density floor (n_floor = 0) in cells
                        // with rho0 <= 0 (adversarial-review finding F1).
                        beta = std::min(
                            theta_dt*kg2*B0mag
                                / (static_cast<RT>(PhysConst::mu0)*rho_lim),
                            RT(1.0e30));
                        bhx = B0x/B0mag;
                        bhy = B0y/B0mag;
                        bhz = B0z/B0mag;
                    }

                    // Resistive / hyper-resistive diagonal legs. Parser
                    // arguments follow the Ohm kernels: eta(rho, |J|, t) at
                    // |J| = 0 and the raw (unfloored) rho; eta_H(rho, |B|).
                    RT eta0 = RT(0.0);
                    if (eta0_set) { eta0 = eta0_const; }
                    else if (eta_from_parser && have_rho) {
                        eta0 = static_cast<RT>(eta_exec(rho0, RT(0.0), time));
                    }
                    RT etah0 = RT(0.0);
                    if (etah0_set) { etah0 = etah0_const; }
                    else if (etah_from_parser && have_rho) {
                        etah0 = static_cast<RT>(etah_exec(rho0, B0mag));
                    }

                    const RT mm_diag = have_mm ? mm_arr(i,j,k,mm_diag_comp) : RT(0.0);

                    c_arr(i,j,k,0) = RT(1.0) + mm_diag
                        + theta_dt*kg2/static_cast<RT>(PhysConst::mu0)
                          * (eta0 + etah0*kg2);
                    c_arr(i,j,k,1) = beta;
                    c_arr(i,j,k,2) = bhx;
                    c_arr(i,j,k,3) = bhy;
                    c_arr(i,j,k,4) = bhz;
                });
            }
        }

        if (m_verbose) {
            amrex::Print() << "Updating "
                << amrex::getEnumNameString(PreconditionerType::pc_hall_jacobi)
                << ": level " << n << ", theta*dt = " << theta_dt
                << ", kg2 = " << kg2
                << ", hall data " << (have_hall ? "present" : "absent") << "\n";
        }
    }
}

template <class T, class Ops>
void HallJacobiPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("HallJacobiPC::Apply()");
    using namespace amrex;
    using ablastr::coarsen::sample::Interp;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "HallJacobiPC::Apply() called on undefined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "HallJacobiPC::Apply() - a_x must be Efield_fp type");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "HallJacobiPC::Apply() - a_b must be Efield_fp type");

    if (!m_work_defined) {
        // Update() has not run yet; act as the identity.
        a_x.Copy(a_b);
        return;
    }

    auto& b_mfarrvec = a_b.getArrayVec();
    auto& x_mfarrvec = a_x.getArrayVec();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        ((b_mfarrvec.size() == m_num_amr_levels) && (x_mfarrvec.size() == m_num_amr_levels)),
        "Error in HallJacobiPC::Apply() - mismatch in number of levels." );

    // Ghost-padded copies of the RHS for cross-stagger averaging. Ghosts
    // outside the domain stay zero: no BC information is used (Open-BC safe;
    // a truncation-of-the-average approximation at PC accuracy).
    for (int n = 0; n < m_num_amr_levels; n++) {
        const auto& geom = m_ops->GetGeometry(n);
        for (int dim = 0; dim < 3; dim++) {
            m_r_ghost[n][dim].setVal(RT(0.0));
            MultiFab::Copy(m_r_ghost[n][dim], *(b_mfarrvec[n][dim]),
                           0, 0, 1, IntVect::TheZeroVector());
            m_r_ghost[n][dim].FillBoundary(geom.periodicity());
        }
    }

    const GpuArray<int,3> coarsen = {1, 1, 1};

    for (int n = 0; n < m_num_amr_levels; n++) {

        const GpuArray<int,3> r_stag[3] = {
            Stag(m_r_ghost[n][0]), Stag(m_r_ghost[n][1]), Stag(m_r_ghost[n][2]) };

        for (int dim = 0; dim < 3; dim++) {

            auto& x_mf = *(x_mfarrvec[n][dim]);
            const auto& c_mf = m_coefs[n][dim];
            const GpuArray<int,3> dst_stag = Stag(c_mf);
            const GpuArray<int,3> s0 = r_stag[0];
            const GpuArray<int,3> s1 = r_stag[1];
            const GpuArray<int,3> s2 = r_stag[2];
            const int d = dim;

            for (MFIter mfi(x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box bx = mfi.tilebox();
                auto x_arr = x_mf.array(mfi);
                auto c_arr = c_mf.const_array(mfi);
                auto r0_arr = m_r_ghost[n][0].const_array(mfi);
                auto r1_arr = m_r_ghost[n][1].const_array(mfi);
                auto r2_arr = m_r_ghost[n][2].const_array(mfi);

                ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    // Local residual vector: own component exact, the other
                    // two sample-averaged to this stagger (PC approximation).
                    const RT rx = (d == 0) ? r0_arr(i,j,k)
                        : Interp(r0_arr, s0, dst_stag, coarsen, i, j, k, 0);
                    const RT ry = (d == 1) ? r1_arr(i,j,k)
                        : Interp(r1_arr, s1, dst_stag, coarsen, i, j, k, 0);
                    const RT rz = (d == 2) ? r2_arr(i,j,k)
                        : Interp(r2_arr, s2, dst_stag, coarsen, i, j, k, 0);

                    const RT a    = c_arr(i,j,k,0);
                    const RT beta = c_arr(i,j,k,1);
                    const RT bhx  = c_arr(i,j,k,2);
                    const RT bhy  = c_arr(i,j,k,3);
                    const RT bhz  = c_arr(i,j,k,4);

                    // X r = r x bhat (sign derived in the class doc)
                    const RT Xrx = ry*bhz - rz*bhy;
                    const RT Xry = rz*bhx - rx*bhz;
                    const RT Xrz = rx*bhy - ry*bhx;
                    const RT rdotb = rx*bhx + ry*bhy + rz*bhz;

                    // P^{-1} r = [a r - beta (X r) + (beta^2/a)(r.bhat) bhat]
                    //            / (a^2 + beta^2); write own component only.
                    const RT denom = a*a + beta*beta;
                    RT num;
                    if (d == 0) { num = a*rx - beta*Xrx + (beta*beta/a)*rdotb*bhx; }
                    else if (d == 1) { num = a*ry - beta*Xry + (beta*beta/a)*rdotb*bhy; }
                    else { num = a*rz - beta*Xrz + (beta*beta/a)*rdotb*bhz; }
                    x_arr(i,j,k) = num/denom;
                });
            }
        }
    }

    if (m_verbose) {
        const RT bnorm = a_b.norm2();
        const RT xnorm = a_x.norm2();
        amrex::Print() << "  pc_hall_jacobi: ||b|| = " << bnorm
                       << ", ||P^-1 b|| = " << xnorm << "\n";
    }
}

#endif
