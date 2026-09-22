/* Copyright 2026 Prabhat Kumar
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 * Implementation detail included by HybridPICPC.H. See its provenance.
 */
#ifndef WARPX_HYBRID_PIC_PC_SETUP_H_
#define WARPX_HYBRID_PIC_PC_SETUP_H_

template <class T, class Ops>
void HybridPICPC<T,Ops>::Define ( const T& a_U, Ops* a_ops )
{
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined, "HybridPICPC::Define() called on defined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.IsDefined(), "HybridPICPC::Define() called with undefined solver vector");

    m_ops = a_ops;

    auto& warpx = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.get_pointer_HybridPICModel() != nullptr,
        "pc_hybrid_pic requires the hybrid-PIC (Ohm's law) field solver");
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE("pc_hybrid_pic is not implemented for 1D radial geometries");
#elif defined(WARPX_DIM_RZ)
    // RZ support is m = 0 only, with the axis at r = 0 (the residual's
    // CylindricalYee stencils and on-axis branches are mirrored exactly)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::n_rz_azimuthal_modes == 1,
        "pc_hybrid_pic in RZ supports the m = 0 azimuthal mode only");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.Geom(0).ProbLo(0) == 0.0,
        "pc_hybrid_pic in RZ requires the axis (r = 0) in the domain");
#endif
    m_staggered =
        (WarpX::grid_type != ablastr::utils::enums::GridType::Collocated);
    if (m_staggered) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type == ablastr::utils::enums::GridType::Staggered,
            "pc_hybrid_pic supports the collocated and staggered (Yee) "
            "grids only");
    }

    // Physical-BC closure: detect PEC(_Insulator) faces so the PC's operator
    // reproduces the residual's boundary rows (identity wall-tangential-E
    // rows with dead columns; parity-mirrored ghosts for the delta-B /
    // delta-J stages). Other non-periodic BCs keep the zero-extension
    // closure (warned once).
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    m_dim2comp[0] = 0; m_dim2comp[1] = 2;
#elif defined(WARPX_DIM_1D_Z)
    m_dim2comp[0] = 2;
#endif
#if defined(WARPX_DIM_RZ)
    m_has_axis = true;
#endif
    {
        const auto& fbc_lo = a_ops->GetFieldBoundaryLo();
        const auto& fbc_hi = a_ops->GetFieldBoundaryHi();
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            if (warpx.Geom(0).isPeriodic(d)) { continue; }
            for (int side = 0; side < 2; side++) {
                const FieldBoundaryType bt =
                    (side == 0) ? fbc_lo[d] : fbc_hi[d];
                if (bt == FieldBoundaryType::PEC
                    || bt == FieldBoundaryType::PEC_Insulator) {
                    m_wall_pec[d][side] = true;
                    m_has_pec_wall = true;
                } else {
                    m_wall_open[d][side] = true;
                    ablastr::warn_manager::WMRecordWarning("HybridPICPC",
                        std::string("pc_hybrid_pic: non-periodic, non-PEC "
                        "field boundary detected; the preconditioner uses ")
                        + (m_staggered
                           ? "a homogeneous zero-extension ghost closure with "
                             "exact identity rows on the unmodeled hi-side "
                             "wall planes on that face (approximate boundary "
                             "rows)."
                           : "a homogeneous zero-extension closure on that "
                             "face (approximate boundary rows)."));
                }
            }
        }
    }

    const amrex::ParmParse pp("pc_hybrid_pic");
    pp.query("verbose", m_verbose);
    pp.query("mg_floor", m_mg_floor);
    pp.query("inner_max", m_inner_max);
    pp.query("vcycles", m_vcycles);
    pp.query("sigma", m_sigma);
    pp.query("pe_row_sweeps", m_pe_row_sweeps);
    {
        amrex::Real swm = static_cast<amrex::Real>(m_pair_sw_min);
        pp.query("pair_sw_min", swm);
        m_pair_sw_min = static_cast<RT>(swm);
        amrex::Real irtol = static_cast<amrex::Real>(m_inner_rtol);
        pp.query("inner_rtol", irtol);
        m_inner_rtol = static_cast<RT>(irtol);
        amrex::Real sgw = static_cast<amrex::Real>(m_sigma_w);
        pp.query("sigma_w", sgw);
        m_sigma_w = static_cast<RT>(sgw);
        amrex::Real wdf = static_cast<amrex::Real>(m_pair_wdf);
        pp.query("pair_whistler_defect", wdf);
        m_pair_wdf = static_cast<RT>(wdf);
        // mirror the residual's marker-CFL cap factor (ThetaImplicitHybrid)
        const amrex::ParmParse pp_impl("implicit_evolve");
        pp_impl.query("pe_ue_cap_fac", m_pe_ue_cap_fac);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_sigma > 0.0 && m_sigma < 2.0,
        "pc_hybrid_pic.sigma must be in (0, 2)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_sigma_w > RT(0.0) && m_sigma_w < RT(2.0),
        "pc_hybrid_pic.sigma_w must be in (0, 2)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pair_wdf > RT(0.0) && m_pair_wdf <= RT(1.0),
        "pc_hybrid_pic.pair_whistler_defect must be in (0, 1]");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_inner_max >= 0 && m_inner_max + 1 <= kMaxKry,
        "pc_hybrid_pic.inner_max must be in [0, 63] (Krylov coefficients are "
        "passed to the block kernels by value)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_inner_rtol > RT(0.0) && m_vcycles > 0,
        "pc_hybrid_pic.inner_rtol and vcycles must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pair_sw_min >= RT(0.0) && m_pair_sw_min <= RT(1.0),
        "pc_hybrid_pic.pair_sw_min must be in [0, 1]");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pe_row_sweeps >= 0
                                     && !(m_pe_row_sweeps > 0 && m_staggered),
        "pc_hybrid_pic.pe_row_sweeps must be >= 0 (collocated grid only)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mg_floor >= 4,
        "pc_hybrid_pic.mg_floor must be >= 4");
    if (m_inner_max > 0) {
        // The inner GMRES stops at an RHS-dependent iteration count, so the
        // preconditioner is nonlinear: classical (fixed-M) right-PC GMRES is
        // silently invalid. Enforce a flexible outer solver; the KSP type
        // itself is checked after KSPSetFromOptions in WarpX_PETSc.cpp.
        std::string nlsolver = "newton";
        const amrex::ParmParse pp_ie("implicit_evolve");
        pp_ie.query("nonlinear_solver", nlsolver);
        if (nlsolver == "newton") {
            std::string linsolver = "amrex_gmres";
            const amrex::ParmParse pp_newton("newton");
            pp_newton.query("linear_solver", linsolver);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(linsolver == "petsc_ksp" || linsolver == "amrex_fgmres",
                "pc_hybrid_pic with an inner Krylov solve (inner_max > 0) is a "
                "nonlinear preconditioner and requires a flexible outer Krylov "
                "solver: set newton.linear_solver = amrex_fgmres, or petsc_ksp and "
                "PETSC_OPTIONS=\"-ksp_type fgmres\", or set inner_max = 0");
        }
    }
    // Physical boundaries (PEC walls, open zero-extension faces, the RZ
    // axis) are modeled by the closures of the collocated path: the E-half
    // is EField-classed at every fill, the RHS b~ is closured before its
    // copy into the E-half, and the output passes through
    // OverrideIdentityRows. The staggered node-sandwich staging and output
    // exchange assume periodic ghosts.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_staggered || warpx.Geom(0).isAllPeriodic(),
        "pc_hybrid_pic on the staggered (Yee) grid requires fully periodic "
        "BCs (physical boundaries and RZ: collocated grid only)");

    // Storage on the solver-vector layout (single AMR level); the MG
    // hierarchy coarsens that grid while every box stays coarsenable,
    // keeping the DistributionMapping so transfers are local.
    const amrex::MultiFab* E0 = a_U.getArrayVec()[0][0];
    const amrex::DistributionMapping& dm = E0->DistributionMap();

    // Container layout: nodal on the collocated grid (the solver vector's
    // own staggering); CELL-indexed on the staggered grid (see class docs)
    amrex::Vector<amrex::BoxArray> ba_node;
    ba_node.push_back(amrex::convert(E0->boxArray(),
        m_staggered ? amrex::IntVect::TheCellVector()
                    : amrex::IntVect::TheNodeVector()));
    m_geom.clear();
    m_geom.push_back(warpx.Geom(0));
    {
        amrex::BoxArray ba_cell = amrex::convert(ba_node[0], amrex::IntVect::TheCellVector());
        while (true) {
            const amrex::Geometry& fg = m_geom.back();
            // min box width mg_floor >= 4: periodic exchanges of kNGrow
            // ghosts must not wrap more than one period
            if (!fg.Domain().coarsenable(2, m_mg_floor)
                || !ba_cell.coarsenable(2, m_mg_floor)) { break; }
            ba_cell.coarsen(2);
            ba_node.push_back(amrex::convert(ba_cell, ba_node[0].ixType()));
            amrex::Array<int,AMREX_SPACEDIM> is_per{};
            for (int d = 0; d < AMREX_SPACEDIM; d++) { is_per[d] = fg.isPeriodic(d); }
            m_geom.emplace_back(amrex::coarsen(fg.Domain(), 2), fg.ProbDomain(),
                                static_cast<int>(fg.Coord()), is_per);
        }
        if (m_geom.size() == 1) {
            amrex::Print() << "pc_hybrid_pic: grid is not coarsenable; each V-cycle is "
                           << kNuSingle << " smoothing sweeps\n";
        }
    }
    m_nlevels = static_cast<int>(m_geom.size());

    // single-kernel ghost fill eligibility (FillGhosts): one box spanning
    // the whole (index-type-converted) domain on that level
    m_fastfill_ok.assign(m_nlevels, false);
    for (int lev = 0; lev < m_nlevels; lev++) {
        const amrex::BoxArray& ba = ba_node[lev];
        const amrex::Box dom_t = amrex::convert(m_geom[lev].Domain(), ba.ixType());
        m_fastfill_ok[lev] = (ba.size() == 1) && (ba[0] == dom_t);
    }
    m_zero_free = !m_staggered;   // EB (set in Update) checked at use

#if defined(WARPX_DIM_3D)
    const int dirs[3] = {0, 1, 2};
    constexpr int ndirs = 3;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const int dirs[2] = {0, 2};
    constexpr int ndirs = 2;
#else // WARPX_DIM_1D_Z
    const int dirs[1] = {2};
    constexpr int ndirs = 1;
#endif
    m_l2.assign(m_nlevels, 0.0);
    m_l1.assign(m_nlevels, 0.0);
    m_grid.assign(m_nlevels, HybridPICPCGrid{});
    for (int lev = 0; lev < m_nlevels; lev++) {
        const auto* dx = m_geom[lev].CellSize();
        for (int d = 0; d < ndirs; d++) {
            const amrex::Real h = dx[d];
            m_l1[lev] += 2.0_rt / h;
            m_grid[lev].inv2dx[dirs[d]] = 0.5_rt / h;
            m_grid[lev].invdx2[dirs[d]] = 1.0_rt / (h*h);
#if defined(WARPX_DIM_RZ)
            // the RZ curl-curl is the compact down/up composition, whose
            // Nyquist symbol reaches 4/h^2 per direction (the Cartesian
            // wide central composition peaks at half that): size the
            // defect diagonal accordingly so the sigma < 2 damping bound
            // keeps its Cartesian semantics. The same applies to the
            // staggered (Yee) composition in Cartesian geometry.
            m_l2[lev] += 4.0_rt / (h*h);
#else
            m_l2[lev] += (m_staggered ? 4.0_rt : 2.0_rt) / (h*h);
#endif
        }
#if defined(WARPX_DIM_RZ)
        m_grid[lev].rmin = static_cast<amrex::Real>(m_geom[lev].ProbLo(0));
        m_grid[lev].dr = dx[0];
        m_grid[lev].inv_dr = 1.0_rt / dx[0];
        m_grid[lev].inv_dz = 1.0_rt / dx[1];
#endif
    }

    m_coef.resize(m_nlevels);
    m_xW.resize(m_nlevels);
    m_bW.resize(m_nlevels);
    m_rW.resize(m_nlevels);
    m_tmp.resize(m_nlevels);
    m_xF.resize(m_nlevels);
    m_bF.resize(m_nlevels);
    m_rF.resize(m_nlevels);
    if (m_staggered) {
        m_tmp2.resize(m_nlevels);
    } else {
        m_xW2.resize(m_nlevels);
        m_xF2.resize(m_nlevels);
    }

    for (int lev = 0; lev < m_nlevels; lev++) {
        const amrex::BoxArray& ba = ba_node[lev];
        // kNGrow coefficient ghost layers: the staged F = curl xE is
        // evaluated in a 2-ring around the valid region, and the staggered
        // up-averages read the coefficients one ring beyond that; internal
        // ghosts are exchanged in Update(), physical ghosts follow the
        // closures
        m_coef[lev] = std::make_unique<MF>(ba, dm, kNCoef, kNGrow);
        m_coef[lev]->setVal(0.0);
        // Domain (non-periodic) ghosts stay zero: homogeneous Dirichlet
        // closure on every level (exchanges fill periodic/internal only)
        auto alloc = [&] (std::vector<std::unique_ptr<MF>>& a_v, int a_ngrow) {
            a_v[lev] = std::make_unique<MF>(ba, dm, 3, a_ngrow);
            a_v[lev]->setVal(0.0);
        };
        alloc(m_xW, kNGrow);
        alloc(m_xF, kNGrow);
        alloc(m_tmp, kNGrow);
        alloc(m_rW, 1);
        alloc(m_rF, 1);
        alloc(m_bW, 0);
        alloc(m_bF, 0);
        if (m_staggered) {
            alloc(m_tmp2, kNGrow);
        } else {
            // ping-pong partners of the iterates (see m_xW2)
            alloc(m_xW2, kNGrow);
            alloc(m_xF2, kNGrow);
        }
    }
    m_bE = std::make_unique<MF>(ba_node[0], dm, 3, 2);
    m_bE->setVal(0.0);
    m_pe_row_scr = std::make_unique<MF>(ba_node[0], dm, 1, 1);
    m_pe_row_scr->setVal(0.0);
    if (m_pe_row_sweeps > 0) {
        m_pe_row_scr2 = std::make_unique<MF>(ba_node[0], dm, 1, 1);
        m_pe_row_scr2->setVal(0.0);
    }

    if (m_inner_max > 0) {
        // block Krylov storage (see m_kryB)
        m_kryB = std::make_unique<MF>(ba_node[0], dm, (m_inner_max + 1) * kNKB, 0);
        m_bKB  = std::make_unique<MF>(ba_node[0], dm, kNKB, 0);
        m_wB   = std::make_unique<MF>(ba_node[0], dm, kNKB, 0);
        m_zB   = std::make_unique<MF>(ba_node[0], dm, kNKB, kNGrow);
        m_kryB->setVal(0.0);
        m_bKB->setVal(0.0);
        m_wB->setVal(0.0);
        m_zB->setVal(0.0);
        // batched-dot launch layout: ~2 cells per thread, at most 512
        // blocks per box; partial sums are slot-major over all local boxes
        m_dot_nblocks.clear();
        m_dot_box_offset.clear();
        m_dot_nblocks_total = 0;
        for (amrex::MFIter mfi(*m_wB, false); mfi.isValid(); ++mfi) {
            const amrex::Long npts = mfi.validbox().numPts();
            int nb = static_cast<int>((npts + 2*kDotThreads - 1) / (2*kDotThreads));
            nb = std::max(1, std::min(nb, 512));
            m_dot_box_offset.push_back(m_dot_nblocks_total);
            m_dot_nblocks.push_back(nb);
            m_dot_nblocks_total += nb;
        }
        const std::size_t ndot = static_cast<std::size_t>(m_inner_max + 2)
            * static_cast<std::size_t>(std::max(m_dot_nblocks_total, 1));
        m_dot_dev.resize(ndot);
        m_dot_host.resize(ndot);
    }

    m_is_defined = true;
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::printParameters () const
{
    using namespace amrex;
    Print() << "Hybrid-PIC EMHD preconditioner (pc_hybrid_pic):\n";
    Print() << "  grid: " << (m_staggered ? "staggered (Yee)" : "collocated")
            << "\n";
    Print() << "  formulation: primal pair (xE, W), joint block Jacobi "
            << "(convection exact)\n";
    if (m_inner_max > 0) {
        Print() << "  solve: MG V(" << kNuPre << "," << kNuPost << ") (" << kNuBottom
                << " bottom sweeps, " << m_nlevels << " level(s)) inside GMRES(<= "
                << m_inner_max << " its, rtol " << m_inner_rtol
                << " in the s_W-weighted block norm); requires a flexible outer "
                << "solver\n";
    } else {
        Print() << "  solve: " << m_vcycles << " MG V(" << kNuPre << "," << kNuPost
                << ") cycle(s) per application (" << kNuBottom << " bottom sweeps, "
                << m_nlevels << " level(s)); a fixed linear operator\n";
    }
    for (int lev = 0; lev < m_nlevels; lev++) {
        Print() << "    level " << lev << ": " << m_geom[lev].Domain()
                << ", diag l2 = " << m_l2[lev] << "\n";
    }
    Print() << "  damping: " << m_sigma << " (E rows), " << m_sigma_w
            << " (W rows)\n";
    Print() << "  block defects: hyper-resistive Nyquist 2*l2, whistler "
            << m_pair_wdf << "*l2 (pair_whistler_defect)\n";
    if (m_pair_sw_min > RT(0.0)) {
        Print() << "  W-hat scale floor: " << m_pair_sw_min << " x global (pair_sw_min)\n";
    }
    if (m_pe_row_sweeps > 0) {
        Print() << "  pe-row upwind/conduction sweeps: " << m_pe_row_sweeps << "\n";
    }
    Print() << "  ion response (deposited mass-matrix diagonal, "
            << "implicit_evolve.use_mass_matrices_pc): "
            << ((m_ops != nullptr && m_ops->GetMassMatricesCoeff() != nullptr)
                ? "true" : "false")
            << "\n";
    Print() << "  boundary closure: "
            << (m_has_pec_wall
                ? "PEC wall model (identity tangential-E rows, dead "
                  "columns, parity-mirrored delta-B/delta-J ghosts)"
                : "periodic/zero-extension")
            << (EB::enabled() ? " + EB staircase mask (eb_update_E)" : "")
            << "\n";
    Print() << "  single-kernel ghost fill: "
            << (m_fastfill_ok.empty() || !m_fastfill_ok[0] ? "off" : "on")
            << " (single domain-spanning box per level)\n";
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::Update ( const T& a_U )
{
    using namespace amrex;
    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    BL_PROFILE("HybridPICPC::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "HybridPICPC::Update() called on undefined object");
    amrex::ignore_unused(a_U);

    auto& warpx = WarpX::GetInstance();
    const HybridPICModel* hybrid = warpx.get_pointer_HybridPICModel();

    // EB mask (staircase): the nodal eb_update_E flag the residual's Ohm
    // solves consume; on the collocated grid all three components share the
    // nodal staggering, so component 0 serves as the per-node mask
    m_eb_flag = nullptr;
    if (EB::enabled()) {
        m_eb_flag = warpx.GetEBUpdateEFlag()[0][0].get();
    }

    // The solver adapter supplies total B at the linearization state,
    // including the prescribed/circuit drive. Do not add the drive twice.
    const bool add_external = false; // GetBfieldThetaForPC already contains the total drive.
    const auto B_fp = m_ops->GetBfieldThetaForPC(0);
    ablastr::fields::VectorField B_ext;
    if (add_external) {
        B_ext = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0);
    }
    const auto [rho, rho_comp] = m_ops->GetOhmDensityForPC(0);

    // pe-row leg (see m_pe_row_active): Update() cannot see whether the
    // solver vector carries the scalar row, so activate on the model gates
    // only; Apply() additionally gates on the scalar's presence, making
    // this inert for E-only vectors.
#if defined(WARPX_DIM_RZ)
    m_pe_row_active = false;
#else
    m_pe_row_active = !m_staggered && m_ops->HasPressureUnknownForPC();
#endif
    const bool pe_row_on = m_pe_row_active;
    // the frozen electron velocity of the convection term
    const auto Ji_fp = m_ops->GetIonCurrentForPC(0);
    const ablastr::fields::VectorField Jp_fp =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
    const RT gamma_e = static_cast<RT>(hybrid->m_gamma);

    const RT theta_dt = m_ops->GetTheta() * this->m_dt;
    const RT t_now = warpx.gett_new(0);
    const RT rho_floor = hybrid->m_n_floor * PhysConst::q_e;
    // same C1 smooth floor as the Ohm kernels: the PC coefficients see the
    // identical rho_lim field the residual differentiates
    const RT floor_w = static_cast<RT>(hybrid->m_n_floor_smooth_width) * rho_floor;
    const RT one_over_mu0 = 1.0_rt / PhysConst::mu0;
    const auto eta = hybrid->m_eta;
    const auto eta_te = hybrid->m_eta_te;
    const bool has_te = hybrid->m_resistivity_has_Te_dependence;
    const auto* te_mf = hybrid->ResistivityTe(0);
    const auto eta_h = hybrid->m_eta_h;
    const bool include_hyper = hybrid->m_include_hyper_resistivity_term;

    // J0 = curl(B_tot)/mu0 at the linearization state, for freezing the
    // (possibly J-dependent) resistivity at |J0| rather than |J| = 0
    // (m_bE doubles as ghosted scratch here; Apply() rewrites it)
    for (int n = 0; n < 3; n++) {
        if (m_staggered) {
            CopyCompIn(*B_fp[n], *m_bE, n, false);
            if (add_external) { CopyCompIn(*B_ext[n], *m_bE, n, true); }
        } else {
            amrex::MultiFab::Copy(*m_bE, *B_fp[n], 0, n, 1, 0);
            if (add_external) {
                amrex::MultiFab::Add(*m_bE, *B_ext[n], 0, n, 1, 0);
            }
        }
    }
    FillGhosts(*m_bE, 0);
    ApplyDomainClosure(*m_bE, 0, ClosureClass::BCoef);
    ComputeCurl(*m_bE, *m_tmp[0], 0, 0, CurlKind::BtoJ);   // m_tmp[0] = mu0 J0

    // Reduced mass-matrix diagonal (raw dJ/dE, deposited/synced each Newton
    // iteration in PreLinearSolve) for the optional ion-response leg,
    // provided through implicit_evolve.use_mass_matrices_pc = 1
    const auto* mmpc = m_ops->GetMassMatricesCoeff();
    m_have_mm = (mmpc != nullptr) && !mmpc->empty();
    const bool have_mm = m_have_mm;
    const amrex::MultiFab* Sxx_mf = have_mm ? (*mmpc)[0][0] : nullptr;
    const amrex::MultiFab* Syy_mf = have_mm ? (*mmpc)[0][1] : nullptr;
    const amrex::MultiFab* Szz_mf = have_mm ? (*mmpc)[0][2] : nullptr;
    const int cxx = have_mm ? (Sxx_mf->nComp() - 1)/2 : 0;
    const int cyy = have_mm ? (Syy_mf->nComp() - 1)/2 : 0;
    const int czz = have_mm ? (Szz_mf->nComp() - 1)/2 : 0;
    if (have_mm) {
        // the deposited diagonals are read index-space (per-component E
        // staggerings on the Yee grid, nodal when collocated): the grids
        // must align cell-wise with the coefficient container
        AMREX_ALWAYS_ASSERT(
            amrex::convert(Sxx_mf->boxArray(), amrex::IntVect::TheCellVector())
                == amrex::convert(m_coef[0]->boxArray(),
                                  amrex::IntVect::TheCellVector())
            && Sxx_mf->DistributionMap() == m_coef[0]->DistributionMap());
    }
    const bool stag = m_staggered;
    const auto dxi_pc = m_geom[0].InvCellSizeArray();
#if defined(WARPX_DIM_RZ)
    amrex::ignore_unused(dxi_pc);   // the pe-row leg is Cartesian only
#endif
    const RT l2_pc = m_l2[0];
    // marker-CFL clamp of |ue| for the pe-row diagonal (see m_pe_row_active):
    // c = pe_ue_cap_fac * dx_min / (theta dt), the residual's cap; <= 0 = none
    amrex::Real dx_min_pc = m_geom[0].CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        dx_min_pc = std::min(dx_min_pc, m_geom[0].CellSize(d));
    }
    const amrex::Real ue_cap_pc =
        static_cast<amrex::Real>(m_pe_ue_cap_fac * dx_min_pc / theta_dt);
    // representative constant conductivity for the PC leg: a spatially
    // varying kappa_e(rho,Te) is sampled at the ambient reference state
    // (n0_ref, Te0) -- the PC need not track the exact coefficient field
    const RT kappa_pc = static_cast<RT>(m_ops->PressureConductivityForPC());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_coef[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& coef = m_coef[0]->array(mfi);
        Array4<const Real> const& Bx = B_fp[0]->const_array(mfi);
        Array4<const Real> const& By = B_fp[1]->const_array(mfi);
        Array4<const Real> const& Bz = B_fp[2]->const_array(mfi);
        Array4<const Real> const& r = rho->const_array(mfi);
        const Array4<const Real> te = has_te ? te_mf->const_array(mfi) : Array4<const Real>{};
        Array4<const Real> Bxe, Bye, Bze;
        if (add_external) {
            Bxe = B_ext[0]->const_array(mfi);
            Bye = B_ext[1]->const_array(mfi);
            Bze = B_ext[2]->const_array(mfi);
        }
        Array4<const Real> const& cJ = m_tmp[0]->const_array(mfi);
        Array4<const Real> Sxx, Syy, Szz;
        if (have_mm) {
            Sxx = Sxx_mf->const_array(mfi);
            Syy = Syy_mf->const_array(mfi);
            Szz = Szz_mf->const_array(mfi);
        }
        Array4<const Real> const& Jix = Ji_fp[0]->const_array(mfi);
        Array4<const Real> const& Jiy = Ji_fp[1]->const_array(mfi);
        Array4<const Real> const& Jiz = Ji_fp[2]->const_array(mfi);
        Array4<const Real> const& Jpx = Jp_fp[0]->const_array(mfi);
        Array4<const Real> const& Jpy = Jp_fp[1]->const_array(mfi);
        Array4<const Real> const& Jpz = Jp_fp[2]->const_array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // staggered: down-average each B component to the node over
                // its cell-centered dimensions (the residual's ghost-filled
                // B provides the lo-side reads; on the axis the odd
                // B_theta ghosts average to the physical zero)
                auto bnx = [&] (Array4<const Real> const& B) {
#if defined(WARPX_DIM_3D)
                    return 0.25_rt*(B(i,j-1,k-1) + B(i,j,k-1)
                                  + B(i,j-1,k) + B(i,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    return 0.5_rt*(B(i,j-1,k) + B(i,j,k));
#else
                    return 0.5_rt*(B(i-1,j,k) + B(i,j,k));
#endif
                };
                auto bny = [&] (Array4<const Real> const& B) {
#if defined(WARPX_DIM_3D)
                    return 0.25_rt*(B(i-1,j,k-1) + B(i,j,k-1)
                                  + B(i-1,j,k) + B(i,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    return 0.25_rt*(B(i-1,j-1,k) + B(i,j-1,k)
                                  + B(i-1,j,k) + B(i,j,k));
#else
                    return 0.5_rt*(B(i-1,j,k) + B(i,j,k));
#endif
                };
                auto bnz = [&] (Array4<const Real> const& B) {
#if defined(WARPX_DIM_3D)
                    return 0.25_rt*(B(i-1,j-1,k) + B(i,j-1,k)
                                  + B(i-1,j,k) + B(i,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    return 0.5_rt*(B(i-1,j,k) + B(i,j,k));
#else
                    return B(i,j,k);
#endif
                };
                Real bx_v = stag ? bnx(Bx) : Bx(i,j,k);
                Real by_v = stag ? bny(By) : By(i,j,k);
                Real bz_v = stag ? bnz(Bz) : Bz(i,j,k);
                if (add_external) {
                    bx_v += stag ? bnx(Bxe) : Bxe(i,j,k);
                    by_v += stag ? bny(Bye) : Bye(i,j,k);
                    bz_v += stag ? bnz(Bze) : Bze(i,j,k);
                }
                const Real Bmag = std::sqrt(bx_v*bx_v + by_v*by_v + bz_v*bz_v);
                const Real rho_raw = r(i,j,k,rho_comp);
                const Real rho_e = HybridSmoothFloor(rho_raw, rho_floor, floor_w);
                if (Bmag > 0.0_rt) {
                    coef(i,j,k,0) = bx_v / Bmag;
                    coef(i,j,k,1) = by_v / Bmag;
                    coef(i,j,k,2) = bz_v / Bmag;
                } else {
                    coef(i,j,k,0) = 0.0_rt;
                    coef(i,j,k,1) = 0.0_rt;
                    coef(i,j,k,2) = 0.0_rt;
                }
                coef(i,j,k,3) = theta_dt * Bmag * one_over_mu0 / rho_e;
                // Resistivity frozen at the linearization state |J0|
                // (|J| = 0 would underestimate anomalous eta(J) in current
                // sheets). The parsers see the raw (unfloored) rho, matching
                // the Ohm's-law solve: flooring would deselect vacuum
                // branches if(rho < rho_vac, ...).
                const Real J0mag = std::sqrt(cJ(i,j,k,0)*cJ(i,j,k,0)
                                           + cJ(i,j,k,1)*cJ(i,j,k,1)
                                           + cJ(i,j,k,2)*cJ(i,j,k,2)) * one_over_mu0;
                coef(i,j,k,4) = theta_dt * (has_te ? eta_te(rho_raw, J0mag, te(i,j,k), t_now)
                    : eta(rho_raw, J0mag, t_now)) * one_over_mu0;
                coef(i,j,k,5) = include_hyper ?
                    theta_dt * eta_h(rho_raw, Bmag) * one_over_mu0 : 0.0_rt;
                // ion-response strengths w_c = Sigma_cc |B_tot| / rho_c
                // (S v = (Sigma v) x B / rho_c through the Hall term; the
                // raw dJ/dE deposit already carries the theta*dt of the
                // implicit velocity integration, so w ~ Omega_ci theta*dt)
                Real w6 = 0.0_rt, w7 = 0.0_rt, w8 = 0.0_rt;
                if (have_mm) {
                    const Real bfac = Bmag / rho_e;
                    w6 = Sxx(i,j,k,cxx) * bfac;
                    w7 = Syy(i,j,k,cyy) * bfac;
                    w8 = Szz(i,j,k,czz) * bfac;
                }
                coef(i,j,k,6) = w6;
                coef(i,j,k,7) = w7;
                coef(i,j,k,8) = w8;
                coef(i,j,k,9) = rho_e;
                // pe row: pointwise diagonal D_p = 1 + coef(10), coef(10) =
                // theta*dt*[gamma*sum_d|ue_d|/dx_d + conduction]. |ue| from
                // the floored density: in floored cells the fictitious ue
                // only GROWS the diagonal, further damping the correction
                // there -- the safe direction for a PC.
                if (pe_row_on) {
                    const Real irho = 1.0_rt / rho_e;
                    // |ue_d| clamped like the residual's pe advance
                    auto uec = [=] (Real u) {
                        return (ue_cap_pc > 0.0_rt) ? ue_cap_pc * std::tanh(u / ue_cap_pc) : u;
                    };
                    Real adv = 0.0_rt;
#if defined(WARPX_DIM_1D_Z)
                    adv = uec(std::abs(Jiz(i,j,k) - Jpz(i,j,k)) * irho) * dxi_pc[0];
#elif defined(WARPX_DIM_XZ)
                    adv = uec(std::abs(Jix(i,j,k) - Jpx(i,j,k)) * irho) * dxi_pc[0]
                        + uec(std::abs(Jiz(i,j,k) - Jpz(i,j,k)) * irho) * dxi_pc[1];
#elif defined(WARPX_DIM_3D)
                    adv = uec(std::abs(Jix(i,j,k) - Jpx(i,j,k)) * irho) * dxi_pc[0]
                        + uec(std::abs(Jiy(i,j,k) - Jpy(i,j,k)) * irho) * dxi_pc[1]
                        + uec(std::abs(Jiz(i,j,k) - Jpz(i,j,k)) * irho) * dxi_pc[2];
#endif
                    // + the electron-conduction stiffness (kappa_e grad(pe/n)
                    // diffusion diagonal) so D_p tracks the pe row when
                    // conduction dominates the grid scale
                    coef(i,j,k,10) = theta_dt * (gamma_e * adv
                        + (gamma_e - 1.0_rt) * kappa_pc
                          * PhysConst::q_e * l2_pc / rho_e);
                } else {
                    coef(i,j,k,10) = 0.0_rt;
                }
                // frozen electron velocity of the convection term, stored
                // pre-scaled as theta*dt * ue0 = theta*dt (J_i - J_plasma)/rho_c.
                // staggered: each J-like component is down-averaged to the
                // node over its cell-centered dimension (the residual's
                // ghost-filled J provides the lo-side reads), matching the
                // B down-averages above
                {
                    const Real irho = theta_dt / rho_e;
                    if (stag) {
                        auto jn = [&] (Array4<const Real> const& Ja,
                                       Array4<const Real> const& Jb,
                                       int c) -> Real {
                            amrex::ignore_unused(c);
#if defined(WARPX_DIM_3D)
                            if (c == 0) {
                                return 0.5_rt*(Ja(i-1,j,k) - Jb(i-1,j,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            if (c == 1) {
                                return 0.5_rt*(Ja(i,j-1,k) - Jb(i,j-1,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            return 0.5_rt*(Ja(i,j,k-1) - Jb(i,j,k-1)
                                         + Ja(i,j,k)   - Jb(i,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                            if (c == 0) {
                                return 0.5_rt*(Ja(i-1,j,k) - Jb(i-1,j,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            if (c == 2) {
                                return 0.5_rt*(Ja(i,j-1,k) - Jb(i,j-1,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            return Ja(i,j,k) - Jb(i,j,k);
#else
                            if (c == 2) {
                                return 0.5_rt*(Ja(i-1,j,k) - Jb(i-1,j,k)
                                             + Ja(i,j,k)   - Jb(i,j,k));
                            }
                            return Ja(i,j,k) - Jb(i,j,k);
#endif
                        };
                        coef(i,j,k,11) = jn(Jix, Jpx, 0) * irho;
                        coef(i,j,k,12) = jn(Jiy, Jpy, 1) * irho;
                        coef(i,j,k,13) = jn(Jiz, Jpz, 2) * irho;
                    } else {
                        coef(i,j,k,11) = (Jix(i,j,k) - Jpx(i,j,k)) * irho;
                        coef(i,j,k,12) = (Jiy(i,j,k) - Jpy(i,j,k)) * irho;
                        coef(i,j,k,13) = (Jiz(i,j,k) - Jpz(i,j,k)) * irho;
                    }
                }
                // LOCAL W-hat scale s_W = max(beta_loc, 1)/l2 = max(c_w,
                // 1/l2), so every node's W-residual is weighted by its own
                // E-row coupling ||U_loc|| (a single global scale cannot
                // weight both the vacuum, beta ~ 1/rho_floor, and the plasma
                // of a strong density-contrast profile)
                coef(i,j,k,14) = amrex::max(coef(i,j,k,3), Real(1.0)/Real(l2_pc));
            });
    }

    // optional floor of the local W-hat scale at pair_sw_min x the global
    // scale max(beta, 1)/l2, beta = l2 max(c_w) (the peak whistler CFL);
    // before the coefficient exchange and the coarse injection, so every
    // level inherits the floored value
    if (m_pair_sw_min > RT(0.0)) {
        const RT beta = static_cast<RT>(m_l2[0] * m_coef[0]->max(3, 0));
        const amrex::Real swfl = static_cast<amrex::Real>(
            m_pair_sw_min * amrex::max(beta, RT(1.0)) / m_l2[0]);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*m_coef[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<Real> const& c = m_coef[0]->array(mfi);
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    c(i,j,k,14) = amrex::max(c(i,j,k,14), swfl);
                });
        }
    }

    // Coarse operators by rediscretization: inject the coefficients down the
    // hierarchy (l2 is recomputed per level in Define()) and exchange their
    // internal ghosts for the stencils.
    FillGhosts(*m_coef[0], 0);
    ApplyDomainClosure(*m_coef[0], 0, ClosureClass::Coef);
    for (int lev = 1; lev < m_nlevels; lev++) {
        RestrictCoef(lev);
        FillGhosts(*m_coef[lev], lev);
        ApplyDomainClosure(*m_coef[lev], lev, ClosureClass::Coef);
    }
}

#endif
