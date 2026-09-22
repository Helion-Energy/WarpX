/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_PIC_RZ_PC_H_
#define WARPX_HYBRID_PIC_RZ_PC_H_
#ifdef WARPX_DIM_RZ
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "HybridPICRZOperator.H"
#include "Preconditioner.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <ablastr/coarsen/sample.H>

/** WarpX coefficient adapter for the native RZ primal-pair multigrid PC.
 * Density derivatives, inertia advection, the longitudinal constraint and
 * the live vacuum/circuit response remain in the true outer Jacobian.
 */
template <class T, class Ops>
class HybridPICRZPC : public Preconditioner<T, Ops> {
  public:
    using RT = typename T::value_type;
    HybridPICRZOperator op;
    Ops* ops = nullptr;
    HybridPICRZOperator::Field input, output;
    int inner_max = 8, vcycles = 1;
    RT inner_rtol = RT(.3);
    bool defined = false;
    bool vacuum_rows = true;

    static amrex::GpuArray<int, 3>
    Stag (amrex::MultiFab const& a) {
        auto const iv = a.ixType().toIntVect();
        return {iv[0], iv[1], 1};
    }
    void
    Define (T const& u, Ops* o) override {
        using namespace amrex;
        using namespace amrex::literals;
        auto& w = WarpX::GetInstance();
        ops = o;
        auto const* h = ops->GetHybridPICModel();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            h != nullptr, "pc_hybrid_pic requires the hybrid field solver");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            w.finestLevel() == 0 && WarpX::n_rz_azimuthal_modes == 1 &&
                w.Geom(0).ProbLo(0) == 0. && !EB::enabled(),
            "Native staggered RZ pc_hybrid_pic requires one level, m=0, rmin=0 "
            "and no EB; Cartesian paths are separate");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !h->m_resistive_wall && !h->m_esolve_curlcurl &&
                !h->m_esolve_tensor,
            "Native staggered RZ pc_hybrid_pic does not yet model the "
            "resistive-shell, tensor_form or nested curlcurl_form rows");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !h->m_hyper_res_curl_curl && !h->m_hyper_resistivity_curlcurl,
            "Native staggered RZ pc_hybrid_pic currently models Laplacian "
            "hyper-resistivity");
        ParmParse pp("pc_hybrid_pic");
        int floor = 4;
        pp.query("vacuum_rows", vacuum_rows);
        pp.query("inner_max", inner_max);
        pp.query("inner_rtol", inner_rtol);
        pp.query("vcycles", vcycles);
        pp.query("mg_floor", floor);
        pp.query("sigma", op.sigma);
        pp.query("sigma_w", op.sigma_w);
        pp.query("pair_whistler_defect", op.whistler_defect);
        pp.query("verbose", op.verbose);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            inner_max >= 0 && inner_max <= 63 && inner_rtol > 0. &&
                inner_rtol < 1. && vcycles > 0 && floor >= 4 && op.sigma > 0. &&
                op.sigma < 2. && op.sigma_w > 0. && op.sigma_w < 2. &&
                op.whistler_defect > 0. && op.whistler_defect <= 1.,
            "Invalid pc_hybrid_pic controls");
        if (inner_max > 0) {
            std::string method = "amrex_gmres";
            ParmParse("newton").query("linear_solver", method);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                method == "amrex_fgmres",
                "Native RZ pc_hybrid_pic inner Krylov requires "
                "newton.linear_solver=amrex_fgmres; inner_max=0 uses fixed "
                "linear V-cycles");
        }
        for (auto const& side :
             {ops->GetFieldBoundaryLo(), ops->GetFieldBoundaryHi()}) {
            for (auto const bc : side) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    bc == FieldBoundaryType::None ||
                        bc == FieldBoundaryType::PEC ||
                        bc == FieldBoundaryType::PMC ||
                        bc == FieldBoundaryType::Periodic,
                    "Native RZ pc_hybrid_pic supports axis/None, PEC, "
                    "PMC/Neumann and periodic field boundaries");
            }
        }
        auto const& e = *u.getArrayVec()[0][0];
        op.Define(w.Geom(0), amrex::convert(e.boxArray(), IntVect(0, 0)),
                  e.DistributionMap(), ops->GetFieldBoundaryLo(),
                  ops->GetFieldBoundaryHi(), floor);
        op.pin_vector_potential = h->m_darwin && h->m_add_external_fields;
        for (int c = 0; c < 3; ++c) {
            auto const& f = *u.getArrayVec()[0][c];
            input[c] = std::make_unique<MultiFab>(f.boxArray(),
                                                  f.DistributionMap(), 1, 0);
            output[c] = std::make_unique<MultiFab>(f.boxArray(),
                                                   f.DistributionMap(), 1, 0);
        }
        defined = true;
    }
    [[nodiscard]] bool
    IsDefined () const override {
        return defined;
    }
    void
    printParameters () const override {
        amrex::Print()
            << "Hybrid-PIC spatial Hall PC: native staggered RZ primal pair, "
            << op.levels.size() << " MG levels, inner_max=" << inner_max
            << ", inner_rtol=" << inner_rtol
            << ", vector-potential derivative pins=" << op.pin_vector_potential
            << "\n";
    }
    void
    Update (T const&) override {
        using namespace amrex;
        using namespace amrex::literals;
        using ablastr::coarsen::sample::Interp;
        using warpx::fields::FieldType;
        BL_PROFILE("HybridPICRZPC::Update");
        auto& w = WarpX::GetInstance();
        auto const* h = ops->GetHybridPICModel();
        auto& l = *op.levels[0];
        auto const B = ops->GetBfieldThetaForPC(0);
        auto const Ji = ops->GetIonCurrentForPC(0);
        auto const Jp =
            w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        auto const density = ops->GetOhmDensityForPC(0);
        auto const* rho = density.first;
        int const rho_comp = density.second;
        auto const* mid = ops->GetRhoMidForPC(0);
        int const mid_comp = mid->nComp() / 2;
        auto const* ped = h->DensityPedestal(0);
        auto const* te = h->ResistivityTe(0);
        bool const has_ped = ped != nullptr,
                   has_te = h->m_resistivity_has_Te_dependence;
        bool const hall = h->m_include_hall_term,
                   hyper = h->m_include_hyper_resistivity_term;
        bool const eta_j = h->m_resistivity_has_J_dependence,
                   hyper_b = h->m_hyper_resistivity_has_B_dependence;
        bool const has_eta = h->m_eta_expression != "0.0";
        auto const eta = h->m_eta;
        auto const eta_h = h->m_eta_h;
        auto const eta_te = h->m_eta_te;
        RT const theta_dt = ops->GetTheta() * this->m_dt, time = w.gett_new(0);
        RT const rho_floor = h->m_n_floor * PhysConst::q_e,
                 floor_width = h->m_n_floor_smooth_width * rho_floor;
        RT const taper_width = h->m_electron_inertia_floor_taper * rho_floor;
        RT const factor =
            h->m_electron_inertia_bdf2 && h->m_inertia_history_levels >= 2
                ? (2._rt * ops->GetTheta() + 1._rt) / 2._rt
                : 1._rt;
        RT const inertia = h->m_include_electron_inertia
                               ? factor * h->m_electron_inertia_mass /
                                     (PhysConst::mu0 * PhysConst::q_e)
                               : 0.;
        bool const vacuum_gate = h->m_holmstrom_vacuum_region;
        RT const gate_width = h->m_holmstrom_transition_width * rho_floor;
        RT const axis_radius = h->m_holmstrom_axis_radius,
                 axis_width = h->m_holmstrom_axis_rolloff;
        bool const conductor = h->m_pec_conductor_wall_rows;
        RT const l2 = 4._rt / (l.geom.CellSize(0) * l.geom.CellSize(0)) +
                      4._rt / (l.geom.CellSize(1) * l.geom.CellSize(1));
        RT const dr = l.geom.CellSize(0);
        // The recovered E rows no longer contain the vacuum Ohm force. Use
        // an identity approximation there; the true outer Jv still evaluates
        // the live recovery. Select exactly its component and density mask.
        bool const recovery = vacuum_rows && h->m_darwin &&
                              h->m_darwin_vacuum_recovery &&
                              h->m_darwin_vacuum_recovery_cadence == "half";
        bool const flux_only = h->m_darwin_vacuum_recovery_components == "flux";
        int const mask_mode = h->m_darwin_vacuum_recovery_mask == "global" ? 2
                              : h->m_darwin_vacuum_recovery_mask == "transition"
                                  ? 1
                                  : 0;
        RT const recovery_floor =
            rho_floor * h->m_darwin_vacuum_recovery_density_fraction;
        auto const* mask_density =
            recovery ? (h->m_darwin_vacuum_recovery_frozen_mask
                            ? w.m_fields.get("hybrid_rho_vacmask_fp", 0)
                            : w.m_fields.get(FieldType::rho_fp, 0))
                     : nullptr;
        GpuArray<int, 3> const nd = {1, 1, 1}, rr = {1, 1, 1};
        auto const brs = Stag(*B[0]), bts = Stag(*B[1]), bzs = Stag(*B[2]);
        auto const jrs = Stag(*Ji[0]), jts = Stag(*Ji[1]), jzs = Stag(*Ji[2]);
        auto const* mm = ops->GetMassMatricesCoeff();
        bool const have_mm = mm != nullptr && !mm->empty();
        for (MFIter mfi(*l.nodes); mfi.isValid(); ++mfi) {
            auto n = l.nodes->array(mfi);
            auto const br = B[0]->const_array(mfi), bt = B[1]->const_array(mfi),
                       bz = B[2]->const_array(mfi);
            auto const jr = Ji[0]->const_array(mfi),
                       jt = Ji[1]->const_array(mfi),
                       jz = Ji[2]->const_array(mfi);
            auto const pr = Jp[0]->const_array(mfi),
                       pt = Jp[1]->const_array(mfi),
                       pz = Jp[2]->const_array(mfi);
            auto const rm = mid->const_array(mfi);
            auto const p =
                has_ped ? ped->const_array(mfi) : Array4<Real const>{};
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                             int k) {
                n(i, j, k, 0) = Interp(br, brs, nd, rr, i, j, k, 0);
                n(i, j, k, 1) = Interp(bt, bts, nd, rr, i, j, k, 0);
                n(i, j, k, 2) = Interp(bz, bzs, nd, rr, i, j, k, 0);
                n(i, j, k, 3) =
                    theta_dt *
                    (Interp(jr, jrs, nd, rr, i, j, k, 0) -
                     (hall ? Interp(pr, jrs, nd, rr, i, j, k, 0) : 0.));
                n(i, j, k, 4) =
                    theta_dt *
                    (Interp(jt, jts, nd, rr, i, j, k, 0) -
                     (hall ? Interp(pt, jts, nd, rr, i, j, k, 0) : 0.));
                n(i, j, k, 5) =
                    theta_dt *
                    (Interp(jz, jzs, nd, rr, i, j, k, 0) -
                     (hall ? Interp(pz, jzs, nd, rr, i, j, k, 0) : 0.));
                RT const raw = rm(i, j, k, mid_comp),
                         den = HybridSmoothFloor(
                             raw + (has_ped ? p(i, j, k) : 0.), rho_floor,
                             floor_width);
                RT const taper =
                    taper_width > 0.
                        ? .5_rt * (1._rt +
                                   std::tanh((raw - rho_floor) / taper_width))
                        : 1._rt;
                n(i, j, k, 6) =
                    raw > 0. && den > 0. ? inertia * taper / den : 0.;
            });
        }
        for (int c = 0; c < 3; ++c) {
            auto const dest = Stag(*l.coeff[c]);
            auto const* sm = have_mm ? (*mm)[0][c] : nullptr;
            int const sm_comp = sm ? (sm->nComp() - 1) / 2 : 0;
            for (MFIter mfi(*l.coeff[c]); mfi.isValid(); ++mfi) {
                auto v = l.coeff[c]->array(mfi);
                auto const r = rho->const_array(mfi);
                auto const p =
                    has_ped ? ped->const_array(mfi) : Array4<Real const>{};
                auto const t =
                    has_te ? te->const_array(mfi) : Array4<Real const>{};
                auto const br = B[0]->const_array(mfi),
                           bt = B[1]->const_array(mfi),
                           bz = B[2]->const_array(mfi);
                auto const jr = Jp[0]->const_array(mfi),
                           jt = Jp[1]->const_array(mfi),
                           jz = Jp[2]->const_array(mfi);
                auto const mass =
                    sm ? sm->const_array(mfi) : Array4<Real const>{};
                auto const recovery_density =
                    mask_density ? mask_density->const_array(mfi)
                                 : Array4<Real const>{};
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                 int k) {
                    RT const raw = Interp(r, nd, dest, rr, i, j, k, rho_comp);
                    RT const den = HybridSmoothFloor(
                        raw + (has_ped ? Interp(p, nd, dest, rr, i, j, k, 0)
                                       : 0.),
                        rho_floor, floor_width);
                    RT const b0 = Interp(br, brs, dest, rr, i, j, k, 0),
                             b1 = Interp(bt, bts, dest, rr, i, j, k, 0),
                             b2 = Interp(bz, bzs, dest, rr, i, j, k, 0);
                    RT const bm = std::sqrt(b0 * b0 + b1 * b1 + b2 * b2);
                    RT jmag = 0.;
                    if (eta_j) {
                        RT const a = Interp(jr, jrs, dest, rr, i, j, k, 0),
                                 b = Interp(jt, jts, dest, rr, i, j, k, 0),
                                 d = Interp(jz, jzs, dest, rr, i, j, k, 0);
                        jmag = std::sqrt(a * a + b * b + d * d);
                    }
                    RT gate = 1.;
                    if (vacuum_gate) {
                        RT const radius = (i + (c == 0 ? .5_rt : 0._rt)) * dr;
                        RT const mask =
                            axis_radius <= 0. ? 1._rt
                            : gate_width > 0. && axis_width > 0.
                                ? .5_rt * (1._rt -
                                           std::tanh((radius - axis_radius) /
                                                     axis_width))
                                : (radius < axis_radius ? 1._rt : 0._rt);
                        RT const g =
                            gate_width > 0.
                                ? .5_rt * (1._rt + std::tanh((raw - rho_floor) /
                                                             gate_width))
                                : (raw >= rho_floor ? 1._rt : 0._rt);
                        gate = 1._rt - (1._rt - g) * mask;
                    }
                    if (conductor && raw <= 0.) {
                        gate = 0.;
                    }
                    RT const ir = den > 0. ? gate / den : 0.;
                    v(i, j, k, HybridPICRZOperator::Hall) =
                        hall ? theta_dt * ir / PhysConst::mu0 : 0.;
                    v(i, j, k, HybridPICRZOperator::InvRho) = ir;
                    v(i, j, k, HybridPICRZOperator::Eta) =
                        has_eta ? theta_dt / PhysConst::mu0 *
                                      (has_te ? eta_te(raw, jmag,
                                                       Interp(t, nd, dest, rr,
                                                              i, j, k, 0),
                                                       time)
                                              : eta(raw, jmag, time))
                                : 0.;
                    v(i, j, k, HybridPICRZOperator::Hyper) =
                        hyper ? theta_dt / PhysConst::mu0 *
                                    eta_h(raw, hyper_b ? bm : 0.)
                              : 0.;
                    v(i, j, k, HybridPICRZOperator::Scale) = amrex::max(
                        v(i, j, k, HybridPICRZOperator::Hall) * bm, 1._rt / l2);
                    v(i, j, k, HybridPICRZOperator::Ion) =
                        have_mm ? mass(i, j, k, sm_comp) : 0.;
                    RT weight = 1.;
                    if (recovery && (!flux_only || c == 1)) {
                        RT const rm =
                            Interp(recovery_density, nd, dest, rr, i, j, k, 0);
                        bool const replaced =
                            mask_mode == 2 ||
                            (mask_mode == 0 && rm < recovery_floor) ||
                            (mask_mode == 1 && rm > 0. && rm < recovery_floor);
                        if (replaced) {
                            weight = 0.;
                        }
                    }
                    v(i, j, k, HybridPICRZOperator::RowWeight) = weight;
                });
            }
        }
        op.ion_response = have_mm;
        op.PrepareCoefficients();
    }
    void
    Apply (T& x, T const& b) override {
        BL_PROFILE("HybridPICRZPC::Apply");
        for (int c = 0; c < 3; ++c) {
            amrex::MultiFab::Copy(*input[c], *b.getArrayVec()[0][c], 0, 0, 1,
                                  0);
        }
        op.Solve(output, input, inner_max, inner_rtol, vcycles);
        for (int c = 0; c < 3; ++c) {
            amrex::MultiFab::Copy(*x.getArrayVec()[0][c], *output[c], 0, 0, 1,
                                  0);
        }
        if (x.getScalarVecType() != warpx::fields::FieldType::None) {
            amrex::MultiFab::Copy(*x.getScalarVec()[0], *b.getScalarVec()[0], 0,
                                  0, 1, 0);
        }
    }
    bool
    ApplyFrozenOperator (T& out, T const& in) override {
        for (int c = 0; c < 3; ++c) {
            amrex::MultiFab::Copy(*input[c], *in.getArrayVec()[0][c], 0, 0, 1,
                                  0);
        }
        op.ApplyField(output, input);
        for (int c = 0; c < 3; ++c) {
            amrex::MultiFab::Copy(*out.getArrayVec()[0][c], *output[c], 0, 0, 1,
                                  0);
        }
        if (out.getScalarVecType() != warpx::fields::FieldType::None) {
            amrex::MultiFab::Copy(*out.getScalarVec()[0], *in.getScalarVec()[0],
                                  0, 0, 1, 0);
        }
        return true;
    }
};
#endif
#endif
