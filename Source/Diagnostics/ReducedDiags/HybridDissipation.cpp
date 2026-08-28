/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HybridDissipation.H"

#include "EmbeddedBoundary/Enabled.H"
#if defined(WARPX_DIM_RZ)
#   include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <array>
#include <fstream>
#include <memory>

using namespace amrex;
using warpx::fields::FieldType;

HybridDissipation::HybridDissipation (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
#if !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "the HybridDissipation reduced diagnostic is implemented for RZ "
        "geometry");
#endif

    m_data.resize(5, 0.0_rt);

    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
                          std::ofstream::out};
        int c = 0;
        ofs << "#";
        ofs << "[" << c++ << "]step()";
        ofs << m_sep << "[" << c++ << "]time(s)";
        ofs << m_sep << "[" << c++ << "]P_eta(W)";
        ofs << m_sep << "[" << c++ << "]P_etaH(W)";
        ofs << m_sep << "[" << c++ << "]P_diss(W)";
        ofs << m_sep << "[" << c++ << "]P_sigma_vac(W)";
        ofs << m_sep << "[" << c++ << "]P_sponge(W)";
        ofs << "\n";
        ofs.close();
    }
}

void
HybridDissipation::ComputeDiags (const int step)
{
    if (!DoDiags(step)) { return; }
#if defined(WARPX_DIM_RZ)
    using namespace ablastr::coarsen::sample;

    auto& warpx = WarpX::GetInstance();
    auto* hybrid = warpx.get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(hybrid != nullptr,
        "the HybridDissipation reduced diagnostic requires the hybrid "
        "(Ohm's law) solver");

    constexpr int lev = 0;
    const auto eta = hybrid->m_eta;
    const auto eta_h = hybrid->m_eta_h;
    const bool has_J_dep = hybrid->m_resistivity_has_J_dependence;
    const bool has_B_dep = hybrid->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper = hybrid->m_include_hyper_resistivity_term;
    const bool external_b_split = hybrid->m_add_external_fields &&
        (hybrid->m_external_field_mode ==
         HybridPICModel::ExternalFieldMode::Split);
    const amrex::Real t_new = warpx.gett_new(lev);

    const std::array<const amrex::MultiFab*, 3> J = {
        warpx.m_fields.get(FieldType::hybrid_current_fp_plasma,
                           ablastr::fields::Direction{0}, lev),
        warpx.m_fields.get(FieldType::hybrid_current_fp_plasma,
                           ablastr::fields::Direction{1}, lev),
        warpx.m_fields.get(FieldType::hybrid_current_fp_plasma,
                           ablastr::fields::Direction{2}, lev)};
    const std::array<const amrex::MultiFab*, 3> B = {
        warpx.m_fields.get(FieldType::Bfield_fp,
                           ablastr::fields::Direction{0}, lev),
        warpx.m_fields.get(FieldType::Bfield_fp,
                           ablastr::fields::Direction{1}, lev),
        warpx.m_fields.get(FieldType::Bfield_fp,
                           ablastr::fields::Direction{2}, lev)};
    std::array<const amrex::MultiFab*, 3> B_ext = {nullptr, nullptr, nullptr};
    if (external_b_split && has_B_dep) {
        for (int d = 0; d < 3; ++d) {
            B_ext[d] = warpx.m_fields.get(FieldType::hybrid_B_fp_external,
                                          ablastr::fields::Direction{d}, lev);
        }
    }
    const amrex::MultiFab* rho =
        warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);

    const auto& geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const amrex::Real dr = dx[0];
    const amrex::Real dz = dx[1];
    const amrex::Real rmin = geom.ProbLo(0);
    const auto dom_lo = amrex::lbound(geom.Domain());
    const auto dom_hi = amrex::ubound(geom.Domain());

    // The stencil coefficients are the inverse cell sizes (cylindrical Yee)
    std::array<amrex::Real, 3> cell_size = {dx[0], 0.0_rt, dx[1]};
    amrex::Vector<amrex::Real> hc_r, hc_z;
    CylindricalYeeAlgorithm::InitializeStencilCoefficients(cell_size,
                                                           hc_r, hc_z);
    amrex::Gpu::DeviceVector<amrex::Real> dv_r(hc_r.size()), dv_z(hc_z.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          hc_r.begin(), hc_r.end(), dv_r.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          hc_z.begin(), hc_z.end(), dv_z.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* coefs_r = dv_r.data();
    const auto n_coefs_r = static_cast<int>(dv_r.size());
    const amrex::Real* coefs_z = dv_z.data();
    const auto n_coefs_z = static_cast<int>(dv_z.size());

    const amrex::GpuArray<int, 3> nodal_iv = {1, 1, 1};
    const amrex::GpuArray<int, 3> coarsen_iv = {1, 1, 1};
    const amrex::GpuArray<int, 3> Jr_stag = hybrid->Jx_IndexType;
    const amrex::GpuArray<int, 3> Jt_stag = hybrid->Jy_IndexType;
    const amrex::GpuArray<int, 3> Jz_stag = hybrid->Jz_IndexType;
    const amrex::GpuArray<int, 3> Br_stag = hybrid->Bx_IndexType;
    const amrex::GpuArray<int, 3> Bt_stag = hybrid->By_IndexType;
    const amrex::GpuArray<int, 3> Bz_stag = hybrid->Bz_IndexType;

    // EB-frozen cells never receive the resistive terms in the solver;
    // mirror the same skip so frozen regions book zero dissipation.
    const bool eb_enabled = EB::enabled();
    auto& eb_update_E = warpx.GetEBUpdateEFlag();

    amrex::Real p_eta = 0.0_rt;
    amrex::Real p_eta_h = 0.0_rt;

    for (int d = 0; d < 3; ++d) {
        const amrex::GpuArray<int, 3> self_stag =
            (d == 0) ? Jr_stag : ((d == 1) ? Jt_stag : Jz_stag);
        const bool nodal_r = (self_stag[0] == 1);
        // owner mask so shared nodal points count once across boxes
        const auto mask = amrex::OwnerMask(*J[d], geom.periodicity());

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (amrex::MFIter mfi(*J[d], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto Jself = J[d]->const_array(mfi);
            const auto Jr = J[0]->const_array(mfi);
            const auto Jt = J[1]->const_array(mfi);
            const auto Jz = J[2]->const_array(mfi);
            const auto rho_arr = rho->const_array(mfi);
            const auto Br = B[0]->const_array(mfi);
            const auto Bt = B[1]->const_array(mfi);
            const auto Bz = B[2]->const_array(mfi);
            const auto Br_x = (B_ext[0] != nullptr)
                ? B_ext[0]->const_array(mfi) : amrex::Array4<const Real>{};
            const auto Bt_x = (B_ext[1] != nullptr)
                ? B_ext[1]->const_array(mfi) : amrex::Array4<const Real>{};
            const auto Bz_x = (B_ext[2] != nullptr)
                ? B_ext[2]->const_array(mfi) : amrex::Array4<const Real>{};
            const auto own = mask->const_array(mfi);
            amrex::Array4<const int> eb_flag {};
            if (eb_enabled && eb_update_E[lev][d]) {
                eb_flag = eb_update_E[lev][d]->const_array(mfi);
            }
            const int dcomp = d;

            reduce_op.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) -> ReduceTuple
                {
                    if (own(i, j, 0) == 0) { return {0.0_rt, 0.0_rt}; }
                    if (eb_flag && eb_flag(i, j, 0) == 0) {
                        return {0.0_rt, 0.0_rt};
                    }

                    const Real jv = Jself(i, j, 0);
                    const Real rho_val =
                        Interp(rho_arr, nodal_iv, self_stag, coarsen_iv,
                               i, j, 0, 0);

                    Real jtot_val = 0._rt;
                    if (has_J_dep) {
                        const Real jr_v = (dcomp == 0) ? jv :
                            Interp(Jr, Jr_stag, self_stag, coarsen_iv,
                                   i, j, 0, 0);
                        const Real jt_v = (dcomp == 1) ? jv :
                            Interp(Jt, Jt_stag, self_stag, coarsen_iv,
                                   i, j, 0, 0);
                        const Real jz_v = (dcomp == 2) ? jv :
                            Interp(Jz, Jz_stag, self_stag, coarsen_iv,
                                   i, j, 0, 0);
                        jtot_val = std::sqrt(jr_v*jr_v + jt_v*jt_v
                                             + jz_v*jz_v);
                    }

                    // geometry: r at this component's radial staggering,
                    // and the volume-centroid radius of the ring it owns
                    // (the nodal axis ring owns [0, dr/2] -> centroid dr/8)
                    const Real r = nodal_r ? (rmin + i*dr)
                                           : (rmin + (i + 0.5_rt)*dr);
                    Real r_vol = r;
                    if (nodal_r && i == dom_lo.x && rmin == 0.0_rt) {
                        r_vol = 0.125_rt*dr;
                    }
                    const Real dV = 2.0_rt*MathConst::pi*r_vol*dr*dz;

                    const Real pe = eta(rho_val, jtot_val, t_new)*jv*jv*dV;

                    Real ph = 0._rt;
                    const bool interior =
                        (i > dom_lo.x) && (j > dom_lo.y) &&
                        (i < dom_hi.x + self_stag[0]) &&
                        (j < dom_hi.y + self_stag[1]);
                    if (include_hyper && interior) {
                        Real btot_val = 0._rt;
                        if (has_B_dep) {
                            Real br_v = Interp(Br, Br_stag, self_stag,
                                               coarsen_iv, i, j, 0, 0);
                            Real bt_v = Interp(Bt, Bt_stag, self_stag,
                                               coarsen_iv, i, j, 0, 0);
                            Real bz_v = Interp(Bz, Bz_stag, self_stag,
                                               coarsen_iv, i, j, 0, 0);
                            if (Br_x) {
                                br_v += Interp(Br_x, Br_stag, self_stag,
                                               coarsen_iv, i, j, 0, 0);
                                bt_v += Interp(Bt_x, Bt_stag, self_stag,
                                               coarsen_iv, i, j, 0, 0);
                                bz_v += Interp(Bz_x, Bz_stag, self_stag,
                                               coarsen_iv, i, j, 0, 0);
                            }
                            btot_val = std::sqrt(br_v*br_v + bt_v*bt_v
                                                 + bz_v*bz_v);
                        }

                        using T_Algo = CylindricalYeeAlgorithm;
                        Real lap = 0._rt;
                        if (dcomp == 0) {
                            lap = T_Algo::Dr_rDr_over_r(Jself, r, dr,
                                      coefs_r, n_coefs_r, i, j, 0, 0)
                                + T_Algo::Dzz(Jself, coefs_z, n_coefs_z,
                                              i, j, 0, 0)
                                - jv/(r*r);
                        } else if (dcomp == 1) {
                            if (r > 0.0_rt) {
                                lap = T_Algo::Dr_rDr_over_r(Jself, r, dr,
                                          coefs_r, n_coefs_r, i, j, 0, 0)
                                    + T_Algo::Dzz(Jself, coefs_z,
                                                  n_coefs_z, i, j, 0, 0)
                                    - jv/(r*r);
                            }
                        } else {
                            lap = T_Algo::Dzz(Jself, coefs_z, n_coefs_z,
                                              i, j, 0, 0);
                            if (r > 0.5_rt*dr) {
                                lap += T_Algo::Dr_rDr_over_r(Jself, r, dr,
                                           coefs_r, n_coefs_r, i, j, 0, 0);
                            } else {
                                lap += T_Algo::Drr(Jself, coefs_r,
                                           n_coefs_r, i, j, 0, 0);
                            }
                        }
                        ph = -eta_h(rho_val, btot_val)*jv*lap*dV;
                    }

                    return {pe, ph};
                });
        }

        const auto rv = reduce_data.value(reduce_op);
        p_eta += amrex::get<0>(rv);
        p_eta_h += amrex::get<1>(rv);
    }

    // Vacuum-eta-floor conduction loss (see m_je_vacuum_eta): the
    // sigma E channel of the relax advance, P_sigma = Int g(rho)/
    // eta_vac |E|^2 dV. Zero unless the floor is active. E is
    // collocated (nodal) under the je contract; the owner mask keeps
    // shared nodes counted once.
    amrex::Real p_sigma = 0.0_rt;
    const amrex::Real eta_vac = hybrid->m_je_vacuum_eta;
    if (eta_vac > 0.0_rt) {
        const amrex::Real rho_vgate = 3.0_rt / (PhysConst::q_e *
            ((hybrid->m_je_vacuum_eta_n_gate > 0.0_rt)
                 ? hybrid->m_je_vacuum_eta_n_gate
                 : hybrid->m_je_n_min));
        const std::array<const amrex::MultiFab*, 3> E = {
            warpx.m_fields.get(FieldType::Efield_fp,
                               ablastr::fields::Direction{0}, lev),
            warpx.m_fields.get(FieldType::Efield_fp,
                               ablastr::fields::Direction{1}, lev),
            warpx.m_fields.get(FieldType::Efield_fp,
                               ablastr::fields::Direction{2}, lev)};
        // Split-field convention: the stored E in cells BELOW the
        // solver density floor still contains the analytic inductive
        // drive E_ext (the per-cell subtraction is gated on
        // rho >= rho_floor) -- exactly the gated cells. Square the
        // wave part E - E_ext, or the column reads the imposed drive
        // (measured: P*eta_vac invariant across 8 decades of the pin
        // = pure drive signature, up to 3e16 W of fiction at 1e-8).
        const bool sub_ext = hybrid->m_add_external_fields;
        std::array<const amrex::MultiFab*, 3> Eext =
            {nullptr, nullptr, nullptr};
        if (sub_ext) {
            Eext = {warpx.m_fields.get(FieldType::hybrid_E_fp_external,
                                       ablastr::fields::Direction{0}, lev),
                    warpx.m_fields.get(FieldType::hybrid_E_fp_external,
                                       ablastr::fields::Direction{1}, lev),
                    warpx.m_fields.get(FieldType::hybrid_E_fp_external,
                                       ablastr::fields::Direction{2}, lev)};
        }
        const auto mask = amrex::OwnerMask(*E[0], geom.periodicity());
        amrex::ReduceOps<amrex::ReduceOpSum> rop;
        amrex::ReduceData<amrex::Real> rdata(rop);
        using RT = typename decltype(rdata)::Type;
        for (amrex::MFIter mfi(*E[0], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto Er = E[0]->const_array(mfi);
            const auto Et = E[1]->const_array(mfi);
            const auto Ez = E[2]->const_array(mfi);
            const auto Xr = sub_ext ? Eext[0]->const_array(mfi)
                : amrex::Array4<const amrex::Real>{};
            const auto Xt = sub_ext ? Eext[1]->const_array(mfi)
                : amrex::Array4<const amrex::Real>{};
            const auto Xz = sub_ext ? Eext[2]->const_array(mfi)
                : amrex::Array4<const amrex::Real>{};
            const auto rho_arr = rho->const_array(mfi);
            const auto m_arr = mask->const_array(mfi);
            rop.eval(box, rdata,
                [=] AMREX_GPU_DEVICE (int i, int j, int) -> RT
            {
                if (!m_arr(i, j, 0)) { return {0.0_rt}; }
                const amrex::Real r = rmin + i*dr;
                const amrex::Real r_vol =
                    (r > 0.0_rt) ? r : dr/8.0_rt;
                const amrex::Real dV =
                    2.0_rt*MathConst::pi*r_vol*dr*dz;
                const amrex::Real g_v =
                    std::exp(-amrex::max(rho_arr(i, j, 0), 0.0_rt)
                             * rho_vgate);
                const amrex::Real ewr = Er(i, j, 0)
                    - (sub_ext ? Xr(i, j, 0) : 0.0_rt);
                const amrex::Real ewt = Et(i, j, 0)
                    - (sub_ext ? Xt(i, j, 0) : 0.0_rt);
                const amrex::Real ewz = Ez(i, j, 0)
                    - (sub_ext ? Xz(i, j, 0) : 0.0_rt);
                const amrex::Real e2 =
                    ewr*ewr + ewt*ewt + ewz*ewz;
                return {g_v / eta_vac * e2 * dV};
            });
        }
        p_sigma = amrex::get<0>(rdata.value(rop));
    }

    // Sponge-cap loss (see m_je_sponge_width): the wave energy the
    // z-cap layers remove, at the rate the damping applies it,
    // P_sponge = Int xi(z)/tau_max (eps_art |dE|^2 + |dB|^2/mu0) dV
    // with dX the deviation from the drive-tracking target (the same
    // arithmetic as ApplyJeSpongeLayer -- book what the layer damps,
    // on the wave part only, or the column reads the imposed drive:
    // the P_sigma_vac lesson). Zero unless the sponge is active and
    // the relax solve has published c_art.
    amrex::Real p_sponge = 0.0_rt;
    const amrex::Real Ls = hybrid->m_je_sponge_width;
    const amrex::Real c_art_pub = hybrid->m_je_c_art;
    if (Ls > 0.0_rt && c_art_pub > 0.0_rt) {
        const amrex::Real vw = (hybrid->m_je_sponge_vw > 0.0_rt)
            ? hybrid->m_je_sponge_vw : c_art_pub;
        const amrex::Real inv_tau_max = vw * hybrid->m_je_sponge_lnf / Ls;
        const amrex::Real eps_art =
            1.0_rt / (PhysConst::mu0 * c_art_pub * c_art_pub);
        const amrex::Real inv_mu0 = 1.0_rt / PhysConst::mu0;
        const amrex::Real zlo_d = geom.ProbLo(1);
        const amrex::Real zhi_d = geom.ProbHi(1);
        const amrex::Real z_lo_edge = zlo_d + Ls;
        const amrex::Real z_hi_edge = zhi_d - Ls;
        const std::array<const amrex::MultiFab*, 3> E = {
            warpx.m_fields.get(FieldType::Efield_fp,
                               ablastr::fields::Direction{0}, lev),
            warpx.m_fields.get(FieldType::Efield_fp,
                               ablastr::fields::Direction{1}, lev),
            warpx.m_fields.get(FieldType::Efield_fp,
                               ablastr::fields::Direction{2}, lev)};
        const amrex::MultiFab* Eref =
            warpx.m_fields.get("hybrid_sponge_Eref_fp", lev);
        const amrex::MultiFab* Bref =
            warpx.m_fields.get("hybrid_sponge_Bref_fp", lev);
        const bool have_ext = hybrid->m_add_external_fields &&
            (hybrid->m_je_sponge_track_eext ||
             hybrid->m_je_sponge_track_bext);
        const amrex::Real trkE =
            (have_ext && hybrid->m_je_sponge_track_eext) ? 1.0_rt : 0.0_rt;
        const amrex::Real trkB =
            (have_ext && hybrid->m_je_sponge_track_bext) ? 1.0_rt : 0.0_rt;
        std::array<const amrex::MultiFab*, 3> Eex =
            {nullptr, nullptr, nullptr};
        std::array<const amrex::MultiFab*, 3> Bex =
            {nullptr, nullptr, nullptr};
        const amrex::MultiFab* Eex0 = nullptr;
        const amrex::MultiFab* Bex0 = nullptr;
        if (have_ext) {
            for (int d = 0; d < 3; ++d) {
                Eex[d] = warpx.m_fields.get(
                    FieldType::hybrid_E_fp_external,
                    ablastr::fields::Direction{d}, lev);
                Bex[d] = warpx.m_fields.get(
                    FieldType::hybrid_B_fp_external,
                    ablastr::fields::Direction{d}, lev);
            }
            Eex0 = warpx.m_fields.get("hybrid_sponge_Eext0_fp", lev);
            Bex0 = warpx.m_fields.get("hybrid_sponge_Bext0_fp", lev);
        }
        const auto mask = amrex::OwnerMask(*E[0], geom.periodicity());
        amrex::ReduceOps<amrex::ReduceOpSum> rop;
        amrex::ReduceData<amrex::Real> rdata(rop);
        using RT = typename decltype(rdata)::Type;
        for (amrex::MFIter mfi(*E[0], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto Er = E[0]->const_array(mfi);
            const auto Et = E[1]->const_array(mfi);
            const auto Ez = E[2]->const_array(mfi);
            const auto Brr = B[0]->const_array(mfi);
            const auto Btt = B[1]->const_array(mfi);
            const auto Bzz = B[2]->const_array(mfi);
            const auto er = Eref->const_array(mfi);
            const auto br = Bref->const_array(mfi);
            amrex::Array4<const amrex::Real> xer, xet, xez,
                xbr, xbt, xbz, xe0, xb0;
            if (have_ext) {
                xer = Eex[0]->const_array(mfi);
                xet = Eex[1]->const_array(mfi);
                xez = Eex[2]->const_array(mfi);
                xbr = Bex[0]->const_array(mfi);
                xbt = Bex[1]->const_array(mfi);
                xbz = Bex[2]->const_array(mfi);
                xe0 = Eex0->const_array(mfi);
                xb0 = Bex0->const_array(mfi);
            }
            const auto m_arr = mask->const_array(mfi);
            const bool use_ext = have_ext;
            rop.eval(box, rdata,
                [=] AMREX_GPU_DEVICE (int i, int j, int) -> RT
            {
                if (!m_arr(i, j, 0)) { return {0.0_rt}; }
                const amrex::Real z = zlo_d + j*dz;
                amrex::Real xi = 0.0_rt;
                if (z > z_hi_edge) { xi = (z - z_hi_edge) / Ls; }
                else if (z < z_lo_edge) { xi = (z_lo_edge - z) / Ls; }
                if (xi <= 0.0_rt) { return {0.0_rt}; }
                xi = amrex::min(xi, 1.0_rt);
                const amrex::Real r = rmin + i*dr;
                const amrex::Real r_vol =
                    (r > 0.0_rt) ? r : dr/8.0_rt;
                const amrex::Real dV =
                    2.0_rt*MathConst::pi*r_vol*dr*dz;
                const amrex::Real der = Er(i, j, 0) - er(i, j, 0, 0)
                    - (use_ext ? trkE*(xer(i, j, 0) - xe0(i, j, 0, 0))
                               : 0.0_rt);
                const amrex::Real det = Et(i, j, 0) - er(i, j, 0, 1)
                    - (use_ext ? trkE*(xet(i, j, 0) - xe0(i, j, 0, 1))
                               : 0.0_rt);
                const amrex::Real dez = Ez(i, j, 0) - er(i, j, 0, 2)
                    - (use_ext ? trkE*(xez(i, j, 0) - xe0(i, j, 0, 2))
                               : 0.0_rt);
                const amrex::Real dbr = Brr(i, j, 0) - br(i, j, 0, 0)
                    - (use_ext ? trkB*(xbr(i, j, 0) - xb0(i, j, 0, 0))
                               : 0.0_rt);
                const amrex::Real dbt = Btt(i, j, 0) - br(i, j, 0, 1)
                    - (use_ext ? trkB*(xbt(i, j, 0) - xb0(i, j, 0, 1))
                               : 0.0_rt);
                const amrex::Real dbz = Bzz(i, j, 0) - br(i, j, 0, 2)
                    - (use_ext ? trkB*(xbz(i, j, 0) - xb0(i, j, 0, 2))
                               : 0.0_rt);
                const amrex::Real u2 =
                    eps_art*(der*der + det*det + dez*dez)
                    + inv_mu0*(dbr*dbr + dbt*dbt + dbz*dbz);
                return {xi * inv_tau_max * u2 * dV};
            });
        }
        p_sponge = amrex::get<0>(rdata.value(rop));
    }

    amrex::ParallelDescriptor::ReduceRealSum(p_eta);
    amrex::ParallelDescriptor::ReduceRealSum(p_eta_h);
    amrex::ParallelDescriptor::ReduceRealSum(p_sigma);
    amrex::ParallelDescriptor::ReduceRealSum(p_sponge);

    m_data[0] = p_eta;
    m_data[1] = p_eta_h;
    m_data[2] = p_eta + p_eta_h + p_sigma + p_sponge;
    m_data[3] = p_sigma;
    m_data[4] = p_sponge;
#else
    amrex::ignore_unused(step);
#endif
}
