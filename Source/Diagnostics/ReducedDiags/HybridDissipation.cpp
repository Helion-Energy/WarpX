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
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <array>
#include <fstream>
#include <memory>
#include <string>

using namespace amrex;
using warpx::fields::FieldType;

#if defined(WARPX_DIM_RZ)
namespace
{
    /** Domain mean of f^2 at one Yee component's OWN staggering.
     *
     * The f^2 dV and dV sums are accumulated together and returned as a
     * ratio, so the mean is exact at whatever centring the component carries.
     * Normalising instead by an analytic domain volume would bias any
     * radially nodal component at O(1/N_r), because the RZ element gives the
     * on-axis ring only its half cell. An owner mask keeps nodal points
     * shared between boxes from counting twice. \p mf_add, when given, is
     * added pointwise to \p mf first -- the external-split path, where
     * Bfield_fp carries only the plasma part of B.
     */
    amrex::Real
    DomainMeanSquare (amrex::MultiFab const & mf,
                      amrex::Geometry const & geom,
                      amrex::MultiFab const * mf_add = nullptr)
    {
        const QdsmcVolumeElement vol = MakeQdsmcVolumeElement(geom, mf.ixType());
        const auto mask = amrex::OwnerMask(mf, geom.periodicity());

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.tilebox();
            const auto f = mf.const_array(mfi);
            const auto f_add = (mf_add != nullptr)
                ? mf_add->const_array(mfi) : amrex::Array4<const Real>{};
            const auto own = mask->const_array(mfi);

            reduce_op.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) -> ReduceTuple
                {
                    if (own(i, j, 0) == 0) { return {0.0_rt, 0.0_rt}; }
                    const Real dV = vol(i);
                    Real fv = f(i, j, 0);
                    if (f_add) { fv += f_add(i, j, 0); }
                    return {fv*fv*dV, dV};
                });
        }

        const auto rt = reduce_data.value(reduce_op);
        amrex::Real rv[2] = {amrex::get<0>(rt), amrex::get<1>(rt)};
        amrex::ParallelDescriptor::ReduceRealSum(rv, 2);
        return (rv[1] > 0.0_rt) ? rv[0]/rv[1] : 0.0_rt;
    }
}
#endif

HybridDissipation::HybridDissipation (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
#if !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "the HybridDissipation reduced diagnostic is implemented for RZ "
        "geometry");
#endif

    m_data.resize(6, 0.0_rt);

    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
                          std::ofstream::out};
        int c = 0;
        ofs << "#";
        ofs << "[" << c++ << "]step()";
        ofs << m_sep << "[" << c++ << "]time(s)";
        ofs << m_sep << "[" << c++ << "]P_eta(W)";
        ofs << m_sep << "[" << c++ << "]P_etaH(W)";
        // P_nu is APPENDED after the total rather than grouped with the other
        // channels: this file is a shared artifact read across repositories,
        // and inserting a column would silently hand a positional reader P_nu
        // where it expects P_diss. Columns 0-4 keep their historical meaning;
        // new channels go on the end.
        ofs << m_sep << "[" << c++ << "]P_diss(W)";
        ofs << m_sep << "[" << c++ << "]P_nu(W)";
        // The gyroviscosity requirement gates are dimensionless and are not
        // power channels, so they take no part in P_diss; they are appended
        // last for the same positional-reader reason as P_nu above.
        ofs << m_sep << "[" << c++ << "]beta_i_over_4()";
        ofs << m_sep << "[" << c++ << "]F_gyro()";
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
        hybrid->m_external_split;
    const amrex::Real t_new = warpx.gett_new(lev);

    // KNOWN LIMITATION, made loud rather than left silent: the P_etaH
    // integrand below is the component-wise vector Laplacian form,
    // -eta_H J . lap(J). Under hyper_resistivity_curl_curl the SOLVER instead
    // applies E_H = +curl(eta_H curl J), whose dissipation is
    // +Int eta_H |curl J|^2 -- a different quantity (the two differ by the
    // eta_H grad(div J) terms, which is exactly why that option exists). This
    // column would then report a number the solver never produced. The
    // curl-curl integrand is not implemented here; until it is, warn rather
    // than quietly hand back a mismatched reading.
    if (include_hyper && hybrid->m_hyper_res_curl_curl) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            ablastr::warn_manager::WMRecordWarning(
                "HybridDissipation",
                "hybrid_pic_model.hyper_resistivity_curl_curl is on, but the "
                "P_etaH column integrates the component-wise vector Laplacian "
                "form -eta_H J.lap(J), not the curl-curl dissipation "
                "+Int eta_H |curl J|^2 that the solver actually applies. The "
                "P_etaH reading does not correspond to this run's "
                "hyper-resistive dissipation; P_eta and P_nu are unaffected.",
                ablastr::warn_manager::WarnPriority::high);
        }
    }

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
        const QdsmcVolumeElement vol =
            MakeQdsmcVolumeElement(geom, J[d]->ixType());
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

                    // r at this component's radial staggering (the physics
                    // below needs it); the control volume comes from the
                    // shared element in QdsmcVolumeElement.H, which is the
                    // same one the electron-energy budget integrates against
                    // -- that is what makes the two ledgers comparable.
                    const Real r = nodal_r ? (rmin + i*dr)
                                           : (rmin + (i + 0.5_rt)*dr);
                    const Real dV = vol(i);

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

    // PHYSICAL electron viscosity. Unlike the two terms above this is NOT
    // re-derived here: the solver writes its heating rate Q_nu [W/m^3] to
    // hybrid_qdsmc_visc_heating_fp and this column is the volume integral of
    // exactly that field. A re-derived integrand can silently disagree with
    // what the solver did (P_etaH already does, under
    // hyper_resistivity_curl_curl); integrating the solver's own output
    // cannot. A zero here therefore means the term did not run.
    amrex::Real p_nu = 0.0_rt;
    if (warpx.m_fields.has("hybrid_qdsmc_visc_heating_fp", lev)) {
        const amrex::MultiFab* Qnu =
            warpx.m_fields.get("hybrid_qdsmc_visc_heating_fp", lev);
        const QdsmcVolumeElement vol =
            MakeQdsmcVolumeElement(geom, Qnu->ixType());
        const auto mask = amrex::OwnerMask(*Qnu, geom.periodicity());

        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (amrex::MFIter mfi(*Qnu, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto q_arr = Qnu->const_array(mfi);
            const auto own = mask->const_array(mfi);

            reduce_op.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) -> ReduceTuple
                {
                    if (own(i, j, 0) == 0) { return {0.0_rt}; }
                    return {q_arr(i, j, 0) * vol(i)};
                });
        }
        p_nu = amrex::get<0>(reduce_data.value(reduce_op));
    }

    amrex::ParallelDescriptor::ReduceRealSum(p_eta);
    amrex::ParallelDescriptor::ReduceRealSum(p_eta_h);
    amrex::ParallelDescriptor::ReduceRealSum(p_nu);

    m_data[0] = p_eta;
    m_data[1] = p_eta_h;
    m_data[2] = p_eta + p_eta_h + p_nu;
    m_data[3] = p_nu;

    // ---------------------------------------------------------------------
    // GYROVISCOSITY REQUIREMENT GATES. Diagnostics only: nothing below is
    // read by a solve, and nothing below introduces a knob.
    //
    // These two dimensionless numbers are the measurement taken INSTEAD of
    // implementing a Braginskii eta_3 / eta_4 ELECTRON gyroviscosity. Two
    // literature results make that the right trade, and the gates are what
    // keep the decision falsifiable rather than assumed:
    //
    //  - the electron gyroviscous term is suppressed in the generalized
    //    Ohm's law by at least the mass ratio relative to the other terms
    //    (King & Kruger, arXiv:1407.3864, Sec. II), so the ION gates below
    //    bound the electron requirement from above by m_e/m_i;
    //  - where a current sheet narrows to the electron gyroradius the FLR
    //    ordering that a Braginskii closure rests on has already failed, so
    //    such a closure would be outside its own validity exactly where it
    //    would be needed.
    //
    // [4] beta_i/4 = rho_i^2 / (2 d_i^2) = mu0 <p_i> / (2 <|B|^2>), the
    //     WAVENUMBER-INDEPENDENT ratio of the gyroviscous to the Hall stress.
    //     Ferraro, ApJ 662, 512 (2007), Sec. 3 states it as A/H = beta_i/4:
    //     both dimensionless groups carry the same wavenumber factor, so
    //     their ratio does not. See also Schnack et al., Phys. Plasmas 13,
    //     058103 (2006), Eqs. (55ab)/(56ab). It costs only beta_i, which is
    //     what makes it the cleanest single "does it matter here" scalar.
    //
    // [5] F = (beta_i/4) (omega_dyn / Omega_ci), the gyroviscous force
    //     measured against the magnetic tension force: Ferraro, ApJ 662, 512
    //     (2007), Eq. (8b), where omega_dyn is the equilibrium rotation rate
    //     of the accretion disk that paper analyses.
    //
    // AMBIGUITY DECLARED, because omega_dyn is where this criterion is soft
    // and the choice is the main source of spread in applying it:
    //   - the VELOCITY is |u_i|, the ION BULK flow speed, taken from the same
    //     NGP moment deposit that supplies T_i, so the bulk subtracted out of
    //     T_i is exactly the bulk put into omega_dyn;
    //   - the LENGTH is the resolved gradient scale of the magnetic field,
    //     L = |B| / (mu0 |J_plasma|) -- the local current-sheet width, i.e.
    //     the shortest scale the field actually resolves and the one that
    //     collapses in the reconnection layer this gate is asked about.
    //   Another defensible pair (a mode frequency and a machine radius, say)
    //   moves F by the corresponding ratio. F is a scaling gate, not a
    //   calibrated number; read it in decades.
    //
    // REDUCTION: every factor is a domain-volume average in its own right and
    // the gates are ASSEMBLED from those averages. They are deliberately not
    // the volume average of the pointwise gate, which diverges at a field
    // null (both expressions go as inverse powers of |B|) and would then
    // report the null rather than the plasma. Volumes come from the shared
    // control-volume element in QdsmcVolumeElement.H -- the same element the
    // power columns above integrate against.
    // ---------------------------------------------------------------------
    amrex::Real beta_i_over_4 = 0.0_rt;
    amrex::Real f_gyro = 0.0_rt;

    // <|B|^2> [T^2] and <|J_plasma|^2> [A^2/m^4], each component averaged at
    // its own staggering. The external B is added back in wherever the solver
    // splits it out of Bfield_fp, so the gate always sees the total field.
    amrex::Real b2_bar = 0.0_rt;
    amrex::Real j2_bar = 0.0_rt;
    for (int d = 0; d < 3; ++d) {
        const amrex::MultiFab* b_ext = external_b_split
            ? warpx.m_fields.get(FieldType::hybrid_B_fp_external,
                                 ablastr::fields::Direction{d}, lev)
            : nullptr;
        b2_bar += DomainMeanSquare(*B[d], geom, b_ext);
        j2_bar += DomainMeanSquare(*J[d], geom);
    }

    // Ion moments. n_i, T_i and the bulk u_i all come from the one NGP
    // deposit the T_<species> field diagnostic already uses
    // (WarpXParticleContainer::DepositTotalNGPTemperature), summed over the
    // charged species: N_<s> holds the per-cell PARTICLE COUNT, so summing
    // N k_B T_i over cells is already Int n_i k_B T_i dV with no volume
    // factor at all; T_<s> is in eV; u_<s> is the weighted mean proper
    // velocity gamma*v.
    const amrex::Real inv_c2 = 1.0_rt/(PhysConst::c*PhysConst::c);
    amrex::Real ion_sums[4] = {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
    auto& mypc = warpx.GetPartContainer();
    for (int is = 0; is < mypc.nSpecies(); ++is)
    {
        auto& pc = mypc.GetParticleContainer(is);
        const auto q_s = static_cast<amrex::Real>(pc.getCharge());
        const auto m_s = static_cast<amrex::Real>(pc.getMass());
        if (q_s <= 0.0_rt || m_s <= 0.0_rt) { continue; }

        pc.DepositTotalNGPTemperature(lev);
        const std::string& s_name = pc.getName();
        const amrex::MultiFab& Ns = *warpx.m_fields.get("N_" + s_name, lev);
        const amrex::MultiFab& Ts = *warpx.m_fields.get("T_" + s_name, lev);
        const amrex::MultiFab& uxs = *warpx.m_fields.get(
            "u_" + s_name, ablastr::fields::Direction{0}, lev);
        const amrex::MultiFab& uys = *warpx.m_fields.get(
            "u_" + s_name, ablastr::fields::Direction{1}, lev);
        const amrex::MultiFab& uzs = *warpx.m_fields.get(
            "u_" + s_name, ablastr::fields::Direction{2}, lev);

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real,
                          amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        // These moments are cell-centred, so every point has exactly one
        // owner and no owner mask is needed (unlike the nodal fields above).
        for (amrex::MFIter mfi(Ns, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto n_arr = Ns.const_array(mfi);
            const auto t_arr = Ts.const_array(mfi);
            const auto ux = uxs.const_array(mfi);
            const auto uy = uys.const_array(mfi);
            const auto uz = uzs.const_array(mfi);

            reduce_op.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) -> ReduceTuple
                {
                    const Real n_w = n_arr(i, j, 0);
                    if (n_w <= 0.0_rt) {
                        return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
                    }
                    // proper velocity gamma*v back to v
                    const Real u2 = ux(i, j, 0)*ux(i, j, 0)
                                  + uy(i, j, 0)*uy(i, j, 0)
                                  + uz(i, j, 0)*uz(i, j, 0);
                    const Real v2 = u2/(1.0_rt + u2*inv_c2);
                    return {n_w*PhysConst::q_e*t_arr(i, j, 0), // Int n k_B T dV
                            m_s*n_w,                           // Int rho_m dV
                            m_s*n_w*v2,                        // Int rho_m u^2 dV
                            q_s*n_w};                          // Int q_i n_i dV
                });
        }

        const auto rt = reduce_data.value(reduce_op);
        ion_sums[0] += amrex::get<0>(rt);
        ion_sums[1] += amrex::get<1>(rt);
        ion_sums[2] += amrex::get<2>(rt);
        ion_sums[3] += amrex::get<3>(rt);
    }
    amrex::ParallelDescriptor::ReduceRealSum(ion_sums, 4);

    // Cell-centred control volume of the whole domain, from the shared
    // element; its radial sum is exact at cell centring.
    const QdsmcVolumeElement vol_cc =
        MakeQdsmcVolumeElement(geom, amrex::IndexType::TheCellType());
    amrex::Real dom_vol = 0.0_rt;
    for (int i = dom_lo.x; i <= dom_hi.x; ++i) { dom_vol += vol_cc(i); }
    dom_vol *= static_cast<amrex::Real>(dom_hi.y - dom_lo.y + 1);

    if (b2_bar > 0.0_rt && dom_vol > 0.0_rt && ion_sums[1] > 0.0_rt) {
        const amrex::Real p_i_bar = ion_sums[0]/dom_vol;    // <n_i k_B T_i> [Pa]
        beta_i_over_4 = PhysConst::mu0*p_i_bar/(2.0_rt*b2_bar);

        if (j2_bar > 0.0_rt && ion_sums[3] > 0.0_rt) {
            const amrex::Real b_bar = std::sqrt(b2_bar);
            const amrex::Real j_bar = std::sqrt(j2_bar);
            const amrex::Real u_bar = std::sqrt(ion_sums[2]/ion_sums[1]);
            // effective ion m_i/q_i [kg/C] of the whole ion population
            const amrex::Real m_over_q = ion_sums[1]/ion_sums[3];
            const amrex::Real omega_dyn = u_bar*PhysConst::mu0*j_bar/b_bar;
            const amrex::Real omega_ci = b_bar/m_over_q;
            f_gyro = beta_i_over_4*omega_dyn/omega_ci;
        }
    }

    m_data[4] = beta_i_over_4;
    m_data[5] = f_gyro;
#else
    amrex::ignore_unused(step);
#endif
}
