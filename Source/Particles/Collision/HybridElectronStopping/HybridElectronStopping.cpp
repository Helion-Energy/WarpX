/* Copyright 2026 The WarpX Community
 * Authors: Eric Clark (Helion Energy)
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HybridElectronStopping.H"

#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_GpuAtomic.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>

HybridElectronStopping::HybridElectronStopping (std::string const& collision_name)
    : CollisionBase(collision_name)
{
    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 1,
        "hybrid_electron_stopping must have exactly one species.");

    const amrex::ParmParse pp_collision_name(collision_name);
    utils::parser::getWithParser(pp_collision_name, "coulomb_log", m_coulomb_log);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_coulomb_log > 0._prt,
        "For hybrid_electron_stopping, coulomb_log must be specified and positive.");

    // The heating side of this operator lands in the electron-energy
    // equation's T_e field: without it there is nothing to heat (stopping
    // on a prescribed background remains background_stopping's job).
    const amrex::ParmParse pp_hybrid("hybrid_pic_model");
    bool solve_eee = false;
    pp_hybrid.query("solve_electron_energy_equation", solve_eee);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(solve_eee,
        "The hybrid_electron_stopping collision requires "
        "hybrid_pic_model.solve_electron_energy_equation = 1: the drag's "
        "conjugate heating is deposited on the electron temperature field. "
        "For stopping on a prescribed (non-evolving) background use the "
        "background_stopping collision instead.");

    amrex::Print() << "[stopping] " << m_species_names[0]
        << " on electrons: coulomb_log=" << m_coulomb_log
        << " (drag -> u_e, heating staged to qdsmc sources)\n";
}

void
HybridElectronStopping::doCollisions (amrex::Real /*cur_time*/, amrex::Real dt, MultiParticleContainer* mypc)
{
    ABLASTR_PROFILE("HybridElectronStopping::doCollisions()");
    using namespace amrex::literals;
    using warpx::fields::FieldType;

    // Goldston & Rutherford ion-on-electron slowing down (Introduction to
    // Plasma Physics, Eq. 14.12), the background_stopping electron branch's
    // rate expression, but evaluated from the SELF-CONSISTENT hybrid fields:
    //   n_e = rho_fp / e and T_e = the electron-energy-equation temperature
    // gathered at the particle. The drag is exponential and drag-only
    // toward the local electron fluid velocity u_e (Ve_fp), applied in
    // velocity space with the proper-velocity conversion handled exactly.
    //
    // Timing/energy semantics: this runs in the COLLISION stage of the
    // step; each particle's weighted kinetic-energy loss w dKE is
    // scatter-added (gather-conjugate linear nodal weights, true node
    // control volumes) into the HybridPICModel staging field, which the
    // SOURCE stage of the SAME step converts into a T_e increment
    // (QDSMCApplyFastIonHeating) -- one-step staging, no energy lost.

    auto & warpx = WarpX::GetInstance();
    auto & species = mypc->GetParticleContainerFromName(m_species_names[0]);

    auto const * hybrid_model = warpx.get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(hybrid_model != nullptr,
        "HybridElectronStopping requires the hybrid-PIC solver to be active.");

    amrex::ParticleReal const species_mass   = species.getMass();
    amrex::ParticleReal const species_charge = species.getCharge();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(species_mass > 0._prt,
        "With hybrid_electron_stopping, the species mass must be > 0");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(species_charge != 0._prt,
        "With hybrid_electron_stopping, the species charge must be nonzero");

    amrex::ParticleReal const clog = m_coulomb_log;
    amrex::Real const inv_qe    = 1.0_rt / PhysConst::q_e;
    amrex::Real const rho_floor = PhysConst::q_e * hybrid_model->m_n_floor;
    // Floor on T_e in the rate so (k_B T_e)^{-3/2} stays finite (the same
    // 1e-3 eV floor the Q_ei relaxation kernels put on their rate argument).
    amrex::Real const Te_floor_K = 1.e-3_rt * PhysConst::q_e / PhysConst::kb;

    int const nox = WarpX::nox;
    int const n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;
    bool const galerkin_interpolation = WarpX::galerkin_interpolation;

    for (int lev = 0; lev <= species.finestLevel(); ++lev) {
        ablastr::fields::VectorField Ve_fp = warpx.m_fields.get_alldirs("Ve_fp", lev);
        ablastr::fields::VectorField B_fp  = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        amrex::MultiFab const & rho_fp     = *warpx.m_fields.get(FieldType::rho_fp, lev);
        amrex::MultiFab const & Te_fp      = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        amrex::MultiFab & stage            = hybrid_model->GetFastIonHeatingStaging(lev);

        amrex::XDim3 const dinv = WarpX::InvCellSize(lev);
        auto const dxi = warpx.Geom(lev).InvCellSizeArray();
        auto const dxn = warpx.Geom(lev).CellSizeArray();
        auto const plo = warpx.Geom(lev).ProbLoArray();
        auto const phi = warpx.Geom(lev).ProbHiArray();
        amrex::GpuArray<int, AMREX_SPACEDIM> is_per{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            is_per[d] = warpx.Geom(lev).isPeriodic(d);
        }

        // NO OpenMP here: the w dKE deposit below scatter-adds into the
        // staging field with Gpu::Atomic::AddNoRet, a plain += on the host,
        // and neighboring tiles' deposits reach shared seam nodes of the
        // same fab, so a threaded tile loop would race (the same reason the
        // qdsmc daughter deposit is unthreaded).
        for (WarpXParIter pti(species, lev); pti.isValid(); ++pti)
        {
            amrex::Box const & box = pti.validbox();
            amrex::XDim3 const xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
            amrex::Dim3 const lo = amrex::lbound(box);

            auto const Vex_arr = Ve_fp[0]->const_array(pti);
            auto const Vey_arr = Ve_fp[1]->const_array(pti);
            auto const Vez_arr = Ve_fp[2]->const_array(pti);
            auto const Bx_arr  = B_fp[0]->const_array(pti);
            auto const By_arr  = B_fp[1]->const_array(pti);
            auto const Bz_arr  = B_fp[2]->const_array(pti);
            auto const rho_arr = rho_fp.const_array(pti);
            auto const Te_arr  = Te_fp.const_array(pti);
            auto const st_arr  = stage.array(pti);

            amrex::IndexType const Vex_type = Ve_fp[0]->ixType();
            amrex::IndexType const Vey_type = Ve_fp[1]->ixType();
            amrex::IndexType const Vez_type = Ve_fp[2]->ixType();
            amrex::IndexType const Bx_type  = B_fp[0]->ixType();
            amrex::IndexType const By_type  = B_fp[1]->ixType();
            amrex::IndexType const Bz_type  = B_fp[2]->ixType();

            auto& attribs = pti.GetAttribs();
            amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
            amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
            amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();
            amrex::ParticleReal const* const AMREX_RESTRICT wp = attribs[PIdx::w].dataPtr();

            auto const getPosition = GetParticlePosition<PIdx>(pti);
            long const np = pti.numParticles();

            // amrex::For (not ParallelFor): iterations scatter-add into
            // shared staging nodes.
            amrex::For(np, [=] AMREX_GPU_DEVICE (long ip)
            {
                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                // Self-consistent n_e at the particle (the same source and
                // solver-floor skip as hybrid_resistive_drag).
                amrex::Real const rho_val = ablastr::particles::doGatherScalarFieldNodal(
                    xp, yp, zp, rho_arr, dxi, plo);
                if (rho_val <= rho_floor) { return; }
                amrex::Real const n_e = rho_val * inv_qe;

                // T_e at the particle from the EEE field (stored in Kelvin).
                amrex::Real const Te_K = amrex::max(
                    ablastr::particles::doGatherScalarFieldNodal(
                        xp, yp, zp, Te_arr, dxi, plo),
                    Te_floor_K);
                amrex::Real const T_e = Te_K * PhysConst::kb;

                // G&R Eq. 14.12 slowing-down rate: the background_stopping
                // electron branch's alpha, with the input Coulomb logarithm.
                amrex::Real constexpr pi   = MathConst::pi;
                amrex::Real constexpr ep0  = PhysConst::epsilon_0;
                amrex::Real constexpr q_e2 = PhysConst::q_e * PhysConst::q_e;
                amrex::Real constexpr ep02 = ep0 * ep0;

                amrex::Real const pi32 = pi * std::sqrt(pi);
                amrex::Real const q2   = species_charge * species_charge;
                amrex::Real const T32  = T_e * std::sqrt(T_e);

                amrex::Real const nu_s = std::sqrt(2._rt) * n_e * q2 * q_e2
                    * std::sqrt(PhysConst::m_e) * clog
                    / (12._rt * pi32 * ep02 * species_mass * T32);

                // u_e at the particle (Ve_fp in the E gather slot; B rides
                // along in the B slot and is not used).
                amrex::ParticleReal Vex = 0._prt, Vey = 0._prt, Vez = 0._prt;
                amrex::ParticleReal Bxp = 0._prt, Byp = 0._prt, Bzp = 0._prt;
                doGatherShapeN(xp, yp, zp, Vex, Vey, Vez, Bxp, Byp, Bzp,
                               Vex_arr, Vey_arr, Vez_arr, Bx_arr, By_arr, Bz_arr,
                               Vex_type, Vey_type, Vez_type, Bx_type, By_type, Bz_type,
                               dinv, xyzmin, lo, n_rz_azimuthal_modes,
                               nox, galerkin_interpolation);

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                // Collision-stage momenta are in the curvilinear frame
                // (CollisionHandler rotates ux, uy to u_r, u_theta before
                // collisions and back after); the gather returns Cartesian
                // components at the particle, so rotate u_e into the same
                // frame before applying the drag. (RSPHERE never reaches
                // this operator: the electron energy equation this collision
                // requires is not supported there.)
                {
                    amrex::ParticleReal const rp = std::sqrt(xp*xp + yp*yp);
                    amrex::ParticleReal const costheta = (rp > 0._prt ? xp/rp : 1._prt);
                    amrex::ParticleReal const sintheta = (rp > 0._prt ? yp/rp : 0._prt);
                    amrex::ParticleReal const Vex_c = Vex;
                    Vex =  Vex_c*costheta + Vey*sintheta;
                    Vey = -Vex_c*sintheta + Vey*costheta;
                }
#endif

                // Exponential drag toward u_e in velocity space:
                //   dv/dt = -nu_s (v - u_e)   integrates exactly to
                //   v(t+dt) = u_e + (v - u_e) exp(-nu_s dt),
                // unconditionally stable at any nu_s dt. The momenta store
                // the proper velocity u = gamma v: convert u -> v, relax,
                // convert back (|v_new| < c since |u_e| < c).
                amrex::ParticleReal constexpr inv_c2 =
                    1._prt / (PhysConst::c * PhysConst::c);
                amrex::ParticleReal const u2b = ux[ip]*ux[ip] + uy[ip]*uy[ip] + uz[ip]*uz[ip];
                amrex::ParticleReal const gb  = std::sqrt(1._prt + u2b*inv_c2);
                amrex::ParticleReal const fac = std::exp(-nu_s*dt);
                amrex::ParticleReal const vxn = Vex + (ux[ip]/gb - Vex)*fac;
                amrex::ParticleReal const vyn = Vey + (uy[ip]/gb - Vey)*fac;
                amrex::ParticleReal const vzn = Vez + (uz[ip]/gb - Vez)*fac;
                amrex::ParticleReal const v2n = vxn*vxn + vyn*vyn + vzn*vzn;
                amrex::ParticleReal const ga  = 1._prt/std::sqrt(1._prt - v2n*inv_c2);
                ux[ip] = ga*vxn;
                uy[ip] = ga*vyn;
                uz[ip] = ga*vzn;

                // Weighted kinetic-energy loss [J], with
                //   KE = (gamma - 1) m c^2 = m |u|^2 / (gamma + 1)
                // (the second form avoids the (gamma - 1) cancellation for
                // mildly relativistic ions).
                amrex::ParticleReal const u2a = ga*ga*v2n;
                amrex::Real const dE = wp[ip] * species_mass
                    * (u2b/(gb + 1._prt) - u2a/(ga + 1._prt));

                // Scatter-add w dKE as an energy DENSITY [J/m^3] with the
                // gather-conjugate linear nodal weights: each corner node
                // receives dE * weight / V_node, V_node the node's true
                // control volume (clipped at non-periodic domain edges; the
                // cylindrical/spherical radial Jacobian included), so the
                // source-stage conversion dTe = (gamma-1) E/(n_e kB)
                // integrates back to exactly w dKE.
                int ii = 0, jj = 0, kk = 0;
                amrex::Real W[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    xp, yp, zp, plo, dxi, ii, jj, kk, W);
#if defined(WARPX_DIM_3D)
                for (int dk = 0; dk < 2; ++dk) {
                    for (int dj = 0; dj < 2; ++dj) {
                        for (int di = 0; di < 2; ++di) {
                            amrex::Real vol = 1._rt;
                            int const nid[3] = {ii + di, jj + dj, kk + dk};
                            for (int d = 0; d < 3; ++d) {
                                if (is_per[d]) {
                                    vol *= dxn[d];
                                } else {
                                    amrex::Real const xn = plo[d] + nid[d]*dxn[d];
                                    vol *= amrex::min(xn + 0.5_rt*dxn[d], phi[d])
                                         - amrex::max(xn - 0.5_rt*dxn[d], plo[d]);
                                }
                            }
                            amrex::Gpu::Atomic::AddNoRet(
                                &st_arr(ii + di, jj + dj, kk + dk),
                                dE * W[0][di]*W[1][dj]*W[2][dk] / vol);
                        }
                    }
                }
#elif defined(WARPX_DIM_RZ)
                for (int dj = 0; dj < 2; ++dj) {
                    for (int di = 0; di < 2; ++di) {
                        amrex::Real const r_node = plo[0] + (ii + di)*dxn[0];
                        amrex::Real const r_in   = amrex::max(r_node - 0.5_rt*dxn[0], plo[0]);
                        amrex::Real const r_out  = amrex::min(r_node + 0.5_rt*dxn[0], phi[0]);
                        amrex::Real vol = MathConst::pi*(r_out*r_out - r_in*r_in);
                        if (is_per[1]) {
                            vol *= dxn[1];
                        } else {
                            amrex::Real const z_node = plo[1] + (jj + dj)*dxn[1];
                            vol *= amrex::min(z_node + 0.5_rt*dxn[1], phi[1])
                                 - amrex::max(z_node - 0.5_rt*dxn[1], plo[1]);
                        }
                        amrex::Gpu::Atomic::AddNoRet(
                            &st_arr(ii + di, jj + dj, 0),
                            dE * W[0][di]*W[1][dj] / vol);
                    }
                }
#elif defined(WARPX_DIM_XZ)
                for (int dj = 0; dj < 2; ++dj) {
                    for (int di = 0; di < 2; ++di) {
                        amrex::Real vol = 1._rt;
                        int const nid[2] = {ii + di, jj + dj};
                        for (int d = 0; d < 2; ++d) {
                            if (is_per[d]) {
                                vol *= dxn[d];
                            } else {
                                amrex::Real const xn = plo[d] + nid[d]*dxn[d];
                                vol *= amrex::min(xn + 0.5_rt*dxn[d], phi[d])
                                     - amrex::max(xn - 0.5_rt*dxn[d], plo[d]);
                            }
                        }
                        amrex::Gpu::Atomic::AddNoRet(
                            &st_arr(ii + di, jj + dj, 0),
                            dE * W[0][di]*W[1][dj] / vol);
                    }
                }
#elif defined(WARPX_DIM_RCYLINDER)
                for (int di = 0; di < 2; ++di) {
                    amrex::Real const r_node = plo[0] + (ii + di)*dxn[0];
                    amrex::Real const r_in   = amrex::max(r_node - 0.5_rt*dxn[0], plo[0]);
                    amrex::Real const r_out  = amrex::min(r_node + 0.5_rt*dxn[0], phi[0]);
                    amrex::Real const vol = MathConst::pi*(r_out*r_out - r_in*r_in);
                    amrex::Gpu::Atomic::AddNoRet(
                        &st_arr(ii + di, 0, 0), dE * W[0][di] / vol);
                }
#elif defined(WARPX_DIM_RSPHERE)
                for (int di = 0; di < 2; ++di) {
                    amrex::Real const r_node = plo[0] + (ii + di)*dxn[0];
                    amrex::Real const r_in   = amrex::max(r_node - 0.5_rt*dxn[0], plo[0]);
                    amrex::Real const r_out  = amrex::min(r_node + 0.5_rt*dxn[0], phi[0]);
                    amrex::Real const vol = (4._rt/3._rt)*MathConst::pi
                        *(r_out*r_out*r_out - r_in*r_in*r_in);
                    amrex::Gpu::Atomic::AddNoRet(
                        &st_arr(ii + di, 0, 0), dE * W[0][di] / vol);
                }
#else  // WARPX_DIM_1D_Z
                for (int di = 0; di < 2; ++di) {
                    amrex::Real vol;
                    if (is_per[0]) {
                        vol = dxn[0];
                    } else {
                        amrex::Real const zn = plo[0] + (ii + di)*dxn[0];
                        vol = amrex::min(zn + 0.5_rt*dxn[0], phi[0])
                            - amrex::max(zn - 0.5_rt*dxn[0], plo[0]);
                    }
                    amrex::Gpu::Atomic::AddNoRet(
                        &st_arr(ii + di, 0, 0), dE * W[0][di] / vol);
                }
#endif
            });
        }
    }
}
