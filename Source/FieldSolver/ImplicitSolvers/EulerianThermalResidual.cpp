/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "EulerianThermalStage.H"
#include "ThermalCurrentRemainder.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalConduction.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalConduction3D.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalWall.H"
#include "Utils/WarpXConst.H"
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "EulerianThermalStageUtils.H"

namespace warpx::thermal {
using amrex::Real;
using stage_detail::check_layout;
using stage_detail::fill_images;
using stage_detail::finite;

bool
EulerianThermalStage::Residual (const amrex::MultiFab& stage,
                                amrex::MultiFab& residual) {
    if(!InvalidateEvaluation())return false;
    m_remainder_ready=false;
    check_layout(residual, m_old, 1);
    if (!Admissible(stage)) {
        amrex::Print() << "Thermal state inadmissible\n";
        return false;
    }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
    amrex::MultiFab::Copy(m_stage, stage, 0, 0, 1, 0);
    fill_images(m_stage, m_geometry);
    m_source_derivative.setVal(0);
    if (m_source_evaluator) {
        m_live_source.setVal(0);
        if (!m_source_evaluator(m_stage, m_density, m_magnetic, m_live_source,
                                m_source_derivative) ||
            !finite(m_live_source) || !finite(m_source_derivative)) {
            amrex::Print() << "Thermal source callback rejected\n";
            return false;
        }
    } else {
        amrex::MultiFab::Copy(m_live_source, m_source, 0, 0, 1, 0);
    }
    auto const parallel = m_parallel;
    auto const perpendicular = m_perpendicular;
    auto const gamma_minus_one = m_options.gamma - 1;
    auto const time = m_options.time;
    auto const conductivity_options = m_options.conductivity;
    for (amrex::MFIter mfi(m_kappa); mfi.isValid(); ++mfi) {
        auto const u = m_stage.const_array(mfi);
        auto const n = m_density.const_array(mfi);
        auto const kap = m_kappa.array(mfi);
        auto const b = m_magnetic.const_array(mfi);
        auto const active = m_conduction_active.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            if (active(i, j, k) == 0) {
                kap(i, j, k, 0) = 0;
                kap(i, j, k, 1) = 0;
                return; // do not evaluate dormant material laws
            }
            Real const te =
                gamma_minus_one * u(i, j, k) / (n(i, j, k) * PhysConst::q_e);
            Real const b2 = b(i, j, k, 0) * b(i, j, k, 0) +
                            b(i, j, k, 1) * b(i, j, k, 1) +
                            b(i, j, k, 2) * b(i, j, k, 2);
            auto const value = EvaluateThermalConductivity(
                n(i, j, k), te, time, b2, conductivity_options, parallel,
                perpendicular);
            kap(i, j, k, 0) = value.parallel;
            kap(i, j, k, 1) = value.perpendicular;
        });
    }
    if (!finite(m_kappa) || m_kappa.min(0) < 0 || m_kappa.min(1) < 0) {
        amrex::Print() << "Thermal conductivity law rejected\n";
        return false;
    }
    fill_images(m_kappa, m_geometry);
    auto const dx = m_geometry.CellSizeArray();
#if defined(WARPX_DIM_RZ)
    auto const rlo = m_geometry.ProbLo(0);
#endif
    auto const domain = m_geometry.Domain();
    auto const theta = m_options.theta;
    auto const theta_c = m_options.conduction_theta;
    auto const weight = theta_c / theta;
    auto const flux_limit = m_options.free_streaming_fraction;
    auto const temperature_floor = m_options.temperature_floor_ev;
    auto const bfloor2 = m_options.magnetic_floor * m_options.magnetic_floor;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
#if defined(WARPX_DIM_RZ)
        mhd::ChaconFDParams p;
        p.normal = d == 0 ? 0 : 2;
        p.tangential = d == 0 ? 2 : 0;
        p.order = m_options.order;
        p.cross_mode = m_options.cross_mode;
        p.radial_faces = d == 0;
        p.periodic_normal = m_geometry.isPeriodic(d);
        p.periodic_tangential = m_geometry.isPeriodic(1 - d);
        p.domain_lo_normal = domain.smallEnd(d);
        p.domain_hi_normal = domain.bigEnd(d);
        p.domain_lo_tangential = domain.smallEnd(1 - d);
        p.domain_hi_tangential = domain.bigEnd(1 - d);
        p.radial_lower = rlo;
        p.radial_cell_size = dx[0];
        p.inverse_normal_size = 1 / dx[d];
        p.inverse_tangential_size = 1 / dx[1 - d];
#else
        mhd::CartesianFDParams p;
        p.normal = d;
        p.order = m_options.order;
        p.cross_mode = m_options.cross_mode;
        p.periodic_normal = m_geometry.isPeriodic(d);
        p.domain_lo_normal = domain.smallEnd(d);
        p.domain_hi_normal = domain.bigEnd(d);
        for (int axis = 0; axis < 3; ++axis) {
            p.periodic[axis] = m_geometry.isPeriodic(axis);
            p.low[axis] = domain.smallEnd(axis);
            p.high[axis] = domain.bigEnd(axis);
            p.inverse_size[axis] = 1 / dx[axis];
        }
#endif
        p.limiter_width = m_options.limiter_width;
        p.b2_floor = bfloor2;
        auto const low = m_options.boundary[d][0];
        auto const high = m_options.boundary[d][1];
        for (amrex::MFIter mfi(*m_flux[d]); mfi.isValid(); ++mfi) {
            auto const u = m_stage.const_array(mfi);
            auto const old = m_old.const_array(mfi);
            auto const old_n = m_old_density.const_array(mfi);
            auto const n = m_density.const_array(mfi);
            auto const b = m_magnetic.const_array(mfi);
            auto const kap = m_kappa.const_array(mfi);
            auto const active = m_conduction_active.const_array(mfi);
            auto const flux = m_flux[d]->array(mfi);
            auto const coeff = m_coefficient[d]->array(mfi);
            auto const wall_derivative = m_wall_derivative[d]->array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                flux(i, j, k) = 0;
                coeff(i, j, k) = 0;
                wall_derivative(i, j, k) = 0;
#if defined(WARPX_DIM_RZ)
                if (d == 0 && rlo + i * dx[0] == 0) {
                    return;
                }
#endif
                int il = i, jl = j, kl = k;
                if (d == 0) {
                    --il;
                } else if (d == 1) {
                    --jl;
                } else {
                    --kl;
                }
                if (active(il, jl, kl) == 0 || active(i, j, k) == 0) {
                    return;
                }
                int const index = d == 0 ? i : (d == 1 ? j : k);
                bool const at_low =
                    !p.periodic_normal && index == p.domain_lo_normal;
                bool const at_high =
                    !p.periodic_normal && index == p.domain_hi_normal + 1;
                auto const e = [=] (int ii, int jj, int kk) {
                    return u(ii, jj, kk) / n(ii, jj, kk);
                };
                auto const en = [=] (int ii, int jj, int kk) {
                    return old(ii, jj, kk) / old_n(ii, jj, kk);
                };
#if defined(WARPX_DIM_RZ)
                auto const masked = [=] (int ii, int jj, int kk) {
                    amrex::ignore_unused(kk);
                    return active(ii, jj, kk) == 0 ||
                           (!p.periodic_normal &&
                            (mhd::index_along(ii, jj, kk, p.normal) <
                                 p.domain_lo_normal ||
                             mhd::index_along(ii, jj, kk, p.normal) >
                                 p.domain_hi_normal)) ||
                           (!p.periodic_tangential &&
                            (mhd::index_along(ii, jj, kk, p.tangential) <
                                 p.domain_lo_tangential ||
                             mhd::index_along(ii, jj, kk, p.tangential) >
                                 p.domain_hi_tangential));
                };
#else
                auto const masked=[=](int ii,int jj,int kk) {
                    int const cell[3]={ii,jj,kk};
                    for(int axis=0;axis<3;++axis) {
                        if(!p.periodic[axis]&&(cell[axis]<p.low[axis]||cell[axis]>p.high[axis])){return true;}
                    }
                    return active(ii,jj,kk)==0;
                };
#endif
                Real nf, bracket, old_bracket, chi_nn, cap_energy;
                if (at_low || at_high) {
                    auto const bc = at_low ? low : high;
                    if (bc.kind == BoundaryKind::Adiabatic) {
                        return;
                    }
                    Real const sign = at_low ? -1 : 1;
                    if (bc.kind == BoundaryKind::PrescribedFlux) {
                        flux(i, j, k) = sign * bc.value;
                        return;
                    }
                    int const ic = at_low ? i : il;
                    int const jc = at_low ? j : jl;
                    int const kc = at_low ? k : kl;
                    nf = n(ic, jc, kc);
                    if (bc.kind == BoundaryKind::Leg) {
                        Real const te = gamma_minus_one / PhysConst::q_e *
                                        (weight * e(ic, jc, kc) +
                                         (1 - weight) * en(ic, jc, kc));
                        LegFaceParameters const wall{bc.value,
                                                     bc.leg_length,
                                                     0.5 * dx[d],
                                                     bc.flux_limit,
                                                     bc.cap_form ==
                                                         WallCapForm::Sonic,
                                                     gamma_minus_one + 1,
                                                     bc.ion_mass,
                                                     temperature_floor};
                        flux(i, j, k) =
                            sign *
                            EvaluateLegFace(te, nf, time, wall, parallel).flux;
                        wall_derivative(i, j, k) =
                            LegFaceTemperatureDerivative(te, nf, time, wall,
                                                         parallel) *
                            weight * gamma_minus_one / PhysConst::q_e;
                        return;
                    }
                    Real const b2 = mhd::smooth_positive_floor(
                        b(ic, jc, kc, 0) * b(ic, jc, kc, 0) +
                            b(ic, jc, kc, 1) * b(ic, jc, kc, 1) +
                            b(ic, jc, kc, 2) * b(ic, jc, kc, 2),
                        bfloor2);
                    if (bc.flux_limit > 0 || bc.drain_only) {
                        Real const raw_b2 =
                            b(ic, jc, kc, 0) * b(ic, jc, kc, 0) +
                            b(ic, jc, kc, 1) * b(ic, jc, kc, 1) +
                            b(ic, jc, kc, 2) * b(ic, jc, kc, 2);
                        ThermalConductivityComponent const wall_parallel{
                            parallel, perpendicular, conductivity_options,
                            raw_b2, 0};
                        ThermalConductivityComponent const wall_perpendicular{
                            parallel, perpendicular, conductivity_options,
                            raw_b2, 1};
                        Real const te = gamma_minus_one / PhysConst::q_e *
                                        (weight * e(ic, jc, kc) +
                                         (1 - weight) * en(ic, jc, kc));
                        LegFaceParameters const wall{bc.value,
                                                     bc.leg_length,
                                                     0.5 * dx[d],
                                                     bc.flux_limit,
                                                     bc.cap_form ==
                                                         WallCapForm::Sonic,
                                                     gamma_minus_one + 1,
                                                     bc.ion_mass,
                                                     temperature_floor};
                        Real const projection = b(ic, jc, kc, p.normal) *
                                                b(ic, jc, kc, p.normal) / b2;
                        Real const delta =
                            6.e-6 * amrex::max(te, temperature_floor);
                        flux(i, j, k) =
                            sign * EvaluateReservoirFace(
                                       te, nf, time, wall, projection,
                                       bc.drain_only, wall_parallel,
                                       wall_perpendicular);
                        wall_derivative(i, j, k) =
                            (EvaluateReservoirFace(te + delta, nf, time, wall,
                                                   projection, bc.drain_only,
                                                   wall_parallel,
                                                   wall_perpendicular) -
                             EvaluateReservoirFace(te - delta, nf, time, wall,
                                                   projection, bc.drain_only,
                                                   wall_parallel,
                                                   wall_perpendicular)) /
                            (2 * delta) * weight * gamma_minus_one /
                            PhysConst::q_e;
                        return;
                    }
                    Real const knn = kap(ic, jc, kc, 1) +
                                     (kap(ic, jc, kc, 0) - kap(ic, jc, kc, 1)) *
                                         b(ic, jc, kc, p.normal) *
                                         b(ic, jc, kc, p.normal) / b2;
                    chi_nn = gamma_minus_one * knn / (nf * PhysConst::kb);
                    Real const bath_e =
                        PhysConst::q_e * bc.value / gamma_minus_one;
                    // q_out = 2 n chi (e_c-e_bath)/dx, cap once after mixing.
                    bracket =
                        -sign * 2 * chi_nn * (e(ic, jc, kc) - bath_e) / dx[d];
                    old_bracket =
                        -sign * 2 * chi_nn * (en(ic, jc, kc) - bath_e) / dx[d];
                    cap_energy =
                        weight * e(ic, jc, kc) + (1 - weight) * en(ic, jc, kc);
                } else {
                    nf = 0.5 * (n(il, jl, kl) + n(i, j, k));
                    // Explicit physical adapter: arithmetic kappa at the face,
                    // chi=(gamma-1)kappa/(n_face kB), e=U/n. Thus a variable-n,
                    // constant-T state has identically zero flux.
                    Real const factor = gamma_minus_one / (nf * PhysConst::kb);
                    Real const kp =
                        factor * 0.5 * (kap(il, jl, kl, 0) + kap(i, j, k, 0));
                    Real const kt =
                        factor * 0.5 * (kap(il, jl, kl, 1) + kap(i, j, k, 1));
#if defined(WARPX_DIM_RZ)
                    bracket =
                        mhd::chacon_fd_face_flux(e, masked, b, il, jl, kl, i, j,
                                                 k, kt, kp - kt, p, chi_nn);
                    old_bracket = 0;
                    if (theta_c != theta) {
                        Real unused;
                        old_bracket = mhd::chacon_fd_face_flux(
                            en, masked, b, il, jl, kl, i, j, k, kt, kp - kt, p,
                            unused);
                    }
#else
                    bracket =
                        mhd::cartesian_chacon_fd_face_flux(e, masked, b, il, jl, kl, i, j,
                                                 k, kt, kp - kt, p, chi_nn);
                    old_bracket = 0;
                    if (theta_c != theta) {
                        Real unused;
                        old_bracket = mhd::cartesian_chacon_fd_face_flux(
                            en, masked, b, il, jl, kl, i, j, k, kt, kp - kt, p,
                            unused);
                    }
#endif
                    cap_energy =
                        0.5 * (weight * (e(il, jl, kl) + e(i, j, k)) +
                               (1 - weight) * (en(il, jl, kl) + en(i, j, k)));
                }
                Real const kbt = gamma_minus_one * cap_energy;
                if (flux_limit > 0 && !(std::isfinite(kbt) && kbt > 0)) {
                    flux(i, j, k) = std::numeric_limits<Real>::quiet_NaN();
                    return; // never silently disable a cap at negative mixed T
                }
                Real const q_limit = flux_limit > 0
                                         ? flux_limit * nf * kbt *
                                               std::sqrt(kbt / PhysConst::m_e)
                                         : 0;
                auto const face = mhd::ComposeConductionFlux(
                    bracket, old_bracket, theta, theta_c, nf, q_limit);
                flux(i, j, k) = face.flux;
                coeff(i, j, k) = nf * chi_nn * face.pc_stage_cap_factor;
            });
        }
        if (!finite(*m_flux[d]) || !finite(*m_wall_derivative[d]) ||
            !finite(*m_coefficient[d]) || m_coefficient[d]->min(0) < 0) {
            amrex::Print() << "Thermal flux rejected: direction=" << d
                           << " U_images=" << m_stage.is_finite(0, 1, 3)
                           << " old_images=" << m_old.is_finite(0, 1, 3)
                           << " n_images=" << m_density.is_finite(0, 1, 3)
                           << " old_n_images="
                           << m_old_density.is_finite(0, 1, 3)
                           << " B_images=" << m_magnetic.is_finite(0, 3, 3)
                           << " kappa_images=" << m_kappa.is_finite(0, 2, 3)
                           << " mask_images="
                           << m_conduction_active.is_finite(0, 1, 3)
                           << " q=" << finite(*m_flux[d])
                           << " dq=" << finite(*m_wall_derivative[d])
                           << " coefficient=" << finite(*m_coefficient[d])
                           << " coefficient_min=" << m_coefficient[d]->min(0)
                           << "\n";
            return false;
        }
    }
    auto const transport_parameters = m_options.transport;
    auto const enabled = m_transport;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        auto const periodic = m_geometry.isPeriodic(d);
        auto const low = m_options.boundary[d][0];
        auto const high = m_options.boundary[d][1];
        for (amrex::MFIter mfi(*m_advection[d]); mfi.isValid(); ++mfi) {
            auto const u = m_stage.const_array(mfi);
            auto const n = m_density.const_array(mfi);
            auto const velocity = m_velocity[d]->const_array(mfi);
            auto const speed = m_speed[d]->const_array(mfi);
            auto const adv = m_advection[d]->array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                adv(i, j, k) = 0;
                if (!enabled
#if defined(WARPX_DIM_RZ)
                    || (d == 0 && rlo + i * dx[0] == 0)
#endif
                ) {
                    return;
                }
                int const il = i - (d == 0), jl = j - (d == 1),
                          kl = k - (d == 2);
                int const index = d == 0 ? i : (d == 1 ? j : k);
                bool const at_low = !periodic && index == domain.smallEnd(d);
                bool const at_high = !periodic && index == domain.bigEnd(d) + 1;
                Real const v = velocity(i, j, k);
                if (v == 0 &&
                    (transport_parameters.central_dissipation_entropy == 0 ||
                     speed(i, j, k) == 0)) {
                    return;
                }
                Real ul = u(il, jl, kl), ur = u(i, j, k);
                if (at_low || at_high) {
                    auto const bc = at_low ? low : high;
                    int const ic = at_low ? i : il, jc = at_low ? j : jl,
                              kc = at_low ? k : kl;
                    Real const sign = at_low ? -1 : 1;
                    // Constant extrapolation on outflow. Inflow energy is a
                    // physical caller condition, independent of heat-flux BC.
                    if (sign * v < 0 && bc.inflow_temperature_ev < 0) {
                        adv(i, j, k) = std::numeric_limits<Real>::quiet_NaN();
                        return;
                    }
                    ul = sign * v < 0
                             ? n(ic, jc, kc) * PhysConst::q_e *
                                   bc.inflow_temperature_ev / gamma_minus_one
                             : u(ic, jc, kc);
                    ur = ul;
                } else if (transport_parameters.fluid_reconstruction != 0) {
                    int const ill = il - (d == 0), jll = jl - (d == 1),
                              kll = kl - (d == 2);
                    int const irr = i + (d == 0), jrr = j + (d == 1),
                              krr = k + (d == 2);
                    Real nl, nr, el, er;
                    mhd_transport::reconstruct_face_pair(
                        n(ill, jll, kll), n(il, jl, kl), n(i, j, k),
                        n(irr, jrr, krr), transport_parameters, nl, nr);
                    nl = mhd_transport::reconstruction_positive_floor(
                        nl, transport_parameters.density_floor);
                    nr = mhd_transport::reconstruction_positive_floor(
                        nr, transport_parameters.density_floor);
                    if (nl <= 0 || nr <= 0) {
                        adv(i, j, k) = std::numeric_limits<Real>::quiet_NaN();
                        return;
                    }
                    mhd_transport::reconstruct_face_pair(
                        u(ill, jll, kll) / n(ill, jll, kll), ul / n(il, jl, kl),
                        ur / n(i, j, k), u(irr, jrr, krr) / n(irr, jrr, krr),
                        transport_parameters, el, er);
                    // Number-density/specific-energy analogue of the MHD
                    // primitive reconstruction; no fixed-composition rho.
                    Real const efloor =
                        PhysConst::q_e * temperature_floor / gamma_minus_one;
                    ul = mhd_transport::reconstruction_positive_floor(
                        nl * el, nl * efloor);
                    ur = mhd_transport::reconstruction_positive_floor(
                        nr * er, nr * efloor);
                }
                auto const face = mhd_transport::CentralElectronFlux(
                    ul, ur, v, speed(i, j, k),
                    transport_parameters.central_dissipation_entropy);
                adv(i, j, k) = face.energy;
            });
        }
        if (!finite(*m_advection[d])) {
            amrex::Print() << "Thermal advection rejected: direction=" << d
                           << " velocity=" << m_velocity[d]->min(0) << "/"
                           << m_velocity[d]->max(0)
                           << " U_images=" << m_stage.is_finite(0, 1, 3)
                           << " n_images=" << m_density.is_finite(0, 1, 3)
                           << "\n";
            return false;
        }
    }
    auto const h = theta * m_options.dt;
    for (amrex::MFIter mfi(residual); mfi.isValid(); ++mfi) {
        auto const u = m_stage.const_array(mfi);
        auto const old = m_old.const_array(mfi);
        auto const source = m_live_source.const_array(mfi);
        auto const qr = m_flux[0]->const_array(mfi);
        auto const qz = m_flux[1]->const_array(mfi);
        auto const fr = m_advection[0]->const_array(mfi);
        auto const fz = m_advection[1]->const_array(mfi);
        auto const vr = m_velocity[0]->const_array(mfi);
        auto const vz = m_velocity[1]->const_array(mfi);
#if defined(WARPX_DIM_3D)
        auto const qt = m_flux[2]->const_array(mfi);
        auto const ft = m_advection[2]->const_array(mfi);
        auto const vt = m_velocity[2]->const_array(mfi);
#endif
        auto const out = residual.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
#if defined(WARPX_DIM_RZ)
            Real const r = rlo + (i + 0.5) * dx[0];
            Real const div = ((rlo + (i + 1) * dx[0]) * qr(i + 1, j, k) -
                              (rlo + i * dx[0]) * qr(i, j, k)) /
                                 (r * dx[0]) +
                             (qz(i, j + 1, k) - qz(i, j, k)) / dx[1];
            Real const div_adv = ((rlo + (i + 1) * dx[0]) * fr(i + 1, j, k) -
                                  (rlo + i * dx[0]) * fr(i, j, k)) /
                                     (r * dx[0]) +
                                 (fz(i, j + 1, k) - fz(i, j, k)) / dx[1];
            Real const div_u = ((rlo + (i + 1) * dx[0]) * vr(i + 1, j, k) -
                                (rlo + i * dx[0]) * vr(i, j, k)) /
                                   (r * dx[0]) +
                               (vz(i, j + 1, k) - vz(i, j, k)) / dx[1];
#else
            Real const div=(qr(i+1,j,k)-qr(i,j,k))/dx[0]+(qz(i,j+1,k)-qz(i,j,k))/dx[1]+(qt(i,j,k+1)-qt(i,j,k))/dx[2];
            Real const div_adv=(fr(i+1,j,k)-fr(i,j,k))/dx[0]+(fz(i,j+1,k)-fz(i,j,k))/dx[1]+(ft(i,j,k+1)-ft(i,j,k))/dx[2];
            Real const div_u=(vr(i+1,j,k)-vr(i,j,k))/dx[0]+(vz(i,j+1,k)-vz(i,j,k))/dx[1]+(vt(i,j,k+1)-vt(i,j,k))/dx[2];
#endif
            Real const pressure_work = -gamma_minus_one * u(i, j, k) * div_u;
            out(i, j, k) =
                u(i, j, k) - old(i, j, k) -
                h * (source(i, j, k) - div - div_adv + pressure_work);
        });
    }
    if(!finite(residual))return false;
    if(m_remainder_context && !CompleteResidualRemainder(residual))return false;
    return CompleteEvaluation();
#else
    return false;
#endif
}

bool EulerianThermalStage::CopyResidualRemainder(amrex::MultiFab& output) const {
    bool valid=m_remainder_context && m_remainder_ready && m_residual_remainder &&
        remainder::Layout(output,m_old) && !remainder::Overlap(output,*m_residual_remainder);
    if(!remainder::All(valid))return false;
    amrex::MultiFab::Copy(output,*m_residual_remainder,0,0,1,0);return true;
}
bool EulerianThermalStage::CompleteResidualRemainder(amrex::MultiFab const& represented) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
    bool valid=remainder::ArithmeticSupported() && m_remainder_context && m_transport &&
        m_options.transport.fluid_reconstruction==0 && m_options.transport.central_dissipation_entropy==0;
    if(!remainder::All(valid))return false;
    // This first capability does not silently drop a source or conductive term.
    if(m_live_source.norminf(0)!=0 || m_source_derivative.norminf(0)!=0 ||
       m_kappa.norminf(0)!=0 || m_kappa.norminf(1)!=0)return false;
    for(auto const& q:m_flux)if(q->norminf(0)!=0)return false;
    auto const dx=m_geometry.CellSizeArray();
    auto const domain=m_geometry.Domain();
    Real const gamma_minus_one=m_options.gamma-1;
    Real const temperature_floor=m_options.temperature_floor_ev;
#if defined(WARPX_DIM_RZ)
    Real const rlo=m_geometry.ProbLo(0);
#endif
    auto const transport_parameters = m_options.transport;
    auto const enabled = m_transport;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        auto const periodic = m_geometry.isPeriodic(d);
        auto const low = m_options.boundary[d][0];
        auto const high = m_options.boundary[d][1];
        for (amrex::MFIter mfi(*m_advection[d]); mfi.isValid(); ++mfi) {
            auto const u = m_stage.const_array(mfi);
            auto const n = m_density.const_array(mfi);
            auto const velocity = m_velocity[d]->const_array(mfi);
            auto const speed = m_speed[d]->const_array(mfi);
            auto const adv = m_advection_remainder[d]->array(mfi);
            auto const represented_flux = m_advection[d]->const_array(mfi);
            auto const small = m_velocity_remainder[d]->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                adv(i, j, k) = 0;
                if (!enabled
#if defined(WARPX_DIM_RZ)
                    || (d == 0 && rlo + i * dx[0] == 0)
#endif
                ) {
                    return;
                }
                int const il = i - (d == 0), jl = j - (d == 1),
                          kl = k - (d == 2);
                int const index = d == 0 ? i : (d == 1 ? j : k);
                bool const at_low = !periodic && index == domain.smallEnd(d);
                bool const at_high = !periodic && index == domain.bigEnd(d) + 1;
                Real const v = velocity(i, j, k);
                if (v == 0 && small(i,j,k)==0) {
                    return;
                }
                Real ul = u(il, jl, kl), ur = u(i, j, k);
                if ((at_low || at_high) && v==0 && small(i,j,k)!=0) {
                    adv(i,j,k)=std::numeric_limits<Real>::quiet_NaN();return;
                }
                if (at_low || at_high) {
                    auto const bc = at_low ? low : high;
                    int const ic = at_low ? i : il, jc = at_low ? j : jl,
                              kc = at_low ? k : kl;
                    Real const sign = at_low ? -1 : 1;
                    // Constant extrapolation on outflow. Inflow energy is a
                    // physical caller condition, independent of heat-flux BC.
                    if (sign * v < 0 && bc.inflow_temperature_ev < 0) {
                        adv(i, j, k) = std::numeric_limits<Real>::quiet_NaN();
                        return;
                    }
                    ul = sign * v < 0
                             ? n(ic, jc, kc) * PhysConst::q_e *
                                   bc.inflow_temperature_ev / gamma_minus_one
                             : u(ic, jc, kc);
                    ur = ul;
                } else if (transport_parameters.fluid_reconstruction != 0) {
                    int const ill = il - (d == 0), jll = jl - (d == 1),
                              kll = kl - (d == 2);
                    int const irr = i + (d == 0), jrr = j + (d == 1),
                              krr = k + (d == 2);
                    Real nl, nr, el, er;
                    mhd_transport::reconstruct_face_pair(
                        n(ill, jll, kll), n(il, jl, kl), n(i, j, k),
                        n(irr, jrr, krr), transport_parameters, nl, nr);
                    nl = mhd_transport::reconstruction_positive_floor(
                        nl, transport_parameters.density_floor);
                    nr = mhd_transport::reconstruction_positive_floor(
                        nr, transport_parameters.density_floor);
                    if (nl <= 0 || nr <= 0) {
                        adv(i, j, k) = std::numeric_limits<Real>::quiet_NaN();
                        return;
                    }
                    mhd_transport::reconstruct_face_pair(
                        u(ill, jll, kll) / n(ill, jll, kll), ul / n(il, jl, kl),
                        ur / n(i, j, k), u(irr, jrr, krr) / n(irr, jrr, krr),
                        transport_parameters, el, er);
                    // Number-density/specific-energy analogue of the MHD
                    // primitive reconstruction; no fixed-composition rho.
                    Real const efloor =
                        PhysConst::q_e * temperature_floor / gamma_minus_one;
                    ul = mhd_transport::reconstruction_positive_floor(
                        nl * el, nl * efloor);
                    ur = mhd_transport::reconstruction_positive_floor(
                        nr * er, nr * efloor);
                }
                namespace dd=warpx::ohm::compensated;
                auto const vpair=dd::Pair{v,small(i,j,k)};
                auto const flux=dd::Multiply(dd::Add(dd::Multiply(vpair,ul),dd::Multiply(vpair,ur)),.5);
                adv(i,j,k)=dd::WithHigh(flux,represented_flux(i,j,k)).lo;
            });
        }
        if(!finite(*m_advection_remainder[d]))return false;
    }

    Real const h=m_options.theta*m_options.dt;
    for(amrex::MFIter mfi(*m_residual_remainder);mfi.isValid();++mfi) {
        auto const u=m_stage.const_array(mfi),old=m_old.const_array(mfi),high=represented.const_array(mfi);
        auto const out=m_residual_remainder->array(mfi);
        amrex::GpuArray<amrex::Array4<Real const>,AMREX_SPACEDIM> v,vl,f,fl;
        for(int d=0;d<AMREX_SPACEDIM;++d) {
            v[d]=m_velocity[d]->const_array(mfi);vl[d]=m_velocity_remainder[d]->const_array(mfi);
            f[d]=m_advection[d]->const_array(mfi);fl[d]=m_advection_remainder[d]->const_array(mfi);
        }
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            namespace dd=warpx::thermal::remainder;
            dd::Pair divv{0.,0.},divf{0.,0.};
            for(int d=0;d<AMREX_SPACEDIM;++d) {
                int const ip=i+(d==0),jp=j+(d==1),kp=k+(d==2);
                dd::Pair av{v[d](ip,jp,kp),vl[d](ip,jp,kp)},bv{v[d](i,j,k),vl[d](i,j,k)};
                dd::Pair af{f[d](ip,jp,kp),fl[d](ip,jp,kp)},bf{f[d](i,j,k),fl[d](i,j,k)};
                Real denominator=dx[d];
#if defined(WARPX_DIM_RZ)
                if(d==0) {
                    Real const rp=rlo+(i+1)*dx[0],rm=rlo+i*dx[0],r=rlo+(i+.5)*dx[0];
                    av=dd::Multiply(av,rp);bv=dd::Multiply(bv,rm);
                    af=dd::Multiply(af,rp);bf=dd::Multiply(bf,rm);denominator=r*dx[0];
                }
#endif
                divv=dd::Add(divv,dd::Divide(dd::Add(av,dd::Negate(bv)),denominator));
                divf=dd::Add(divf,dd::Divide(dd::Add(af,dd::Negate(bf)),denominator));
            }
            auto const work=dd::Product(dd::Multiply({-gamma_minus_one,0.},u(i,j,k)),divv);
            auto const rhs=dd::Add(dd::Negate(divf),work);
            auto const value=dd::Add(dd::Sum(u(i,j,k),-old(i,j,k)),dd::Negate(dd::Multiply(rhs,h)));
            out(i,j,k)=dd::WithHigh(value,high(i,j,k)).lo;
        });
    }
    m_remainder_ready=finite(*m_residual_remainder);return m_remainder_ready;
#else
    return false;
#endif
}
} // namespace warpx::thermal
