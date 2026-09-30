/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>

#include <array>
#include <cmath>
#include <iomanip>
#include <string>

/** Probe the actual native advance, without a copied stencil or a production
 * test hook. For a linear operator, a single SSPRK2 step is quadratic in dt:
 * [4 (T(dt/2)-T0) - (T(dt)-T0)]/dt is its spatial RHS to roundoff. Repeat at
 * half dt and report the difference; the runner checks each native call used
 * one accepted step. For nonlinear coefficients this is a temporal extrapolate,
 * so its temporal sensitivity is reported separately from spatial error.
 */
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& hp = *sim.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& te =
            *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const plo = geom.ProbLoArray();
        auto const volume = MakeQdsmcVolumeElement(geom, te.ixType());
        int const nr = geom.Domain().length(0);
        int const nz = geom.Domain().length(AMREX_SPACEDIM - 1);
        amrex::ParmParse pp("conduction_test");
        std::string profile = "axial";
        pp.query("profile", profile);
        int kind = profile == "axial"    ? 0
                   : profile == "r2"     ? 1
                   : profile == "r4"     ? 2
                   : profile == "closed" ? 3
                                         : 4;
        amrex::Real density_slope = 0.0;
        amrex::Real limiter = 0.0;
        amrex::Real perpendicular = 1.0;
        amrex::Real beta = 0.0;
        pp.query("density_slope", density_slope);
        pp.query("limiter", limiter);
        pp.query("perpendicular", perpendicular);
        pp.query("beta", beta);
        // These are a spatial-operator fixture, not campaign settings.
        hp.m_cond_fd_time = 0;
        hp.m_cond_flux_limit_factor = limiter;
        hp.m_cond_te_floor = 0.0;
        hp.m_cond_fd_rtol = 1.e-3;
        hp.m_cond_fd_atol = 1.e-12;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            for (int side = 0; side < 2; ++side) {
                hp.m_cond_bc[d][side] = 0;
            }
        }
        // A regular tilted field: Br = beta*r, Bz = 1. Linear staggered
        // interpolation is exact here, isolating the conduction stencil.
        for (int d = 0; d < 3; ++d) {
            auto& field = *sim.m_fields.get(FieldType::Bfield_fp,
                                            ablastr::fields::Direction{d}, 0);
            amrex::Real const offset =
                field.ixType().nodeCentered(0) ? 0.0 : 0.5;
            for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
                auto const f = field.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    f(i, j, k) = d == 0 ? beta * (plo[0] + (i + offset) * dx[0])
                                 : d == 2 ? 1.0
                                          : 0.0;
                });
            }
        }
        amrex::Real const pi = std::acos(-1.0);
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const t = te.array(mfi);
            auto const n = rho.array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    amrex::Real const r = plo[0] + i * dx[0];
                    int const node[3] = {i, j, k};
                    amrex::Real const z =
                        plo[AMREX_SPACEDIM - 1] +
                        node[AMREX_SPACEDIM - 1] * dx[AMREX_SPACEDIM - 1];
                    amrex::Real const a = (1.0 - r * r) * (1.0 - r * r);
                    t(i, j, k) =
                        2.0 + (kind == 0   ? std::cos(2 * pi * z)
                               : kind == 1 ? r * r
                               : kind == 2 ? r * r * r * r
                               : kind == 3
                                   ? a
                                   : a * (1.0 + 0.1 * std::cos(2 * pi * z)));
                });
            // rho can have a different ghost width from Te.
            amrex::ParallelFor(rho[mfi].box(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   amrex::Real const r = plo[0] + i * dx[0];
                                   n(i, j, k) = 2.e18 * PhysConst::q_e *
                                                (1.0 + density_slope * r * r);
                               });
        }
        auto energy = [&] () {
            auto const e = hp.QDSMCClassEnergy(0, &rho);
            return e[0] + e[1];
        };
        amrex::MultiFab initial(te.boxArray(), te.DistributionMap(), 1,
                                te.nGrowVect());
        amrex::MultiFab::Copy(initial, te, 0, 0, 1, te.nGrowVect());
        std::array<amrex::MultiFab, 3> evolved;
        amrex::Real min_dx = dx[0];
        for (int d = 1; d < AMREX_SPACEDIM; ++d) {
            min_dx = amrex::min(min_dx, dx[d]);
        }
        amrex::Real const dt = 1.e-3 * min_dx * min_dx;
        amrex::Real max_energy_relative = 0.0;
        auto const e0 = energy();
        for (int step = 0; step < 3; ++step) {
            amrex::MultiFab::Copy(te, initial, 0, 0, 1, te.nGrowVect());
            hp.QdsmcConductionOnceFDAtState(0, dt / std::pow(2.0, step), false,
                                            rho, 0.0);
            AMREX_ALWAYS_ASSERT(te.is_finite());
            max_energy_relative =
                amrex::max(max_energy_relative, std::abs(energy() - e0) / e0);
            evolved[step].define(te.boxArray(), te.DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(evolved[step], te, 0, 0, 1, 0);
        }
        // Optional native-state evidence for deterministic unchanged-path checks.
        // Output occurs only after all three advances, with no solver mutation.
        bool write_fields = false;
        pp.query("write_fields", write_fields);
        if (write_fields) {
            amrex::VisMF::Write(initial, "T0");
            for (int step = 0; step < 3; ++step) {
                amrex::VisMF::Write(evolved[step], "T" + std::to_string(step + 1));
            }
        }
        auto owner = te.OwnerMask(geom.periodicity());
        amrex::ReduceOps<
            amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax,
            amrex::ReduceOpMax, amrex::ReduceOpSum, amrex::ReduceOpSum,
            amrex::ReduceOpMax, amrex::ReduceOpSum, amrex::ReduceOpSum>
            op;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                          amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                          amrex::Real>
            data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const t0 = initial.const_array(mfi),
                       t1 = evolved[0].const_array(mfi),
                       t2 = evolved[1].const_array(mfi),
                       t4 = evolved[2].const_array(mfi);
            auto const own = owner->const_array(mfi);
            op.eval(
                mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                    int const node[3] = {i, j, k};
                    amrex::Real const r = plo[0] + i * dx[0];
                    amrex::Real const z =
                        plo[AMREX_SPACEDIM - 1] +
                        node[AMREX_SPACEDIM - 1] * dx[AMREX_SPACEDIM - 1];
                    amrex::Real const c = std::cos(2 * pi * z),
                                      wave = 4 * pi * pi;
                    amrex::Real const a = (1 - r * r) * (1 - r * r),
                                      ar = -4 * r + 4 * r * r * r;
                    amrex::Real radial = kind == 0   ? 0.0
                                         : kind == 1 ? 4.0
                                         : kind == 2 ? 16 * r * r
                                         : kind == 3
                                             ? 16 * r * r - 8
                                             : (16 * r * r - 8) * (1 + 0.1 * c);
                    amrex::Real const tzz = kind == 0   ? -wave * c
                                            : kind == 4 ? -0.1 * wave * a * c
                                                        : 0.0;
                    amrex::Real const tz =
                        kind == 0   ? -2 * pi * std::sin(2 * pi * z)
                        : kind == 4 ? -0.2 * pi * a * std::sin(2 * pi * z)
                                    : 0.0;
                    amrex::Real const trz =
                        kind == 4 ? -0.2 * pi * ar * std::sin(2 * pi * z) : 0.0;
                    amrex::Real const tr = kind == 0   ? 0.0
                                           : kind == 1 ? 2 * r
                                           : kind == 2 ? 4 * r * r * r
                                           : kind == 3 ? ar
                                                       : ar * (1 + 0.1 * c);
#if defined(WARPX_DIM_3D)
                    radial -= kind == 0   ? 0.0
                              : kind == 1 ? 2.0
                              : kind == 2 ? 4 * r * r
                              : kind == 3 ? -4 + 4 * r * r
                                          : (-4 + 4 * r * r) * (1 + 0.1 * c);
#endif
                    amrex::Real const denom = 1 + beta * beta * r * r;
                    amrex::Real const xrr =
                        (perpendicular + beta * beta * r * r) / denom;
                    amrex::Real const xzz =
                        (1 + perpendicular * beta * beta * r * r) / denom;
                    amrex::Real const xrz =
                        (1 - perpendicular) * beta * r / denom;
                    amrex::Real const dxrr = 2 * beta * beta * r *
                                             (1 - perpendicular) /
                                             (denom * denom);
                    amrex::Real const radial_xrz =
#if defined(WARPX_DIM_RZ)
                        (1 - perpendicular) * beta / denom +
#endif
                        (1 - perpendicular) * beta * (1 - beta * beta * r * r) /
                            (denom * denom);
                    amrex::Real exact = xrr * radial + dxrr * tr +
                                        2 * xrz * trz + radial_xrz * tz +
                                        xzz * tzz;
                    exact += 2 * density_slope * r /
                             (1 + density_slope * r * r) *
                             (xrr * tr + xrz * tz);
                    amrex::Real const rhs = (4 * (t2(i, j, k) - t0(i, j, k)) -
                                             (t1(i, j, k) - t0(i, j, k))) /
                                            dt;
                    amrex::Real const rhs_half =
                        (4 * (t4(i, j, k) - t0(i, j, k)) -
                         (t2(i, j, k) - t0(i, j, k))) /
                        (0.5 * dt);
                    bool const regular = kind == 0 || i <= nr - 4;
                    amrex::Real const error =
                        regular ? std::abs(rhs - exact) : 0.0;
                    amrex::Real const w = own(i, j, k) ? volume(i, j, k) : 0.0;
                    amrex::Real const wr = regular ? w : 0.0;
                    return {i == 0 ? error : 0.0,
                            i == 1 ? error : 0.0,
                            i == 2 ? error : 0.0,
                            error,
                            wr * error * error,
                            wr,
                            std::abs(rhs - rhs_half),
                            w * (1 + density_slope * r * r) * rhs,
                            w * (1 + density_slope * r * r) * exact};
                });
        }
        auto const v = data.value(op);
        amrex::Real maxima[5] = {amrex::get<0>(v), amrex::get<1>(v),
                                 amrex::get<2>(v), amrex::get<3>(v),
                                 amrex::get<6>(v)};
        amrex::Real sums[4] = {amrex::get<4>(v), amrex::get<5>(v),
                               amrex::get<7>(v), amrex::get<8>(v)};
        amrex::ParallelDescriptor::ReduceRealMax(maxima, 5);
        amrex::ParallelDescriptor::ReduceRealSum(sums, 4);
        AMREX_ALWAYS_ASSERT(max_energy_relative < 1.e-11);
        amrex::Print() << std::setprecision(17)
                       << "CONDUCTION_ORDER_RESULT {\"N\":" << nz
                       << ",\"Nr\":" << nr << ",\"Nz\":" << nz
                       << ",\"profile\":\"" << profile
                       << "\",\"density_slope\":" << density_slope
                       << ",\"beta\":" << beta
                       << ",\"perpendicular\":" << perpendicular
                       << ",\"limiter\":" << limiter << ",\"dt\":" << dt
                       << ",\"axis_error\":" << maxima[0]
                       << ",\"row1_error\":" << maxima[1]
                       << ",\"row2_error\":" << maxima[2]
                       << ",\"linf_error\":" << maxima[3]
                       << ",\"weighted_l2_error\":"
                       << std::sqrt(sums[0] / sums[1])
                       << ",\"temporal_sensitivity\":" << maxima[4]
                       << ",\"rhs_energy_sum\":" << sums[2]
                       << ",\"exact_point_energy_sum\":" << sums[3]
                       << ",\"energy_relative\":" << max_energy_relative
                       << ",\"min_temperature\":" << te.min(0)
                       << ",\"max_temperature\":" << te.max(0) << "}\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
