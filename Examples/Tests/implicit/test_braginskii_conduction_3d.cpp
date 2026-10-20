/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/BraginskiiConductivity.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcConductionReport.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>

using Real = amrex::Real;
using Vector3 = amrex::GpuArray<Real, 3>;

AMREX_GPU_HOST_DEVICE Vector3
magnetic_field (Vector3 const& x, int mode) {
    if (mode == 0) {
        return {0, 0, 0};
    }
    if (mode == 1) {
        return {0.00002, 0.00006, 0.000075};
    }
    if (mode == 2) {
        return {0.001 * (x[0] - .5), 0.001 * (x[1] - .5), -0.002 * (x[2] - .5)};
    }
    if (mode == 3) {
        return {.04, .06, .07};
    }
    return {2, 3, 5};
}

// Direct rational Z=1 conductivity and analytic temperature gradients.
// This oracle does not call the production conductivity or spatial operator.
AMREX_GPU_HOST_DEVICE Vector3
continuum_flux (Vector3 const& x, int mode, Real density, Real base_temperature,
                Real amplitude) {
    Real const k = 2 * MathConst::pi;
    Real const c = std::cos(k * x[0]) * std::cos(k * x[1]) * std::cos(k * x[2]);
    Real const temperature = base_temperature * (1 + amplitude * c);
    Vector3 gradient{};
    for (int d = 0; d < 3; ++d) {
        gradient[d] = -base_temperature * amplitude * k * std::sin(k * x[d]) *
                      std::cos(k * x[(d + 1) % 3]) *
                      std::cos(k * x[(d + 2) % 3]) * PhysConst::q_e /
                      PhysConst::kb;
    }
    auto const b = magnetic_field(x, mode);
    Real b2 = 0, dot = 0;
    for (int d = 0; d < 3; ++d) {
        b2 += b[d] * b[d];
        dot += b[d] * gradient[d];
    }
    Real const tau =
        3.44e5 * std::pow(temperature, 1.5) / (density * 1.e-6 * 10.0);
    Real const hall2 = std::pow(PhysConst::q_e * tau / PhysConst::m_e, 2) * b2;
    Real const scale = density * PhysConst::kb * PhysConst::q_e * temperature *
                       tau / PhysConst::m_e;
    Real const perpendicular = scale * (4.664 * hall2 + 11.92) /
                               (hall2 * hall2 + 14.79 * hall2 + 3.7703);
    Real const parallel =
        amrex::min(scale * 11.92 / 3.7703, 1.e5 * perpendicular);
    Real const difference = b2 > 0 ? (parallel - perpendicular) / b2 : 0;
    Vector3 flux{};
    for (int d = 0; d < 3; ++d) {
        flux[d] = perpendicular * gradient[d] + difference * b[d] * dot;
    }
    return flux;
}

AMREX_GPU_HOST_DEVICE Real
continuum_rhs (Vector3 const& x, int mode, Real density, Real capacity_density,
               Real temperature, Real amplitude) {
    Real constexpr h = 1.e-5;
    Real divergence = 0;
    for (int d = 0; d < 3; ++d) {
        auto plus = x, minus = x;
        plus[d] += h;
        minus[d] -= h;
        auto const fp =
            continuum_flux(plus, mode, density, temperature, amplitude);
        auto const fm =
            continuum_flux(minus, mode, density, temperature, amplitude);
        divergence += (fp[d] - fm[d]) / (2 * h);
    }
    return divergence / (1.5 * PhysConst::kb * capacity_density);
}

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using warpx::fields::FieldType;
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& hp = *sim.get_pointer_HybridPICModel();
        auto& te =
            *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const lo = geom.ProbLoArray();
        auto const box = amrex::surroundingNodes(geom.Domain());
        auto const dlo = box.smallEnd(), dhi = box.bigEnd();
        auto owner = amrex::OwnerMask(te, geom.periodicity());
        amrex::GpuArray<int, 3> periodic{};
        amrex::ParmParse pp("physical_test");
        int mode = 0, expect_hall_overflow = -1;
        bool evolve = false, guard_only = false, unfreeze = true;
        Real density = 3.e21, temperature = 1000, capfactor = 1, nped = 0;
        pp.query("mode", mode);
        pp.query("expect_hall_overflow", expect_hall_overflow);
        pp.query("evolve", evolve);
        pp.query("guard_only", guard_only);
        pp.query("unfreeze", unfreeze);
        pp.query("density", density);
        pp.query("temperature_eV", temperature);
        pp.query("capacity_factor", capfactor);
        pp.query("pedestal_density", nped);
        AMREX_ALWAYS_ASSERT(density >= 0 && (guard_only || density > 0) &&
                            temperature > 0 && capfactor > 0 && nped >= 0);
        AMREX_ALWAYS_ASSERT(hp.m_cond_model == 1 &&
                            hp.m_cond_fixed_coulomb_log == 10);
        AMREX_ALWAYS_ASSERT(hp.m_cond_max_anisotropy == 1.e5);
        hp.m_qdsmc_n_floor = std::max(Real(1.e14), density * capfactor);
        hp.m_qdsmc_halo_unfreeze = unfreeze;
        hp.m_cond_te_floor = 0;
        hp.m_cond_flux_limit_factor = 0;
        auto* pedestal = const_cast<amrex::MultiFab*>(hp.DensityPedestal(0));
        if (pedestal) {
            pedestal->setVal(nped * PhysConst::q_e);
        } else {
            AMREX_ALWAYS_ASSERT(nped == 0);
        }
        Real const capacity_density =
            std::max(density + nped, hp.m_qdsmc_n_floor);
        Real const amplitude = evolve ? .2 : .01;
        for (int d = 0; d < 3; ++d) {
            periodic[d] = geom.isPeriodic(d);
            for (int s = 0; s < 2; ++s) {
                hp.m_cond_bc[d][s] = 0;
            }
            auto& field = *sim.m_fields.get(FieldType::Bfield_fp,
                                            ablastr::fields::Direction{d}, 0);
            Vector3 offset{};
            for (int a = 0; a < 3; ++a) {
                offset[a] = field.ixType().nodeCentered(a) ? 0 : .5;
            }
            for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
                auto const f = field.array(mfi);
                amrex::ParallelFor(
                    mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Vector3 const x = {lo[0] + (i + offset[0]) * dx[0],
                                           lo[1] + (j + offset[1]) * dx[1],
                                           lo[2] + (k + offset[2]) * dx[2]};
                        f(i, j, k) = magnetic_field(x, mode)[d];
                    });
            }
        }
        rho.setVal(density * PhysConst::q_e);
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const t = te.array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real const wave =
                        std::cos(2 * MathConst::pi * (lo[0] + i * dx[0])) *
                        std::cos(2 * MathConst::pi * (lo[1] + j * dx[1])) *
                        std::cos(2 * MathConst::pi * (lo[2] + k * dx[2]));
                    t(i, j, k) = temperature * (1 + amplitude * wave) *
                                 PhysConst::q_e / PhysConst::kb;
                });
        }
        auto energy = [&] () {
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                auto const own = owner->const_array(mfi);
                op.eval(mfi.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            if (!own(i, j, k)) {
                                return {0};
                            }
                            int const node[3] = {i, j, k};
                            Real volume = dx[0] * dx[1] * dx[2];
                            for (int d = 0; d < 3; ++d) {
                                if (!periodic[d] &&
                                    (node[d] == dlo[d] || node[d] == dhi[d])) {
                                    volume *= .5;
                                }
                            }
                            return {1.5 * PhysConst::kb * capacity_density *
                                    volume * t(i, j, k)};
                        });
            }
            Real sum = amrex::get<0>(data.value(op));
            amrex::ParallelDescriptor::ReduceRealSum(sum);
            return sum;
        };
        amrex::MultiFab initial(te.boxArray(), te.DistributionMap(), 1,
                                te.nGrowVect());
        amrex::MultiFab::Copy(initial, te, 0, 0, 1, te.nGrowVect());
        Real bmin = 0;
        if (mode != 0 && mode != 2) {
            auto const b = magnetic_field({.5, .5, .5}, mode);
            bmin = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        }
        // Native coefficient used only to select a stable interval, never as
        // the analytic continuum oracle or the independently integrated energy.
        Real chi = 0;
        for (Real factor : {1 - amplitude, 1 + amplitude}) {
            auto const c = EvaluateBraginskiiConductivity(
                density, temperature * factor, bmin, 1.e5, 10);
            chi = std::max(chi, c.parallel_used /
                                    (1.5 * PhysConst::kb * capacity_density));
        }
        AMREX_ALWAYS_ASSERT(std::isfinite(chi) && (chi > 0 || guard_only));
        Real const dt =
            chi > 0 ? (evolve ? 4.0 : .005) * dx[0] * dx[0] / chi : 1.e-8;
        Real const e0 = energy();
        Real energy_error = 0;
        auto advance = [&] (Real interval) {
            QdsmcConductionReport report;
            bool const ok = hp.TryQdsmcConductionOnceFDAtState(
                0, interval, false, rho, 0, report);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ok && report.completed,
                                             report.failure);
            AMREX_ALWAYS_ASSERT(report.completed_time == interval &&
                                te.is_finite());
            if (expect_hall_overflow >= 0) {
                AMREX_ALWAYS_ASSERT(report.physical_coefficients_sampled &&
                                    report.coefficient_nodes > 0);
                bool const overflow = expect_hall_overflow != 0;
                AMREX_ALWAYS_ASSERT(report.hall_min_overflow == overflow &&
                                    report.hall_max_overflow == overflow);
                for (int n = 0; n < 6; ++n) {
                    AMREX_ALWAYS_ASSERT(std::isfinite(report.coefficient_min[n]) &&
                                        std::isfinite(report.coefficient_max[n]));
                }
                if (overflow) {
                    AMREX_ALWAYS_ASSERT(report.coefficient_min[5] ==
                                        std::numeric_limits<Real>::max() &&
                                        report.coefficient_max[5] ==
                                        std::numeric_limits<Real>::max());
                }
            }
            energy_error = std::max(energy_error, std::abs(energy() - e0) / e0);
            AMREX_ALWAYS_ASSERT(energy_error < 1.e-11);
        };
        if (guard_only) {
            hp.m_cond_fd_time = 2;
            advance(dt);
            amrex::MultiFab difference(te.boxArray(), te.DistributionMap(), 1,
                                       0);
            amrex::MultiFab::Copy(difference, te, 0, 0, 1, 0);
            amrex::MultiFab::Subtract(difference, initial, 0, 0, 1, 0);
            Real const change = difference.norminf();
            bool const inactive = density == 0 || chi == 0 ||
                                  (!unfreeze && pedestal == nullptr &&
                                   density <= hp.m_qdsmc_n_floor);
            AMREX_ALWAYS_ASSERT(!inactive || change == 0);
            amrex::Print() << std::setprecision(17)
                           << "BRAGINSKII_GUARD_RESULT {\"density\":" << density
                           << ",\"mode\":" << mode
                           << ",\"pedestal_density\":" << nped
                           << ",\"capacity_density\":" << capacity_density
                           << ",\"inactive\":" << (inactive ? "true" : "false")
                           << ",\"change_K\":" << change
                           << ",\"energy_relative\":" << energy_error << "}\n";
        } else if (!evolve) {
            hp.m_cond_fd_time = 0;
            hp.m_cond_fd_rtol = 1.e-3;
            hp.m_cond_fd_atol = 1.e-12;
            std::array<amrex::MultiFab, 3> states;
            for (int q = 0; q < 3; ++q) {
                amrex::MultiFab::Copy(te, initial, 0, 0, 1, te.nGrowVect());
                advance(dt / std::pow(2.0, q));
                states[q].define(te.boxArray(), te.DistributionMap(), 1, 0);
                amrex::MultiFab::Copy(states[q], te, 0, 0, 1, 0);
            }
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpSum>
                op;
            amrex::ReduceData<Real, Real, Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t0 = initial.const_array(mfi),
                           t1 = states[0].const_array(mfi);
                auto const t2 = states[1].const_array(mfi),
                           t4 = states[2].const_array(mfi);
                auto const own = owner->const_array(mfi);
                op.eval(
                    mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        Vector3 const x = {lo[0] + i * dx[0], lo[1] + j * dx[1],
                                           lo[2] + k * dx[2]};
                        if (!own(i, j, k)) {
                            return {0, 0, 0};
                        }
                        for (int d = 0; d < 3; ++d) {
                            if (x[d] < .2 || x[d] > .8) {
                                return {0, 0, 0};
                            }
                        }
                        Real const exact =
                            continuum_rhs(x, mode, density, capacity_density,
                                          temperature, amplitude);
                        Real const coarse = (4 * (t2(i, j, k) - t0(i, j, k)) -
                                             (t1(i, j, k) - t0(i, j, k))) /
                                            dt;
                        Real const fine = (4 * (t4(i, j, k) - t0(i, j, k)) -
                                           (t2(i, j, k) - t0(i, j, k))) /
                                          (.5 * dt);
                        return {(fine - exact) * (fine - exact), exact * exact,
                                (fine - coarse) * (fine - coarse)};
                    });
            }
            auto const v = data.value(op);
            Real sums[3] = {amrex::get<0>(v), amrex::get<1>(v),
                            amrex::get<2>(v)};
            amrex::ParallelDescriptor::ReduceRealSum(sums, 3);
            Real const error = std::sqrt(sums[0] / sums[1]),
                       sensitivity = std::sqrt(sums[2] / sums[1]);
            amrex::Print() << std::setprecision(17)
                           << "BRAGINSKII_3D_RESULT {\"mode\":" << mode
                           << ",\"density\":" << density
                           << ",\"temperature_eV\":" << temperature
                           << ",\"relative_L2\":" << error
                           << ",\"temporal_sensitivity\":" << sensitivity
                           << ",\"energy_relative\":" << energy_error << "}\n";
            AMREX_ALWAYS_ASSERT(error < .03 && sensitivity < .03 * error);
        } else {
            AMREX_ALWAYS_ASSERT(mode != 2);
            int reference_steps = 8192;
            pp.query("reference_steps", reference_steps);
            AMREX_ALWAYS_ASSERT(reference_steps >= 2048 &&
                                reference_steps % 2 == 0);
            std::array<int, 5> const counts = {32, 64, 128, reference_steps / 2,
                                               reference_steps};
            std::array<amrex::MultiFab, 5> states;
            for (int q = 0; q < 5; ++q) {
                hp.m_cond_fd_time = q < 3 ? 2 : 0;
                amrex::MultiFab::Copy(te, initial, 0, 0, 1, te.nGrowVect());
                for (int step = 0; step < counts[q]; ++step) {
                    advance(dt / counts[q]);
                }
                states[q].define(te.boxArray(), te.DistributionMap(), 1, 0);
                amrex::MultiFab::Copy(states[q], te, 0, 0, 1, 0);
            }
            Real const scale = temperature * PhysConst::q_e / PhysConst::kb;
            std::array<Real, 4> errors{};
            for (int q = 0; q < 4; ++q) {
                amrex::MultiFab::Subtract(states[q], states[4], 0, 0, 1, 0);
                errors[q] = states[q].norminf() / scale;
            }
            amrex::Print() << std::setprecision(17)
                           << "BRAGINSKII_RKL_RESULT {\"mode\":" << mode
                           << ",\"dt\":" << dt << ",\"rkl32\":" << errors[0]
                           << ",\"rkl64\":" << errors[1]
                           << ",\"rkl128\":" << errors[2]
                           << ",\"reference_steps\":" << reference_steps
                           << ",\"reference_half_full\":" << errors[3]
                           << ",\"energy_relative\":" << energy_error << "}\n";
            AMREX_ALWAYS_ASSERT(errors[3] <= 1.e-8 && errors[2] <= 1.e-6 &&
                                errors[2] < errors[1] && errors[1] < errors[0]);
        }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
