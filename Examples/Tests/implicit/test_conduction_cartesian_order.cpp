/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
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

// Analytic constant-tensor Fourier mode with independent Cartesian energy.
// Three advances remove the SSPRK2 dt term and measure temporal sensitivity.
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using Real = amrex::Real;
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
        auto const hi = geom.ProbHiArray();
        auto const box = amrex::surroundingNodes(geom.Domain());
        auto const dlo = box.smallEnd();
        auto const dhi = box.bigEnd();
        auto const owner = amrex::OwnerMask(te, geom.periodicity());
        amrex::ParmParse pp("order_test");
        amrex::Vector<int> mode{0, 0, 1};
        amrex::Vector<Real> magnetic{0, 0, 1};
        Real perpendicular = 1.0, dt_factor = 1.e-3;
        pp.queryarr("mode", mode);
        pp.queryarr("magnetic", magnetic);
        pp.query("perpendicular", perpendicular);
        pp.query("dt_factor", dt_factor);
        AMREX_ALWAYS_ASSERT(mode.size() == 3 && magnetic.size() == 3);
        AMREX_ALWAYS_ASSERT(hp.DensityPedestal(0) == nullptr);
        hp.m_cond_fd_time = 0;
        hp.m_cond_flux_limit_factor = 0;
        hp.m_cond_te_floor = 0;
        hp.m_cond_fd_rtol = 1.e-3;
        hp.m_cond_fd_atol = 1.e-12;
        amrex::GpuArray<Real, 3> wave{};
        amrex::GpuArray<int, 3> periodic{};
        Real norm2 = 0, wave2 = 0, dot = 0, min_dx = dx[0];
        for (int d = 0; d < 3; ++d) {
            periodic[d] = geom.isPeriodic(d);
            AMREX_ALWAYS_ASSERT(mode[d] == 0 || periodic[d]);
            wave[d] = 2.0 * std::acos(-1.0) * mode[d] / (hi[d] - lo[d]);
            wave2 += wave[d] * wave[d];
            norm2 += magnetic[d] * magnetic[d];
            dot += magnetic[d] * wave[d];
            min_dx = std::min(min_dx, dx[d]);
            for (int s = 0; s < 2; ++s) {
                hp.m_cond_bc[d][s] = 0;
            }
            sim.m_fields
                .get(FieldType::Bfield_fp, ablastr::fields::Direction{d}, 0)
                ->setVal(magnetic[d]);
        }
        AMREX_ALWAYS_ASSERT(norm2 > 0 && wave2 > 0);
        Real const eigenvalue =
            perpendicular * wave2 + (1.0 - perpendicular) * dot * dot / norm2;
        Real const density = 2.e18;
        rho.setVal(density * PhysConst::q_e);
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const t = te.array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real const phase = wave[0] * (i - dlo[0]) * dx[0] +
                                       wave[1] * (j - dlo[1]) * dx[1] +
                                       wave[2] * (k - dlo[2]) * dx[2];
                    t(i, j, k) = 2.0 + std::cos(phase);
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
                                return {0.0};
                            }
                            int const node[3] = {i, j, k};
                            Real volume = dx[0] * dx[1] * dx[2];
                            for (int d = 0; d < 3; ++d) {
                                if (!periodic[d] &&
                                    (node[d] == dlo[d] || node[d] == dhi[d])) {
                                    volume *= 0.5;
                                }
                            }
                            return {1.5 * PhysConst::kb * density * volume *
                                    t(i, j, k)};
                        });
            }
            Real sum = amrex::get<0>(data.value(op));
            amrex::ParallelDescriptor::ReduceRealSum(sum);
            return sum;
        };
        amrex::MultiFab initial(te.boxArray(), te.DistributionMap(), 1,
                                te.nGrowVect());
        amrex::MultiFab::Copy(initial, te, 0, 0, 1, te.nGrowVect());
        std::array<amrex::MultiFab, 3> evolved;
        Real const dt = dt_factor * min_dx * min_dx;
        Real const e0 = energy();
        Real energy_error = 0;
        for (int step = 0; step < 3; ++step) {
            amrex::MultiFab::Copy(te, initial, 0, 0, 1, te.nGrowVect());
            QdsmcConductionReport report;
            Real const interval = dt / std::pow(2.0, step);
            bool const ok = hp.TryQdsmcConductionOnceFDAtState(
                0, interval, false, rho, 0, report);
            AMREX_ALWAYS_ASSERT(ok && report.completed &&
                                report.completed_time == interval);
            AMREX_ALWAYS_ASSERT(report.accepted == 1 && report.attempts == 1);
            AMREX_ALWAYS_ASSERT(te.is_finite());
            energy_error = std::max(energy_error, std::abs(energy() - e0) / e0);
            evolved[step].define(te.boxArray(), te.DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(evolved[step], te, 0, 0, 1, 0);
        }
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                         amrex::ReduceOpSum, amrex::ReduceOpSum>
            op;
        amrex::ReduceData<Real, Real, Real, Real> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const t0 = initial.const_array(mfi);
            auto const t1 = evolved[0].const_array(mfi);
            auto const t2 = evolved[1].const_array(mfi);
            auto const t4 = evolved[2].const_array(mfi);
            auto const own = owner->const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        if (!own(i, j, k)) {
                            return {0.0, 0.0, 0.0, 0.0};
                        }
                        Real const rhs = (4.0 * (t2(i, j, k) - t0(i, j, k)) -
                                          (t1(i, j, k) - t0(i, j, k))) /
                                         dt;
                        Real const half = (4.0 * (t4(i, j, k) - t0(i, j, k)) -
                                           (t2(i, j, k) - t0(i, j, k))) /
                                          (0.5 * dt);
                        Real const error =
                            std::abs(rhs + eigenvalue * (t0(i, j, k) - 2.0));
                        return {error, std::abs(rhs - half), error * error,
                                1.0};
                    });
        }
        auto const value = data.value(op);
        Real maxima[2] = {amrex::get<0>(value), amrex::get<1>(value)};
        Real sums[2] = {amrex::get<2>(value), amrex::get<3>(value)};
        amrex::ParallelDescriptor::ReduceRealMax(maxima, 2);
        amrex::ParallelDescriptor::ReduceRealSum(sums, 2);
        AMREX_ALWAYS_ASSERT(energy_error < 1.e-11);
        amrex::Print() << std::setprecision(17)
                       << "CARTESIAN_ORDER_RESULT {\"linf_error\":" << maxima[0]
                       << ",\"l2_error\":" << std::sqrt(sums[0] / sums[1])
                       << ",\"temporal_sensitivity\":" << maxima[1]
                       << ",\"energy_relative\":" << energy_error
                       << ",\"dt\":" << dt << ",\"eigenvalue\":" << eigenvalue
                       << "}\n";
    }
    warpx::initialization::finalize_external_libraries();
}
