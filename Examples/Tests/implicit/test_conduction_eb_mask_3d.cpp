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
#include <AMReX_VisMF.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

// Paired native advances differ only in covered temperatures. The energy
// oracle integrates Cartesian nodal dual volumes independently of the solver.
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
        auto const& phi = *sim.m_fields.get(FieldType::distance_to_eb, 0);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const lo = geom.ProbLoArray();
        auto const domain = amrex::surroundingNodes(geom.Domain());
        auto const dlo = domain.smallEnd();
        auto const dhi = domain.bigEnd();
        auto const owner = amrex::OwnerMask(te, geom.periodicity());
        amrex::MultiFab reference(te.boxArray(), te.DistributionMap(), 1, 0);
        amrex::ParmParse pp("eb_test");
        Real temperature = 300.0, amplitude = 0.0, density = 2.e18;
        Real dt_factor = 1.e-4;
        bool require_invariance = true;
        std::string output;
        pp.query("output", output);
        pp.query("temperature_K", temperature);
        pp.query("amplitude", amplitude);
        pp.query("density", density);
        pp.query("dt_factor", dt_factor);
        pp.query("require_invariance", require_invariance);
        amrex::Vector<Real> magnetic{0.0, 0.0, 1.0};
        pp.queryarr("magnetic", magnetic);
        AMREX_ALWAYS_ASSERT(magnetic.size() == 3);
        AMREX_ALWAYS_ASSERT(temperature > 0.0 && std::abs(amplitude) < 1.0);
        AMREX_ALWAYS_ASSERT(density > hp.m_qdsmc_n_floor);
        AMREX_ALWAYS_ASSERT(hp.DensityPedestal(0) == nullptr);
        rho.setVal(density * PhysConst::q_e);
        for (int d = 0; d < 3; ++d) {
            sim.m_fields
                .get(FieldType::Bfield_fp, ablastr::fields::Direction{d}, 0)
                ->setVal(magnetic[d]);
        }
        Real const dt = dt_factor * dx[0] * dx[0];
        Real const volume = dx[0] * dx[1] * dx[2];
        Real const capacity = 1.5 * PhysConst::kb * density;
        Real const pi = 3.1415926535897932384626433832795;
        auto energy = [&] () {
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                auto const f = phi.const_array(mfi);
                auto const own = owner->const_array(mfi);
                op.eval(mfi.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            if (!own(i, j, k) || f(i, j, k) <= 0.0) {
                                return {0.0};
                            }
                            Real weight = volume;
                            int const node[3] = {i, j, k};
                            for (int d = 0; d < 3; ++d) {
                                if (node[d] == dlo[d] || node[d] == dhi[d]) {
                                    weight *= 0.5;
                                }
                            }
                            return {capacity * t(i, j, k) * weight};
                        });
            }
            Real sum = amrex::get<0>(data.value(op));
            amrex::ParallelDescriptor::ReduceRealSum(sum);
            return sum;
        };
        std::array<QdsmcConductionReport, 2> reports;
        std::array<Real, 2> before{}, after{}, closure{};
        Real difference = 0.0, uniform_change = 0.0, scale = temperature;
        for (int variant = 0; variant < 2; ++variant) {
            te.setVal(temperature);
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.array(mfi);
                auto const f = phi.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    Real const x = lo[0] + (i - dlo[0]) * dx[0];
                    Real const y = lo[1] + (j - dlo[1]) * dx[1];
                    Real const z = lo[2] + (k - dlo[2]) * dx[2];
                    t(i, j, k) = f(i, j, k) <= 0.0
                                     ? temperature * (variant == 0 ? 1.0 : 10.0)
                                     : temperature *
                                           (1.0 + amplitude * std::sin(pi * x) *
                                                      std::cos(pi * y) *
                                                      std::cos(pi * z));
                });
            }
            before[variant] = energy();
            bool const ok = hp.TryQdsmcConductionOnceFDAtState(
                0, dt, false, rho, 0.0, reports[variant]);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ok, reports[variant].failure);
            AMREX_ALWAYS_ASSERT(reports[variant].completed &&
                                reports[variant].completed_time == dt);
            AMREX_ALWAYS_ASSERT(te.is_finite());
            after[variant] = energy();
            Real heat = reports[variant].floor_heat;
            for (int wall = 0; wall < 7; ++wall) {
                heat += reports[variant].entry_heat[wall] -
                        reports[variant].outward_heat[wall];
            }
            closure[variant] =
                std::abs(after[variant] - before[variant] - heat) /
                std::max(std::abs(before[variant]), Real(1.e-300));
            if (variant == 0) {
                amrex::MultiFab::Copy(reference, te, 0, 0, 1, 0);
            }
            amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                             amrex::ReduceOpMax>
                op;
            amrex::ReduceData<Real, Real, Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                auto const ref = reference.const_array(mfi);
                auto const f = phi.const_array(mfi);
                op.eval(mfi.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            if (f(i, j, k) <= 0.0) {
                                return {0.0, 0.0, 0.0};
                            }
                            return {std::abs(t(i, j, k) - ref(i, j, k)),
                                    std::abs(t(i, j, k) - temperature),
                                    std::abs(t(i, j, k))};
                        });
            }
            auto const result = data.value(op);
            Real values[3] = {amrex::get<0>(result), amrex::get<1>(result),
                              amrex::get<2>(result)};
            amrex::ParallelDescriptor::ReduceRealMax(values, 3);
            if (variant == 0) {
                uniform_change = values[1];
            } else {
                difference = values[0];
            }
            scale = std::max(scale, values[2]);
        }
        Real account_difference =
            std::abs(reports[1].floor_heat - reports[0].floor_heat);
        for (int wall = 0; wall < 7; ++wall) {
            account_difference = std::max(
                account_difference, std::abs(reports[1].outward_heat[wall] -
                                             reports[0].outward_heat[wall]));
            account_difference = std::max(
                account_difference, std::abs(reports[1].entry_heat[wall] -
                                             reports[0].entry_heat[wall]));
        }
        Real outward_heat = 0.0, entry_heat = 0.0;
        for (int wall = 0; wall < 7; ++wall) {
            outward_heat += reports[1].outward_heat[wall];
            entry_heat += reports[1].entry_heat[wall];
        }
        Real const tolerance = std::max(
            Real(1.e-10), 64.0 * std::numeric_limits<Real>::epsilon() * scale);
        Real const account_relative =
            account_difference / std::max(std::abs(before[0]), Real(1.e-300));
        Real const closure_tolerance = hp.m_cond_eb_bc == 0 ? 1.e-11 : 1.e-8;
        bool const passed =
            difference <= tolerance && account_relative <= 1.e-11 &&
            closure[0] <= closure_tolerance && closure[1] <= closure_tolerance;
        if (!output.empty()) {
            auto const version = amrex::VisMF::GetHeaderVersion();
            amrex::VisMF::SetHeaderVersion(
                amrex::VisMF::Header::NoFabHeader_v1);
            amrex::VisMF::Write(te, output);
            amrex::VisMF::SetHeaderVersion(version);
        }
        amrex::Print() << std::setprecision(17) << "EB_MASK_RESULT {\"passed\":"
                       << (passed ? "true" : "false")
                       << ",\"fd_order\":" << hp.m_cond_fd_order
                       << ",\"integrator\":" << hp.m_cond_fd_time
                       << ",\"ranks\":" << amrex::ParallelDescriptor::NProcs()
                       << ",\"dt_s\":" << dt
                       << ",\"max_difference_K\":" << difference
                       << ",\"tolerance_K\":" << tolerance
                       << ",\"baseline_deviation_from_reference_K\":"
                       << uniform_change << ",\"account_difference_relative\":"
                       << account_relative
                       << ",\"energy_closure0\":" << closure[0]
                       << ",\"energy_closure1\":" << closure[1]
                       << ",\"energy_before_J\":" << before[1]
                       << ",\"energy_after_J\":" << after[1]
                       << ",\"outward_heat_J\":" << outward_heat
                       << ",\"entry_heat_J\":" << entry_heat
                       << ",\"floor_heat_J\":" << reports[1].floor_heat
                       << ",\"steps0\":" << reports[0].attempts
                       << ",\"steps1\":" << reports[1].attempts << "}\n";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!require_invariance || passed,
                                         "Covered temperature changes affect "
                                         "open conduction or heat accounts");
    }
    warpx::initialization::finalize_external_libraries();
}
