/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "Diagnostics/MultiDiagnostics.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            sim.evolve_scheme == EvolveScheme::Explicit,
            "Conduction boundary fixtures require the explicit hybrid solver");
        auto& hp = *sim.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& te =
            *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray(), plo = geom.ProbLoArray(),
                   phi = geom.ProbHiArray();
        auto const lo = geom.Domain().smallEnd();
        int const nr = geom.Domain().length(0), nz = geom.Domain().length(1);
        amrex::Real const kelvin = PhysConst::q_e / PhysConst::kb;
        amrex::ParmParse pp("boundary_evolve");
        std::string mode = "time";
        pp.query("mode", mode);
        int steps = 10, repeats = 15;
        pp.query("steps", steps);
        pp.query("repeats", repeats);
        amrex::Real dt = 1.e-7, beta = 0.0, eta = 0.0, amplitude = 2.0,
                    bath = 0.5;
        pp.query("dt", dt);
        pp.query("beta", beta);
        pp.query("eta", eta);
        pp.query("amplitude", amplitude);
        pp.query("bath", bath);
        amrex::Real reservoir = bath;
        pp.query("reservoir", reservoir);
        std::string initial_file;
        pp.query("initial_file", initial_file);
        bool const spatial = mode == "spatial", perf = mode == "perf",
                   slab = mode == "slab";
        bool const resumed = mode == "restart_read";
        bool write_fields = true;
        pp.query("write_fields", write_fields);
        rho.setVal(2.e18 * PhysConst::q_e);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            for (int side = 0; side < 2; ++side) {
                hp.m_cond_bc[d][side] = 0;
            }
        }
        hp.m_cond_bc[0][1] = 1;
        hp.m_cond_bc_Te[0][1] = bath;
        hp.m_cond_bc[1][0] = 3;
        hp.m_cond_bc[1][1] = 3;
        hp.m_cond_leg_Te_wall = reservoir;
        hp.m_cond_leg_length = 6.0;
        hp.m_cond_leg_flux_limit = eta;
        hp.m_cond_wall_flux_limit = 0.0;
        hp.m_cond_flux_limit_factor = 0.0;
        if (slab) {
            hp.m_cond_bc[0][1] = 0;
            hp.m_cond_bc[1][0] = 1;
            hp.m_cond_bc_Te[1][0] = bath;
            for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
                auto const density = rho.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    amrex::Real const z = plo[1] + (j - lo[1]) * dx[1];
                    density(i, j, k) = 2.e18 * PhysConst::q_e *
                                       (1 + (z - plo[1]) / (phi[1] - plo[1]));
                });
            }
        }
        // Positive first Robin eigenmode: k*tan(k*length_z/2)=1/L_leg.
        amrex::Real const height = phi[1] - plo[1],
                          zmid = (phi[1] + plo[1]) / 2;
        amrex::Real kl = 0.0, kh = MathConst::pi / height;
        for (int it = 0; it < 80; ++it) {
            amrex::Real const km = (kl + kh) / 2;
            if (km * std::tan(km * height / 2) > 1.0 / 6.0) {
                kh = km;
            } else {
                kl = km;
            }
        }
        amrex::Real const kz = (kl + kh) / 2,
                          ar = 2.4048255576957727686 / phi[0];
        for (int d = 0; d < 3; ++d) {
            auto& b = *sim.m_fields.get(FieldType::Bfield_fp,
                                        ablastr::fields::Direction{d}, 0);
            amrex::Real const shift = b.ixType().nodeCentered(0) ? 0.0 : 0.5;
            for (amrex::MFIter mfi(b); mfi.isValid(); ++mfi) {
                auto const f = b.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    f(i, j, k) =
                        d == 0   ? beta * (plo[0] + (i - lo[0] + shift) * dx[0])
                        : d == 2 ? 1.0
                                 : 0.0;
                });
            }
        }
        if (!resumed) {
            amrex::Gpu::DeviceVector<amrex::Real> loaded;
            if (!initial_file.empty()) {
                std::ifstream input(initial_file);
                AMREX_ALWAYS_ASSERT(input.good());
                std::string line;
                std::getline(input, line);
                std::vector<amrex::Real> host;
                while (std::getline(input, line)) {
                    std::replace(line.begin(), line.end(), ',', ' ');
                    std::istringstream row(line);
                    amrex::Real r, z, t;
                    row >> r >> z >> t;
                    AMREX_ALWAYS_ASSERT(!row.fail());
                    host.push_back(t);
                }
                AMREX_ALWAYS_ASSERT(host.size() ==
                                    std::size_t((nr + 1) * (nz + 1)));
                loaded.resize(host.size());
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(),
                                 host.end(), loaded.begin());
            }
            auto const* loaded_data = loaded.empty() ? nullptr : loaded.data();
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    amrex::Real const r = plo[0] + (i - lo[0]) * dx[0];
                    amrex::Real const z = plo[1] + (j - lo[1]) * dx[1];
                    amrex::Real const radial = 1.0 - r * r / (phi[0] * phi[0]);
                    amrex::Real const shape =
                        spatial ? ::j0(ar * r) * std::cos(kz * (z - zmid))
                                : radial * radial *
                                      (1 + 0.1 * std::cos(2 * MathConst::pi *
                                                          (z - zmid) / height));
                    if (loaded_data) {
                        t(i, j, k) = i >= lo[0] && i <= lo[0] + nr &&
                                             j >= lo[1] && j <= lo[1] + nz
                                         ? loaded_data[(i - lo[0]) * (nz + 1) +
                                                       j - lo[1]]
                                         : kelvin * bath;
                    } else {
                        t(i, j, k) =
                            slab
                                ? kelvin *
                                      (bath -
                                       (bath - reservoir) *
                                           std::log(1 + (z - plo[1]) / height) /
                                           (std::log(2.0) + 6 / (2 * height)))
                                : kelvin * (bath + amplitude * shape);
                    }
                });
            }
        }
#ifdef BOUNDARY_CANDIDATE_REPORT
        if (resumed) {
            AMREX_ALWAYS_ASSERT(!hp.m_cond_last_report.completed);
        }
#endif
        amrex::MultiFab initial(te.boxArray(), te.DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(initial, te, 0, 0, 1, 0);
        std::vector<double> times;
        std::array<amrex::Real, 7> heat{};
        amrex::Real floor = 0.0, max_residual = 0.0;
        int total_stages = 0, s_max = 0;
        int const calls = perf ? repeats + 3 : steps;
        for (int step = 0; step < calls; ++step) {
            if (perf) {
                amrex::MultiFab::Copy(te, initial, 0, 0, 1, 0);
            }
            amrex::Gpu::streamSynchronize();
            auto const start = std::chrono::steady_clock::now();
            hp.QdsmcConductionOnceFDAtState(0, dt, false, rho, 0.0);
            amrex::Gpu::streamSynchronize();
            auto const stop = std::chrono::steady_clock::now();
            if (!perf || step >= 3) {
                times.push_back(
                    std::chrono::duration<double>(stop - start).count());
            }
#ifdef BOUNDARY_CANDIDATE_REPORT
            auto const& report = hp.m_cond_last_report;
            AMREX_ALWAYS_ASSERT(report.completed &&
                                report.completed_time >= dt * (1 - 1.e-12));
            total_stages += report.accepted;
            s_max = std::max(s_max, report.s_max);
            for (int a = 0; a < 7; ++a) {
                heat[a] += report.outward_heat[a];
            }
            floor += report.floor_heat;
            amrex::Real scale = std::abs(report.energy_before[0]) +
                                std::abs(report.energy_after[0]);
            max_residual = std::max(max_residual, std::abs(report.residual) /
                                                      std::max(scale, 1.e-100));
            AMREX_ALWAYS_ASSERT(std::abs(report.residual) <= 1.e-11 * scale);
#endif
        }
        if (mode == "restart_write") {
            sim.GetMultiDiags().NewIteration();
            sim.GetMultiDiags().FilterComputePackFlush(0, true);
        }
        if (write_fields) {
            auto const owner = amrex::OwnerMask(te, geom.periodicity());
            amrex::Gpu::DeviceVector<amrex::Real> values((nr + 1) * (nz + 1),
                                                         0.0);
            auto* ptr = values.data();
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                auto const own = owner->const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    if (own(i, j, k)) {
                        ptr[(i - lo[0]) * (nz + 1) + j - lo[1]] = t(i, j, k);
                    }
                });
            }
            std::vector<amrex::Real> host(values.size());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                             values.end(), host.begin());
            amrex::ParallelDescriptor::ReduceRealSum(host.data(), host.size());
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ofstream file("nodal_temperature.csv");
                file << "r_m,z_m,Te_K,exact_K,region\n"
                     << std::setprecision(17);
                for (int i = 0; i <= nr; ++i) {
                    for (int j = 0; j <= nz; ++j) {
                        double const r = plo[0] + i * dx[0],
                                     z = plo[1] + j * dx[1];
                        double const exact =
                            slab
                                ? kelvin *
                                      (bath -
                                       (bath - reservoir) *
                                           std::log(1 + (z - plo[1]) / height) /
                                           (std::log(2.0) + 6 / (2 * height)))
                                : kelvin *
                                      (bath +
                                       amplitude * ::j0(ar * r) *
                                           std::cos(kz * (z - zmid)) *
                                           std::exp(-100 * (ar * ar + kz * kz) *
                                                    steps * dt));
                        int const region =
                            (i == 0 || i == nr) && (j == 0 || j == nz)     ? 0
                            : i == 0                                       ? 1
                            : i == nr                                      ? 2
                            : (j == 0 || j == nz)                          ? 3
                            : (i < 3 || i > nr - 3 || j < 3 || j > nz - 3) ? 4
                                                                           : 5;
                        file << r << ',' << z << ',' << host[i * (nz + 1) + j]
                             << ',' << (spatial || slab ? exact : 0.0) << ','
                             << region << '\n';
                    }
                }
            }
            amrex::VisMF::Write(te, "native_Te");
        }
        std::sort(times.begin(), times.end());
        auto const median = times[times.size() / 2];
        auto const p95 =
            times[std::min(times.size() - 1, std::size_t(0.95 * times.size()))];
        std::ostringstream summary;
        summary << std::setprecision(17) << "BOUNDARY_EVOLVE mode=" << mode
                << " calls=" << calls << " dt=" << dt << " s_max=" << s_max
                << " stages=" << total_stages << " median_s=" << median
                << " p95_s=" << p95 << " floor_J=" << floor
                << " max_relative_residual=" << max_residual;
        for (int a = 0; a < 7; ++a) {
            summary << " heat" << a << "_J=" << heat[a];
        }
        amrex::Print() << summary.str() << "\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
