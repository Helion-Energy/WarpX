/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace {
using namespace amrex::literals;
using amrex::MultiFab;
using warpx::fields::FieldType;
using Field = ablastr::fields::VectorField;

// Deliberately a single-rank, single-box native field-map fixture. Each map
// retains its actual Yee staggering; the runner applies physical dual volumes.
void
dump (MultiFab const& mf, std::string const& path) {
    AMREX_ALWAYS_ASSERT(mf.size() == 1 &&
                        amrex::ParallelDescriptor::NProcs() == 1);
    amrex::Gpu::streamSynchronize();
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const full = mf[mfi].box(), valid = mfi.validbox();
        std::vector<amrex::Real> host(mf[mfi].size());
        amrex::Gpu::dtoh_memcpy(host.data(), mf[mfi].dataPtr(),
                                host.size() * sizeof(amrex::Real));
        std::vector<amrex::Real> packed;
        for (int j = valid.smallEnd(1); j <= valid.bigEnd(1); ++j) {
            for (int i = valid.smallEnd(0); i <= valid.bigEnd(0); ++i) {
                packed.push_back(host[(i - full.smallEnd(0)) +
                                      full.length(0) * (j - full.smallEnd(1))]);
            }
        }
        std::array<std::int64_t, 4> h{valid.length(0), valid.length(1),
                                      mf.ixType().nodeCentered(0),
                                      mf.ixType().nodeCentered(1)};
        std::ofstream out(path + ".bin", std::ios::binary);
        out.write(reinterpret_cast<char const*>(h.data()), sizeof(h));
        out.write(reinterpret_cast<char const*>(packed.data()),
                  packed.size() * sizeof(amrex::Real));
        AMREX_ALWAYS_ASSERT(out.good());
    }
}

void
dump_vector (Field const& f, std::string const& name) {
    for (int c = 0; c < 3; ++c) {
        dump(*f[c], name + std::to_string(c));
    }
}

amrex::Real
integral (MultiFab const& f, amrex::Geometry const& geom) {
    auto const volume = MakeQdsmcVolumeElement(geom, f.ixType());
    MultiFab out(f.boxArray(), f.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
        auto const a = f.const_array(mfi);
        auto const b = out.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               b(i, j, k) = volume(i, j, k) * a(i, j, k);
                           });
    }
    return out.sum_unique(0, false, geom.periodicity());
}

struct VectorScratch {
    std::array<MultiFab, 3> storage;
    Field f;
    explicit VectorScratch (Field const& like, int extra = 0)
        : f{&storage[0], &storage[1], &storage[2]} {
        for (int c = 0; c < 3; ++c) {
            storage[c].define(like[c]->boxArray(), like[c]->DistributionMap(),
                              1, like[c]->nGrowVect() + amrex::IntVect(extra));
            storage[c].setVal(0._rt);
        }
    }
};
} // namespace

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        amrex::ParmParse test("holmstrom_test");
        bool active = true;
        amrex::Real cond_dt = 1.e-13_rt;
        std::string source_probe = "none";
        test.query("source_probe", source_probe);
        AMREX_ALWAYS_ASSERT(source_probe == "none" || source_probe == "visc" ||
                            source_probe == "hyper");
        test.query("active", active);
        test.query("conduction_dt", cond_dt);
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        auto const& geom = w.Geom(0);
        auto const dx = geom.CellSizeArray(), lo = geom.ProbLoArray();
        auto const radius = geom.ProbHi(0),
                   length = geom.ProbHi(1) - geom.ProbLo(1);
        auto const wave = 2._rt * MathConst::pi / length;
        auto const nf = hp.m_n_floor;
        constexpr amrex::Real t0 = 100._rt * PhysConst::q_e / PhysConst::kb;
        AMREX_ALWAYS_ASSERT(hp.m_include_electron_inertia_elliptic &&
                            !hp.m_esolve_curlcurl);
        AMREX_ALWAYS_ASSERT(hp.m_cond_fd_order == 4 &&
                            hp.m_cond_fd_limiter == 5);
        AMREX_ALWAYS_ASSERT(cond_dt > 0 && radius > .015_rt);
        if (hp.m_holmstrom_vacuum_region) {
            AMREX_ALWAYS_ASSERT(hp.m_holmstrom_axis_radius == .006_rt &&
                                hp.m_holmstrom_axis_rolloff == .003_rt &&
                                hp.m_holmstrom_transition_width == .5_rt);
        } else {
            AMREX_ALWAYS_ASSERT(hp.m_holmstrom_axis_radius == 0 &&
                                hp.m_holmstrom_axis_rolloff == 0 &&
                                hp.m_holmstrom_transition_width == 0);
        }
        auto& rho = *w.m_fields.get(FieldType::rho_fp, 0);
        auto& te =
            *w.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto* ped = const_cast<MultiFab*>(hp.DensityPedestal(0));
        AMREX_ALWAYS_ASSERT(ped);
        ped->setVal(nf * PhysConst::q_e);
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
            auto const a = rho.array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), rho.nComp(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    auto const r = lo[0] + i * dx[0];
                    auto const ne =
                        nf *
                        (active ? 5._rt - 4.8_rt * std::exp(-r * r /
                                                            (.018_rt * .018_rt))
                                : 20._rt);
                    a(i, j, k, n) = ne * PhysConst::q_e;
                });
        }
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const a = te.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                  int k) {
                auto const r = lo[0] + i * dx[0], z = lo[1] + j * dx[1];
                auto const g = std::pow(1._rt - r * r / (radius * radius), 4);
                a(i, j, k) =
                    t0 * (1._rt +
                          g * (.05_rt + .1_rt * std::cos(wave * (z - lo[1]))));
            });
        }
        auto B = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto E = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto Ji = w.m_fields.get_alldirs(FieldType::current_fp, 0);
        auto J = w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        VectorScratch potential(E, 2);
        for (int c = 0; c < 3; ++c) {
            auto const ix = potential.f[c]->ixType().toIntVect();
            for (amrex::MFIter mfi(*potential.f[c]); mfi.isValid(); ++mfi) {
                auto const a = potential.f[c]->array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    auto const r =
                        lo[0] + (i + (ix[0] ? 0._rt : .5_rt)) * dx[0];
                    auto const z =
                        lo[1] + (j + (ix[1] ? 0._rt : .5_rt)) * dx[1];
                    auto const g = 1._rt - r * r / (radius * radius);
                    a(i, j, k) =
                        c == 1   ? .5_rt * r + .03_rt * r * std::pow(g, 4) *
                                                   std::cos(wave * (z - lo[1]))
                        : c == 2 ? .002_rt * radius / 10._rt * std::pow(g, 5) *
                                       std::sin(wave * (z - lo[1]))
                                 : 0._rt;
                });
            }
        }
        auto* fdtd = w.get_pointer_fdtd_solver_fp(0);
        fdtd->ComputeCurlA(B, potential.f, w.GetEBUpdateBFlag()[0], 0,
                           B[0]->nGrowVect());
        for (int c = 0; c < 3; ++c) {
            B[c]->FillBoundary(geom.periodicity());
        }
        w.ApplyBfieldBoundary(0, PatchType::fine, SubcyclingHalf::FirstHalf,
                              0._rt);
        hp.CalculatePlasmaCurrent(B, w.GetEBUpdateEFlag()[0], 0);
        // Native moments are prescribed at their Yee locations, not a claim
        // about PIC sampling. The analysis uses native interpolated rho for
        // the gate; it does not substitute these point samples for that rho.
        for (int c = 0; c < 3; ++c) {
            auto const ix = Ji[c]->ixType().toIntVect();
            for (amrex::MFIter mfi(*Ji[c]); mfi.isValid(); ++mfi) {
                auto const a = Ji[c]->array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    auto const r =
                        lo[0] + (i + (ix[0] ? 0._rt : .5_rt)) * dx[0];
                    auto const z =
                        lo[1] + (j + (ix[1] ? 0._rt : .5_rt)) * dx[1];
                    auto const g =
                        std::pow(1._rt - r * r / (radius * radius), 4);
                    auto const ne =
                        nf *
                        (active ? 5._rt - 4.8_rt * std::exp(-r * r /
                                                            (.018_rt * .018_rt))
                                : 20._rt);
                    auto const u =
                        c == 1   ? 1.e4_rt * r / radius * g
                        : c == 2 ? 1.e3_rt * g * std::cos(wave * (z - lo[1]))
                                 : 0._rt;
                    a(i, j, k) = ne * PhysConst::q_e * u;
                });
            }
            Ji[c]->FillBoundary(geom.periodicity());
            J[c]->FillBoundary(geom.periodicity());
        }
        hp.QDSMCFillElectronPressureFromTe(0, rho);
        auto const& pe =
            *w.m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
        dump(te, "T0");
        dump(rho, "rho");
        dump(*ped, "pedestal");
        dump(pe, "Pe");
        dump_vector(B, "B");
        dump_vector(J, "J");
        dump_vector(Ji, "Ji");
        // Production wrapper assembles the selected hyper-resistivity and
        // applies physical E boundaries, then the embedded elliptic inertia
        // solve.
        hp.HybridPICSolveE(E, Ji, B, rho, w.GetEBUpdateEFlag()[0], 0, true,
                           true);
        dump_vector(E, "applied_E");
        VectorScratch dbdt(B);
        fdtd->ComputeCurlA(dbdt.f, E, w.GetEBUpdateBFlag()[0], 0);
        for (int c = 0; c < 3; ++c) {
            dbdt.f[c]->mult(-1._rt);
        }
        dump_vector(dbdt.f, "dBdt");
        VectorScratch raw(E), hyper(E), visc(E);
        for (int term = 0; term < 3; ++term) {
            hp.m_include_hall_term = term != 1;
            hp.m_include_electron_pressure_term = term != 2;
            fdtd->HybridPICSolveE(raw.f, J, Ji, B, rho, pe,
                                  w.GetEBUpdateEFlag()[0], 0, &hp, true, true,
                                  &hyper.f, &visc.f);
            dump_vector(raw.f, term == 0   ? "raw_E"
                               : term == 1 ? "nohall_E"
                                           : "nope_E");
            if (term == 0) {
                dump_vector(hyper.f, "hyper_E");
                dump_vector(visc.f, "visc_E");
                for (int c = 0; c < 3; ++c) {
                    MultiFab::Copy(*E[c], *raw.f[c], 0, 0, 1,
                                   E[c]->nGrowVect());
                }
                w.ApplyEfieldBoundary(0, PatchType::fine, 0._rt);
                dump_vector(E, "boundary_E");
            }
        }
        hp.m_include_hall_term = true;
        hp.m_include_electron_pressure_term = true;
        // Direct edge work is always measured. The native nodal receiver
        // probe is a separate declared gate below: an unbooked hyper term
        // must not be mistaken for a live source path in this fixture.
        amrex::Real wh = 0, wv = 0;
        for (int c = 0; c < 3; ++c) {
            MultiFab work(J[c]->boxArray(), J[c]->DistributionMap(), 1, 0);
            MultiFab::Copy(work, *J[c], 0, 0, 1, 0);
            MultiFab::Multiply(work, *hyper.f[c], 0, 0, 1, 0);
            wh += integral(work, geom);
            dump(work, "hyper_edge_work" + std::to_string(c));
            MultiFab::Copy(work, *J[c], 0, 0, 1, 0);
            MultiFab::Multiply(work, *visc.f[c], 0, 0, 1, 0);
            wv += integral(work, geom);
            dump(work, "visc_edge_work" + std::to_string(c));
        }
        hp.QDSMCAddViscousHeating(0, 0._rt, rho, false);
        auto const& qnu = *w.m_fields.get("hybrid_qdsmc_visc_heating_fp", 0);
        dump(qnu, "Qnu");
        auto const q = integral(qnu, geom);
        MultiFab initial(te.boxArray(), te.DistributionMap(), 1,
                         te.nGrowVect());
        MultiFab::Copy(initial, te, 0, 0, 1, te.nGrowVect());
        auto const u = hp.QDSMCClassEnergy(0, &rho);
        auto const f0 = hp.GetQdsmcCondFloorTally();
        // Diagnostic h,h/2,h/4 extrapolation leaves the spatial scheme and
        // limiter intact. The fixture declares SSPRK2, not campaign RKL2.
        for (int step = 0; step < 3; ++step) {
            MultiFab::Copy(te, initial, 0, 0, 1, te.nGrowVect());
            hp.QdsmcConductionOnceFDAtState(0, cond_dt / std::pow(2._rt, step),
                                            false, rho, 0._rt);
            AMREX_ALWAYS_ASSERT(te.is_finite());
            dump(te, "T" + std::to_string(step + 1));
        }
        auto const floor = (hp.GetQdsmcCondFloorTally() - f0) * dx[0] * dx[1];
        amrex::Print() << std::setprecision(17) << "HOLMSTROM_AXIS {\"active\":"
                       << (active ? "true" : "false") << ",\"on\":"
                       << (hp.m_holmstrom_vacuum_region ? "true" : "false")
                       << ",\"radius_m\":" << hp.m_holmstrom_axis_radius
                       << ",\"rolloff_m\":" << hp.m_holmstrom_axis_rolloff
                       << ",\"transition_width\":"
                       << hp.m_holmstrom_transition_width
                       << ",\"n_floor\":" << hp.m_n_floor
                       << ",\"nr\":" << geom.Domain().length(0)
                       << ",\"nz\":" << geom.Domain().length(1)
                       << ",\"domain_radius_m\":" << radius
                       << ",\"domain_length_m\":" << length
                       << ",\"conduction_dt\":" << cond_dt
                       << ",\"inertia_rtol\":" << hp.m_electron_inertia_rtol
                       << ",\"hyper_work_W\":" << wh
                       << ",\"visc_work_W\":" << wv << ",\"Qnu_W\":" << q
                       << ",\"thermal_energy_J\":" << u[0] + u[1]
                       << ",\"conduction_floor_J\":" << floor << "}\n";
        AMREX_ALWAYS_ASSERT(floor == 0._rt);
        if (source_probe != "none") {
            MultiFab::Copy(te, initial, 0, 0, 1, te.nGrowVect());
            MultiFab work(te.boxArray(), te.DistributionMap(), 1, 0);
            amrex::Real clamp = 0;
            amrex::Print() << "HOLMSTROM_SOURCE_PROBE " << source_probe
                           << " before native receiver guard\n";
            hp.QDSMCDepositDragWork(0, 0._rt, rho,
                                    source_probe == "visc" ? visc.f : hyper.f,
                                    work, clamp, true, false);
            dump(work, source_probe + "_nodal_work");
            amrex::Print() << std::setprecision(17)
                           << "HOLMSTROM_SOURCE_RESULT {\"source\":\""
                           << source_probe
                           << "\",\"nodal_work_W\":" << integral(work, geom)
                           << ",\"clamp_J\":" << clamp << "}\n";
        }

        WarpX::ResetInstance();
    }
    warpx::initialization::finalize_external_libraries();
    return 0;
}
