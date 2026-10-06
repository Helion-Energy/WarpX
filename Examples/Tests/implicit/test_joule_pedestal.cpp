/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#include <cmath>
#include <iomanip>
#include <limits>

using namespace amrex::literals;
using amrex::MultiFab;
using warpx::fields::FieldType;

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    int result = 0;
    {
        amrex::ParmParse test("joule_test");
        amrex::Real ne = 1.e19_rt, pedfrac = .2_rt, dt = 1.e-7_rt;
        amrex::Real accepted = 1._rt, redirected = 0._rt, declined = 0._rt;
        bool variable = false, redirect = false, check_relaxation = false;
        amrex::Real raw_species_scale = 1._rt;
        test.query("raw_species_scale", raw_species_scale);
        test.query("check_relaxation", check_relaxation);
        AMREX_ALWAYS_ASSERT(raw_species_scale > 0._rt);
        amrex::Real receiving_ratio = 1._rt;
        test.query("receiving_density_ratio", receiving_ratio);
        test.query("ne", ne);
        test.query("pedestal_fraction", pedfrac);
        test.query("dt", dt);
        test.query("accepted_fraction", accepted);
        test.query("redirected_fraction", redirected);
        test.query("declined_fraction", declined);
        test.query("variable", variable);
        test.query("redirect", redirect);
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        auto const& g = w.Geom(0);
        auto const dx = g.CellSizeArray(), lo = g.ProbLoArray();
        auto const radius = g.ProbHi(0), length = g.ProbHi(1) - g.ProbLo(1);
        auto const dom = g.Domain();
        auto const z_hi = dom.bigEnd(1) + 1;
        constexpr amrex::Real eta = 1.e-5_rt, current = 1.e5_rt;
        constexpr amrex::Real t0 = 100._rt * PhysConst::q_e / PhysConst::kb;
        auto& rho = *w.m_fields.get(FieldType::rho_fp, 0);
        auto& te =
            *w.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto* ped = const_cast<MultiFab*>(hp.DensityPedestal(0));
        if (!ped) {
            pedfrac = 0;
        }
        auto& a = *w.m_fields.get("rho_fp_ions", 0);
        auto& b = *w.m_fields.get("rho_fp_ions2", 0);
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const rr = rho.array(mfi), aa = a.array(mfi),
                       bb = b.array(mfi);
            auto const pp =
                ped ? ped->array(mfi) : amrex::Array4<amrex::Real>{};
            bool const use_ped = ped != nullptr;
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                  int k) {
                auto const r = lo[0] + i * dx[0], z = lo[1] + j * dx[1];
                auto const mod =
                    variable ? 1._rt + .15_rt * (r / radius) * (r / radius) *
                                           std::cos(2._rt * MathConst::pi *
                                                    (z - lo[1]) / length)
                             : 1._rt;
                rr(i, j, k) = PhysConst::q_e * ne * mod;
                aa(i, j, k) = raw_species_scale * .6_rt * rr(i, j, k);
                bb(i, j, k) = raw_species_scale * .4_rt * rr(i, j, k);
                if (use_ped) {
                    pp(i, j, k) =
                        PhysConst::q_e * ne * pedfrac *
                        (variable ? 1._rt + .25_rt * (r / radius) * (r / radius)
                                  : 1._rt);
                }
            });
        }
        te.setVal(t0);
        auto const J =
            w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        auto const B = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        for (int c = 0; c < 3; ++c) {
            J[c]->setVal(c == 2 ? current : 0._rt);
            B[c]->setVal(c == 2 ? 1._rt : 0._rt);
            w.m_fields.get_alldirs("current_fp_ions", 0)[c]->setVal(0._rt);
            w.m_fields.get_alldirs("current_fp_ions2", 0)[c]->setVal(0._rt);
        }
        MultiFab redirect_energy(te.boxArray(), te.DistributionMap(), 2,
                                 te.nGrowVect());
        redirect_energy.setVal(0._rt);
        MultiFab receiving(rho.boxArray(), rho.DistributionMap(), 1,
                           rho.nGrowVect());
        MultiFab::Copy(receiving, rho, 0, 0, 1, rho.nGrowVect());
        receiving.mult(receiving_ratio, 0, 1, receiving.nGrow());
        auto const before = hp.QDSMCClassEnergy(0, &receiving);
        hp.QDSMCAddJouleHeating(0, dt, rho,
                               redirect ? &redirect_energy : nullptr,
                               receiving_ratio == 1._rt ? nullptr : &receiving);
        auto const after = hp.QDSMCClassEnergy(0, &receiving);
        // Independent physical dual-ring measure. The periodic high-z image
        // is excluded; duplicate inter-box nodes use sum_unique below.
        MultiFab diagnostic(te.boxArray(), te.DistributionMap(), 4, 0);
        for (amrex::MFIter mfi(diagnostic); mfi.isValid(); ++mfi) {
            auto const out = diagnostic.array(mfi);
            auto const tt = te.const_array(mfi);
            auto const rr = rho.const_array(mfi);
            auto const red = redirect_energy.const_array(mfi);
            auto const pp = ped ? ped->const_array(mfi)
                                : amrex::Array4<amrex::Real const>{};
            bool const use_ped = ped != nullptr;
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                auto const r = lo[0] + i * dx[0];
                auto const vl = std::max(0._rt, r - .5_rt * dx[0]);
                auto const vh = std::min(radius, r + .5_rt * dx[0]);
                auto const volume =
                    j == z_hi ? 0._rt
                              : MathConst::pi * (vh * vh - vl * vl) * dx[1];
                auto const capacity =
                    1.5_rt * (receiving_ratio * rr(i, j, k) +
                              (use_ped ? pp(i, j, k) : 0._rt)) /
                    PhysConst::q_e * PhysConst::kb;
                out(i, j, k, 0) = volume * capacity * (tt(i, j, k) - t0);
                // Redirect stores mass-independent energy per ion; this is
                // its independent staged energy, not realized particle heat.
                out(i, j, k, 1) =
                    volume * 1.5_rt *
                    ((.6_rt * rr(i, j, k)) / PhysConst::q_e * red(i, j, k, 0) +
                     (.4_rt * rr(i, j, k)) / (2._rt * PhysConst::q_e) * red(i, j, k, 1));
                out(i, j, k, 2) = volume * capacity * t0;
                auto const expected_t =
                    t0 + accepted * dt * eta * current * current / capacity;
                out(i, j, k, 3) = std::abs(tt(i, j, k) - expected_t);
            });
        }
        auto const deposited = diagnostic.sum_unique(0, false, g.periodicity());
        auto const ion_staged =
            diagnostic.sum_unique(1, false, g.periodicity());
        auto const initial = diagnostic.sum_unique(2, false, g.periodicity());
        auto const native = after[0] + after[1] - before[0] - before[1];
        auto const source = dt * eta * current * current * MathConst::pi *
                            radius * radius * length;
        auto const dropped =
            hp.m_joule_dropped_heat_gate_J + hp.m_joule_dropped_redirect_gate_J;
        auto const bound =
            128._rt * std::numeric_limits<amrex::Real>::epsilon() * initial +
            2.e-11_rt * source;
        bool pass = std::abs(deposited - accepted * source) <= bound &&
                          std::abs(ion_staged - redirected * source) <= bound &&
                          std::abs(dropped - declined * source) <= bound &&
                          std::abs(native - deposited) <= bound;
        amrex::VisMF::Write(te, "temperature");
        amrex::Print() << std::setprecision(17)
                       << "JOULE_PEDESTAL {\"deposited_J\":" << deposited
                       << ",\"native_dU_J\":" << native
                       << ",\"physical_source_J\":" << source
                       << ",\"redirect_staged_J\":" << ion_staged
                       << ",\"declined_J\":" << dropped
                       << ",\"expected_electron_J\":" << accepted * source
                       << ",\"expected_redirect_J\":" << redirected * source
                       << ",\"expected_declined_J\":" << declined * source
                       << ",\"temperature_max_error_K\":"
                       << diagnostic.norminf(3) << ",\"bound_J\":" << bound
                       << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
        if (check_relaxation) {
            // The same raw species fractions set the physical relaxation rate.
            // Cold ions make the independently known solution a single
            // exponential: alpha = (gamma-1)*3*(0.6/1 + 0.4/2) = 1.6.
            auto const cc = amrex::convert(te.boxArray(), amrex::IntVect::TheCellVector());
            MultiFab ti_a(cc, te.DistributionMap(), 1, 1);
            MultiFab ti_b(cc, te.DistributionMap(), 1, 1);
            ti_a.setVal(0._rt); ti_b.setVal(0._rt); te.setVal(t0);
            std::map<std::string, MultiFab*> const ti{{"ions", &ti_a}, {"ions2", &ti_b}};
            constexpr amrex::Real relax_dt = 1.e-4_rt, nu = 1.e3_rt;
            hp.QDSMCAddTemperatureRelaxation(0, relax_dt, rho, ti);
            auto const expected = t0 * std::exp(-1.6_rt * nu * relax_dt);
            te.plus(-expected, 0, 1, 0);
            auto const error = te.norminf();
            bool const relax_pass = error < 128._rt * std::numeric_limits<amrex::Real>::epsilon() * t0;
            amrex::Print() << "SPECIES_RELAXATION {\"raw_species_scale\":" << raw_species_scale
                           << ",\"error_K\":" << error << ",\"pass\":"
                           << (relax_pass ? "true" : "false") << "}\n";
            pass = pass && relax_pass;
        }
        result = pass ? 0 : 2;
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
    return result;
}
