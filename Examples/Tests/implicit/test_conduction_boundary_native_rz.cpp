/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <string>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            sim.evolve_scheme == EvolveScheme::Explicit,
            "Conduction boundary fixtures require the explicit hybrid solver");
        auto& hp = *sim.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const plo = geom.ProbLoArray();
        auto const phi = geom.ProbHiArray();
        auto const dlo = geom.Domain().smallEnd();
        auto const dhi = geom.Domain().bigEnd();
        auto const own = amrex::OwnerMask(te, geom.periodicity());
        bool const periodic_z = geom.isPeriodic(1);
        amrex::Real const kelvin = PhysConst::q_e / PhysConst::kb;
        amrex::ParmParse pp("boundary_test");
        amrex::Real density = 2.e18;
        pp.query("density", density);
        rho.setVal(density * PhysConst::q_e);
        auto const* pedestal = hp.DensityPedestal(0);
        // Independent analytic annular-prism volumes, without the production helper.
        auto energy = [&] ()
        {
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<amrex::Real> data(op);
            using Tuple = typename decltype(data)::Type;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                auto const mask = own->const_array(mfi);
                auto const ped = pedestal ? pedestal->const_array(mfi)
                                          : amrex::Array4<amrex::Real const>{};
                amrex::Real const floor = hp.m_qdsmc_n_floor;
                op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple {
                    if (!mask(i,j,k)) { return {0.0}; }
                    amrex::Real const r = plo[0] + (i-dlo[0])*dx[0];
                    amrex::Real const rl = amrex::max(plo[0], r-dx[0]/2);
                    amrex::Real const rh = amrex::min(phi[0], r+dx[0]/2);
                    amrex::Real const z = plo[1] + (j-dlo[1])*dx[1];
                    amrex::Real const dz = periodic_z ? dx[1]
                        : amrex::min(phi[1], z+dx[1]/2)-amrex::max(plo[1], z-dx[1]/2);
                    amrex::Real const ne = amrex::max(density +
                        (ped ? ped(i,j,k)/PhysConst::q_e : 0.0), floor);
                    return {1.5*PhysConst::kb*ne*t(i,j,k)*MathConst::pi*(rh*rh-rl*rl)*dz};
                });
            }
            amrex::Real value = amrex::get<0>(data.value(op));
            amrex::ParallelDescriptor::ReduceRealSum(value);
            return value;
        };
        std::string mode = "flux";
        pp.query("mode", mode);
        amrex::Real dt = 1.e-7;
        pp.query("dt", dt);
        for (int d=0; d<AMREX_SPACEDIM; ++d) {
            for (int s=0; s<2; ++s) { hp.m_cond_bc[d][s] = 0; }
        }
        te.setVal(100.0*kelvin);
        if (mode == "flux") {
            hp.m_cond_bc[0][1] = 2;
            hp.m_cond_bc_q[0][1] = 1.e4;
            amrex::Real const area = 2*MathConst::pi*phi[0]*(phi[1]-plo[1]);
            amrex::Real const expected = 1.e4*area*dt;
            auto const before = energy();
            auto const tally0 = hp.GetQdsmcWallTally(0,1);
            hp.QdsmcConductionOnceFDAtState(0,dt,false,rho,0.0);
            auto const actual = energy()-before;
            auto const heat = (hp.GetQdsmcWallTally(0,1)-tally0)*dx[0]*dx[1];
            amrex::Print() << std::setprecision(17) << "NATIVE_BOUNDARY mode=flux dt=" << dt
                << " energy_ratio=" << actual/expected << " tally_ratio=" << heat/expected
                << " delta_energy_J=" << actual << " expected_J=" << expected << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(actual/expected-1) < 1.e-9);
            AMREX_ALWAYS_ASSERT(std::abs(heat/expected-1) < 1.e-9);
        } else if (mode == "robin") {
            AMREX_ALWAYS_ASSERT(!periodic_z);
            hp.m_cond_bc[1][1] = 3;
            hp.m_cond_leg_flux_limit = 0;
            hp.m_cond_leg_length = 6;
            hp.m_cond_leg_Te_wall = 0.5;
            amrex::Real const tb = 1.999, ti = 2.0;
            int const wall = dhi[1]+1;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) {
                    t(i,j,k) = kelvin*(j == wall ? tb : ti);
                });
            }
            hp.QdsmcConductionOnceFDAtState(0,dt,false,rho,0.0);
            amrex::MultiFab rate(te.boxArray(),te.DistributionMap(),1,0);
            for (amrex::MFIter mfi(rate); mfi.isValid(); ++mfi) {
                auto const out=rate.array(mfi); auto const t=te.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) {
                    out(i,j,k) = (j == wall) ? (t(i,j,k)/kelvin-tb)/dt : 0.0;
                });
            }
            // chi=100 in the input deck, uniform density, exactly one inward face.
            amrex::Real const interior=100*(ti-tb)/dx[1]*2/dx[1];
            amrex::Real const external=100*(tb-0.5)/6*2/dx[1];
            auto const expected=interior-external;
            auto const actual=-rate.norminf();
            amrex::Print() << std::setprecision(17) << "NATIVE_BOUNDARY mode=robin dt=" << dt
                << " measured_eV_s=" << actual << " one_face_eV_s=" << expected
                << " doubled_face_eV_s=" << 2*interior-external << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(actual-expected) < 1.e-4*(std::abs(interior)+external));
        } else if (mode == "closed") {
            auto const before = energy();
            hp.QdsmcConductionOnceFDAtState(0,dt,false,rho,0.0);
            auto const after = energy();
            amrex::Print() << "NATIVE_BOUNDARY mode=closed relative_energy=" << (after-before)/before << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(after-before) < 1.e-11*before);
        }

        else {
            QdsmcConductionReport report;
            bool expect_failure = false;
            amrex::Real expected_corner_heat = 0.0;
            amrex::Real expected_leg_heat = -1.0;
            amrex::Real tensor_rate = 0.0, tensor_initial = 0.0;
            int const probe_i = (dlo[0] + dhi[0] + 1) / 2, probe_j = dhi[1] + 1;
            te.setVal(2.0 * kelvin);
            hp.m_cond_bc[0][1] = 1;
            hp.m_cond_bc_Te[0][1] = 2.0;
            hp.m_cond_wall_flux_limit = 0.0;
            hp.m_cond_bc[1][1] = 3;
            hp.m_cond_leg_Te_wall = 0.5;
            hp.m_cond_leg_length = 6.0;
            hp.m_cond_leg_flux_limit = 0.0;
            if (mode == "leg_law") {
                AMREX_ALWAYS_ASSERT(pedestal == nullptr);
                amrex::Real tb = 2.0, ti = 2.0, eta = 0.0;
                pp.query("tb", tb);
                pp.query("ti", ti);
                pp.query("eta", eta);
                hp.m_cond_bc[0][1] = 0;
                hp.m_cond_leg_flux_limit = eta;
                for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                    auto const t = te.array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                         int i, int j, int k) {
                        t(i, j, k) = kelvin * (j == dhi[1] + 1 ? tb : ti);
                    });
                }
                amrex::Real const ncap =
                    amrex::max(density, hp.m_qdsmc_n_floor);
                amrex::Real const qu = 1.5 * PhysConst::kb * ncap * 100 *
                                       amrex::max(tb - 0.5, 0.0) * kelvin / 6;
                amrex::Real const v =
                    hp.m_cond_wall_flux_cap_form == 1
                        ? std::sqrt(hp.m_gamma * PhysConst::q_e * tb /
                                    PhysConst::m_p)
                        : std::sqrt(PhysConst::q_e * tb / PhysConst::m_e);
                amrex::Real const qc = eta * density * PhysConst::q_e * tb * v;
                amrex::Real const q = eta > 0 ? amrex::min(qu, qc) : qu;
                bool const gate =
                    density > 0 && ti > amrex::max(0.5, hp.m_cond_te_floor);
                expected_leg_heat =
                    gate ? q * MathConst::pi * phi[0] * phi[0] * dt : 0.0;
                amrex::Print()
                    << std::setprecision(17) << "LEG_ORACLE Tb_eV=" << tb
                    << " Ti_eV=" << ti << " n_eff=" << density
                    << " n_cap=" << ncap << " gate=" << gate
                    << " q_uncapped_W_m2=" << qu
                    << " q_applied_W_m2=" << (gate ? q : 0.0) << "\n";
            } else if (mode == "tensor") {
                amrex::Real const beta = 0.4, gr = 0.2, gz = 0.1;
                hp.m_cond_bc[0][1] = 0;
                for (int d = 0; d < 3; ++d) {
                    sim.m_fields
                        .get(FieldType::Bfield_fp,
                             ablastr::fields::Direction{d}, 0)
                        ->setVal(d == 0   ? beta
                                 : d == 2 ? 1.0
                                          : 0.0);
                }
                for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                    auto const t = te.array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                         int i, int j, int k) {
                        amrex::Real const r = plo[0] + (i - dlo[0]) * dx[0];
                        amrex::Real const z = (j - dlo[1]) * dx[1];
                        t(i, j, k) = kelvin * (2 + gr * r + gz * z);
                    });
                }
                amrex::Real const r = plo[0] + (probe_i - dlo[0]) * dx[0];
                tensor_initial = 2 + gr * r + gz * (phi[1] - plo[1]);
                amrex::Real const rr =
                    .01 + 99.99 * beta * beta / (1 + beta * beta);
                amrex::Real const zz = .01 + 99.99 / (1 + beta * beta);
                amrex::Real const rz = 99.99 * beta / (1 + beta * beta);
                // At the outward end of the inward axial face, positive
                // mixed flux uses the wall node as donor. Its upstream node
                // clamps to itself, so existing SSMART returns Tb rather than
                // the face average. This is the independent native-face oracle;
                // the continuum linear-gradient value lacks this O(h) factor.
                amrex::Real const mixed_wall_factor =
                    tensor_initial / (tensor_initial - 0.5 * gz * dx[1]);
                tensor_rate = (rr * gr + rz * gz) / r -
                              2 / dx[1] *
                                  (rz * gr * mixed_wall_factor + zz * gz +
                                   100 * (tensor_initial - 0.5) / 6);
                amrex::Print()
                    << "TENSOR_FACE_ORACLE SSMART_wall_factor="
                    << mixed_wall_factor << " inward_normal=" << zz * gz
                    << " inward_mixed=" << rz * gr * mixed_wall_factor
                    << " external_leg=" << 100 * (tensor_initial - 0.5) / 6
                    << " K_to_eV_m_s\n";
            } else if (mode == "entry") {
                te.setVal(5.0 * kelvin);
            } else if (mode == "floor_equal" || mode == "floor_invalid") {
                hp.m_cond_bc_Te[0][1] = hp.m_cond_te_floor;
                if (mode == "floor_invalid") {
                    hp.m_cond_bc_Te[0][1] = 0.5 * hp.m_cond_te_floor;
                    expect_failure = true;
                }
            } else if (mode == "conflict") {
                hp.m_cond_bc[1][1] = 1;
                hp.m_cond_bc_Te[1][1] = 1.0;
                expect_failure = true;
            } else if (mode == "budget") {
                hp.m_cond_fd_max_subcycles = 1;
                expect_failure = true;
            } else if (mode == "corner") {
                AMREX_ALWAYS_ASSERT(pedestal == nullptr);
                rho.setVal(0.0);
                for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
                    auto const r = rho.array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                         int i, int j, int k) {
                        if (i == dhi[0] + 1 && j == dhi[1] + 1) {
                            r(i, j, k) = density * PhysConst::q_e;
                        }
                    });
                }
                amrex::Real const area =
                    MathConst::pi *
                    (phi[0] * phi[0] -
                     (phi[0] - dx[0] / 2) * (phi[0] - dx[0] / 2));
                expected_corner_heat = 1.5 * PhysConst::kb * density * 100 *
                                       (2.0 - 0.5) * kelvin / 6 * area * dt;
            } else if (mode == "floor") {
                hp.m_cond_bc[0][1] = 2;
                hp.m_cond_bc_q[0][1] = -1.e8;
                hp.m_cond_bc[1][1] = 0;
                te.setVal(100.0 * kelvin);
            } else if (mode == "gate_failure") {
                hp.m_cond_bc[0][1] = 0;
                hp.m_cond_leg_length = 1.e-6;
                expect_failure = true;
                for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                    auto const t = te.array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                         int i, int j, int k) {
                        t(i, j, k) = kelvin * (j == dhi[1] ? 0.499 : 2.0);
                    });
                }
            } else if (mode == "growth_failure" || mode == "growth_capped") {
                hp.m_cond_bc[0][1] = 2;
                hp.m_cond_bc_q[0][1] = 1.e10;
                hp.m_cond_bc[1][1] = 0;
                expect_failure = mode == "growth_failure";
            } else if ((mode == "density_cap" || mode == "density_closed")) {
                hp.m_cond_bc[0][1] = 0;
                hp.m_cond_leg_flux_limit = 1.e-6;
                hp.m_qdsmc_halo_unfreeze = mode == "density_cap";
                AMREX_ALWAYS_ASSERT(pedestal == nullptr &&
                                    density < hp.m_qdsmc_n_floor);
            } else if (mode == "eb_invalid" || mode == "eb_equal" ||
                       mode == "eb_capped") {
                hp.m_cond_bc[0][1] = 0;
                hp.m_cond_bc[1][1] = 0;
                expect_failure = mode == "eb_invalid";
            } else if (mode == "ghost") {
                setenv("WARPX_TEST_COND_GHOST_POISON", "-1e250", 1);
            } else if (mode == "capped_wall") {
                te.setVal(5.0 * kelvin);
                hp.m_cond_wall_flux_limit = 1.e-6;
            } else {
                AMREX_ALWAYS_ASSERT(mode == "pinned" || mode == "pedestal");
            }
            amrex::MultiFab before(te.boxArray(), te.DistributionMap(), 1,
                                   te.nGrowVect());
            amrex::MultiFab::Copy(before, te, 0, 0, 1, te.nGrowVect());
            auto const wall_before = hp.GetQdsmcWallTally(0, 1);
            auto const leg_before = hp.GetQdsmcLegTally(1, 1);
            auto const floor_before = hp.m_cond_floor_tally;
            amrex::Real const independent_before =
                mode == "corner" ? 0.0 : energy();
            bool const ok = hp.TryQdsmcConductionOnceFDAtState(
                0, dt, false, rho, 0.0, report);
            amrex::Print() << std::setprecision(17)
                           << "NATIVE_BOUNDARY mode=" << mode
                           << " completed=" << ok
                           << " failure=" << report.failure
                           << " residual_J=" << report.residual
                           << " floor_J=" << report.floor_heat
                           << " floor_raw_J=" << report.floor_raw_heat
                           << " s_max=" << report.s_max << "\n";
            if (expect_failure) {
                AMREX_ALWAYS_ASSERT(!ok);
                amrex::MultiFab::Subtract(before, te, 0, 0, 1, te.nGrowVect());
                AMREX_ALWAYS_ASSERT(before.norminf(0, te.nGrow()) == 0.0);
                AMREX_ALWAYS_ASSERT(wall_before == hp.GetQdsmcWallTally(0, 1));
                AMREX_ALWAYS_ASSERT(leg_before == hp.GetQdsmcLegTally(1, 1));
                AMREX_ALWAYS_ASSERT(floor_before == hp.m_cond_floor_tally);
                if (mode == "gate_failure" || mode == "growth_failure") {
                    AMREX_ALWAYS_ASSERT(
                        report.failure.find("stage stability") !=
                        std::string::npos);
                }
            } else {
                AMREX_ALWAYS_ASSERT(ok && report.completed);
                if (mode != "corner") {
                    amrex::Real const independent_after = energy();
                    AMREX_ALWAYS_ASSERT(
                        std::abs(independent_before -
                                 report.energy_before[0]) <=
                        256 * std::numeric_limits<amrex::Real>::epsilon() *
                            std::abs(independent_before));
                    AMREX_ALWAYS_ASSERT(
                        std::abs(independent_after - report.energy_after[0]) <=
                        256 * std::numeric_limits<amrex::Real>::epsilon() *
                            std::abs(independent_after));
                }
                if (mode == "growth_capped") {
                    amrex::Real const imposed = 1.e10 * 2*MathConst::pi*phi[0] *
                                                (phi[1]-plo[1]) * dt;
                    AMREX_ALWAYS_ASSERT(report.floor_heat == 0.0 && report.floor_raw_heat == 0.0);
                    AMREX_ALWAYS_ASSERT(std::abs((energy()-independent_before)/imposed-1) < 1.e-9);
                }
                if (mode == "pedestal") {
                    AMREX_ALWAYS_ASSERT(report.energy_before[0] >
                                        report.energy_before[1]);
                    AMREX_ALWAYS_ASSERT(report.energy_before[2] == 0 &&
                                        report.energy_before[3] == 0);
                    // The existing pedestal convention opens every positive
                    // n_eff row, even when its capacity is raised by the
                    // conduction floor.
                    AMREX_ALWAYS_ASSERT(report.energy_before[5] == 0);
                }
                amrex::Real exchange = std::abs(report.floor_heat);
                for (int a = 0; a < 7; ++a) {
                    exchange += std::abs(report.outward_heat[a]) +
                                std::abs(report.entry_heat[a]);
                }
                amrex::Real const energy_scale =
                    std::abs(report.energy_before[0]) +
                    std::abs(report.energy_after[0]);
                AMREX_ALWAYS_ASSERT(
                    std::abs(report.residual) <=
                    256 * std::numeric_limits<amrex::Real>::epsilon() *
                            energy_scale +
                        1.e-11 * exchange);
                if (hp.m_cond_bc[0][1] == 1 &&
                    hp.m_cond_wall_flux_limit == 0.0) {
                    amrex::Real const wall_temp =
                        hp.m_cond_bc_Te[0][1] * kelvin;
                    amrex::MultiFab error(te.boxArray(), te.DistributionMap(),
                                          1, 0);
                    for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                        auto const e = error.array(mfi);
                        auto const t = te.const_array(mfi);
                        auto const r = rho.const_array(mfi);
                        amrex::ParallelFor(
                            mfi.validbox(),
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                e(i, j, k) =
                                    i == dhi[0] + 1 && r(i, j, k) > 0
                                        ? std::abs(t(i, j, k) - wall_temp)
                                        : 0.0;
                            });
                    }
                    AMREX_ALWAYS_ASSERT(
                        error.norminf() <=
                        256 * std::numeric_limits<amrex::Real>::epsilon() *
                            amrex::max(1.0, wall_temp));
                }
                if (mode == "leg_law") {
                    if (expected_leg_heat == 0.0) {
                        AMREX_ALWAYS_ASSERT(report.outward_heat[3] == 0.0);
                    } else {
                        AMREX_ALWAYS_ASSERT(std::abs(report.outward_heat[3] /
                                                         expected_leg_heat -
                                                     1) < 1.e-4);
                    }
                }
                if (mode == "tensor") {
                    amrex::ReduceOps<amrex::ReduceOpSum> op;
                    amrex::ReduceData<amrex::Real> data(op);
                    using Tuple = typename decltype(data)::Type;
                    for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                        auto const t = te.const_array(mfi);
                        auto const owner = own->const_array(mfi);
                        op.eval(
                            mfi.validbox(), data,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                return {i == probe_i && j == probe_j &&
                                                owner(i, j, k)
                                            ? (t(i, j, k) / kelvin -
                                               tensor_initial) /
                                                  dt
                                            : 0.0};
                            });
                    }
                    amrex::Real observed = amrex::get<0>(data.value(op));
                    amrex::ParallelDescriptor::ReduceRealSum(observed);
                    amrex::Print() << std::setprecision(17)
                                   << "TENSOR_ORACLE measured_eV_s=" << observed
                                   << " expected_eV_s=" << tensor_rate << "\n";
                    AMREX_ALWAYS_ASSERT(std::abs(observed - tensor_rate) <
                                        1.e-4 * std::abs(tensor_rate));
                }
                if (mode == "ghost") {
                    amrex::MultiFab reference(te.boxArray(),
                                              te.DistributionMap(), 1, 0);
                    amrex::MultiFab::Copy(reference, te, 0, 0, 1, 0);
                    amrex::MultiFab::Copy(te, before, 0, 0, 1, te.nGrowVect());
                    setenv("WARPX_TEST_COND_GHOST_POISON", "1e250", 1);
                    QdsmcConductionReport second;
                    AMREX_ALWAYS_ASSERT(hp.TryQdsmcConductionOnceFDAtState(
                        0, dt, false, rho, 0.0, second));
                    amrex::MultiFab::Subtract(reference, te, 0, 0, 1, 0);
                    AMREX_ALWAYS_ASSERT(reference.norminf() == 0.0);
                    for (int a = 0; a < 7; ++a) {
                        AMREX_ALWAYS_ASSERT(second.outward_heat[a] ==
                                            report.outward_heat[a]);
                    }
                    unsetenv("WARPX_TEST_COND_GHOST_POISON");
                }
                if (mode == "entry") {
                    amrex::Real const fraction =
                        (phi[0] * phi[0] -
                         (phi[0] - dx[0] / 2) * (phi[0] - dx[0] / 2)) /
                        (phi[0] * phi[0] - plo[0] * plo[0]);
                    amrex::Real const expected =
                        report.energy_before[0] * fraction * (2.0 - 5.0) / 5.0;
                    AMREX_ALWAYS_ASSERT(
                        std::abs(report.entry_heat[1] - expected) <
                        1.e-11 * std::abs(expected));
                }
                if (mode == "corner") {
                    AMREX_ALWAYS_ASSERT(
                        std::abs(report.outward_heat[3] / expected_corner_heat -
                                 1.0) < 1.e-10);
                    AMREX_ALWAYS_ASSERT(
                        std::abs(report.outward_heat[1] / expected_corner_heat +
                                 1.0) < 1.e-10);
                }
                if (mode == "floor") {
                    AMREX_ALWAYS_ASSERT(report.floor_heat > 0 &&
                                        report.floor_count > 0);
                }
                if (mode == "eb_equal") {
                    AMREX_ALWAYS_ASSERT(report.entry_heat[6] < 0 &&
                                        report.floor_heat == 0);
                }
                if (mode == "eb_capped") {
                    AMREX_ALWAYS_ASSERT(report.outward_heat[6] > 0 &&
                                        report.entry_heat[6] == 0);
                }
                if ((mode == "density_cap" || mode == "density_closed")) {
                    AMREX_ALWAYS_ASSERT(report.energy_before[2] == 0 &&
                                        report.energy_before[3] == 0);
                    AMREX_ALWAYS_ASSERT(std::abs(report.energy_before[0] /
                                                     report.energy_before[1] -
                                                 hp.m_qdsmc_n_floor / density) <
                                        1.e-11);
                    amrex::Real const q =
                        1.e-6 * density * PhysConst::kb * 2 * kelvin *
                        std::sqrt(PhysConst::kb * 2 * kelvin / PhysConst::m_e);
                    amrex::Real const expected =
                        q * MathConst::pi * phi[0] * phi[0] * dt;
                    AMREX_ALWAYS_ASSERT(
                        std::abs(report.outward_heat[3] / expected - 1) <
                        1.e-4);
                }
            }
        }

        AMREX_ALWAYS_ASSERT(te.is_finite());
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
