/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcConductionReport.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_Utility.H>
#include <fstream>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <string>

// Cartesian oracle: physical nodal dual volumes and rectangular wall areas.
// No production volume helper or cylindrical metric is used here.
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
        auto const plo = geom.ProbLoArray();
        auto const phi = geom.ProbHiArray();
        auto const nodes = amrex::surroundingNodes(geom.Domain());
        auto const dlo = nodes.smallEnd();
        auto const dhi = nodes.bigEnd();
        auto const owner = amrex::OwnerMask(te, geom.periodicity());
        auto const* pedestal = hp.DensityPedestal(0);
        Real const kelvin = PhysConst::q_e / PhysConst::kb;
        amrex::ParmParse pp("boundary_test");
        std::string mode = "flux";
        int axis = 0, side = 1;
        Real density = 2.e18, dt = 1.e-7, flux = 1.e4, eta = 0.0;
        pp.query("mode", mode);
        pp.query("axis", axis);
        pp.query("side", side);
        pp.query("density", density);
        pp.query("dt", dt);
        pp.query("flux", flux);
        pp.query("eta", eta);
        AMREX_ALWAYS_ASSERT(axis >= 0 && axis < 3 && (side == 0 || side == 1));
        AMREX_ALWAYS_ASSERT(density >= 0.0 && dt > 0.0);
        int const face = 2 * axis + side;
        int const wall = side == 0 ? dlo[axis] : dhi[axis];
        Real area = 1.0;
        for (int d = 0; d < 3; ++d) {
            AMREX_ALWAYS_ASSERT(!geom.isPeriodic(d));
            if (d != axis) {
                area *= phi[d] - plo[d];
            }
            for (int s = 0; s < 2; ++s) {
                hp.m_cond_bc[d][s] = 0;
            }
        }
        rho.setVal(density * PhysConst::q_e);
        te.setVal(100.0 * kelvin);
        auto energy = [&] () {
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                auto const r = rho.const_array(mfi);
                auto const own = owner->const_array(mfi);
                auto const ped = pedestal ? pedestal->const_array(mfi)
                                          : amrex::Array4<Real const>{};
                Real const floor = hp.m_qdsmc_n_floor;
                op.eval(mfi.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            if (!own(i, j, k)) {
                                return {0.0};
                            }
                            int const node[3] = {i, j, k};
                            Real volume = dx[0] * dx[1] * dx[2];
                            for (int d = 0; d < 3; ++d) {
                                if (node[d] == dlo[d] || node[d] == dhi[d]) {
                                    volume *= 0.5;
                                }
                            }
                            Real const ne = amrex::max(
                                (r(i, j, k) + (ped ? ped(i, j, k) : 0.0)) /
                                    PhysConst::q_e,
                                floor);
                            return {1.5 * PhysConst::kb * ne * t(i, j, k) *
                                    volume};
                        });
            }
            Real sum = amrex::get<0>(data.value(op));
            amrex::ParallelDescriptor::ReduceRealSum(sum);
            return sum;
        };
        bool expect_failure = false;
        bool const retry = mode == "retry_floor" || mode == "retry_budget";
        Real expected_flux_heat = 0.0, expected_leg_heat = 0.0;
        if (mode == "flux") {
            hp.m_cond_bc[axis][side] = 2;
            hp.m_cond_bc_q[axis][side] = flux;
            expected_flux_heat = flux * area * dt;
        } else if (mode == "closed") {
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    t(i, j, k) = kelvin * (100.0 + 20.0 * std::cos(i * dx[0]) *
                                                       std::cos(j * dx[1]) *
                                                       std::cos(k * dx[2]));
                });
            }
        } else if (mode == "pinned" || mode == "floor_invalid" ||
                   mode == "conflict") {
            te.setVal(5.0 * kelvin);
            hp.m_cond_bc[axis][side] = 1;
            hp.m_cond_bc_Te[axis][side] =
                mode == "floor_invalid" ? 0.5 * hp.m_cond_te_floor : 2.0;
            hp.m_cond_wall_flux_limit = 0.0;
            expect_failure = mode != "pinned";
            if (mode == "conflict") {
                hp.m_cond_bc[(axis + 1) % 3][side] = 1;
                hp.m_cond_bc_Te[(axis + 1) % 3][side] = 1.0;
            }
        } else if (mode == "leg_corner") {
            // Keep exactly one edge of positive-density nodes pinned at 2 eV.
            // Its Robin flux stays constant at every stage; the oracle is
            // q*A*dt.
            AMREX_ALWAYS_ASSERT(pedestal == nullptr);
            int const pin_axis = (axis + 1) % 3;
            int const line_axis = (axis + 2) % 3;
            hp.m_cond_bc[pin_axis][1] = 1;
            hp.m_cond_bc_Te[pin_axis][1] = 2.0;
            hp.m_cond_wall_flux_limit = 0.0;
            hp.m_cond_bc[axis][side] = 3;
            hp.m_cond_leg_length = 6.0;
            hp.m_cond_leg_Te_wall = 0.5;
            hp.m_cond_leg_flux_limit = eta;
            te.setVal(2.0 * kelvin);
            rho.setVal(0.0);
            for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
                auto const r = rho.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    int const node[3] = {i, j, k};
                    if (node[axis] == wall && node[pin_axis] == dhi[pin_axis]) {
                        r(i, j, k) = density * PhysConst::q_e;
                    }
                });
            }
            Real const ncap = std::max(density, hp.m_qdsmc_n_floor);
            Real const unbounded =
                1.5 * PhysConst::kb * ncap * 100.0 * 1.5 * kelvin / 6.0;
            Real const speed =
                hp.m_cond_wall_flux_cap_form == 1
                    ? std::sqrt(hp.m_gamma * PhysConst::q_e * 2.0 /
                                PhysConst::m_p)
                    : std::sqrt(PhysConst::q_e * 2.0 / PhysConst::m_e);
            Real const cap = eta * density * PhysConst::q_e * 2.0 * speed;
            Real const q = eta > 0.0 ? std::min(unbounded, cap) : unbounded;
            expected_leg_heat = q * (dx[pin_axis] / 2.0) *
                                (phi[line_axis] - plo[line_axis]) * dt;
        } else if (mode == "floor" || retry || mode == "nonfinite_rhs") {
            hp.m_cond_bc[axis][side] = 2;
            hp.m_cond_bc_q[axis][side] =
                mode == "nonfinite_rhs" ? std::numeric_limits<Real>::infinity()
                                        : -1.e8;
            expect_failure = mode == "nonfinite_rhs" || mode == "retry_budget";
            if (retry) {
                setenv("WARPX_TEST_COND_REJECT_AT", "5", 1);
                hp.m_cond_te_floor_mask = true;
                hp.m_cond_rkl_max_stages = 8;
                if (mode == "retry_budget") {
                    hp.m_cond_fd_max_subcycles = 8;
                }
            }
        } else {
            amrex::Abort("Unknown Cartesian boundary test mode");
        }
        // Nonzero manufactured native E/B/J values make omissions and component
        // swaps observable. This fixture calls conduction only; both off/on
        // arms and the baseline library receive these exact same fields.
        using ablastr::fields::Direction;
        // B controls the nonlinear parallel flux limiter. Preserve the native
        // retry oracle's B=0 geometry unless this is a dedicated field-capture
        // case. E and both currents are passive in this conduction-only call.
        int manufacture_b=0;
        amrex::ParmParse("capture_test").query("manufacture_b",manufacture_b);
        int field_serial=0;
        for (auto field : {FieldType::Efield_fp,FieldType::Bfield_fp,
                           FieldType::current_fp,FieldType::hybrid_current_fp_plasma}) {
            for(int c=0;c<3;++c) {
                ++field_serial;
                if(field==FieldType::Bfield_fp && !manufacture_b)continue;
                if(!sim.m_fields.has(field,Direction{c},0))continue;
                auto& mf=*sim.m_fields.get(field,Direction{c},0);
                int const serial=field_serial;
                for(amrex::MFIter mfi(mf);mfi.isValid();++mfi) {
                    auto const a=mf.array(mfi);
                    amrex::ParallelFor(mfi.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k) {
                        a(i,j,k)=Real(serial)*0.01+Real(i)*0.0001+Real(j)*0.00001+Real(k)*0.000001;
                    });
                }
            }
        }
        amrex::MultiFab saved(te.boxArray(), te.DistributionMap(), 1,
                              te.nGrowVect());
        amrex::MultiFab::Copy(saved, te, 0, 0, 1, te.nGrowVect());
        Real const before = energy();
        std::array<Real, 6> wall_tallies{}, leg_tallies{};
        for (int d = 0; d < 3; ++d) {
            for (int s = 0; s < 2; ++s) {
                wall_tallies[2 * d + s] = hp.GetQdsmcWallTally(d, s);
                leg_tallies[2 * d + s] = hp.GetQdsmcLegTally(d, s);
            }
        }
        Real const floor_tally = hp.m_cond_floor_tally;
        Real const eb_tally = hp.m_cond_eb_tally;
        int force_reject=0;
        amrex::ParmParse("capture_test").query("force_reject",force_reject);
        if(force_reject) {
            setenv("WARPX_TEST_COND_REJECT_AT","5",1);
            hp.m_cond_rkl_max_stages=8;
        }
        QdsmcConductionReport report;
        bool const ok =
            hp.TryQdsmcConductionOnceFDAtState(0, dt, false, rho, 0.0, report);
        if(force_reject) {
            unsetenv("WARPX_TEST_COND_REJECT_AT");
            AMREX_ALWAYS_ASSERT(report.retries==1 && report.rejected_work==5);
        }
        if (retry) {
            unsetenv("WARPX_TEST_COND_REJECT_AT");
            AMREX_ALWAYS_ASSERT(report.retries == 1 &&
                                report.rejected_work == 5);
        }
        Real const after = energy();
        Real heat = report.floor_heat;
        for (int a = 0; a < 7; ++a) {
            heat += report.entry_heat[a] - report.outward_heat[a];
        }
        Real const scale = std::max(std::abs(before), Real(1.e-300));
        Real const closure = std::abs(after - before - heat) / scale;
        amrex::Print() << std::setprecision(17)
                       << "CARTESIAN_BOUNDARY_OBSERVATION mode=" << mode
                       << " axis=" << axis << " side=" << side
                       << " completed=" << ok << " failure=" << report.failure
                       << " closure=" << closure << " dE=" << after - before
                       << " expected_flux=" << expected_flux_heat
                       << " outward_heat=" << report.outward_heat[face]
                       << " expected_leg=" << expected_leg_heat
                       << " floor_heat=" << report.floor_heat
                       << " retries=" << report.retries
                       << " rejected_work=" << report.rejected_work << "\n";
        std::string output;
        amrex::ParmParse("capture_test").get("output",output);
        if(amrex::ParallelDescriptor::IOProcessor()) AMREX_ALWAYS_ASSERT(amrex::UtilCreateDirectory(output,0750));
        amrex::ParallelDescriptor::Barrier();
        auto const header=amrex::VisMF::GetHeaderVersion();
        amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::NoFabHeader_v1);
        amrex::VisMF::Write(te,output+"/Te");
        amrex::VisMF::Write(saved,output+"/Te_initial");
        amrex::VisMF::Write(rho,output+"/rho");
        for(auto const& item:std::array<std::pair<FieldType,char const*>,4>{{
            {FieldType::Efield_fp,"E"},{FieldType::Bfield_fp,"B"},
            {FieldType::current_fp,"J_ion"},{FieldType::hybrid_current_fp_plasma,"J_plasma"}}}) {
            for(int c=0;c<3;++c)if(sim.m_fields.has(item.first,Direction{c},0))
                amrex::VisMF::Write(*sim.m_fields.get(item.first,Direction{c},0),output+"/"+item.second+"_"+"xyz"[c]);
        }
        amrex::VisMF::SetHeaderVersion(header);
        if(amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream f(output+"/ACCOUNTS.json");f<<std::setprecision(17)
              <<"{\"completed\":"<<(ok?"true":"false")<<",\"failure\":\""<<report.failure
              <<"\",\"time\":"<<report.completed_time<<",\"attempts\":"<<report.attempts
              <<",\"retries\":"<<report.retries<<",\"rejected_work\":"<<report.rejected_work
              <<",\"energy_before\":"<<before<<",\"energy_after\":"<<after
              <<",\"floor_tally\":"<<hp.m_cond_floor_tally<<",\"eb_tally\":"<<hp.m_cond_eb_tally
              <<",\"floor_heat\":"<<report.floor_heat<<",\"outward_heat\":[";
            for(int n=0;n<7;++n){if(n)f<<',';f<<report.outward_heat[n];}
            f<<"],\"entry_heat\":[";for(int n=0;n<7;++n){if(n)f<<',';f<<report.entry_heat[n];}
            f<<"],\"wall_tallies\":[";for(int n=0;n<6;++n){if(n)f<<',';f<<hp.GetQdsmcWallTally(n/2,n%2);}
            f<<"],\"leg_tallies\":[";for(int n=0;n<6;++n){if(n)f<<',';f<<hp.GetQdsmcLegTally(n/2,n%2);}
            f<<"]}\n";
        }
        if (expect_failure) {
            AMREX_ALWAYS_ASSERT(!ok && !report.completed);
            amrex::MultiFab::Subtract(saved, te, 0, 0, 1, te.nGrowVect());
            AMREX_ALWAYS_ASSERT(saved.norminf(0, te.nGrow()) == 0.0);
            AMREX_ALWAYS_ASSERT(hp.m_cond_floor_tally == floor_tally);
            AMREX_ALWAYS_ASSERT(hp.m_cond_eb_tally == eb_tally);
            for (int d = 0; d < 3; ++d) {
                for (int s = 0; s < 2; ++s) {
                    AMREX_ALWAYS_ASSERT(wall_tallies[2 * d + s] ==
                                        hp.GetQdsmcWallTally(d, s));
                    AMREX_ALWAYS_ASSERT(leg_tallies[2 * d + s] ==
                                        hp.GetQdsmcLegTally(d, s));
                }
            }
        } else {
            AMREX_ALWAYS_ASSERT(ok && report.completed &&
                                report.completed_time == dt);
            AMREX_ALWAYS_ASSERT(te.is_finite() && closure <= 1.e-8);
            if (mode == "closed") {
                AMREX_ALWAYS_ASSERT(std::abs(after - before) / scale < 1.e-11);
            }
            if (mode == "flux") {
                AMREX_ALWAYS_ASSERT(
                    std::abs((after - before) / expected_flux_heat - 1.0) <
                    1.e-9);
                AMREX_ALWAYS_ASSERT(
                    std::abs(-report.outward_heat[face] / expected_flux_heat -
                             1.0) < 1.e-9);
            }
            if (mode == "leg_corner") {
                AMREX_ALWAYS_ASSERT(expected_leg_heat > 0.0);
                AMREX_ALWAYS_ASSERT(
                    std::abs(report.outward_heat[face] / expected_leg_heat -
                             1.0) < 1.e-9);
            }
            if (mode == "floor" || mode == "retry_floor") {
                AMREX_ALWAYS_ASSERT(report.floor_heat > 0.0);
                AMREX_ALWAYS_ASSERT(
                    te.min(0) >=
                    hp.m_cond_te_floor * kelvin *
                        (1.0 - 64 * std::numeric_limits<Real>::epsilon()));
            }
            if (mode == "pinned") {
                amrex::MultiFab error(te.boxArray(), te.DistributionMap(), 1,
                                      0);
                for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                    auto const e = error.array(mfi);
                    auto const t = te.const_array(mfi);
                    amrex::ParallelFor(
                        mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            int const node[3] = {i, j, k};
                            e(i, j, k) =
                                node[axis] == wall
                                    ? std::abs(t(i, j, k) - 2.0 * kelvin)
                                    : 0.0;
                        });
                }
                AMREX_ALWAYS_ASSERT(error.norminf() <=
                                    256 * std::numeric_limits<Real>::epsilon() *
                                        2.0 * kelvin);
            }
        }
        amrex::Print() << std::setprecision(17)
                       << "CARTESIAN_BOUNDARY_RESULT {\"mode\":\"" << mode
                       << "\",\"axis\":" << axis << ",\"side\":" << side
                       << ",\"passed\":true,\"expected_failure\":"
                       << (expect_failure ? "true" : "false")
                       << ",\"energy_before_J\":" << before
                       << ",\"energy_after_J\":" << after
                       << ",\"closure\":" << closure
                       << ",\"expected_flux_J\":" << expected_flux_heat
                       << ",\"expected_leg_J\":" << expected_leg_heat
                       << ",\"outward_heat_J\":" << report.outward_heat[face]
                       << ",\"floor_heat_J\":" << report.floor_heat
                       << ",\"attempts\":" << report.attempts
                       << ",\"retries\":" << report.retries << "}\n";
    }
    warpx::initialization::finalize_external_libraries();
}
