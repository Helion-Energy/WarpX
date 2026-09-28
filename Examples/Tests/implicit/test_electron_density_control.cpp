/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronDensityControl.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>
#include <string>
using namespace warpx::thermal;
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        int grid = 4;
        std::string negative;
        pp.query("grid", grid);
        pp.query("negative", negative);
        ElectronDensityControl state;
        state.initial_number_floor = state.initial_representation_floor = 2.e18;
        state.initial_gate_floor = 3.e17;
        state.number_floor = state.representation_floor = 5.e17;
        state.gate_floor = 3.e17;
        state.pedestal = true;
        state.pedestal_tracks_floor = true;
        state.epoch = 17;
        state.floor_inventory_joule = -.12;
        state.pedestal_inventory_joule = 2.3;
        state.controller_configuration =
            "{\"alpha\":0.05,\"gate_track\":false}";
        state.controller_last = state.number_floor;
        state.controller_ema = 4.9e17;
        state.controller_has_ema = true;
        state.controller_step = 23;
        auto text = state.Encode();
        auto restored = ElectronDensityControl::Decode(text);
        AMREX_ALWAYS_ASSERT(text == restored.Encode());
        ElectronDensityControl requested = state;
        requested.number_floor = requested.representation_floor =
            requested.initial_number_floor;
        requested.gate_floor = requested.initial_gate_floor;
        requested.epoch = 0;
        requested.controller_configuration.clear();
        requested.controller_last = 0.;
        requested.controller_ema = 0.;
        requested.controller_has_ema = false;
        requested.controller_step = -1;
        restored.ValidateConfiguration(requested);
        if (negative == "original_floor") {
            requested.initial_number_floor *= 2.;
            requested.initial_representation_floor *= 2.;
            restored.ValidateConfiguration(requested);
        }
        if (negative == "pedestal") {
            requested.pedestal_tracks_floor = false;
            restored.ValidateConfiguration(requested);
        }
        if (negative == "representation") {
            state.representation_floor *= 2.;
            state.Validate();
        }
        if (negative == "nonfinite") {
            state.controller_ema =
                std::numeric_limits<amrex::Real>::quiet_NaN();
            state.Validate();
        }
        if (negative == "trailing") {
            ElectronDensityControl::Decode(text + "extra\n");
        }
        if (negative == "truncated") {
            ElectronDensityControl::Decode(text.substr(0, text.size() / 2));
        }
        if (negative == "epoch") {
            auto i = text.find("epoch 17");
            text.replace(i, 8, "epoch -1");
            ElectronDensityControl::Decode(text);
        }
        if (negative == "policy") {
            auto i = text.find("temperature_preserving_cell_v1");
            text.replace(i, 29, "unknown");
            ElectronDensityControl::Decode(text);
        }
        if (!negative.empty()) {
            amrex::Abort("negative density-control test unexpectedly survived");
        }
        amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
        amrex::RealBox real({AMREX_D_DECL(0., 0., 0.)},
                            {AMREX_D_DECL(2., 3., 4.)});
        amrex::Array<int, AMREX_SPACEDIM> periodic{};
#ifdef WARPX_DIM_RZ
        int coord = 1;
#else
        int coord = 0;
#endif
        amrex::Geometry geom(domain, &real, coord, periodic.data());
        amrex::BoxArray ba(domain);
        ba.maxSize(grid);
        amrex::DistributionMapping dm(ba);
        amrex::MultiFab u(ba, dm, 1, 0), n(ba, dm, 1, 0), nf(ba, dm, 1, 0),
            nn(ba, dm, 1, 0), out(ba, dm, 1, 0);
        for (amrex::MFIter mfi(u); mfi.isValid(); ++mfi) {
            auto a = u.array(mfi), b = n.array(mfi), c = nf.array(mfi),
                 d = nn.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    b(i, j, k) = 1.e18 * (1. + .01 * (i + 2 * j + 3 * k));
                    a(i, j, k) = 17. * (1. + .03 * (i + j + k));
                    c(i, j, k) = b(i, j, k) * (i % 2 ? 1.25 : .8);
                    d(i, j, k) = b(i, j, k) * (j % 2 ? .6 : 1.8);
                });
        }
        auto result = RemapElectronDensityControl(geom, u, n, n, n, out);
        AMREX_ALWAYS_ASSERT(result.valid && result.floor_joule == 0. &&
                            result.pedestal_joule == 0. &&
                            result.total_joule == 0.);
        amrex::MultiFab::Subtract(out, u, 0, 0, 1, 0);
        AMREX_ALWAYS_ASSERT(out.norm0() == 0.);
        result = RemapElectronDensityControl(geom, u, n, nf, nn, out);
        AMREX_ALWAYS_ASSERT(result.valid);
        // Independent oracle: compare intensive U/n and integrate actual output
        // differences. Density varies in every direction, with up/down changes.
        auto const dx = geom.CellSizeArray();
        auto const lo = geom.ProbLoArray();
        amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
        bool const rz = geom.IsRZ();
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Real, amrex::Real> data(op);
        for (amrex::MFIter mfi(u); mfi.isValid(); ++mfi) {
            auto a = u.const_array(mfi), b = n.const_array(mfi),
                 c = nn.const_array(mfi), d = out.const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j,
                                         int k) -> decltype(data)::Type {
                        auto error = std::abs((d(i, j, k) / c(i, j, k)) /
                                                  (a(i, j, k) / b(i, j, k)) -
                                              1.);
                        auto vol = rz ? volume * 6.2831853071795864769 *
                                            (lo[0] + (i + .5) * dx[0])
                                      : volume;
                        return {error, (d(i, j, k) - a(i, j, k)) * vol};
                    });
        }
        auto values = data.value();
        amrex::Real error = amrex::get<0>(values),
                    total = amrex::get<1>(values);
        amrex::ParallelDescriptor::ReduceRealMax(error);
        amrex::ParallelDescriptor::ReduceRealSum(total);
        AMREX_ALWAYS_ASSERT(error <
                            8 * std::numeric_limits<amrex::Real>::epsilon());
        auto scale =
            std::max({1., std::abs(total), std::abs(result.floor_joule),
                      std::abs(result.pedestal_joule)});
        AMREX_ALWAYS_ASSERT(std::abs(total - result.total_joule) <
                            64 * std::numeric_limits<amrex::Real>::epsilon() *
                                scale);
        AMREX_ALWAYS_ASSERT(
            std::abs(total - result.floor_joule - result.pedestal_joule) <
            64 * std::numeric_limits<amrex::Real>::epsilon() * scale);
        nn.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
        auto rejected = RemapElectronDensityControl(geom, u, n, nf, nn, out);
        AMREX_ALWAYS_ASSERT(!rejected.valid);
        amrex::Print() << "DENSITY_CONTROL_PASS T_relative=" << error
                       << " floor_J=" << result.floor_joule
                       << " pedestal_J=" << result.pedestal_joule
                       << " total_J=" << total << "\n";
    }
    amrex::Finalize();
}
