/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ElectronDensityControl.H"
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
namespace warpx::thermal {
void
ElectronDensityControl::Validate () const {
    auto positive = [] (amrex::Real x) { return std::isfinite(x) && x > 0.; };
    auto nonnegative = [] (amrex::Real x) {
        return std::isfinite(x) && x >= 0.;
    };
    bool const controller = !controller_configuration.empty();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        positive(initial_number_floor) && positive(initial_gate_floor) &&
            initial_representation_floor == initial_number_floor &&
            positive(number_floor) && positive(gate_floor) &&
            representation_floor == number_floor &&
            nonnegative(pedestal_taper_cells) &&
            (!pedestal_tracks_floor ||
             (pedestal && pedestal_profile.empty())) &&
            std::isfinite(floor_inventory_joule) &&
            std::isfinite(pedestal_inventory_joule) &&
            epoch < std::uint64_t(std::numeric_limits<std::int64_t>::max()) &&
            controller_configuration.size() <= 16384 &&
            controller_configuration.find('\n') == std::string::npos &&
            ((!controller && controller_last == 0. && controller_ema == 0. &&
              !controller_has_ema && controller_step == -1) ||
             (controller && controller_last == number_floor &&
              nonnegative(controller_ema) &&
              (controller_has_ema || controller_ema == 0.) &&
              controller_step >= -1)),
        "Invalid Eulerian density-control state");
}
void
ElectronDensityControl::ValidateConfiguration (
    const ElectronDensityControl& r) const {
    Validate();
    r.Validate();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        initial_number_floor == r.initial_number_floor &&
            initial_gate_floor == r.initial_gate_floor &&
            initial_representation_floor == r.initial_representation_floor &&
            pedestal == r.pedestal &&
            pedestal_tracks_floor == r.pedestal_tracks_floor &&
            pedestal_profile == r.pedestal_profile &&
            pedestal_taper_cells == r.pedestal_taper_cells &&
            halo_unfreeze == r.halo_unfreeze,
        "Eulerian density-control restart requires matching original floor and "
        "pedestal configuration");
}
std::string
ElectronDensityControl::Encode () const {
    Validate();
    std::ostringstream out;
    out << std::setprecision(17)
        << "WarpXElectronDensityControl 1\npolicy "
           "temperature_preserving_cell_v1\n"
        << "initial_floors " << initial_number_floor << ' '
        << initial_gate_floor << ' ' << initial_representation_floor
        << "\npedestal " << pedestal << ' ' << pedestal_tracks_floor << ' '
        << pedestal_taper_cells << ' ' << std::quoted(pedestal_profile)
        << "\nhalo_unfreeze " << halo_unfreeze << "\nlive_floors "
        << number_floor << ' ' << gate_floor << ' ' << representation_floor
        << "\nepoch " << epoch << "\ninventory_joule " << floor_inventory_joule
        << ' ' << pedestal_inventory_joule << "\ncontroller "
        << std::quoted(controller_configuration) << ' ' << controller_last
        << ' ' << controller_ema << ' ' << controller_has_ema << ' '
        << controller_step << '\n';
    return out.str();
}
ElectronDensityControl
ElectronDensityControl::Decode (const std::string& text) {
    ElectronDensityControl c;
    std::istringstream in(text);
    std::string label, magic, policy;
    int version = 0;
    in >> magic >> version;
    bool okay = magic == "WarpXElectronDensityControl" && version == 1;
    in >> label >> policy;
    okay =
        okay && label == "policy" && policy == "temperature_preserving_cell_v1";
    in >> label >> c.initial_number_floor >> c.initial_gate_floor >>
        c.initial_representation_floor;
    okay = okay && label == "initial_floors";
    in >> label >> c.pedestal >> c.pedestal_tracks_floor >>
        c.pedestal_taper_cells >> std::quoted(c.pedestal_profile);
    okay = okay && label == "pedestal";
    in >> label >> c.halo_unfreeze;
    okay = okay && label == "halo_unfreeze";
    in >> label >> c.number_floor >> c.gate_floor >> c.representation_floor;
    okay = okay && label == "live_floors";
    in >> label >> c.epoch;
    okay = okay && label == "epoch";
    in >> label >> c.floor_inventory_joule >> c.pedestal_inventory_joule;
    okay = okay && label == "inventory_joule";
    in >> label >> std::quoted(c.controller_configuration) >>
        c.controller_last >> c.controller_ema >> c.controller_has_ema >>
        c.controller_step;
    okay = okay && label == "controller" && bool(in);
    in >> std::ws;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        okay && in.eof(),
        "Invalid Eulerian density-control checkpoint metadata");
    c.Validate();
    return c;
}
DensityControlInventory
RemapElectronDensityControl (const amrex::Geometry& geometry,
                             const amrex::MultiFab& energy,
                             const amrex::MultiFab& old_density,
                             const amrex::MultiFab& floor_density,
                             const amrex::MultiFab& new_density,
                             amrex::MultiFab& candidate) {
    AMREX_ALWAYS_ASSERT(&energy != &candidate && &old_density != &candidate &&
                        &floor_density != &candidate &&
                        &new_density != &candidate &&
                        energy.ixType().cellCentered());
    for (auto const* f : {&old_density, &floor_density, &new_density,
                          static_cast<const amrex::MultiFab*>(&candidate)}) {
        AMREX_ALWAYS_ASSERT(f->boxArray() == energy.boxArray() &&
                            f->DistributionMap() == energy.DistributionMap() &&
                            f->nComp() == 1);
    }
    auto const dx = geometry.CellSizeArray(), lo = geometry.ProbLoArray();
    auto const index_lo = geometry.Domain().smallEnd();
    bool const rz = geometry.IsRZ();
    amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpMax>
        op;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, int> data(op);
    using Tuple = decltype(data)::Type;
    for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
        auto const u = energy.const_array(mfi),
                   n = old_density.const_array(mfi);
        auto const nf = floor_density.const_array(mfi),
                   nn = new_density.const_array(mfi);
        auto const out = candidate.array(mfi);
        op.eval(
            mfi.validbox(), data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                amrex::Real const before = u(i, j, k), a = n(i, j, k),
                                  b = nf(i, j, k), c = nn(i, j, k);
                bool const valid = std::isfinite(before) && before >= 0. &&
                                   std::isfinite(a) && a > 0. &&
                                   std::isfinite(b) && b > 0. &&
                                   std::isfinite(c) && c > 0.;
                if (!valid) {
                    out(i, j, k) = 0.;
                    return {0., 0., 0., 1};
                }
                amrex::Real const middle = b == a ? before : before * (b / a);
                amrex::Real const after = c == a ? before : before * (c / a);
                out(i, j, k) = after;
                amrex::Real const v =
                    rz ? volume * 6.283185307179586476925286766559 *
                             (lo[0] + (i - index_lo[0] + 0.5) * dx[0])
                       : volume;
                return {(middle - before) * v, (after - middle) * v,
                        (after - before) * v,
                        int(!std::isfinite(after) || !std::isfinite(middle))};
            });
    }
    auto const values = data.value();
    amrex::Real sum[3] = {amrex::get<0>(values), amrex::get<1>(values),
                          amrex::get<2>(values)};
    int invalid = amrex::get<3>(values);
    amrex::ParallelDescriptor::ReduceRealSum(sum, 3);
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    return {invalid == 0 && std::isfinite(sum[0]) && std::isfinite(sum[1]) &&
                std::isfinite(sum[2]),
            sum[0], sum[1], sum[2]};
}
} // namespace warpx::thermal
