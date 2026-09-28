/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "AbsorbingElectronInventory.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <cmath>

namespace warpx::thermal {
AbsorbingElectronInventoryResult
RemapAbsorbingElectronInventory (const amrex::Geometry& geometry,
                                 const amrex::MultiFab& virtual_energy,
                                 const amrex::MultiFab& virtual_density,
                                 const amrex::MultiFab& survivor_density,
                                 const amrex::MultiFab& raw_lost_number_density,
                                 amrex::MultiFab& survivor_energy) {
    AMREX_ALWAYS_ASSERT(virtual_energy.ixType().cellCentered() &&
                        virtual_energy.nComp() == 1);
    for (auto const* f :
         {&virtual_density, &survivor_density, &raw_lost_number_density,
          static_cast<const amrex::MultiFab*>(&survivor_energy)}) {
        AMREX_ALWAYS_ASSERT(f->boxArray() == virtual_energy.boxArray() &&
                            f->DistributionMap() ==
                                virtual_energy.DistributionMap() &&
                            f->nComp() == 1);
    }
    for (auto const* f : {&virtual_energy, &virtual_density, &survivor_density,
                          &raw_lost_number_density}) {
        AMREX_ALWAYS_ASSERT(f != &survivor_energy);
        for (amrex::MFIter mfi(survivor_energy); mfi.isValid(); ++mfi) {
            AMREX_ALWAYS_ASSERT((*f)[mfi].dataPtr() !=
                                survivor_energy[mfi].dataPtr());
        }
    }
#if defined(WARPX_DIM_RZ)
    AMREX_ALWAYS_ASSERT(geometry.IsRZ());
#elif defined(WARPX_DIM_3D)
    AMREX_ALWAYS_ASSERT(geometry.Coord() == 0);
#else
    amrex::Abort("Absorbing electron inventory supports RZ and Cartesian 3D");
#endif
    auto const dx = geometry.CellSizeArray(), lo = geometry.ProbLoArray();
    auto const index_lo = geometry.Domain().smallEnd();
    bool const rz = geometry.IsRZ();
    amrex::Real const cell_volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    using R = amrex::Real;
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMin>
        op;
    amrex::ReduceData<R, R, R, R, R, R, R, R, int> data(op);
    using Tuple = decltype(data)::Type;
    op.eval(1, data, [] AMREX_GPU_DEVICE(int) -> Tuple {
        return {0., 0., 0., 0., 0., 0., 0., 0., 1};
    });
    for (amrex::MFIter mfi(virtual_energy); mfi.isValid(); ++mfi) {
        auto const u = virtual_energy.const_array(mfi);
        auto const v = virtual_density.const_array(mfi);
        auto const s = survivor_density.const_array(mfi);
        auto const lost = raw_lost_number_density.const_array(mfi);
        auto const out = survivor_energy.array(mfi);
        op.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                    R const uv = u(i, j, k), nv = v(i, j, k), ns = s(i, j, k);
                    R const raw_lost_n = lost(i, j, k);
                    bool const good = std::isfinite(uv) && uv >= 0. &&
                                      std::isfinite(nv) && nv > 0. &&
                                      std::isfinite(ns) && ns > 0. &&
                                      std::isfinite(raw_lost_n);
                    if (!good) {
                        out(i, j, k) = 0.;
                        return {0., 0., 0., 0., 0., 0., 0., 0., 0};
                    }
                    R const us = ns == nv ? uv : uv * (ns / nv);
                    R const specific = uv / nv;
                    R const after_specific = us / ns;
                    R const raw = specific * raw_lost_n;
                    R const represented = uv - us;
                    R const replacement = raw - represented;
                    R const temperature_error =
                        specific == after_specific
                            ? 0.
                            : (specific > 0.
                                   ? std::abs(after_specific / specific - 1.)
                                   : 1.);
                    R const density_increase = ns > nv ? (ns - nv) / nv : 0.;
                    R const volume =
                        rz ? cell_volume * 6.283185307179586476925286766559 *
                                 (lo[0] + (i - index_lo[0] + 0.5) * dx[0])
                           : cell_volume;
                    R const a = raw * volume, b = represented * volume,
                            c = replacement * volume;
                    bool const finite = std::isfinite(us) && us >= 0. &&
                                        std::isfinite(specific) &&
                                        std::isfinite(after_specific) &&
                                        std::isfinite(temperature_error) &&
                                        std::isfinite(density_increase) &&
                                        std::isfinite(volume) && volume > 0. &&
                                        std::isfinite(a) && std::isfinite(b) &&
                                        std::isfinite(c);
                    if (!finite) {
                        out(i, j, k) = 0.;
                        return {0., 0., 0., 0., 0., 0., 0., 0., 0};
                    }
                    out(i, j, k) = us;
                    return {a,
                            b,
                            c,
                            std::abs(a),
                            std::abs(b),
                            std::abs(c),
                            temperature_error,
                            density_increase,
                            1};
                });
    }
    auto const values = data.value();
    R sums[6] = {amrex::get<0>(values), amrex::get<1>(values),
                 amrex::get<2>(values), amrex::get<3>(values),
                 amrex::get<4>(values), amrex::get<5>(values)};
    R maxima[2] = {amrex::get<6>(values), amrex::get<7>(values)};
    int valid = amrex::get<8>(values);
    amrex::ParallelDescriptor::ReduceRealSum(sums, 6);
    amrex::ParallelDescriptor::ReduceRealMax(maxima, 2);
    amrex::ParallelDescriptor::ReduceIntMin(valid);
    for (R value : sums) {
        valid = valid && std::isfinite(value);
    }
    return {valid != 0, sums[0], sums[1],   sums[2],  sums[3],
            sums[4],    sums[5], maxima[0], maxima[1]};
}
} // namespace warpx::thermal
