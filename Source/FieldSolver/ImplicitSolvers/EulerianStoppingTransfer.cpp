/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "EulerianStoppingTransfer.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "KineticThermalMoments.H"
#include "Utils/WarpXConst.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace warpx::thermal {
namespace {
using amrex::Real;
using MF = amrex::MultiFab;
bool
finite (const MF& f) {
    return !f.contains_nan(0, 1, 0) && !f.contains_inf(0, 1, 0);
}
bool
overlaps (const MF& a, const MF& b) {
    using Range = std::pair<std::uintptr_t, std::uintptr_t>;
    std::vector<Range> ranges;
    for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto const begin = reinterpret_cast<std::uintptr_t>(a[mfi].dataPtr());
        ranges.emplace_back(begin, begin + a[mfi].box().numPts() * a.nComp() *
                                               sizeof(Real));
    }
    std::sort(ranges.begin(), ranges.end());
    for (amrex::MFIter mfi(b); mfi.isValid(); ++mfi) {
        auto const begin = reinterpret_cast<std::uintptr_t>(b[mfi].dataPtr());
        auto const end =
            begin + b[mfi].box().numPts() * b.nComp() * sizeof(Real);
        auto const next =
            std::lower_bound(ranges.begin(), ranges.end(), Range{end, 0});
        if (next != ranges.begin() && std::prev(next)->second > begin) {
            return true;
        }
    }
    return false;
}
Real
integral (const MF& f, const amrex::Geometry& g,
          const amrex::iMultiFab* owner = nullptr) {
    auto volume = MakeQdsmcVolumeElement(g, f.ixType());
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<Real> data(op);
    using Tuple = decltype(data)::Type;
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto a = f.const_array(mfi);
        auto own = owner ? owner->const_array(mfi) : amrex::Array4<const int>{};
        op.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                    return {(!own || own(i, j, k))
                                ? volume(i, j, k) * a(i, j, k)
                                : Real(0)};
                });
    }
    Real result = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}
} // namespace

EulerianStoppingTransfer::EulerianStoppingTransfer (
    const amrex::Geometry& geometry, const amrex::BoxArray& cells,
    const amrex::DistributionMapping& distribution,
    StoppingTransferOptions options)
    : m_geometry(geometry), m_cells(cells),
      m_nodes(amrex::convert(cells, amrex::IntVect::TheNodeVector())),
      m_distribution(distribution), m_options(options),
      m_impulse(m_nodes, distribution, 1, 0),
      m_charge(m_nodes, distribution, 1, 0),
      m_eligible(m_nodes, distribution, 1, 0),
      m_declined(m_nodes, distribution, 1, 0),
      m_seam_difference(m_nodes, distribution, 1, 0),
      m_requested(cells, distribution, 1, 0),
      m_candidate(cells, distribution, 1, 0),
      m_delivered(cells, distribution, 1, 0),
      m_floor_declined(cells, distribution, 1, 0),
      m_owner(m_impulse.OwnerMask(geometry.periodicity())) {
#ifndef WARPX_DIM_RZ
    amrex::Abort("EulerianStoppingTransfer supports RZ only; existing 3D paths "
                 "are unchanged");
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !options.embedded_boundary && options.physical_levels == 1 &&
            options.azimuthal_modes == 1,
        "EulerianStoppingTransfer requires single-level m0/no-EB");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        cells.ixType().cellCentered() &&
            cells.minimalBox() == geometry.Domain() &&
            cells.numPts() == geometry.Domain().numPts() &&
            geometry.Coord() == 1 && !geometry.isPeriodic(0) &&
            geometry.ProbLo(0) == 0 &&
            geometry.Domain().smallEnd() == amrex::IntVect::TheZeroVector(),
        "EulerianStoppingTransfer requires a complete zero-based physical RZ "
        "mesh");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(options.gamma) && options.gamma > 1 &&
            std::isfinite(options.raw_density_floor) &&
            options.raw_density_floor >= 0 &&
            std::isfinite(options.temperature_floor_ev) &&
            options.temperature_floor_ev >= 0,
        "EulerianStoppingTransfer invalid options");
}

void
EulerianStoppingTransfer::CheckInput (const MF& f, bool nodal) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !f.hasEBFabFactory() && f.nComp() == 1 &&
            f.boxArray() == (nodal ? m_nodes : m_cells) &&
            f.DistributionMap() == m_distribution,
        "EulerianStoppingTransfer input layout/EB mismatch");
    int alias = 0;
    for (auto const* out :
         {&m_impulse, &m_charge, &m_eligible, &m_declined, &m_seam_difference,
          &m_requested, &m_candidate, &m_delivered, &m_floor_declined}) {
        alias = alias || overlaps(f, *out);
    }
    amrex::ParallelDescriptor::ReduceIntMax(alias);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !alias, "EulerianStoppingTransfer input aliases owned scratch");
}

bool
EulerianStoppingTransfer::CopyConsolidated (MF& out, const MF& in) {
    MF::Copy(out, in, 0, 0, 1, 0);
    out.OverrideSync(m_geometry.periodicity());
    MF::LinComb(m_seam_difference, 1, out, 0, -1, in, 0, 0, 1, 0);
    // SumBoundary may sum shared contributions in a different floating-point
    // order on each copy. Reconcile that roundoff only; this is NOT a
    // substitute for the caller's one SumBoundary of partial deposits.
    Real const scale = std::max(out.norminf(), in.norminf());
    return m_seam_difference.norminf() <=
           64 * std::numeric_limits<Real>::epsilon() * scale;
}

bool
EulerianStoppingTransfer::Evaluate (KineticThermalMoments& map,
                                    const MF& impulse, const MF& raw_charge,
                                    const MF& capacity, const MF& before) {
    m_valid = false;
    m_ledger = {};
    CheckInput(impulse, true);
    CheckInput(raw_charge, true);
    CheckInput(capacity, false);
    CheckInput(before, false);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        map.NumberDensity().boxArray() == m_cells &&
            map.NumberDensity().DistributionMap() == m_distribution,
        "EulerianStoppingTransfer physical map layout mismatch");
    if (!finite(impulse) || !finite(raw_charge) || !finite(capacity) ||
        !finite(before) || capacity.min(0) <= 0 || before.min(0) < 0) {
        return false;
    }
    if (!CopyConsolidated(m_impulse, impulse) ||
        !CopyConsolidated(m_charge, raw_charge)) {
        return false;
    }
    Real const rho_floor = PhysConst::q_e * m_options.raw_density_floor;
    for (amrex::MFIter mfi(m_eligible); mfi.isValid(); ++mfi) {
        auto in = m_impulse.const_array(mfi), rho = m_charge.const_array(mfi);
        auto eligible = m_eligible.array(mfi), declined = m_declined.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                bool const active = rho(i, j, k) > rho_floor;
                eligible(i, j, k) = active ? in(i, j, k) : Real(0);
                declined(i, j, k) = active ? Real(0) : in(i, j, k);
            });
    }
    // Physical impulse quadrature, even when rho used native axis3 moments.
    map.RestrictNodalScalar(m_eligible, 0, m_requested);
    Real const floor_per_particle =
        PhysConst::q_e * m_options.temperature_floor_ev / (m_options.gamma - 1);
    for (amrex::MFIter mfi(m_candidate); mfi.isValid(); ++mfi) {
        auto u = before.const_array(mfi), n = capacity.const_array(mfi),
             du = m_requested.const_array(mfi);
        auto candidate = m_candidate.array(mfi),
             delivered = m_delivered.array(mfi),
             declined = m_floor_declined.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const minimum =
                    amrex::min(u(i, j, k), n(i, j, k) * floor_per_particle);
                Real const requested = u(i, j, k) + du(i, j, k);
                Real const accepted = amrex::max(requested, minimum);
                candidate(i, j, k) = accepted;
                delivered(i, j, k) =
                    accepted - u(i, j, k); // measured stored change
                declined(i, j, k) = amrex::max(minimum - requested, Real(0));
            });
    }
    if (!finite(m_requested) || !finite(m_candidate) || !finite(m_delivered) ||
        !finite(m_floor_declined)) {
        return false;
    }
    m_ledger.requested = integral(m_impulse, m_geometry, m_owner.get());
    m_ledger.density_eligible = integral(m_eligible, m_geometry, m_owner.get());
    m_ledger.density_declined = integral(m_declined, m_geometry, m_owner.get());
    m_ledger.mapped = integral(m_requested, m_geometry);
    m_ledger.delivered = integral(m_delivered, m_geometry);
    m_ledger.floor_declined = integral(m_floor_declined, m_geometry);
    m_ledger.map_defect = m_ledger.mapped - m_ledger.density_eligible;
    m_ledger.exchange_defect = m_ledger.delivered - m_ledger.requested +
                               m_ledger.density_declined -
                               m_ledger.floor_declined;
    m_valid = std::isfinite(m_ledger.requested) &&
              std::isfinite(m_ledger.density_eligible) &&
              std::isfinite(m_ledger.density_declined) &&
              std::isfinite(m_ledger.mapped) &&
              std::isfinite(m_ledger.delivered) &&
              std::isfinite(m_ledger.floor_declined) &&
              std::isfinite(m_ledger.map_defect) &&
              std::isfinite(m_ledger.exchange_defect);
    return m_valid;
}
const MF&
EulerianStoppingTransfer::CandidateEnergy () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_candidate;
}
const MF&
EulerianStoppingTransfer::CellImpulse () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_delivered;
}
const MF&
EulerianStoppingTransfer::RequestedCellImpulse () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_requested;
}
const MF&
EulerianStoppingTransfer::NodalEligibleImpulse () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_eligible;
}
const MF&
EulerianStoppingTransfer::NodalDeclinedImpulse () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_declined;
}
const MF&
EulerianStoppingTransfer::CellFloorDeclinedImpulse () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_floor_declined;
}
const StoppingTransferLedger&
EulerianStoppingTransfer::Ledger () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_ledger;
}
} // namespace warpx::thermal
