/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "KineticThermalSpecies.H"
#include "Utils/WarpXConst.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>
#include <cmath>
#include <set>
#include <utility>
namespace warpx::thermal {
namespace {
using MF = amrex::MultiFab;
using Real = amrex::Real;
amrex::IntVect
yee (int c) {
    amrex::IntVect t(1);
#ifdef WARPX_DIM_RZ
    if (c == 0) {
        t[0] = 0;
    }
    if (c == 2) {
        t[1] = 0;
    }
#else
    if (c < AMREX_SPACEDIM) {
        t[c] = 0;
    }
#endif
    return t;
}
amrex::GpuArray<int, 3>
stagger (const MF& f, int unused = 1) {
    amrex::GpuArray<int, 3> t{unused, unused, unused};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        t[d] = f.ixType().nodeCentered(d);
    }
    return t;
}
bool
finite (const MF& f, int ng = 0) {
    return !f.contains_nan(0, 1, ng) && !f.contains_inf(0, 1, ng);
}
bool
all (bool x) {
    int ok = x;
    amrex::ParallelDescriptor::ReduceIntMin(ok);
    return ok;
}
} // namespace
KineticThermalSpecies::KineticThermalSpecies (
    const amrex::Geometry& g, const amrex::BoxArray& cells,
    const amrex::DistributionMapping& dm, amrex::IntVect ghosts,
    std::vector<KineticSpeciesDescriptor> species, KineticSpeciesOptions o)
    : m_geometry(g), m_cells(cells),
      m_nodes(amrex::convert(cells, amrex::IntVect(1))), m_distribution(dm),
      m_ghosts(ghosts), m_descriptors(std::move(species)), m_options(o) {
#ifndef WARPX_DIM_RZ
    amrex::Abort(
        "KineticThermalSpecies supports RZ only; existing 3D paths unchanged");
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !o.embedded_boundary && o.physical_levels == 1 &&
            o.azimuthal_modes == 1,
        "KineticThermalSpecies requires single-level m0/no-EB");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        cells.ixType().cellCentered() && cells.minimalBox() == g.Domain() &&
            cells.numPts() == g.Domain().numPts() && g.Coord() == 1 &&
            !g.isPeriodic(0) && ghosts.allGE(amrex::IntVect(1)),
        "KineticThermalSpecies geometry/layout/ghost mismatch");
    std::set<std::string> names;
    for (auto const& d : m_descriptors) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !d.name.empty() && names.insert(d.name).second &&
                std::isfinite(d.charge_number) && d.charge_number > 0 &&
                std::isfinite(d.mass) && d.mass > 0 &&
                (!d.has_resistivity_overlay || bool(d.resistivity_overlay)),
            "KineticThermalSpecies invalid charged-species descriptor");
        OwnedSpecies s;
        s.rho = std::make_unique<MF>(m_nodes, dm, 1, ghosts);
        if (d.has_resistivity_overlay) {
            for (int c = 0; c < 3; ++c) {
                s.current[c] = std::make_unique<MF>(
                    amrex::convert(cells, yee(c)), dm, 1, ghosts);
            }
        }
        if (o.relaxation && !d.relaxation_excluded) {
            s.ti_cell = std::make_unique<MF>(cells, dm, 1, 1);
            s.ti_node = std::make_unique<MF>(m_nodes, dm, 1, 0);
        }
        m_owned.push_back(std::move(s));
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_owned.empty(), "KineticThermalSpecies needs all charged species");
    if (o.relaxation) {
        for (int c = 0; c < 3; ++c) {
            auto ba = amrex::convert(cells, yee(c));
            m_temperature[c] = std::make_unique<MF>(ba, dm, 1, ghosts);
            m_weight[c] = std::make_unique<MF>(ba, dm, 1, ghosts);
            m_variance[c] = std::make_unique<MF>(ba, dm, 1, ghosts);
            m_velocity_sum[c] = std::make_unique<MF>(ba, dm, 1, ghosts);
            m_count[c] = std::make_unique<amrex::iMultiFab>(ba, dm, 1, ghosts);
        }
    }
    m_old_ready = !o.relaxation;
}
void
KineticThermalSpecies::Sum (MF& f) const {
    ablastr::utils::communication::SumBoundary(
        f, 0, 1, f.nGrowVect(), f.nGrowVect(), m_options.single_precision_comms,
        m_geometry.periodicity());
}
bool
KineticThermalSpecies::FreezeOldTemperature (const TemperatureDeposit& deposit,
                                             const TemperatureFilter& filter) {
    m_old_ready = !m_options.relaxation;
    m_trial_ready = false;
    m_views.clear();
    if (!m_options.relaxation) {
        return true;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        bool(deposit), "KineticThermalSpecies missing old-temperature deposit");
    SpeciesVarianceFields f;
    for (int c = 0; c < 3; ++c) {
        f.temperature[c] = m_temperature[c].get();
        f.weight[c] = m_weight[c].get();
        f.variance[c] = m_variance[c].get();
        f.velocity_sum[c] = m_velocity_sum[c].get();
        f.count[c] = m_count[c].get();
    }
    for (std::size_t s = 0; s < m_owned.size(); ++s) {
        if (m_descriptors[s].relaxation_excluded) {
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            for (auto* p : {f.temperature[c], f.weight[c], f.variance[c],
                            f.velocity_sum[c]}) {
                p->setVal(0);
            }
            f.count[c]->setVal(0);
        }
        if (!all(deposit(s, SpeciesVariancePass::First, f))) {
            return false;
        }
        amrex::Gpu::streamSynchronize();
        for (int c = 0; c < 3; ++c) {
            f.count[c]->SumBoundary(0, 1, m_ghosts, m_ghosts,
                                    m_geometry.periodicity());
            Sum(*f.weight[c]);
            Sum(*f.velocity_sum[c]);
        }
        if (!all(deposit(s, SpeciesVariancePass::Second, f))) {
            return false;
        }
        amrex::Gpu::streamSynchronize();
        for (int c = 0; c < 3; ++c) {
            Sum(*f.variance[c]);
            for (amrex::MFIter mfi(*f.temperature[c]); mfi.isValid(); ++mfi) {
                auto n = f.count[c]->const_array(mfi);
                auto w = f.weight[c]->const_array(mfi);
                auto v = f.variance[c]->const_array(mfi);
                auto t = f.temperature[c]->array(mfi);
                // Legacy sample correction n/((n-1)*sumw), including ghosts.
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    if (n(i, j, k) > 1) {
                        Real const count = n(i, j, k);
                        Real const norm = count / ((count - 1) * w(i, j, k));
                        t(i, j, k) = norm * v(i, j, k);
                    }
                });
            }
            f.temperature[c]->mult(m_descriptors[s].mass / PhysConst::kb, 0, 1,
                                   0);
            ablastr::utils::communication::FillBoundary(
                *f.temperature[c], m_options.single_precision_comms,
                m_geometry.periodicity(), true);
        }
        if (filter) {
            filter(s, f.temperature);
        }
        for (auto* t : f.temperature) {
            ablastr::utils::communication::FillBoundary(
                *t, m_options.single_precision_comms, m_geometry.periodicity(),
                true);
            if (!finite(*t, t->nGrow())) {
                return false;
            }
        }
        ConvertTemperature(s);
        if (!finite(*m_owned[s].ti_node)) {
            return false;
        }
    }
    m_old_ready = true;
    return true;
}
namespace {
void
convert_temperature (
    amrex::MultiFab& cc, amrex::MultiFab& node,
    const std::array<std::unique_ptr<amrex::MultiFab>, 3>& temperature,
    const amrex::Geometry& geometry) {
    cc.setVal(0);
    auto const a = stagger(*temperature[0]), b = stagger(*temperature[1]),
               c = stagger(*temperature[2]);
    amrex::GpuArray<int, 3> cell{0, 0, 0}, nodal{1, 1, 1}, ratio{1, 1, 1};
    for (int d = AMREX_SPACEDIM; d < 3; ++d) {
        cell[d] = 1;
    }
    Real const convert = PhysConst::kb / (3 * PhysConst::q_e);
    for (amrex::MFIter mfi(cc); mfi.isValid(); ++mfi) {
        auto t = cc.array(mfi);
        auto tx = temperature[0]->const_array(mfi);
        auto ty = temperature[1]->const_array(mfi),
             tz = temperature[2]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            t(i, j, k) =
                convert * (ablastr::coarsen::sample::Interp(tx, a, cell, ratio,
                                                            i, j, k, 0) +
                           ablastr::coarsen::sample::Interp(ty, b, cell, ratio,
                                                            i, j, k, 0) +
                           ablastr::coarsen::sample::Interp(tz, c, cell, ratio,
                                                            i, j, k, 0));
        });
    }
    cc.FillBoundary(
        geometry.periodicity()); // nonperiodic exterior remains zero
    for (amrex::MFIter mfi(node); mfi.isValid(); ++mfi) {
        auto t = node.array(mfi);
        auto u = cc.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               t(i, j, k) = ablastr::coarsen::sample::Interp(
                                   u, cell, nodal, ratio, i, j, k, 0);
                           });
    }
}
} // namespace
void
KineticThermalSpecies::ConvertTemperature (std::size_t s) {
    convert_temperature(*m_owned[s].ti_cell, *m_owned[s].ti_node, m_temperature,
                        m_geometry);
}
bool
KineticThermalSpecies::RefreshMaterializedTrial (MaterializedSpeciesTrial state,
                                                 const TrialDeposit& deposit) {
    m_trial_ready = false;
    m_views.clear();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        state == MaterializedSpeciesTrial::FullParticleState,
        "KineticThermalSpecies aggregate MM has no per-species response; "
        "materialize full probe particles");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_old_ready && bool(deposit),
        "KineticThermalSpecies requires fixed-old Ti and a trial depositor");
    for (std::size_t s = 0; s < m_owned.size(); ++s) {
        auto& own = m_owned[s];
        SpeciesMutableVector j{};
        own.rho->setVal(0);
        for (int c = 0; c < 3; ++c) {
            if (own.current[c]) {
                j[c] = own.current[c].get();
                j[c]->setVal(0);
            }
        }
        if (!all(deposit(s, *own.rho, j))) {
            return false;
        }
        Sum(*own.rho);
        if (!finite(*own.rho, own.rho->nGrow())) {
            return false;
        }
        for (auto* f : j) {
            if (f) {
                Sum(*f);
                if (!finite(*f, f->nGrow())) {
                    return false;
                }
            }
        }
        auto const& d = m_descriptors[s];
        ThermalSourceSpecies view;
        view.charge_number = d.charge_number;
        view.relaxation_excluded = d.relaxation_excluded;
        view.has_resistivity_overlay = d.has_resistivity_overlay;
        view.resistivity_overlay = d.resistivity_overlay;
        view.raw_charge = own.rho.get();
        view.ion_temperature_ev = own.ti_node.get();
        for (int c = 0; c < 3; ++c) {
            view.raw_current[c] = j[c];
        }
        m_views.push_back(view);
    }
    m_trial_ready = true;
    return true;
}
const std::vector<ThermalSourceSpecies>&
KineticThermalSpecies::Species () const {
    AMREX_ALWAYS_ASSERT(m_trial_ready);
    return m_views;
}
const MF&
KineticThermalSpecies::OldCellTemperature (std::size_t s) const {
    AMREX_ALWAYS_ASSERT(m_old_ready && m_owned.at(s).ti_cell);
    return *m_owned[s].ti_cell;
}
const MF&
KineticThermalSpecies::OldNodalTemperature (std::size_t s) const {
    AMREX_ALWAYS_ASSERT(m_old_ready && m_owned.at(s).ti_node);
    return *m_owned[s].ti_node;
}
} // namespace warpx::thermal
