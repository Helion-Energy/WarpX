/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "EulerianThermalSources.H"
#include "ThermalWorkRedistribution.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Utils/WarpXConst.H"
#include "ablastr/coarsen/sample.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace warpx::thermal {
namespace {
using amrex::Real;
using ConstVector = std::array<const amrex::MultiFab*, 3>;
using DeviceVector = amrex::GpuArray<amrex::Array4<const Real>, 3>;
using Staggers = amrex::GpuArray<amrex::GpuArray<int, 3>, 3>;

Staggers
staggers (const ConstVector& vector) {
    Staggers result;
    for (int c = 0; c < 3; ++c) {
        result[c] = {1, 1, 1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            result[c][d] = vector[c]->ixType().nodeCentered(d);
        }
    }
    return result;
}

DeviceVector
arrays (const ConstVector& vector, const amrex::MFIter& mfi) {
    return {vector[0]->const_array(mfi), vector[1]->const_array(mfi),
            vector[2]->const_array(mfi)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
square_magnitude (DeviceVector const& v, Staggers const& s, int i, int j,
                  int k) {
    amrex::GpuArray<int, 3> const nodal{1, 1, 1}, coarsen{1, 1, 1};
    Real result = 0;
    for (int c = 0; c < 3; ++c) {
        Real const x = ablastr::coarsen::sample::Interp(v[c], s[c], nodal,
                                                        coarsen, i, j, k, 0);
        result += x * x;
    }
    return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
halo_weight (Real n, Real threshold) {
    Real const x =
        amrex::min(Real(1), amrex::max(Real(0), (n - threshold) / threshold));
    return x * x * (3 - 2 * x);
}
} // namespace

EulerianThermalSources::EulerianThermalSources (
    const amrex::Geometry& geometry, const amrex::MultiFab& nodal_layout,
    ThermalSourceOptions options)
    : m_geometry(geometry), m_nodes(nodal_layout.boxArray()),
      m_distribution(nodal_layout.DistributionMap()),
      m_options(std::move(options)),
      m_species_sum(m_nodes, m_distribution, 1, 0),
      m_work_eligible(m_nodes, m_distribution, 1, 2),
      m_owner(nodal_layout.OwnerMask(geometry.periodicity())) {
#ifndef WARPX_DIM_RZ
    amrex::Abort("EulerianThermalSources supports RZ only; existing 3D paths "
                 "are unchanged");
#endif
    CheckNodal(&nodal_layout, 1);
    auto const& o = m_options;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !o.embedded_boundary && o.physical_levels == 1 &&
            !o.temperature_shunt && !o.pedestal_temperature_cap &&
            !o.redirect_kick_cap,
        "EulerianThermalSources: EB, AMR, temperature shunt/pedestal cap and "
        "redirect kick cap "
        "require separately derived thermal stages and are unsupported");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(o.density_floor) && o.density_floor >= 0 &&
            std::isfinite(o.physical_dt) && o.physical_dt > 0 &&
            std::isfinite(o.time) && std::isfinite(o.joule_density_gate) &&
            o.joule_density_gate >= 0 &&
            std::isfinite(o.source_taper_density) &&
            o.source_taper_density >= 0 &&
            std::isfinite(o.redirect_temperature_ev) &&
            o.redirect_temperature_ev >= 0 &&
            std::isfinite(o.redirect_density_floor_factor) &&
            o.redirect_density_floor_factor >= 0 &&
            (!o.joule_taper ||
             std::max(o.joule_density_gate, o.density_floor) > 0) &&
            (o.viscosity == ViscousSourceKind::None ||
             o.viscosity == ViscousSourceKind::AppliedWork ||
             o.viscosity == ViscousSourceKind::Strain),
        "EulerianThermalSources: invalid source options");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!o.joule || o.joule_applied_edge_work || (o.use_heating_eta
                          ? bool(o.heating_eta)
                          : (o.field_eta_uses_kelvin ? bool(o.field_eta_kelvin)
                                                     : bool(o.field_eta)))) &&
            (!o.relaxation || bool(o.relaxation_rate)) &&
            (!o.external_sink || bool(o.sink)),
        "EulerianThermalSources enabled source requires a live compiled "
        "parser");
    if (o.joule_applied_edge_work) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(o.joule && !o.use_heating_eta &&
            !o.redirect_joule && !o.joule_taper && o.source_taper_density == 0 &&
            o.joule_density_gate <= o.density_floor,
            "Applied Joule work requires common eta without independent heating, redirect or extra gates/tapers");
        m_joule_capacity.define(m_nodes, m_distribution, 1, 0);
        m_joule_work.define(m_nodes, m_distribution, 3, 0);
    }
    auto const cells = amrex::convert(m_nodes, amrex::IntVect::TheCellVector());
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        cells.minimalBox() == geometry.Domain() &&
            cells.numPts() == geometry.Domain().numPts(),
        "EulerianThermalSources requires one complete physical mesh");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 1 && !geometry.isPeriodic(0) &&
            geometry.ProbLo(0) >= 0,
        "EulerianThermalSources requires nonperiodic radial RZ geometry");
#if defined(WARPX_DIM_RZ)
    for (int c = 0; c < 3; ++c) {
        amrex::IntVect type(1);
        if (c == 0) {
            type[0] = 0;
        }
        if (c == 2) {
            type[1] = 0;
        }
        m_work[c] = std::make_unique<amrex::MultiFab>(
            amrex::convert(m_nodes, type), m_distribution, 4, 2);
    }
#endif
}

void
EulerianThermalSources::CheckNodal (const amrex::MultiFab* input,
                                    int components) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        input && !input->hasEBFabFactory() && input->is_nodal() &&
            input->nComp() >= components && input->boxArray() == m_nodes &&
            input->DistributionMap() == m_distribution,
        "EulerianThermalSources requires matching fully nodal, non-EB source "
        "arrays");
}

void
EulerianThermalSources::CheckYee (const ConstVector& input) const {
    auto const cells = amrex::convert(m_nodes, amrex::IntVect::TheCellVector());
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            input[c] && !input[c]->hasEBFabFactory() &&
                input[c]->nComp() == 1 &&
                amrex::convert(input[c]->boxArray(),
                               amrex::IntVect::TheCellVector()) == cells &&
                input[c]->DistributionMap() == m_distribution &&
                input[c]->nGrow() >= 1,
            "EulerianThermalSources requires matching single-mode Yee vectors "
            "with filled ghosts");
    }
}

namespace {
struct WorkStencil {
    amrex::GpuArray<amrex::GpuArray<int, 3>, 2> inner{};
    amrex::GpuArray<amrex::GpuArray<int, 3>, 10> outer{};
    int ninner = 1, nouter = 0;
};

WorkStencil
work_stencil (int component) {
    WorkStencil s;
    int direction = component;
#if AMREX_SPACEDIM == 2
    direction = component == 0 ? 0 : (component == 2 ? 1 : -1);
#elif AMREX_SPACEDIM == 1
    direction = component == 2 ? 0 : -1;
#endif
    if (direction < 0) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            s.outer[s.nouter++][d] = -1;
            s.outer[s.nouter++][d] = 1;
        }
    } else {
        s.ninner = 2;
        s.inner[1][direction] = 1;
        s.outer[s.nouter++][direction] = -1;
        s.outer[s.nouter++][direction] = 2;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (d == direction) { continue; }
            for (int endpoint = 0; endpoint < 2; ++endpoint) {
                for (int side : {-1, 1}) {
                    s.outer[s.nouter][direction] = endpoint;
                    s.outer[s.nouter++][d] = side;
                }
            }
        }
    }
    return s;
}
} // namespace

void
RedistributeThermalWork (
    const amrex::Geometry& geometry, Real density_floor,
    const amrex::MultiFab& raw_charge, const amrex::MultiFab& temperature,
    const amrex::MultiFab& capacity, const ConstVector& current,
    const ConstVector& electric, amrex::MultiFab& eligibility,
    std::array<std::unique_ptr<amrex::MultiFab>, 3>& work_fields,
    amrex::MultiFab& rates, int accepted, int declined, int unassignable_abs) {
    AMREX_ALWAYS_ASSERT(eligibility.nGrowVect().allGE(amrex::IntVect(2)));
    Real const floor = PhysConst::q_e * density_floor;
    eligibility.setVal(0);
    // Freeze the entire valid mask before any edge reads it. Only scratch
    // receives boundary exchange; trial/accepted input arrays stay untouched.
    for (amrex::MFIter mfi(eligibility, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const dst = eligibility.array(mfi);
        auto const r = raw_charge.const_array(mfi);
        auto const t = temperature.const_array(mfi);
        auto const c = capacity.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            dst(i, j, k) = r(i, j, k) > floor && t(i, j, k) > 0 &&
                                  c(i, j, k) > 0 && std::isfinite(t(i, j, k)) &&
                                  std::isfinite(c(i, j, k))
                              ? Real(1) : Real(0);
        });
    }
    eligibility.FillBoundary(geometry.periodicity());
    amrex::GpuArray<WorkStencil, 3> stencils;
    for (int c = 0; c < 3; ++c) {
        auto& work = *work_fields[c];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            current[c]->boxArray() == work.boxArray() &&
                electric[c]->boxArray() == work.boxArray() && work.nComp() >= 4 &&
                work.nGrowVect().allGE(amrex::IntVect(2)),
            "Thermal work requires matching Yee arrays and four scratch channels/two ghosts");
        auto const stencil = stencils[c] = work_stencil(c);
        auto const volume = MakeQdsmcVolumeElement(geometry, work.ixType());
        work.setVal(0);
        for (amrex::MFIter mfi(work, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const w = work.array(mfi);
            auto const mask = eligibility.const_array(mfi);
            auto const j = current[c]->const_array(mfi);
            auto const e = electric[c]->const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int z, int k) {
                Real const power = volume(i, z, k) * j(i, z, k) * e(i, z, k);
                int count = 0;
                for (int n = 0; n < stencil.ninner; ++n) {
                    auto const o = stencil.inner[n];
                    count += int(mask(i + o[0], z + o[1], k + o[2]));
                }
                if (count > 0) {
                    w(i, z, k, 0) = power / Real(count);
                } else {
                    for (int n = 0; n < stencil.nouter; ++n) {
                        auto const o = stencil.outer[n];
                        count += int(mask(i + o[0], z + o[1], k + o[2]));
                    }
                    if (count > 0) {
                        w(i, z, k, 1) = power / Real(count);
                    } else {
                        w(i, z, k, 2) = power;
                        w(i, z, k, 3) = std::abs(power);
                    }
                }
            });
        }
        work.FillBoundary(geometry.periodicity());
    }
    auto const volume = MakeQdsmcVolumeElement(geometry, rates.ixType());
    for (amrex::MFIter mfi(rates, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const q = rates.array(mfi);
        auto const mask = eligibility.const_array(mfi);
        DeviceVector const work{work_fields[0]->const_array(mfi),
                                work_fields[1]->const_array(mfi),
                                work_fields[2]->const_array(mfi)};
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Real assigned = 0, orphan = 0, orphan_abs = 0;
            bool const eligible = mask(i, j, k) != 0;
            for (int c = 0; c < 3; ++c) {
                auto const st = stencils[c];
                auto const w = work[c];
                for (int n = 0; n < st.ninner; ++n) {
                    auto const o = st.inner[n];
                    int const x = i - o[0], y = j - o[1], z = k - o[2];
                    if (eligible) { assigned += w(x, y, z, 0); }
                    // Diagnostic-only split over the original physical nodes;
                    // this books orphan work even when no node can accept heat.
                    orphan += w(x, y, z, 2) / Real(st.ninner);
                    orphan_abs += w(x, y, z, 3) / Real(st.ninner);
                }
                if (eligible) {
                    for (int n = 0; n < st.nouter; ++n) {
                        auto const o = st.outer[n];
                        assigned += w(i - o[0], j - o[1], k - o[2], 1);
                    }
                }
            }
            Real const v = volume(i, j, k);
            q(i, j, k, accepted) = assigned / v;
            q(i, j, k, declined) = orphan / v;
            q(i, j, k, unassignable_abs) = orphan_abs / v;
        });
    }
}

void
EulerianThermalSources::MatchedWork (const ThermalSourceState& state,
                                     const ConstVector& electric,
                                     amrex::MultiFab& rates, int component) {
    CheckYee(electric);
    auto const& capacity = state.heat_capacity_charge
                               ? *state.heat_capacity_charge : *state.raw_charge;
    bool const viscous = component == SourceComponent::ViscousWork;
    RedistributeThermalWork(
        m_geometry, m_options.density_floor, *state.raw_charge,
        *state.temperature_kelvin, capacity, state.plasma_current, electric,
        m_work_eligible, m_work, rates, component,
        viscous ? SourceComponent::DeclinedViscousWork
                : SourceComponent::DeclinedHyperResistive,
        viscous ? SourceComponent::UnassignableViscousWorkAbs
                : SourceComponent::UnassignableHyperResistiveAbs);
}

bool
EulerianThermalSources::Evaluate (const ThermalSourceState& state,
                                  amrex::MultiFab& rates,
                                  amrex::MultiFab& ion_rates) {
    CheckNodal(state.raw_charge, 1);
    CheckNodal(state.temperature_kelvin, 1);
    CheckNodal(&rates, SourceComponent::Count);
    CheckNodal(&ion_rates, std::max(1, static_cast<int>(state.species.size()) *
                                           IonSourceComponent::Count));
    CheckYee(state.plasma_current);
    CheckYee(state.magnetic_field);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        &rates != &ion_rates && &rates != state.raw_charge &&
            &rates != state.temperature_kelvin &&
            &ion_rates != state.raw_charge &&
            &ion_rates != state.temperature_kelvin,
        "EulerianThermalSources outputs must be disjoint scratch");
#if defined(WARPX_DIM_RZ)
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.plasma_current[c]->boxArray() ==
                                             m_work[c]->boxArray(),
                                         "EulerianThermalSources requires the "
                                         "single-mode RZ Yee plasma current");
    }
#endif
    for (auto const* input :
         {state.additive_resistivity, state.stopping_impulse,
          state.viscous_strain_power, state.heat_capacity_charge}) {
        if (input) {
            CheckNodal(input, 1);
        }
    }
    auto const o = m_options;
    if (o.viscosity == ViscousSourceKind::Strain) {
        CheckNodal(state.viscous_strain_power, 1);
    }
    m_species_sum.setVal(0);
    for (auto const& species : state.species) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(species.charge_number) && species.charge_number > 0,
            "EulerianThermalSources requires positive charged species; omit "
            "neutrals");
        CheckNodal(species.raw_charge, 1);
        if (o.relaxation && !species.relaxation_excluded) {
            CheckNodal(species.ion_temperature_ev, 1);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!o.joule_applied_edge_work ||
            !species.has_resistivity_overlay,
            "Applied global Joule work cannot silently replace a species overlay");
        if (species.has_resistivity_overlay) {
            CheckYee(species.raw_current);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                bool(species.resistivity_overlay),
                "EulerianThermalSources species overlay requires a live "
                "compiled parser");
        }
        amrex::MultiFab::Add(m_species_sum, *species.raw_charge, 0, 0, 1, 0);
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!(o.joule || o.relaxation) ||
                                         !state.species.empty(),
                                     "EulerianThermalSources Joule/relaxation "
                                     "require the charged-species context");
    auto disjoint = [&] (const amrex::MultiFab* input) {
        if (!input) {
            return;
        }
        for (auto const* output : {&rates, &ion_rates}) {
            for (amrex::MFIter mfi(*output); mfi.isValid(); ++mfi) {
                auto const& a = (*output)[mfi];
                auto const& b = (*input)[mfi];
                auto const abegin =
                    reinterpret_cast<std::uintptr_t>(a.dataPtr());
                auto const bbegin =
                    reinterpret_cast<std::uintptr_t>(b.dataPtr());
                auto const aend =
                    abegin + sizeof(Real) * a.box().numPts() * a.nComp();
                auto const bend =
                    bbegin + sizeof(Real) * b.box().numPts() * b.nComp();
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    aend <= bbegin || bend <= abegin,
                    "EulerianThermalSources outputs must be disjoint scratch "
                    "(including aliases)");
            }
        }
    };
    for (auto const* input :
         {state.raw_charge, state.temperature_kelvin,
          state.additive_resistivity, state.stopping_impulse,
          state.viscous_strain_power, state.heat_capacity_charge}) {
        disjoint(input);
    }
    for (auto const* vector :
         {&state.plasma_current, &state.magnetic_field, &state.viscous_electric,
          &state.hyper_electric, &state.joule_electric}) {
        bool const present = (*vector)[0] || (*vector)[1] || (*vector)[2];
        if (present) {
            CheckYee(*vector);
        }
        for (auto const* input : *vector) {
            disjoint(input);
        }
    }
    for (auto const& species : state.species) {
        disjoint(species.raw_charge);
        if (o.relaxation && !species.relaxation_excluded) {
            disjoint(species.ion_temperature_ev);
        }
        if (species.has_resistivity_overlay) {
            for (auto const* input : species.raw_current) {
                disjoint(input);
            }
        }
    }
    // Output/output aliases also violate the transaction boundary.
    for (amrex::MFIter mfi(rates); mfi.isValid(); ++mfi) {
        auto const& a = rates[mfi];
        auto const& b = ion_rates[mfi];
        auto const abegin = reinterpret_cast<std::uintptr_t>(a.dataPtr());
        auto const bbegin = reinterpret_cast<std::uintptr_t>(b.dataPtr());
        auto const aend = abegin + sizeof(Real) * a.box().numPts() * a.nComp();
        auto const bend = bbegin + sizeof(Real) * b.box().numPts() * b.nComp();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            aend <= bbegin || bend <= abegin,
            "EulerianThermalSources outputs must be disjoint scratch "
            "(including aliases)");
    }
    rates.setVal(0);
    ion_rates.setVal(0);
    Real const rho_floor = PhysConst::q_e * o.density_floor;
    Real const n_gate = std::max(o.joule_density_gate, o.density_floor);
    Real const nan = std::numeric_limits<Real>::quiet_NaN();
    auto const current_stag = staggers(state.plasma_current);
    auto const magnetic_stag = staggers(state.magnetic_field);
    if (o.joule_applied_edge_work) {
        CheckYee(state.joule_electric);
        // A raw species sum has different units from physical rho in RZ.
        // Use it ONLY as a population predicate, never as a capacity/floor.
        // Empty receivers are excluded before conservative redistribution;
        // otherwise dividing into empty fractions would silently lose work.
        auto const& capacity = state.heat_capacity_charge
                                   ? *state.heat_capacity_charge : *state.raw_charge;
        for (amrex::MFIter mfi(m_joule_capacity, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const out = m_joule_capacity.array(mfi);
            auto const sum = m_species_sum.const_array(mfi);
            auto const cap = capacity.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                out(i,j,k) = std::isfinite(sum(i,j,k)) && sum(i,j,k)>0
                                 ? cap(i,j,k) : Real(0);
            });
        }
        RedistributeThermalWork(m_geometry, o.density_floor, *state.raw_charge,
            *state.temperature_kelvin, m_joule_capacity, state.plasma_current,
            state.joule_electric, m_work_eligible, m_work, m_joule_work, 0, 1, 2);
        amrex::MultiFab::Copy(rates,m_joule_work,1,SourceComponent::DeclinedJouleWork,1,0);
        amrex::MultiFab::Copy(rates,m_joule_work,2,SourceComponent::UnassignableJouleWorkAbs,1,0);
    }

    for (int s = 0; s < static_cast<int>(state.species.size()); ++s) {
        auto const& species = state.species[s];
        Real const Z = species.charge_number;
        bool const relax = o.relaxation && !species.relaxation_excluded;
        bool const excluded = species.relaxation_excluded;
        bool const overlay = species.has_resistivity_overlay;
        auto const eta_per = species.resistivity_overlay;
        Staggers species_stag{};
        if (overlay) {
            species_stag = staggers(species.raw_current);
        }
        int const ion_offset = s * IonSourceComponent::Count;
        for (amrex::MFIter mfi(rates, amrex::TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            auto const q = rates.array(mfi), iq = ion_rates.array(mfi);
            auto const rho = state.raw_charge->const_array(mfi);
            auto const te = state.temperature_kelvin->const_array(mfi);
            auto const rs = species.raw_charge->const_array(mfi);
            auto const sum = m_species_sum.const_array(mfi);
            auto const applied_joule = o.joule_applied_edge_work
                ? m_joule_work.const_array(mfi) : amrex::Array4<const Real>{};
            auto const current = arrays(state.plasma_current, mfi);
            auto const magnetic = arrays(state.magnetic_field, mfi);
            auto const ti = relax ? species.ion_temperature_ev->const_array(mfi)
                                  : amrex::Array4<const Real>{};
            auto const eta_add =
                state.additive_resistivity
                    ? state.additive_resistivity->const_array(mfi)
                    : amrex::Array4<const Real>{};
            bool const add_eta = state.additive_resistivity != nullptr;
            DeviceVector sj{};
            if (overlay) {
                sj = arrays(species.raw_current, mfi);
            }
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                   int k) {
                Real const r = rho(i, j, k);
                if (r <= rho_floor) {
                    return;
                }
                Real const T = te(i, j, k),
                           Tev = T * PhysConst::kb / PhysConst::q_e;
                // Both operands are the same native RAW charge measure
                // (C/m^2 in RZ, before cylindrical volume scaling). A
                // physical C/m^3 floor here is dimensionally invalid and
                // suppresses even one populated species near the axis.
                // Physical availability is gated by rho above, separately.
                Real const species_charge = rs(i, j, k);
                Real const total_charge = sum(i, j, k);
                if (!std::isfinite(species_charge) || species_charge < 0 ||
                    !std::isfinite(total_charge) || total_charge < 0) {
                    q(i, j, k, SourceComponent::Joule) = nan;
                    return;
                }
                Real const fraction = total_charge > 0
                                          ? species_charge / total_charge
                                          : Real(0);
                Real const ne = r / PhysConst::q_e, ns = fraction * ne / Z;
                if (!std::isfinite(ns) || ns < 0 || !std::isfinite(T) ||
                    T <= 0) {
                    q(i, j, k, SourceComponent::Joule) = nan;
                    return;
                }
                if (o.joule_applied_edge_work) {
                    q(i,j,k,SourceComponent::Joule) += fraction * applied_joule(i,j,k);
                } else if (o.joule) {
                    Real const j2 =
                        square_magnitude(current, current_stag, i, j, k);
                    Real const J = std::sqrt(j2);
                    Real eta = o.use_heating_eta
                                   ? o.heating_eta(r, J, Tev, o.time)
                                   : (o.field_eta_uses_kelvin
                                          ? o.field_eta_kelvin(r, J, T, o.time)
                                          : o.field_eta(r, J, o.time));
                    if (add_eta) {
                        eta += eta_add(i, j, k);
                    }
                    if (overlay) {
                        Real const Js = std::sqrt(
                            square_magnitude(sj, species_stag, i, j, k));
                        Real const B = std::sqrt(
                            square_magnitude(magnetic, magnetic_stag, i, j, k));
                        eta += eta_per(rs(i, j, k), r, T, J, Js, B, o.time);
                    }
                    if (!std::isfinite(eta) || eta < 0) {
                        q(i, j, k, SourceComponent::Joule) = nan;
                        return;
                    }
                    // Same drift and fraction as QDSMCAddJouleHeating. This is
                    // Z e^2 eta ns ne |J/rho|^2, NOT arbitrary eta*J^2 when
                    // species overlays/fractions or heating eta differ.
                    Real const power = Z * PhysConst::q_e * PhysConst::q_e *
                                       eta * ns * ne * j2 / (r * r);
                    Real weight = ne <= n_gate ? 0 : 1;
                    if (o.joule_taper) {
                        weight *= halo_weight(ne, n_gate);
                    }
                    if (o.source_taper_density > 0) {
                        weight *= halo_weight(ne, o.source_taper_density);
                    }
                    q(i, j, k, SourceComponent::DeclinedJouleGate) +=
                        (1 - weight) * power;
                    Real const delivery = weight * power;
                    if (o.redirect_joule && Tev >= o.redirect_temperature_ev) {
                        if (excluded || ne <= o.redirect_density_floor_factor *
                                                  o.density_floor) {
                            q(i, j, k, SourceComponent::DeclinedRedirect) +=
                                delivery;
                        } else {
                            iq(i, j, k,
                               ion_offset + IonSourceComponent::RedirectPower) =
                                delivery;
                            // Finite even when ns==0, identical to the legacy
                            // per-physical-ion variance increment divided by
                            // dt.
                            iq(i, j, k,
                               ion_offset +
                                   IonSourceComponent::RedirectVarianceRate) =
                                weight * (2.0 / 3.0) * ne * Z * PhysConst::q_e *
                                PhysConst::q_e * eta * j2 / (r * r);
                        }
                    } else {
                        q(i, j, k, SourceComponent::Joule) += delivery;
                    }
                }
                if (relax) {
                    Real const Ti = ti(i, j, k);
                    Real const nu = o.relaxation_rate(
                        r, amrex::max(Tev, Real(1.e-3)), Ti, o.time);
                    if (!std::isfinite(nu) || nu < 0 || !std::isfinite(Ti) ||
                        Ti < 0) {
                        q(i, j, k, SourceComponent::Relaxation) = nan;
                        return;
                    }
                    Real const ion_power =
                        3 * ns * PhysConst::q_e * nu * (Tev - Ti);
                    q(i, j, k, SourceComponent::Relaxation) -= ion_power;
                    iq(i, j, k,
                       ion_offset + IonSourceComponent::RelaxationPower) =
                        ion_power;
                    iq(i, j, k,
                       ion_offset + IonSourceComponent::CollisionFrequency) =
                        nu;
                }
            });
        }
    }
    if (o.viscosity == ViscousSourceKind::AppliedWork ||
        state.viscous_electric[0]) {
        MatchedWork(state, state.viscous_electric, rates,
                    SourceComponent::ViscousWork);
    }
    if (o.hyperresistive_work) {
        MatchedWork(state, state.hyper_electric, rates,
                    SourceComponent::HyperResistive);
    }
    for (amrex::MFIter mfi(rates, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const q = rates.array(mfi);
        auto const rho = state.raw_charge->const_array(mfi);
        auto const te = state.temperature_kelvin->const_array(mfi);
        auto const magnetic = arrays(state.magnetic_field, mfi);
        bool const has_stopping = state.stopping_impulse != nullptr;
        bool const has_strain = state.viscous_strain_power != nullptr;
        auto const stopping = has_stopping
                                  ? state.stopping_impulse->const_array(mfi)
                                  : amrex::Array4<const Real>{};
        auto const strain = has_strain
                                ? state.viscous_strain_power->const_array(mfi)
                                : amrex::Array4<const Real>{};
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const r = rho(i, j, k), T = te(i, j, k);
                if (has_stopping) {
                    int const channel = r <= rho_floor
                                            ? SourceComponent::DeclinedStopping
                                            : SourceComponent::Stopping;
                    q(i, j, k, channel) = stopping(i, j, k) / o.physical_dt;
                }
                if (r > rho_floor) {
                    if (has_strain) {
                        q(i, j, k, SourceComponent::ViscousStrain) =
                            strain(i, j, k) < 0 ? nan : strain(i, j, k);
                    }
                    if (o.viscosity == ViscousSourceKind::AppliedWork) {
                        q(i, j, k, SourceComponent::Viscous) =
                            q(i, j, k, SourceComponent::ViscousWork);
                    } else if (o.viscosity == ViscousSourceKind::Strain) {
                        q(i, j, k, SourceComponent::Viscous) =
                            q(i, j, k, SourceComponent::ViscousStrain);
                    }
                    if (o.external_sink) {
                        Real const B = std::sqrt(
                            square_magnitude(magnetic, magnetic_stag, i, j, k));
                        q(i, j, k, SourceComponent::ExternalSink) = -o.sink(
                            r, T * PhysConst::kb / PhysConst::q_e, B, o.time);
                    }
                }
                Real total = 0;
                for (int c = SourceComponent::Joule;
                     c <= SourceComponent::Stopping; ++c) {
                    total += q(i, j, k, c);
                }
                q(i, j, k, SourceComponent::ElectronTotal) =
                    (!std::isfinite(r) ||
                     (r > rho_floor && (!std::isfinite(T) || T <= 0)))
                        ? nan
                        : total;
            });
    }
    // Fill only scratch images. Duplicate valid nodes are required to be
    // consistent on input, as for QDSMCDepositDragWork. No SumBoundary here.
    rates.FillBoundary(m_geometry.periodicity());
    ion_rates.FillBoundary(m_geometry.periodicity());
    return rates.is_finite(0, SourceComponent::Count, 0) &&
           ion_rates.is_finite(0, ion_rates.nComp(), 0);
}

namespace {
Real
integral (const amrex::MultiFab& field, int component,
          const amrex::iMultiFab& owner_mask, const amrex::Geometry& geometry) {
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<Real> data(ops);
    using Tuple = typename decltype(data)::Type;
    auto const volume = MakeQdsmcVolumeElement(geometry, field.ixType());
    for (amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const f = field.const_array(mfi);
        auto const owner = owner_mask.const_array(mfi);
        ops.eval(mfi.tilebox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                     return {owner(i, j, k)
                                 ? volume(i, j, k) * f(i, j, k, component)
                                 : Real(0)};
                 });
    }
    Real result = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}

} // namespace

Real
EulerianThermalSources::Integral (const amrex::MultiFab& field,
                                  int component) const {
    return integral(field, component, *m_owner, m_geometry);
}

ThermalSourceLedger
EulerianThermalSources::Ledger (const amrex::MultiFab& rates,
                                const amrex::MultiFab& ion_rates,
                                int species_count) const {
    AMREX_ALWAYS_ASSERT(species_count >= 0);
    CheckNodal(&rates, SourceComponent::Count);
    CheckNodal(&ion_rates,
               std::max(1, species_count * IonSourceComponent::Count));
    ThermalSourceLedger result;
    for (int c = 0; c < SourceComponent::Count; ++c) {
        result.energy[c] = m_options.physical_dt * Integral(rates, c);
    }
    for (int s = 0; s < species_count; ++s) {
        result.ion_relaxation.push_back(
            m_options.physical_dt *
            Integral(ion_rates, s * IonSourceComponent::Count +
                                    IonSourceComponent::RelaxationPower));
        result.ion_redirect.push_back(
            m_options.physical_dt *
            Integral(ion_rates, s * IonSourceComponent::Count +
                                    IonSourceComponent::RedirectPower));
    }
    return result;
}
} // namespace warpx::thermal
