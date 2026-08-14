/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CircuitCoupling.H"

#include "Coils/CoilFieldSolver.H"
#include "Coils/LoopInductance.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

bool
CircuitCoupling::IsConfigured ()
{
    const amrex::ParmParse pp_circuit("circuit");
    std::vector<std::string> coil_names;
    pp_circuit.queryarr("coils", coil_names);
    return !coil_names.empty();
}

CircuitCoupling::CircuitCoupling ()
{
    m_coils.ReadParameters();
}

void
CircuitCoupling::InitData ()
{
    using namespace warpx::circuit;

    auto& warpx = WarpX::GetInstance();
    auto* hybrid = warpx.get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        hybrid != nullptr && hybrid->m_add_external_fields,
        "circuit.coils requires the hybrid solver (algo.maxwell_solver = "
        "hybrid) with hybrid_pic_model.add_external_fields = 1: the coils "
        "drive the split external fields");
    auto& ext = *hybrid->m_external_vector_potential;

    // Validate the coil <-> external-field pairing.
    for (const Coil& c : m_coils.coils()) {
        bool found = false;
        for (int i = 0; i < ext.nFields(); ++i) {
            if (ext.FieldName(i) == c.field_name) { found = true; break; }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(found,
            "circuit coil '" + c.name + "' pairs with external field '" +
            c.field_name + "', which is not listed in "
            "external_vector_potential.fields");
    }

    // Fill the unit fields at I_ref and refresh their curls.
    bool any_filled = false;
    for (const Coil& c : m_coils.coils()) {
        if (!c.fill_unit_field) { continue; }
        FillCoilUnitField(c);
        ext.CalculateExternalCurlA(c.field_name);
        any_filled = true;
    }
    if (any_filled) {
        ext.UpdateHybridExternalFields(warpx.gett_new(0), warpx.getdt(0));
    }

    // The discrete inductance table of the coil set on the run mesh. The
    // discrete self-inductance is the flux the mesh actually links
    // (resolution dependent by construction) and is the value a coupled
    // circuit port must use -- never a continuum (wire/Maxwell) value.
#if defined(WARPX_DIM_RZ)
    const auto& geom = warpx.Geom(0);
    if (geom.ProbLo(0) == 0.0 && m_coils.size() > 0) {
        const LoopGridRZ grid{
            geom.Domain().length(0), geom.Domain().length(1),
            static_cast<double>(geom.ProbHi(0)),
            static_cast<double>(geom.ProbLo(1)),
            static_cast<double>(geom.ProbHi(1))};

        const int n = m_coils.size();
        std::vector<double> L(n);
        std::vector<double> M(static_cast<std::size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i) {
            const Coil& ci = m_coils.coil(i);
            L[i] = DiscreteSelfInductance(ci.r, ci.z, ci.n_turns, grid);
            M[static_cast<std::size_t>(i) * n + i] = L[i];
        }
        double max_asym = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const Coil& ci = m_coils.coil(i);
                const Coil& cj = m_coils.coil(j);
                const double m_ij = DiscreteMutualInductance(
                    ci.r, ci.z, ci.n_turns, cj.r, cj.z, cj.n_turns, grid);
                const double m_ji = DiscreteMutualInductance(
                    cj.r, cj.z, cj.n_turns, ci.r, ci.z, ci.n_turns, grid);
                const double m_sym = 0.5 * (m_ij + m_ji);
                if (m_sym != 0.0) {
                    max_asym = std::max(max_asym,
                        std::abs(m_ij - m_ji) / std::abs(m_sym));
                }
                M[static_cast<std::size_t>(i) * n + j] = m_sym;
                M[static_cast<std::size_t>(j) * n + i] = m_sym;
            }
        }

        amrex::Print() << "Circuit coils: DISCRETE (run-mesh) inductances "
                       << "[H]; circuit ports must use these, not continuum "
                       << "values:\n";
        for (int i = 0; i < n; ++i) {
            const Coil& ci = m_coils.coil(i);
            amrex::Print() << "  " << ci.name
                           << " (r = " << ci.r << " m, z = " << ci.z
                           << " m, n = " << ci.n_turns
                           << ", I_ref = " << ci.I_ref << " A): L_disc = "
                           << L[i] << "\n";
        }
        if (n > 1) {
            amrex::Print() << "  symmetrized mutual matrix "
                           << "(max relative asymmetry " << max_asym << "):\n";
            for (int i = 0; i < n; ++i) {
                amrex::Print() << "   ";
                for (int j = 0; j < n; ++j) {
                    amrex::Print() << " " << M[static_cast<std::size_t>(i) * n + j];
                }
                amrex::Print() << "\n";
            }
        }
    } else if (m_coils.size() > 0) {
        amrex::Print() << "Circuit coils: discrete inductance table skipped "
                       << "(requires the radial domain to start on the axis)\n";
    }
#else
    if (m_coils.size() > 0) {
        amrex::Print() << "Circuit coils: discrete inductance table skipped "
                       << "(an RZ-mesh convention; not defined for this "
                       << "geometry)\n";
    }
#endif
}
