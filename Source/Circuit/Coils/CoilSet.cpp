/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CoilSet.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>

#include <set>

namespace warpx::circuit
{

void
CoilSet::ReadParameters ()
{
    const amrex::ParmParse pp_circuit("circuit");

    std::vector<std::string> coil_names;
    pp_circuit.queryarr("coils", coil_names);

    std::set<std::string> seen;
    for (const auto& name : coil_names) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(seen.insert(name).second,
            "circuit.coils lists coil '" + name + "' more than once");

        Coil c;
        c.name = name;
        utils::parser::getWithParser(pp_circuit, (name + ".r").c_str(), c.r);
        utils::parser::getWithParser(pp_circuit, (name + ".z").c_str(), c.z);
        utils::parser::queryWithParser(pp_circuit, (name + ".n_turns").c_str(), c.n_turns);
        utils::parser::queryWithParser(pp_circuit, (name + ".I_ref").c_str(), c.I_ref);
        c.field_name = name;
        pp_circuit.query((name + ".field_name").c_str(), c.field_name);
        bool fill = true;
        utils::parser::queryWithParser(pp_circuit, (name + ".fill_unit_field").c_str(), fill);
        c.fill_unit_field = fill;

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(c.r > 0.0,
            "circuit." + name + ".r must be > 0 (the filament radius)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(c.I_ref != 0.0,
            "circuit." + name + ".I_ref must be nonzero");

        m_coils.push_back(c);
    }
}

} // namespace warpx::circuit
