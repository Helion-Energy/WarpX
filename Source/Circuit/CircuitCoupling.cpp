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

#include "BoundaryConditions/GreensFunctionOpenBC.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#   include <dlfcn.h>
#endif

namespace
{
    /** Load a compiled ExternalCircuit engine from a shared library
     * exporting the C factory symbol warpx_create_external_circuit. */
    std::unique_ptr<ExternalCircuit>
    LoadExternalCircuitPlugin (std::string const& path)
    {
#if defined(_WIN32)
        amrex::ignore_unused(path);
        WARPX_ABORT_WITH_MESSAGE(
            "circuit.engine = external is not supported on Windows");
        return nullptr;
#else
        void* handle = dlopen(path.c_str(), RTLD_NOW);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(handle != nullptr,
            "circuit.plugin_library: could not load '" + path + "': " +
            std::string(dlerror()));
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto factory = reinterpret_cast<warpx_create_external_circuit_t>(
            dlsym(handle, "warpx_create_external_circuit"));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(factory != nullptr,
            "circuit.plugin_library: '" + path + "' does not export "
            "warpx_create_external_circuit");
        return std::unique_ptr<ExternalCircuit>(factory());
#endif
    }
}

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

    const amrex::ParmParse pp_circuit("circuit");
    pp_circuit.query("engine", m_engine);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_engine == "none" || m_engine == "callbacks" || m_engine == "external",
        "circuit.engine must be one of: none, callbacks, external");
    if (m_engine == "external") {
        pp_circuit.get("plugin_library", m_plugin_library);
    }
    utils::parser::queryWithParser(pp_circuit,
        "coupling.corrector_iterations", m_coupler_params.corrector_iterations);
    utils::parser::queryWithParser(pp_circuit,
        "coupling.corrector_rtol", m_coupler_params.corrector_rtol);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_coupler_params.corrector_iterations >= 0,
        "circuit.coupling.corrector_iterations must be >= 0");
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

    // Resolve the per-coil linkage probes. The reciprocity probe is exact
    // only in free space: it requires the Green's-function open boundary
    // (a conducting wall's image response is not in the unit A).
    const bool open_bc = GreensFunctionOpenBC::IsActive();
    const amrex::ParmParse pp_circuit("circuit");
    m_probes.assign(m_coils.size(), ProbeKind::none);
    for (int ic = 0; ic < m_coils.size(); ++ic) {
        const Coil& c = m_coils.coil(ic);
        std::string probe = "default";
        pp_circuit.query((c.name + ".probe").c_str(), probe);
        if (probe == "default") {
            probe = open_bc ? "reciprocity" : "disk";
        }
        if (probe == "none") {
            m_probes[ic] = ProbeKind::none;
        } else if (probe == "disk") {
            m_probes[ic] = ProbeKind::disk;
        } else if (probe == "reciprocity") {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(open_bc,
                "circuit." + c.name + ".probe = reciprocity requires the "
                "Green's-function open field boundary (free-space "
                "reciprocity is invalid against a conducting wall); use "
                "the disk probe instead");
            m_probes[ic] = ProbeKind::reciprocity;
        } else {
            WARPX_ABORT_WITH_MESSAGE(
                "circuit." + c.name + ".probe must be one of: default, "
                "disk, reciprocity, none");
        }
    }

    // Construct the coupling engine.
    if (m_engine != "none") {
#if !defined(WARPX_DIM_RZ)
        WARPX_ABORT_WITH_MESSAGE(
            "the circuit coupling engine (circuit.engine) is implemented "
            "for RZ geometry");
#else
        // The coupled (measured) coils must be scale-driven: the engine
        // pushes their segments every coupling interval.
        for (int ic = 0; ic < m_coils.size(); ++ic) {
            if (m_probes[ic] == ProbeKind::none) { continue; }
            const Coil& c = m_coils.coil(ic);
            bool scale_driven = false;
            for (int i = 0; i < ext.nFields(); ++i) {
                if (ext.FieldName(i) == c.field_name) {
                    scale_driven = ext.UsesPythonScale(i);
                    break;
                }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(scale_driven,
                "coupled circuit coil '" + c.name + "' requires "
                "external_vector_potential." + c.field_name +
                ".python_scale = 1 (the engine drives its scale segments)");
        }

        std::unique_ptr<ExternalCircuit> plugin;
        if (m_engine == "external") {
            plugin = LoadExternalCircuitPlugin(m_plugin_library);
        }
        m_coupler = std::make_unique<CircuitCoupler>(
            m_coils, m_probes, m_coupler_params, std::move(plugin));
        amrex::Print() << "Circuit coupling engine: " << m_engine
                       << " (corrector_iterations = "
                       << m_coupler_params.corrector_iterations
                       << ", corrector_rtol = "
                       << m_coupler_params.corrector_rtol << ")\n";
#endif
    }
}


void
CircuitCoupling::WriteCheckpointData (std::string const& dir) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
    auto& warpx = WarpX::GetInstance();
    auto* hybrid = warpx.get_pointer_HybridPICModel();
    if (hybrid == nullptr || !hybrid->m_add_external_fields) { return; }
    auto& ext = *hybrid->m_external_vector_potential;

    std::ofstream ofs{dir + "/circuit_coupling.dat", std::ofstream::out};
    ofs << std::setprecision(17);
    ofs << "version 1\n";
    for (int i = 0; i < ext.nFields(); ++i) {
        amrex::Real s_old, s_new, t_old, t_new;
        if (ext.GetScaleSegment(i, s_old, s_new, t_old, t_new)) {
            ofs << ext.FieldName(i) << " " << s_old << " " << s_new
                << " " << t_old << " " << t_new << "\n";
        }
    }
    ofs.close();
}

void
CircuitCoupling::ReadCheckpointData (std::string const& dir)
{
    // Tolerate checkpoints from before the circuit subsystem.
    std::ifstream ifs{dir + "/circuit_coupling.dat", std::ifstream::in};
    if (!ifs.good()) { return; }

    auto& warpx = WarpX::GetInstance();
    auto* hybrid = warpx.get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        hybrid != nullptr && hybrid->m_add_external_fields,
        "restarting a checkpoint with circuit_coupling.dat requires the "
        "hybrid solver with external fields");
    auto& ext = *hybrid->m_external_vector_potential;

    std::string token;
    int version = 0;
    ifs >> token >> version;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(token == "version" && version == 1,
        "unsupported circuit_coupling.dat checkpoint format");

    std::string name;
    amrex::Real s_old, s_new, t_old, t_new;
    while (ifs >> name >> s_old >> s_new >> t_old >> t_new) {
        // Aborts with a clear message if the restart inputs dropped the
        // field or its python_scale declaration.
        ext.SetScale(name, s_old, s_new, t_old, t_new);
    }
}
