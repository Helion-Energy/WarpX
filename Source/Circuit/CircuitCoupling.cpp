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
#include "FieldSolver/ImplicitSolvers/ImplicitMHDWallMask.H"
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
        // The ABI stamp first: a plugin built against a different revision
        // of ExternalCircuit.H must fail loudly here, not misbehave later.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto version = reinterpret_cast<warpx_external_circuit_abi_version_t>(
            dlsym(handle, "warpx_external_circuit_abi_version"));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(version != nullptr,
            "circuit.plugin_library: '" + path + "' does not export "
            "warpx_external_circuit_abi_version (required since ABI 2)");
        const int abi = version();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            abi == WARPX_EXTERNAL_CIRCUIT_ABI_VERSION,
            "circuit.plugin_library: '" + path + "' was built against "
            "ExternalCircuit ABI " + std::to_string(abi) + " but this WarpX "
            "expects ABI " +
            std::to_string(WARPX_EXTERNAL_CIRCUIT_ABI_VERSION) +
            "; rebuild the plugin against the matching headers");
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
        pp_circuit.query("plugin_config", m_plugin_config);
        // Optional restart-run replacement for plugin_config (both are
        // opaque engine strings): lets a restart deck skip engine-side
        // boot work that ReadCheckpoint supersedes (e.g. a pre-roll).
        pp_circuit.query("plugin_restart_config", m_plugin_restart_config);
    }
    utils::parser::queryWithParser(pp_circuit,
        "coupling.corrector_iterations", m_coupler_params.corrector_iterations);
    utils::parser::queryWithParser(pp_circuit,
        "coupling.corrector_rtol", m_coupler_params.corrector_rtol);
    pp_circuit.query("probe_crosscheck", m_coupler_params.probe_crosscheck);
    utils::parser::queryWithParser(pp_circuit,
        "probe_crosscheck_rtol", m_coupler_params.crosscheck_rtol);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_coupler_params.corrector_iterations >= 0,
        "circuit.coupling.corrector_iterations must be >= 0");
    // Optional EMF low-pass handed to a compiled engine (see
    // CircuitCoupler::Params::eps_lowpass_tau). The Python-callback engine
    // computes its own EMF from the linkage registers, so the knob would be
    // a silent no-op there: refuse the pairing.
    utils::parser::queryWithParser(pp_circuit,
        "eps_lowpass_tau", m_coupler_params.eps_lowpass_tau);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_coupler_params.eps_lowpass_tau >= 0.0,
        "circuit.eps_lowpass_tau must be >= 0 (seconds; 0 = off)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_coupler_params.eps_lowpass_tau == 0.0 || m_engine == "external",
        "circuit.eps_lowpass_tau filters the EMF the coupler hands a "
        "compiled engine and requires circuit.engine = external (the "
        "Python-callback engine computes its own EMF)");
    // Newton-scope coupling model knobs (see CircuitCoupler::Params):
    // which linkage the step's EMF is differenced against, and how far
    // the in-residual advance reaches. Both default to the previous
    // behaviour; the alternatives are the python coupling reference's
    // conventions, for driver parity.
    {
        std::string ref = "first_iterate";
        pp_circuit.query("linkage_reference", ref);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ref == "first_iterate" || ref == "accepted",
            "circuit.linkage_reference must be 'first_iterate' or 'accepted'");
        m_coupler_params.linkage_reference_accepted = (ref == "accepted");
        std::string adv = "theta_stage";
        pp_circuit.query("residual_advance", adv);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            adv == "theta_stage" || adv == "full_step",
            "circuit.residual_advance must be 'theta_stage' or 'full_step'");
        m_coupler_params.residual_advance_full_step = (adv == "full_step");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (ref == "first_iterate" && adv == "theta_stage") ||
                m_engine == "external",
            "circuit.linkage_reference / circuit.residual_advance shape the "
            "EMF the coupler hands a compiled engine and require "
            "circuit.engine = external");
    }
    // Probe region / weight / report (see CircuitCoupler::Params). All
    // default off: domain, none, no report -- bit-identical to a coupler
    // without the knobs.
    {
        std::string region = "domain";
        pp_circuit.query("probe_region", region);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            region == "domain" || region == "wall_interior",
            "circuit.probe_region must be 'domain' or 'wall_interior'");
        m_coupler_params.probe_region = (region == "wall_interior")
            ? CircuitCoupler::ProbeRegion::wall_interior
            : CircuitCoupler::ProbeRegion::domain;
        std::string weight = "none";
        pp_circuit.query("probe_weight", weight);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            weight == "none" || weight == "physical_share",
            "circuit.probe_weight must be 'none' or 'physical_share'");
        m_coupler_params.probe_weight = (weight == "physical_share")
            ? CircuitCoupler::ProbeWeight::physical_share
            : CircuitCoupler::ProbeWeight::none;
        pp_circuit.query("probe_region_report",
                         m_coupler_params.probe_region_report);
        // The polyline: the coupling's own key, else the theta-implicit
        // MHD wall's polyline (queried here so a deck that relies on the
        // fallback never trips the unused-input check on either key).
        pp_circuit.query("probe_region_polyline_file",
                         m_probe_region_polyline_file);
        if (m_probe_region_polyline_file.empty()) {
            const amrex::ParmParse pp_mhd("implicit_mhd");
            pp_mhd.query("wall_polyline_file", m_probe_region_polyline_file);
        }
    }
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
    // The analytic-loop probe's mask radius [m]: a global default with a
    // per-coil override; 0 = no mask. Read for every coil (the other
    // probe kinds ignore it).
    double exclusion_default = 0.0;
    utils::parser::queryWithParser(pp_circuit, "probe_exclusion_radius",
                                   exclusion_default);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(exclusion_default >= 0.0,
        "circuit.probe_exclusion_radius must be >= 0 (meters)");
    m_probes.assign(m_coils.size(), ProbeKind::none);
    m_probe_exclusion.assign(m_coils.size(), exclusion_default);
    for (int ic = 0; ic < m_coils.size(); ++ic) {
        const Coil& c = m_coils.coil(ic);
        std::string probe = "default";
        pp_circuit.query((c.name + ".probe").c_str(), probe);
        utils::parser::queryWithParser(pp_circuit,
            (c.name + ".probe_exclusion_radius").c_str(),
            m_probe_exclusion[ic]);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_probe_exclusion[ic] >= 0.0,
            "circuit." + c.name + ".probe_exclusion_radius must be >= 0 "
            "(meters)");
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
        } else if (probe == "loop") {
            // Free-space reciprocity against the analytic loop A of the
            // declared filament (the python coupling reference's probe
            // integrand): the same open-boundary requirement.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(open_bc,
                "circuit." + c.name + ".probe = loop is a free-space "
                "reciprocity functional and requires the Green's-function "
                "open field boundary; use the disk probe against a "
                "conducting wall");
            m_probes[ic] = ProbeKind::loop;
        } else {
            WARPX_ABORT_WITH_MESSAGE(
                "circuit." + c.name + ".probe must be one of: default, "
                "disk, reciprocity, loop, none");
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
            // One-time port configuration: coil order fixes the eps/scale
            // vector indexing of every AdvanceInterval call.
            std::vector<std::string> names;
            std::vector<amrex::Real> i_ref;
            for (int ic = 0; ic < m_coils.size(); ++ic) {
                names.push_back(m_coils.coil(ic).name);
                i_ref.push_back(m_coils.coil(ic).I_ref);
            }
            // On restart, hand the engine the restart variant of its
            // opaque config when one is declared (circuit.
            // plugin_restart_config): ReadCheckpoint below supersedes any
            // engine-side boot state, so engines can skip boot work
            // (e.g. a pre-roll) that Define would otherwise redo.
            const bool restarting = !m_restart_dir.empty();
            plugin->Define(names, i_ref,
                           (restarting && !m_plugin_restart_config.empty())
                               ? m_plugin_restart_config
                               : m_plugin_config);
            // On restart, restore the engine's own state on every rank
            // (the engine runs replicated in lockstep, exactly like the
            // Python-callback engine).
            if (restarting) {
                plugin->ReadCheckpoint(m_restart_dir);
            }
        }
        // Wall-polyline region table of the probe mask and of the region
        // report (either requested): the nodal J_theta-mesh classification
        // r_i < r_wall(z_j) documented on warpx::circuit::ProbeRegionTable.
        const bool wall_interior =
            m_coupler_params.probe_region ==
            CircuitCoupler::ProbeRegion::wall_interior;
        const bool report = !m_coupler_params.probe_region_report.empty();
        if (wall_interior || report) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !m_probe_region_polyline_file.empty(),
                "circuit.probe_region = wall_interior and "
                "circuit.probe_region_report need a wall polyline: set "
                "circuit.probe_region_polyline_file (CSV of 'z, r' rows) or "
                "implicit_mhd.wall_polyline_file");
            std::vector<double> z_poly;
            std::vector<double> r_poly;
            ImplicitMHDWallMask::ReadPolylineFile(
                m_probe_region_polyline_file, z_poly, r_poly,
                "circuit.probe_region_polyline_file");
            m_probe_region.Define(warpx.Geom(0), z_poly, r_poly);
            amrex::Print() << "Circuit probe region: polyline '"
                           << m_probe_region_polyline_file << "' ("
                           << z_poly.size() << " points); rule: J_theta node "
                           << "(r_i, z_j) is inside the wall iff r_i < "
                           << "r_wall(z_j), r_wall = piecewise-linear in z, "
                           << "clamped at the ends (numpy.interp); band = "
                           << "r_wall - 2 dr <= r_i < r_wall (2 dr = "
                           << m_probe_region.BandWidth() << " m)\n";
        }
        if (m_coupler_params.probe_weight ==
            CircuitCoupler::ProbeWeight::physical_share) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                warpx.m_fields.has(CircuitCoupler::LinkageWeightName, 0),
                "circuit.probe_weight = physical_share requires the "
                "theta-implicit MHD solver (algo.evolve_scheme = "
                "theta_implicit_mhd), which fills the nodal "
                "circuit_linkage_weight register w = eta_phys / eta_field "
                "before every linkage measurement; no other solver mocks "
                "the vacuum with a boosted field resistivity");
        }
        m_coupler = std::make_unique<CircuitCoupler>(
            m_coils, m_probes, m_probe_exclusion, m_coupler_params,
            std::move(plugin),
            (wall_interior || report) ? &m_probe_region : nullptr);
        m_coupler->InitRegionReport(!m_restart_dir.empty());
        if (report) {
            amrex::Print() << "Circuit probe region report: '"
                           << m_coupler_params.probe_region_report
                           << "' (per J-based coil at every accepting "
                           << "evaluation and at the first residual "
                           << "evaluation of every step: lambda split by "
                           << "interior / band / exterior and by weight "
                           << "class plasma / mixed / boost; J_theta RMS per "
                           << "region)\n";
        }
        // The coupler's own per-step memory (EMF low-pass state) is part
        // of the checkpoint; restore it with the engine state. Only a
        // compiled engine writes it (WriteCheckpointData), so only a
        // compiled engine reads it back: the Python-callback engine keeps
        // its memory on the Python side and must not be told a state was
        // lost.
        if (!m_restart_dir.empty() && m_coupler->Plugin() != nullptr) {
            m_coupler->ReadMemoryCheckpoint(m_restart_dir);
        }
        amrex::Print() << "Circuit coupling engine: " << m_engine
                       << " (corrector_iterations = "
                       << m_coupler_params.corrector_iterations
                       << ", corrector_rtol = "
                       << m_coupler_params.corrector_rtol
                       << ", eps_lowpass_tau = "
                       << m_coupler_params.eps_lowpass_tau << " s"
                       << (m_coupler_params.eps_lowpass_tau > 0.0
                               ? " [one-pole EMA on the port EMF, memory "
                                 "committed on accept]"
                               : " [off: raw interval EMF]")
                       << ", linkage_reference = "
                       << (m_coupler_params.linkage_reference_accepted
                               ? "accepted [previous accepting evaluation; "
                                 "first step open loop]"
                               : "first_iterate")
                       << ", residual_advance = "
                       << (m_coupler_params.residual_advance_full_step
                               ? "full_step [EMF over the theta interval]"
                               : "theta_stage")
                       << ", probe_region = "
                       << (wall_interior
                               ? "wall_interior [J-based probes over r < "
                                 "r_wall(z) only]"
                               : "domain")
                       << ", probe_weight = "
                       << (m_coupler_params.probe_weight ==
                                   CircuitCoupler::ProbeWeight::physical_share
                               ? "physical_share [w = eta_phys / eta_field "
                                 "at the J_theta nodes]"
                               : "none")
                       << ")\n";
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

    // A compiled engine checkpoints its own state (I/O rank only, like the
    // segments above); the Python engine re-seeds itself on restart. The
    // coupler's per-step memory (EMF low-pass state) goes with it.
    if (m_coupler && m_coupler->Plugin() != nullptr) {
        m_coupler->Plugin()->WriteCheckpoint(dir);
        m_coupler->WriteMemoryCheckpoint(dir);
    }
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

    // A compiled engine's own state is restored in InitData: this runs
    // from InitFromCheckpoint, before the coupler (and the plugin) exist.
    m_restart_dir = dir;
}
