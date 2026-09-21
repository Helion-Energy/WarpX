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
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
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
    LoadExternalCircuitPlugin (std::string const& path,
                               WarpxCircuitAffineApiV1 const*& affine_api)
    {
#if defined(_WIN32)
        amrex::ignore_unused(path, affine_api);
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
        using affine_factory_t = WarpxCircuitAffineApiV1 const* (*)();
        // Optional export; the legacy virtual ABI remains unchanged.
        auto affine_factory = reinterpret_cast<affine_factory_t>(
            dlsym(handle, "warpx_external_circuit_affine_api_v1"));
        affine_api = affine_factory ? affine_factory() : nullptr;
        auto plugin = std::unique_ptr<ExternalCircuit>(factory());
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(plugin != nullptr, "Circuit plugin factory returned null");
        return plugin;
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
    std::string trial_backend = "host";
    pp_circuit.query("trial_backend", trial_backend);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(trial_backend == "host" || trial_backend == "device_affine",
        "circuit.trial_backend must be host or device_affine");
    m_coupler_params.device_affine = trial_backend == "device_affine";
    pp_circuit.query("device_iterations", m_coupler_params.device_iterations);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_coupler_params.device_iterations > 0
        && (!m_coupler_params.device_affine || m_engine == "external"),
        "Device circuit trials require engine=external and device_iterations>0");

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
    // Admit the free-space probes without the open boundary (validation /
    // diagnostic use; see the member doc). Default off = the historical
    // refusal.
    pp_circuit.query("probe_ignore_walls", m_probe_ignore_walls);
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
    if (m_engine != "none") {
        std::string scheme;
        amrex::ParmParse("algo").get("evolve_scheme", scheme);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(scheme == "theta_implicit_hybrid" && hybrid->m_darwin,
            "This circuit integration requires theta_implicit_hybrid with Darwin");
    }
    auto& ext = *hybrid->m_external_vector_potential;

    if (m_engine == "external") {
        amrex::ParmParse implicit("implicit_evolve");
        std::string driver = "native";
        bool segregated = false;
        bool external_iteration = false;
        bool consistent = true;
        implicit.query("circuit_driver", driver);
        implicit.query("darwin_segregated_solve", segregated);
        implicit.query("external_field_iteration", external_iteration);
        implicit.query("darwin_circuit_consistent_stage", consistent);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(driver == "native" && segregated
            && external_iteration && consistent && AMREX_SPACEDIM == 3
            && warpx.maxLevel() == 0 && sizeof(amrex::Real) == sizeof(double),
            "External circuit plugin requires the supported native Darwin driver");
    }

    // Each engine port owns one distinct external-field scale segment.
    std::set<std::string> driven_fields;
    for (const Coil& c : m_coils.coils()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(driven_fields.insert(c.field_name).second,
            "Circuit ports must have distinct field_name values");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!c.fill_unit_field || !EB::enabled(),
            "Painted circuit unit fields with EB require ghost-field qualification; "
            "use an explicitly supplied and validated unit field with fill_unit_field=0");
        bool found = false;
        for (int i = 0; i < ext.nFields(); ++i) {
            if (ext.FieldName(i) == c.field_name) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_engine != "external" || ext.UsesPythonScale(i),
                    "Every external circuit port requires a scale-driven external field, including probe=none");
                found = true;
                break;
            }
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
#elif defined(WARPX_DIM_3D)
    // The 3D table: the same discrete-L rule on the 3D Yee mesh (the disk
    // flux of the order-2 Yee curl of the nominally placed unit field
    // through the nominal circle; LoopInductance.H), evaluated on the
    // global grid so it is identical for any rank count, and -- for coils
    // whose unit field this class painted -- cross-checked against the
    // mesh's own painted register through the 3D disk probe, so the port
    // value and the probe are provably one convention on THIS mesh.
    if (m_coils.size() > 0 &&
        WarpX::grid_type != ablastr::utils::enums::GridType::Staggered) {
        amrex::Print() << "Circuit coils: discrete inductance table skipped "
                       << "(3D table defined for the staggered Yee grid "
                       << "only)\n";
    } else if (m_coils.size() > 0) {
        const auto& geom = warpx.Geom(0);
        const LoopGrid3D grid{
            geom.Domain().length(0), geom.Domain().length(1),
            geom.Domain().length(2),
            static_cast<double>(geom.ProbLo(0)),
            static_cast<double>(geom.ProbLo(1)),
            static_cast<double>(geom.ProbLo(2)),
            static_cast<double>(geom.CellSize(0)),
            static_cast<double>(geom.CellSize(1)),
            static_cast<double>(geom.CellSize(2))};

        const int n = m_coils.size();
        std::vector<double> L(n);
        std::vector<DiskFluxInfo> info(n);
        std::vector<double> M(static_cast<std::size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i) {
            const Coil& ci = m_coils.coil(i);
            L[i] = DiscreteSelfInductance3D(ci.r, ci.z, ci.n_turns, grid,
                                            &info[i]);
            M[static_cast<std::size_t>(i) * n + i] = L[i];
        }
        double max_asym = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const Coil& ci = m_coils.coil(i);
                const Coil& cj = m_coils.coil(j);
                const double m_ij = DiscreteMutualInductance3D(
                    ci.r, ci.z, ci.n_turns, cj.r, cj.z, cj.n_turns, grid);
                const double m_ji = DiscreteMutualInductance3D(
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

        // The circle is truncated to the domain when it reaches outside
        // (the RZ convention for r > r_max); say so, the value is then
        // the flux through the in-domain part of the disk only.
        auto disk_truncated = [&](const Coil& c) {
            const double r = c.r;
            return (-r < grid.x_min) || (r > grid.x_min + grid.nx * grid.dx) ||
                   (-r < grid.y_min) || (r > grid.y_min + grid.ny * grid.dy);
        };

        amrex::Print() << "Circuit coils: DISCRETE (run-mesh) inductances "
                       << "[H] (3D Yee disk rule on the nominal circle); "
                       << "circuit ports must use these, not continuum "
                       << "values:\n";
        for (int i = 0; i < n; ++i) {
            const Coil& ci = m_coils.coil(i);
            amrex::Print() << "  " << ci.name
                           << " (r = " << ci.r << " m, z = " << ci.z
                           << " m, n = " << ci.n_turns
                           << ", I_ref = " << ci.I_ref << " A): L_disc = "
                           << L[i] << " (" << info[i].n_disk_cells
                           << " disk cells, z weight " << info[i].z_weight
                           << (disk_truncated(ci)
                                   ? ", circle truncated to the domain)"
                                   : ")")
                           << "\n";
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

        // Mesh cross-check of the painted coils: the 3D disk probe applied
        // to the mesh's own curl of the painted unit register (at
        // I_ref n) must reproduce the host table to summation roundoff,
        //     L_disc = DiskFluxLinkage(coil, <name>_curlAext_z) / I_ref^2.
        // Without embedded boundaries the two are the same arithmetic in
        // a different summation order (the mesh curl is EB-masked, so the
        // pin is informational there). Asserted under
        // circuit.probe_crosscheck.
        for (int i = 0; i < n; ++i) {
            const Coil& ci = m_coils.coil(i);
            if (!ci.fill_unit_field) { continue; }
            const amrex::MultiFab* curl_z = warpx.m_fields.get(
                ci.field_name + "_curlAext", ablastr::fields::Direction{2}, 0);
            const double l_mesh = static_cast<double>(
                DiskFluxLinkage(ci, *curl_z))
                / (static_cast<double>(ci.I_ref) * static_cast<double>(ci.I_ref));
            const double scale = std::max(std::abs(L[i]), std::abs(l_mesh));
            const double rel = (scale > 0.0) ? std::abs(l_mesh - L[i]) / scale : 0.0;
            amrex::Print() << "  " << ci.name << ": mesh disk-probe cross-check "
                           << "L_mesh = " << l_mesh << " (rel diff "
                           << rel << (EB::enabled() ? ", EB-masked curl" : "")
                           << ")\n";
            if (m_coupler_params.probe_crosscheck && !EB::enabled()) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rel <= 1.0e-9,
                    "circuit.probe_crosscheck: the 3D discrete inductance "
                    "table of coil '" + ci.name + "' disagrees with the "
                    "mesh disk probe of its painted unit field");
            }
        }
    }
#else
    if (m_coils.size() > 0) {
        amrex::Print() << "Circuit coils: discrete inductance table skipped "
                       << "(an RZ/3D-mesh convention; not defined for this "
                       << "geometry)\n";
    }
#endif

    // Resolve the per-coil linkage probes. The reciprocity probe is exact
    // only in free space: it requires the Green's-function open boundary
    // (a conducting wall's image response is not in the unit A), unless
    // circuit.probe_ignore_walls explicitly admits the free-space
    // functional against walls (validation / diagnostic use).
    const bool open_bc = GreensFunctionOpenBC::IsActive();
    const bool free_space_ok = open_bc || m_probe_ignore_walls;
    bool warned_walls = false;
    auto warn_walls = [&](const std::string& coil_name, const char* probe) {
        if (warned_walls) { return; }
        warned_walls = true;
        const std::string msg =
            "circuit.probe_ignore_walls = 1: coil '" + coil_name +
            "' uses the free-space " + std::string(probe) +
            " probe without the Green's-function open boundary; the wall "
            "images are NOT in the unit A, so the measured linkage is the "
            "free-space functional of the plasma current only, not the "
            "model's own flux linkage (an exterior coil behind a conducting "
            "wall is physically screened). Validation/diagnostic use.";
        ablastr::warn_manager::WMRecordWarning(
            "CircuitCoupling", msg, ablastr::warn_manager::WarnPriority::high);
        amrex::Print() << "WARNING: " << msg << "\n";
    };
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
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(free_space_ok,
                "circuit." + c.name + ".probe = reciprocity requires the "
                "Green's-function open field boundary (free-space "
                "reciprocity is invalid against a conducting wall); use "
                "the disk probe instead");
            if (!open_bc) { warn_walls(c.name, "reciprocity"); }
            m_probes[ic] = ProbeKind::reciprocity;
        } else if (probe == "loop") {
            // Free-space reciprocity against the analytic loop A of the
            // declared filament (the python coupling reference's probe
            // integrand): the same open-boundary requirement.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(free_space_ok,
                "circuit." + c.name + ".probe = loop is a free-space "
                "reciprocity functional and requires the Green's-function "
                "open field boundary; use the disk probe against a "
                "conducting wall");
            if (!open_bc) { warn_walls(c.name, "loop"); }
            m_probes[ic] = ProbeKind::loop;
        } else {
            WARPX_ABORT_WITH_MESSAGE(
                "circuit." + c.name + ".probe must be one of: default, "
                "disk, reciprocity, loop, none");
        }
    }

    // Construct the coupling engine (RZ m = 0 and 3D Cartesian: the
    // probes, the discrete inductance table and the coupling-power
    // integral exist for both; the coupler and the engines are
    // geometry-neutral).
    if (m_engine != "none") {
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)
        WARPX_ABORT_WITH_MESSAGE(
            "the circuit coupling engine (circuit.engine) is implemented "
            "for RZ and 3D geometry");
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
        WarpxCircuitAffineApiV1 const* affine_api = nullptr;
        if (m_engine == "external") {
            plugin = LoadExternalCircuitPlugin(m_plugin_library, affine_api);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_coupler_params.device_affine ||
                (affine_api && affine_api->struct_bytes == sizeof(WarpxCircuitAffineApiV1)
                    && affine_api->api_version == WARPX_CIRCUIT_AFFINE_API_V1
                    && affine_api->prepare && affine_api->release),
                "Device circuit trials require the affine v1 plugin capability; no host fallback");
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
        m_coupler = std::make_unique<CircuitCoupler>(
            m_coils, m_probes, m_probe_exclusion, m_coupler_params,
            std::move(plugin), affine_api);
        // The coupler's own per-step memory (EMF low-pass state) is part
        // of the checkpoint; restore it with the engine state.
        if (!m_restart_dir.empty()) {
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
