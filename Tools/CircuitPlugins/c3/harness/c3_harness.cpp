/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

/** Offline validation harness: drives the c3 plugin through the
 * ExternalCircuit ABI with a WarpX-like coupling cadence and dumps the
 * per-step accepted scales (hexfloat) for comparison against the Python
 * reference stepper.
 *
 * Cadence per step [t0, t1 = t0 + dt]: BeginStep, then --evals
 * deliberately-perturbed accept=false evaluations (the first to the
 * interval midpoint -- a re-sized substep, the rest full-width with
 * scaled eps), then the accept=true evaluation with the table eps, then
 * FinishStep. The perturbed evaluations must leave no trace (the
 * restore-and-re-advance iron rule); the accepted trajectory is compared
 * against the single-pass Python reference.
 *
 * Modes:
 *   traj   one engine, [0, t_end]
 *   ckpt   leg 1 [0, t_end] with WriteCheckpoint at --ckpt-t; leg 2 = a
 *          fresh engine (--config2), ReadCheckpoint, [ckpt_t, t_end];
 *          end-state checkpoints of both legs for byte comparison
 *
 * With --plugin the engine is created through dlopen/dlsym (the real ABI
 * path, RTLD_LOCAL); without it the concrete class is instantiated
 * directly, which also enables --lock-out introspection (per-step lock
 * ring currents and linked fluxes, plus the arming record).
 */

#include "C3ExternalCircuit.H"

#include "ExternalCircuit.H"

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string
Hex (double v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%a", v);
    return buf;
}

struct Args
{
    std::string mode = "traj";
    std::string plugin;             // dlopen path ("" = direct)
    std::string config, config2;
    std::string manifest;
    std::string eps_path;
    std::string out, out2;
    std::string lock_out;
    std::string ckpt_dir;
    double t_end = 0.0, dt = 0.0, ckpt_t = 0.0;
    int evals = 2;
};

Args
ParseArgs (int argc, char** argv)
{
    Args a;
    auto need = [&] (int& i) -> std::string {
        if (++i >= argc) {
            throw std::runtime_error("missing value for option");
        }
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string const opt = argv[i];
        if (opt == "--mode") { a.mode = need(i); }
        else if (opt == "--plugin") { a.plugin = need(i); }
        else if (opt == "--config") { a.config = need(i); }
        else if (opt == "--config2") { a.config2 = need(i); }
        else if (opt == "--manifest") { a.manifest = need(i); }
        else if (opt == "--eps") { a.eps_path = need(i); }
        else if (opt == "--out") { a.out = need(i); }
        else if (opt == "--out2") { a.out2 = need(i); }
        else if (opt == "--lock-out") { a.lock_out = need(i); }
        else if (opt == "--ckpt-dir") { a.ckpt_dir = need(i); }
        else if (opt == "--t-end") { a.t_end = std::atof(need(i).c_str()); }
        else if (opt == "--dt") { a.dt = std::atof(need(i).c_str()); }
        else if (opt == "--ckpt-t") { a.ckpt_t = std::atof(need(i).c_str()); }
        else if (opt == "--evals") { a.evals = std::atoi(need(i).c_str()); }
        else {
            throw std::runtime_error("unknown option '" + opt + "'");
        }
    }
    if (a.config.empty() || a.manifest.empty() || a.out.empty()
        || a.t_end <= 0.0 || a.dt <= 0.0) {
        throw std::runtime_error(
            "required: --config --manifest --out --t-end --dt");
    }
    return a;
}

//! one engine instance, dlopen'd or direct
class Engine
{
public:
    Engine (std::string const& plugin)
    {
        if (plugin.empty()) {
            m_direct = std::make_unique<c3::C3ExternalCircuit>();
            m_ec = m_direct.get();
            return;
        }
        m_dl = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (m_dl == nullptr) {
            throw std::runtime_error(std::string("dlopen failed: ")
                                     + dlerror());
        }
        auto* abi = reinterpret_cast<warpx_external_circuit_abi_version_t>(
            dlsym(m_dl, "warpx_external_circuit_abi_version"));
        auto* make = reinterpret_cast<warpx_create_external_circuit_t>(
            dlsym(m_dl, "warpx_create_external_circuit"));
        if (abi == nullptr || make == nullptr) {
            throw std::runtime_error("plugin lacks the ABI symbols");
        }
        int const v = abi();
        if (v != WARPX_EXTERNAL_CIRCUIT_ABI_VERSION) {
            throw std::runtime_error("ABI version mismatch: plugin "
                                     + std::to_string(v) + ", harness "
                                     + std::to_string(
                                           WARPX_EXTERNAL_CIRCUIT_ABI_VERSION));
        }
        std::cerr << "[harness] dlopen OK, ABI version " << v << "\n";
        m_owned.reset(make());
        m_ec = m_owned.get();
    }

    ~Engine ()
    {
        m_owned.reset();
        m_direct.reset();
        if (m_dl != nullptr) { dlclose(m_dl); }
    }

    ExternalCircuit& Ec () { return *m_ec; }
    c3::C3ExternalCircuit* Direct () { return m_direct.get(); }

private:
    void* m_dl = nullptr;
    std::unique_ptr<ExternalCircuit> m_owned;
    std::unique_ptr<c3::C3ExternalCircuit> m_direct;
    ExternalCircuit* m_ec = nullptr;
};

struct Case
{
    std::vector<std::string> names;
    std::vector<amrex::Real> i_ref;
    std::vector<long> eps_ports;                  // Define indices
    std::vector<std::vector<double>> eps_rows;    // per step, per eps port
};

Case
LoadCase (Args const& a)
{
    Case c;
    std::ifstream mf(a.manifest);
    if (!mf) {
        throw std::runtime_error("cannot open manifest '" + a.manifest + "'");
    }
    std::string line;
    while (std::getline(mf, line)) {
        std::stringstream ss(line);
        std::string nm, iref;
        if (line.empty() || line[0] == '#' || !(ss >> nm >> iref)) {
            continue;
        }
        c.names.push_back(nm);
        c.i_ref.push_back(std::strtod(iref.c_str(), nullptr));
    }
    if (!a.eps_path.empty()) {
        std::ifstream ef(a.eps_path);
        if (!ef) {
            throw std::runtime_error("cannot open eps table '"
                                     + a.eps_path + "'");
        }
        std::getline(ef, line);
        std::stringstream hs(line);
        std::string tok;
        hs >> tok;
        if (tok != "ports") {
            throw std::runtime_error("eps table must start with 'ports ...'");
        }
        while (hs >> tok) {
            long idx = -1;
            for (std::size_t k = 0; k < c.names.size(); ++k) {
                if (c.names[k] == tok) { idx = static_cast<long>(k); }
            }
            if (idx < 0) {
                throw std::runtime_error("eps port '" + tok
                                         + "' not in the manifest");
            }
            c.eps_ports.push_back(idx);
        }
        while (std::getline(ef, line)) {
            if (line.empty() || line[0] == '#') { continue; }
            std::stringstream ss(line);
            std::vector<double> row;
            while (ss >> tok) {
                row.push_back(std::strtod(tok.c_str(), nullptr));
            }
            if (row.size() != c.eps_ports.size()) {
                throw std::runtime_error("eps table row width mismatch");
            }
            c.eps_rows.push_back(row);
        }
    }
    return c;
}

std::vector<amrex::Real>
EpsForStep (Case const& c, long step, double scale)
{
    std::vector<amrex::Real> eps(c.names.size(), 0.0);
    if (step < static_cast<long>(c.eps_rows.size())) {
        for (std::size_t j = 0; j < c.eps_ports.size(); ++j) {
            eps[c.eps_ports[j]] = scale * c.eps_rows[step][j];
        }
    }
    return eps;
}

/** Drive steps [step_lo, step_hi) with the perturbed-evaluation cadence;
 * append one CSV row per accepted step. */
void
RunSteps (Engine& eng, Case const& c, Args const& a, long step_lo,
          long step_hi, std::ofstream& csv, std::ofstream* lock_csv)
{
    std::vector<amrex::Real> scales;
    auto* direct = eng.Direct();
    std::vector<long> lock_idx;
    if (direct != nullptr && lock_csv != nullptr) {
        auto const& net = direct->Network();
        for (long k = net.NCoilCircuits(); k < net.NEast(); ++k) {
            lock_idx.push_back(k);
        }
    }
    for (long i = step_lo; i < step_hi; ++i) {
        double const t0 = i * a.dt;
        double const t1 = (i + 1) * a.dt;
        int const phase_before =
            (direct != nullptr) ? direct->Network().LockPhase() : -1;
        eng.Ec().BeginStep(t0, a.dt);
        if (direct != nullptr && lock_csv != nullptr && phase_before == 0
            && direct->Network().LockPhase() == 1) {
            auto const& net = direct->Network();
            *lock_csv << "# armed t_sim=" << Hex(t0);
            for (long const k : lock_idx) {
                *lock_csv << ' ' << net.EastNames()[k] << '='
                          << Hex(net.LinkedFlux(k));
            }
            *lock_csv << '\n';
        }
        // perturbed non-accepted evaluations: a re-sized (midpoint)
        // substep first, then full-width with scaled eps -- none of
        // these may leave a trace in the accepted trajectory
        for (int e = 0; e < a.evals; ++e) {
            double const t1_eval = (e == 0) ? t0 + 0.5 * a.dt : t1;
            auto const eps = EpsForStep(c, i, 1.0 + 0.5 * (e + 1));
            eng.Ec().AdvanceInterval(t0, t1_eval, eps, false, scales);
        }
        auto const eps = EpsForStep(c, i, 1.0);
        eng.Ec().AdvanceInterval(t0, t1, eps, true, scales);
        eng.Ec().FinishStep();
        csv << Hex(t1);
        for (double const s : scales) {
            csv << ' ' << Hex(s);
        }
        csv << '\n';
        if (direct != nullptr && lock_csv != nullptr) {
            auto const& net = direct->Network();
            *lock_csv << Hex(t1);
            for (long const k : lock_idx) {
                *lock_csv << ' ' << Hex(net.ChannelCurrent(k)) << ' '
                          << Hex(net.LinkedFlux(k));
            }
            *lock_csv << '\n';
        }
    }
}

std::ofstream
OpenCsv (std::string const& path, Case const& c)
{
    std::ofstream csv(path);
    if (!csv) {
        throw std::runtime_error("cannot write '" + path + "'");
    }
    csv << "# t_sim";
    for (auto const& nm : c.names) {
        csv << ' ' << nm;
    }
    csv << '\n';
    return csv;
}

} // namespace

int
main (int argc, char** argv)
{
    try {
        Args const a = ParseArgs(argc, argv);
        Case const c = LoadCase(a);
        long const n_steps = std::llround(a.t_end / a.dt);

        Engine eng(a.plugin);
        eng.Ec().Define(c.names, c.i_ref, a.config);
        std::ofstream csv = OpenCsv(a.out, c);
        std::ofstream lock_csv;
        std::ofstream* lock_ptr = nullptr;
        if (!a.lock_out.empty()) {
            if (eng.Direct() == nullptr) {
                throw std::runtime_error("--lock-out needs direct mode "
                                         "(omit --plugin)");
            }
            lock_csv.open(a.lock_out);
            lock_ptr = &lock_csv;
        }

        if (a.mode == "traj") {
            RunSteps(eng, c, a, 0, n_steps, csv, lock_ptr);
        } else if (a.mode == "ckpt") {
            if (a.ckpt_dir.empty() || a.out2.empty() || a.ckpt_t <= 0.0) {
                throw std::runtime_error(
                    "ckpt mode: required --ckpt-dir --out2 --ckpt-t");
            }
            long const ckpt_step = std::llround(a.ckpt_t / a.dt);
            std::filesystem::create_directories(a.ckpt_dir);
            std::filesystem::create_directories(a.ckpt_dir + "/end_a");
            std::filesystem::create_directories(a.ckpt_dir + "/end_b");
            RunSteps(eng, c, a, 0, ckpt_step, csv, lock_ptr);
            eng.Ec().WriteCheckpoint(a.ckpt_dir);
            RunSteps(eng, c, a, ckpt_step, n_steps, csv, lock_ptr);
            eng.Ec().WriteCheckpoint(a.ckpt_dir + "/end_a");

            Engine eng2(a.plugin);
            eng2.Ec().Define(c.names, c.i_ref,
                             a.config2.empty() ? a.config : a.config2);
            eng2.Ec().ReadCheckpoint(a.ckpt_dir);
            std::ofstream csv2 = OpenCsv(a.out2, c);
            RunSteps(eng2, c, a, ckpt_step, n_steps, csv2, nullptr);
            eng2.Ec().WriteCheckpoint(a.ckpt_dir + "/end_b");
        } else {
            throw std::runtime_error("unknown mode '" + a.mode + "'");
        }
        std::cerr << "[harness] done: " << n_steps << " steps -> " << a.out
                  << "\n";
        return 0;
    } catch (std::exception const& e) {
        std::cerr << "[harness] ERROR: " << e.what() << "\n";
        return 1;
    }
}
