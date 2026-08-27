/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

/* Test-fixture ExternalCircuit plugin: independent driven RL loops.
 *
 * One loop per Define'd coil, all sharing the R/L/V0 of the
 * circuit.plugin_config string ("R=...,L=...,V0=..."):
 *
 *     L dI/dt + R I = V0 - eps,
 *
 * advanced by ONE backward-Euler step over each coupling interval from
 * the interval-entry (committed) state:
 *
 *     I(t1) = (I_entry + (t1 - t0)/L * (V0 - eps)) / (1 + (t1 - t0) R/L).
 *
 * accept = false evaluations are pure functions of (t0, t1, eps) and the
 * committed state (restore-and-re-advance); the committed state moves
 * ONLY on accept = true. This is the exact dynamics of the python-hook
 * RL reference in inputs_test_rz_theta_implicit_mhd_circuit_hook_picmi.py,
 * so the native-driver parity test can compare committed scales directly.
 *
 * Builds as a standalone MODULE library against the self-contained ABI
 * header only (the plugin calls no WarpX/AMReX symbols).
 */

#include "Circuit/ExternalCircuit.H"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

class RLTestCircuit final : public ExternalCircuit
{
public:
    void Define (std::vector<std::string> const& coil_names,
                 std::vector<amrex::Real> const& i_ref,
                 std::string const& config) override
    {
        m_n = static_cast<int>(coil_names.size());
        m_i_committed.assign(m_n, 0.0);
        static_cast<void>(i_ref);

        // key=value comma list; unknown keys are rejected loudly.
        std::stringstream ss(config);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const auto eq = item.find('=');
            if (eq == std::string::npos) { continue; }
            const std::string key = item.substr(0, eq);
            const double value = std::stod(item.substr(eq + 1));
            if (key == "R") { m_R = value; }
            else if (key == "L") { m_L = value; }
            else if (key == "V0") { m_V0 = value; }
            else {
                std::fprintf(stderr,
                             "rl_test_circuit: unknown config key '%s'\n",
                             key.c_str());
                std::abort();
            }
        }
    }

    void BeginStep (amrex::Real /*t0*/, amrex::Real /*dt*/) override
    {
        // The committed state IS the interval-entry snapshot.
    }

    void AdvanceInterval (amrex::Real t0, amrex::Real t1,
                          std::vector<amrex::Real> const& eps,
                          bool accept,
                          std::vector<amrex::Real>& scales) override
    {
        const double dt = static_cast<double>(t1 - t0);
        scales.resize(m_n);
        for (int k = 0; k < m_n; ++k) {
            const double e = (k < static_cast<int>(eps.size()))
                ? static_cast<double>(eps[k]) : 0.0;
            const double i_new =
                (m_i_committed[k] + (dt / m_L) * (m_V0 - e))
                / (1.0 + dt * m_R / m_L);
            scales[k] = static_cast<amrex::Real>(i_new);
            if (accept) { m_i_committed[k] = i_new; }
        }
    }

    void FinishStep () override {}

    void WriteCheckpoint (std::string const& dir) const override
    {
        std::ofstream ofs(dir + "/rl_test_circuit.dat");
        ofs.precision(17);
        ofs << m_n << "\n";
        for (const double i : m_i_committed) { ofs << i << "\n"; }
    }

    void ReadCheckpoint (std::string const& dir) override
    {
        std::ifstream ifs(dir + "/rl_test_circuit.dat");
        if (!ifs.good()) { return; }
        int n = 0;
        ifs >> n;
        if (n == m_n) {
            for (int k = 0; k < m_n; ++k) { ifs >> m_i_committed[k]; }
        }
    }

private:
    int m_n = 0;
    double m_R = 1.0;
    double m_L = 1.0;
    double m_V0 = 0.0;
    std::vector<double> m_i_committed;
};

} // namespace

extern "C"
{
    ExternalCircuit* warpx_create_external_circuit ()
    {
        return new RLTestCircuit();
    }

    int warpx_external_circuit_abi_version ()
    {
        return WARPX_EXTERNAL_CIRCUIT_ABI_VERSION;
    }
}
