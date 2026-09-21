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
#include "Circuit/ExternalCircuitAffine.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
        m_i_ref.assign(i_ref.begin(), i_ref.end());
        if (m_i_ref.size() != coil_names.size()) { std::abort(); }
        for (auto value : m_i_ref) { if (!std::isfinite(value) || value == 0.) { std::abort(); } }

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
        if (!(m_L > 0.) || !(m_R >= 0.) || !std::isfinite(m_V0)) { std::abort(); }
    }

    void BeginStep (amrex::Real t0, amrex::Real dt) override
    {
        if (m_step_open) { std::abort(); }
        m_step_open = true;
        m_t0 = t0; m_t1 = t0+dt;
        ++m_token;
        m_accepted_this_step = false;
        // The committed state IS the interval-entry snapshot.
    }

    void AdvanceInterval (amrex::Real t0, amrex::Real t1,
                          std::vector<amrex::Real> const& eps,
                          bool accept,
                          std::vector<amrex::Real>& scales) override
    {
        if (!m_step_open || m_accepted_this_step) { std::abort(); }
        if (accept) {
            m_accepted_this_step = true;
            m_last_dt = static_cast<double>(t1 - t0);
            m_last_eps.assign(eps.begin(), eps.end());
        }
        const double dt = static_cast<double>(t1 - t0);
        scales.resize(m_n);
        for (int k = 0; k < m_n; ++k) {
            const double e = (k < static_cast<int>(eps.size()))
                ? static_cast<double>(eps[k]) : 0.0;
            const double i_new =
                (m_i_committed[k] + (dt / m_L) * (m_V0 - e))
                / (1.0 + dt * m_R / m_L);
            scales[k] = static_cast<amrex::Real>(i_new / m_i_ref[k]);
            if (accept) { m_i_committed[k] = i_new; }
        }
    }

    void FinishStep () override
    {
        if (!m_step_open || !m_accepted_this_step) { std::abort(); }
        m_step_open = false;
        std::printf("NATIVE_RL_COMMIT: %d\n", ++m_commits);
        for (int k = 0; k < m_n; ++k) {
            std::printf("NATIVE_RL_PORT: %d %d %.17g %.17g %.17g %.17g\n",
                        m_commits, k, m_last_dt, m_last_eps.at(k),
                        m_i_committed[k], m_i_ref[k]);
        }
    }

    int PrepareAffine (double t0, double t1, WarpxCircuitAffineViewV1* out)
    {
        if (!out || out->struct_bytes != sizeof(*out) || !m_step_open
            || m_accepted_this_step || t0 != m_t0 || t1 != m_t1 || !(t1>t0)) {
            return WARPX_AFFINE_BAD_REQUEST;
        }
        double const dt = t1-t0, denominator = 1.+dt*m_R/m_L;
        m_p0.resize(m_n); m_g.assign(m_n*m_n,0.);
        for (int k = 0; k < m_n; ++k) {
            m_p0[k] = (m_i_committed[k]+(dt/m_L)*m_V0)/denominator;
            m_g[k*m_n+k] = -(dt/m_L)/denominator;
        }
        *out = {};
        out->struct_bytes = sizeof(*out); out->scalar_kind = WARPX_CIRCUIT_AFFINE_F64;
        out->guard_kind = WARPX_CIRCUIT_AFFINE_SIGN_GUARD_V1;
        out->token = m_token; out->n_port = m_n; out->circuit_substeps = 1;
        out->t0_sim = t0; out->t1_sim = t1;
        out->entry_current = m_i_committed.data();
        out->p0 = m_p0.data(); out->g = m_g.data(); out->i_ref = m_i_ref.data();
        return WARPX_AFFINE_OK;
    }

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
    uint64_t m_token = 0;
    double m_t0 = 0., m_t1 = 0.;
    std::vector<double> m_p0, m_g;
    int m_n = 0;
    double m_R = 1.0;
    double m_L = 1.0;
    double m_V0 = 0.0;
    std::vector<double> m_i_committed;
    std::vector<double> m_i_ref;
    bool m_step_open = false;
    bool m_accepted_this_step = false;
    int m_commits = 0;
    double m_last_dt = 0.;
    std::vector<double> m_last_eps;
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

#ifndef WARPX_TEST_WITHOUT_AFFINE
namespace
{
    int32_t PrepareAffine (void* instance, double t0, double t1,
        WarpxCircuitAffineViewV1* out, char* error, uint64_t capacity)
    {
        try {
            if (!instance) { return WARPX_AFFINE_BAD_REQUEST; }
            return static_cast<RLTestCircuit*>(static_cast<ExternalCircuit*>(instance))
                ->PrepareAffine(t0,t1,out);
        } catch (...) {
            if (error && capacity) { std::snprintf(error,capacity,"RL affine preparation failed"); }
            return WARPX_AFFINE_INTERNAL_ERROR;
        }
    }
    void ReleaseAffine (void*, uint64_t) {}
}
extern "C" WarpxCircuitAffineApiV1 const* warpx_external_circuit_affine_api_v1 ()
{
    static WarpxCircuitAffineApiV1 const api{sizeof(WarpxCircuitAffineApiV1),
        WARPX_CIRCUIT_AFFINE_API_V1,0,PrepareAffine,ReleaseAffine};
    return &api;
}
#endif
