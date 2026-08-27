/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef C3_C3_EXTERNAL_CIRCUIT_H_
#define C3_C3_EXTERNAL_CIRCUIT_H_

#include "C3BankNetwork.H"
#include "C3Config.H"

#include "ExternalCircuit.H"

#include <memory>
#include <string>
#include <vector>

namespace c3 {

/** The c3 act-bank network behind the WarpX external-circuit ABI (v2).
 *
 * Latch-point inventory (the restore-and-re-advance discipline):
 *
 *  1. swd switch/diode transitions: every AdvanceInterval evaluation
 *     restores the interval-entry snapshot first, so switches are always
 *     evaluated against the entry state; the persistent state (and with
 *     it any transition that occurred inside the interval) commits ONLY
 *     when accept = true copies the advanced state into the entry
 *     snapshot. Non-accepted evaluations leave no trace.
 *  2. LU refactorization: state-derived (never latched); the bit-identity
 *     factor cache reproduces re-advanced trajectories exactly.
 *  3. machine-0 lock-ring stage swap: fires ONLY in BeginStep (between
 *     steps, never inside an evaluation), at the first step boundary
 *     t0 > 0 whose machine time t0 - shift >= -dt/2 -- the boundary
 *     nearest machine 0, matching the deck's
 *     max(1, round(shift/dt_warpx)) schedule. BeginStep is post-accept
 *     by contract, so the swap latches immediately and the entry
 *     snapshot is retaken.
 *  4. eps hold: overwritten on every AdvanceInterval from the argument
 *     (the coupler-held predictor value); no plugin-side EMA -- the
 *     hetools reference applies the raw held EMF as-is.
 *  5. checkpoint: WriteCheckpoint serializes the accepted entry snapshot
 *     + lock_phase + the live stepper clock (hexfloat, bit-exact).
 */
class C3ExternalCircuit final : public ExternalCircuit
{
public:
    void Define (std::vector<std::string> const& coil_names,
                 std::vector<amrex::Real> const& i_ref,
                 std::string const& config) override;
    void BeginStep (amrex::Real t0, amrex::Real dt) override;
    void AdvanceInterval (amrex::Real t0, amrex::Real t1,
                          std::vector<amrex::Real> const& eps,
                          bool accept,
                          std::vector<amrex::Real>& scales) override;
    void FinishStep () override;
    void WriteCheckpoint (std::string const& dir) const override;
    void ReadCheckpoint (std::string const& dir) override;

    //! offline-harness introspection (not part of the ABI)
    C3BankNetwork const& Network () const { return *m_net; }
    C3BankNetwork& Network () { return *m_net; }
    C3Config const& Config () const { return m_cfg; }

private:
    void Log (std::string const& msg) const;

    C3Config m_cfg;
    std::unique_ptr<C3BankNetwork> m_net;
    std::vector<double> m_i_ref;          //!< per Define coil [A]
    long m_n_port = 0;
    C3Stepper::Snapshot m_entry;          //!< last accepted state
    long m_n_steps = 0;
};

} // namespace c3

#endif // C3_C3_EXTERNAL_CIRCUIT_H_
