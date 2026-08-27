/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef C3_C3_BANK_NETWORK_H_
#define C3_C3_BANK_NETWORK_H_

#include "C3Config.H"
#include "C3Stepper.H"
#include "YamlLite.H"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace c3 {

using Logger = std::function<void (std::string const&)>;

/** Port of the production formation deck's c3_bank_source.py build() +
 * C3CircuitAdapter: the FULL mirror-machine act-bank system.
 *
 * East block = the Define coils, then the loading-only extra circuits
 * (D03/D04: solved for their mutual loading, no eps, no pushed scale),
 * then the passive rings (resistive plate rings first, R = 0 flux-lock
 * rings last). Under mirror the west twins ride identical banks (C13 is
 * the single shared central coil, no twin) and the west EMF entries
 * mirror the east ones (eps_W := eps_E). The electrical mutual matrix is
 * folded from the amp-turn M_EE/M_EW blocks of the matrix file with the
 * same floating-point expressions as the Python reference, so mel is
 * bit-identical across the ports.
 *
 * Flux-lock rings assemble two-stage exactly like build(): stage A (the
 * lock-free system) is the LIVE stepper; stage B (same shared assembly
 * order, lock rows LAST) is assembled at construction and armed by
 * ArmLocks at the step boundary nearest machine 0 -- from then on each
 * lock row conserves its total linked flux exactly. The network works in
 * MACHINE time throughout; the ABI layer converts sim time via the
 * configured shift.
 */
class C3BankNetwork
{
public:
    C3BankNetwork (YamlNode const& doc,
                   std::vector<std::string> const& coil_names,
                   MatrixBlocks const& blocks,
                   std::vector<RingSpec> const& rings,
                   C3Config const& cfg, Logger const& log);

    //! run the machine pre-roll (bias soak) up to machine -shift (sim 0)
    long PreRoll ();

    //! held EMF of one east port: eps_elec = eps_sign * d * emf_per_turn
    void SetPortEmf (long east_k, double emf_per_turn);
    //! west-mirror the held eps and advance the live stepper [machine s]
    void Advance (double t_machine);
    //! painted amp-turn current of east member k (0 for a lock ring
    //! before the machine-0 swap)
    double ChannelCurrent (long east_k) const;

    bool LockSwapPending () const { return m_lock_phase == 0; }
    int LockPhase () const { return m_lock_phase; }
    /** Between-step machine-0 swap (arm_locks): copy the LIVE pre-lock
     * state onto stage B's index-identical shared prefix, zero the lock
     * slots, carry the switch state machines over verbatim, swap with a
     * continuous clock. Returns (name, locked flux [Wb]) per lock ring. */
    std::vector<std::pair<std::string, double>> ArmLocks (double t_machine);

    //! checkpoint restore: select the live stepper for the recorded phase
    //! and set its t0 (hexfloat-exact from the checkpoint)
    void SelectPhase (int lock_phase, double t0_live);

    C3Stepper& Live () { return *m_live; }
    C3Stepper const& Live () const { return *m_live; }

    long NEast () const { return m_n_e; }
    long NCoilCircuits () const { return m_n_c; }
    long NPre () const { return m_n_pre; }
    double T0Real () const { return m_t0_real; }
    std::vector<std::string> const& EastNames () const { return m_names_e; }
    std::vector<double> const& DFull () const { return m_d_full; }
    //! locked flux [Wb] of east lock ring k, from the live full system
    double LinkedFlux (long east_k) const;

private:
    std::vector<double> const& MirroredEps ();

    // layout
    std::vector<std::string> m_names_e;   //!< east names (coils+extra+rings)
    long m_n_c = 0;                       //!< act circuits (coils + extra)
    long m_n_e = 0;                       //!< east total (incl. rings)
    long m_n_plate = 0, m_n_lock = 0;
    long m_kc13 = -1;
    std::vector<long> m_twins;            //!< east indices with west twins
    std::vector<long> m_keep;             //!< pre-lock eps/mel index list
    long m_n_shared = 0;
    std::vector<double> m_d_full;         //!< amp-turn conversion, full size

    // steppers
    std::unique_ptr<C3Stepper> m_stage_a, m_stage_b;
    C3Stepper* m_live = nullptr;
    int m_lock_phase = 1;                 //!< 0 pre-swap (locks pending)

    // eps
    std::vector<double> m_eps;            //!< held electrical EMF, full size
    std::vector<double> m_eps_slice;      //!< scratch (pre-swap keep slice)
    double m_eps_sign = 1.0;

    // clock
    double m_dt = 0.0, m_shift = 0.0, m_t0_real = 0.0;
    long m_n_pre = 0;

    Logger m_log;
};

} // namespace c3

#endif // C3_C3_BANK_NETWORK_H_
