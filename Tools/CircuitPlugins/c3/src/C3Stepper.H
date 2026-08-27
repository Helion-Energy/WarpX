/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef C3_C3_STEPPER_H_
#define C3_C3_STEPPER_H_

#include "ReferenceCircuit.H"
#include "DenseLU.H"

#include <memory>
#include <vector>

namespace c3 {

/** Port of hetools-greens Machines/<machine>/c3_stepper.py C3Stepper: one
 * coupled act-bank system with interval stepping on the backward-Euler
 * lattice t = t0 + m dt.
 *
 *   * Advance(t_target, eps): integrate whole dt steps up to t_target
 *     with a remainder carry; eps is a per-coil HELD external EMF
 *     [V, electrical] entering the coil-loop (v0) rows.
 *   * TakeSnapshot()/Restore(): byte-exact state capture (solution
 *     vector, step counter, the six SwdBank state arrays); restore +
 *     refresh + refactorize reproduces the forward trajectory
 *     bit-identically -- the predictor/corrector re-advance requirement.
 *
 * Construction mirrors C3Stepper.from_assembled: the caller assembles a
 * NetBuilder (AddCoil/AddPassiveRing in mel order) and hands it over with
 * the electrical mutual matrix; the -mel columns are appended here.
 *
 * Solver difference vs the Python reference: scipy splu is replaced by the
 * self-contained DenseLU, and a small bit-identity factor cache (keyed on
 * the four coefficient-bearing swd state arrays, LRU) skips the
 * refactorization whenever the matrix is one already factorized -- the
 * dominant case under the coupler's restore-and-re-advance churn. A
 * refactorization of an identical matrix returns identical factors, so
 * trajectories are unaffected; only n_fac counts differ from Python.
 */
class C3Stepper
{
public:
    //! complete integrator state (snapshot()/restore() payload)
    struct Snapshot
    {
        std::vector<double> x;
        long m = 0;
        std::vector<std::int8_t> stt;
        std::vector<double> cnt, rd, tau, rqe, cjn;
    };

    /** Wrap an assembled NetBuilder system (from_assembled). The -mel
     * columns are appended here; do NOT append them in the caller.
     *
     * \param nb        builder after all circuits/rings are added (consumed)
     * \param x0_caps   (vc_col, v0) initial capacitor charges
     * \param emf_rows  coil EMF row indices (one per coil, mel order)
     * \param i_cols    coil current column indices (mel order)
     * \param mel       (n, n) electrical mutual matrix [H], row-major
     * \param t0        circuit time of step index 0 [s]
     */
    C3Stepper (NetBuilder&& nb,
               std::vector<std::pair<long, double>> const& x0_caps,
               std::vector<long> const& emf_rows,
               std::vector<long> const& i_cols,
               std::vector<double> mel, long n_coil, double t0);

    //! circuit time after the last completed whole step
    double T () const { return m_t0 + m_m * m_dt; }
    double Dt () const { return m_dt; }
    long NUnk () const { return m_n_unk; }
    long NCoil () const { return static_cast<long>(m_illd.size()); }
    long NFac () const { return m_n_fac; }
    long NReuse () const { return m_n_reuse; }

    /** Integrate whole dt steps with t <= t_target (+tiny); the remainder
     * < dt carries over. eps: per-coil held EMF [V] (mel order) or
     * nullptr for 0. Returns the number of whole steps taken. */
    long Advance (double t_target, std::vector<double> const* eps);

    //! electrical bank currents [A] in circuit (mel) order
    std::vector<double> Currents () const;
    double Current (long k) const { return m_x[m_illd[k]]; }

    Snapshot TakeSnapshot () const;
    //! restore a snapshot; forces a coefficient rebuild (bit-identity cache
    //! may still reuse the factors of an unchanged matrix)
    void Restore (Snapshot const& s);

    // checkpoint / lock-swap surface
    double T0 () const { return m_t0; }
    void SetT0 (double t0) { m_t0 = t0; }
    long M () const { return m_m; }
    void SetM (long m) { m_m = m; }
    std::vector<double> const& X () const { return m_x; }
    std::vector<double>& MutableX () { return m_x; }
    SwdBank& Bank () { return m_bank; }
    SwdBank const& Bank () const { return m_bank; }
    std::vector<double> const& Mel () const { return m_mel; }
    std::vector<long> const& Illd () const { return m_illd; }
    //! state-derived coefficient refresh + factor invalidation (the
    //! restore-style finish of the machine-0 arm_locks swap)
    void RefreshAndInvalidate ();

private:
    void FactorOrReuse ();

    double m_dt = 0.0;
    double m_t0 = 0.0;
    long m_m = 0;               //!< completed whole steps
    long m_n_unk = 0;
    std::vector<long> m_illd;   //!< coil current columns (mel order)
    std::vector<double> m_mel;  //!< (n_coil, n_coil) row-major [H]
    std::vector<long> m_rows, m_cols;
    std::vector<double> m_vals;
    std::vector<double> m_coef;
    std::vector<long> m_ref;
    std::vector<long> m_v0_rows;
    SwdBank m_bank;
    std::vector<double> m_x;

    //! one cached factorization and the swd state it belongs to
    struct FactorEntry
    {
        std::vector<double> rd, tau, rqe, cjn;
        DenseLU lu;
        long last_use = 0;
    };
    static constexpr int kFactorCacheSize = 3;
    std::vector<FactorEntry> m_cache;
    int m_active = -1;          //!< cache entry the next Solve uses
    long m_use_seq = 0;
    bool m_have_lu = false;     //!< python: self.lu is not None
    long m_n_fac = 0;
    long m_n_reuse = 0;

    std::vector<double> m_rhs, m_xnew;   // scratch
};

} // namespace c3

#endif // C3_C3_STEPPER_H_
