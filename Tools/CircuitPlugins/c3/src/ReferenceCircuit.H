/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef C3_REFERENCE_CIRCUIT_H_
#define C3_REFERENCE_CIRCUIT_H_

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace c3 {

/** Port of hetools-greens Machines/<machine>/the reference circuit module (the
 * validated act-bank solver, itself a port of the reference code's circ_active.f90 v7.3
 * and circ_component_{swd,cap,ind,lmg,cs1}.f90) plus the passive-ring row
 * of the production formation deck's c3_bank_source.py. The Python module is the
 * REFERENCE; every function here mirrors its namesake line by line,
 * including assembly order (unknown / row / triplet indices must be
 * identical so that state vectors and the validation trajectories are
 * directly comparable).
 *
 * One deliberate difference: the Python module bakes the global c3.DT into
 * the assembly triplets; here dt is a NetBuilder constructor argument (no
 * global mutable state), with the same baked-at-assembly semantics.
 */

constexpr double BGR = 1.0e10;   //!< swd off resistance [ohm] (circ_params.f90 bgr)
constexpr double BGP = 1.0e30;   //!< stand-in for HUGE() sentinel times
constexpr double BGM = -1.0e30;
constexpr double FRR = 5.0;      //!< turn-off counter trr multiplier

//! deck-parameter dict of one act circuit (yaml coils[].parameters)
using Params = std::map<std::string, double>;

//! resolved switch-diode device parameters (swd_of output)
struct SwdDev
{
    double rd0 = 0.0;
    double ld = 0.0;
    double tot = 0.0;
    double tnc = 0.0;
    double tdc = 0.0;
    double trr = 0.0;
};

//! one registered swd device (NetBuilder.swd entry)
struct SwdReg
{
    double rd0 = 0.0, ld = 0.0, tot = 0.0, tnc = 0.0, tdc = 0.0, trr = 0.0;
    double crr = 0.0;
    long col = 0;      //!< device current column
    long sense = 0;    //!< conduction-sense column (iqe for rr devices)
    long tri = 0;      //!< triplet position of the DT*rd + ld coefficient
    bool rr = false;   //!< carries the v7.2 reverse-recovery companion
    long r2 = -1, r4 = -1, t2m = -1, t2e = -1, t3e = -1, t4v = -1;
};

/** Resolve one switch-diode device from deck parameters (swd_of).
 *
 * \param keys       (r, l, tot, tnc, tdc, trr) namelist key names
 * \param diode_default  true for get_defaults_swd dflg=.TRUE. (tot=0,
 *                   tnc=0: diode), false for switch defaults (tot=BGP,
 *                   tnc=-1)
 * \param[out] dev   the resolved device (valid only when returning true)
 * \returns          whether the device is active
 */
bool SwdOf (Params const& p, char const* const (&keys)[6],
            bool diode_default, SwdDev& dev);

//! series R-L resolve (ind default = active 0-ohm/0-H short)
std::pair<double, double> RlOf (Params const& p, char const* rkey,
                                char const* lkey);

/** Assemble the global backward-Euler circuit system (node-voltage +
 * device-current unknowns, dt-scaled rows exactly as circ_active.f90
 * update_linear_model). */
class NetBuilder
{
public:
    explicit NetBuilder (double dt) : m_dt(dt) {}

    //! allocate one unknown; return its column index
    long Unk ();
    //! allocate one equation; return its row index
    long Row ();
    //! add one matrix triplet; return its position index
    long Put (long r, long c, double v);
    //! KCL row: sum of signed device currents equals zero
    void Kcl (std::vector<std::pair<double, long>> const& terms);
    //! series R-L KVL: dt(V+ - V-) + (dt r + L) I = L I_old (vminus < 0 = ground)
    void DevRow (long i_dev, long vplus, long vminus, double r, double ll);
    //! switch-diode KVL row(s) with mutable state coefficients (vc_col < 0 = none)
    void SwdRow (long i_dev, long vplus, long vminus, SwdDev const& dev,
                 long vc_col = -1);
    //! capacitor KVL + charge rows (circ_component_cap.f90)
    void CapRows (long i_c, long vplus, double c, double rc, double lc,
                  long vc_col);

    double m_dt;
    long m_n_unk = 0;
    long m_n_row = 0;
    std::vector<long> m_rows, m_cols;
    std::vector<double> m_vals;
    std::vector<double> m_rhs_coef;
    std::vector<long> m_rhs_ref;
    std::vector<SwdReg> m_swd;
    std::vector<long> m_v0_rows, m_illd_cols;
};

/** Build one act circuit (add_coil); returns its initial capacitor
 * charges as (vc_col, v0) pairs. Signed r*/ /*v* deck parameters are
 * honored: negative switch resistance = reversed conduction direction,
 * negative charge voltage = reversed bank. */
std::vector<std::pair<long, double>> AddCoil (NetBuilder& nb, Params const& p);

/** One passive eddy/flux-lock ring: ONE unknown (ring current), ONE KVL
 * row carrying only -dt*R (c3_bank_source.add_passive_ring; the ring L
 * lives on the mel diagonal, injected later by the stepper's adopt).
 * Returns the ring current column index. */
long AddPassiveRing (NetBuilder& nb, double r_ohm);

/** All switch-diode devices (port of update_swd incl. rr states). The
 * Python reference updates with numpy masks; this port translates each
 * mask block into per-device guarded assignments in the same sequence,
 * preserving the semantics of the in-place stt mutations between masks. */
class SwdBank
{
public:
    SwdBank (std::vector<SwdReg> const& reg, double dt);

    //! advance device states; returns true if any matrix coefficient moved
    bool Update (double t, std::vector<double> const& x_old);
    //! write current device states into matrix/rhs coefficients
    void Refresh (std::vector<double>& vals,
                  std::vector<double>& rhs_coef) const;

    long NDev () const { return static_cast<long>(m_rd0.size()); }

    // immutable device parameters
    std::vector<double> m_rd0, m_ld, m_tot, m_tnc, m_tdc, m_trr, m_crr;
    std::vector<long> m_sense, m_tri, m_t2m, m_t2e, m_t3e, m_t4v, m_r2, m_r4;
    std::vector<std::uint8_t> m_rr;
    std::vector<std::uint8_t> m_is_sw;
    double m_dt;

    // mutable state (the six snapshot arrays)
    std::vector<std::int8_t> m_stt;
    std::vector<double> m_cnt, m_rd, m_tau, m_rqe, m_cjn;
};

} // namespace c3

#endif // C3_REFERENCE_CIRCUIT_H_
