/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "C3Stepper.H"

#include <stdexcept>
#include <utility>

namespace c3 {

namespace {
constexpr double TINY = 1.0e-12;   // c3_stepper._TINY
}

C3Stepper::C3Stepper (NetBuilder&& nb,
                      std::vector<std::pair<long, double>> const& x0_caps,
                      std::vector<long> const& emf_rows,
                      std::vector<long> const& i_cols,
                      std::vector<double> mel, long n_coil, double t0)
    : m_dt(nb.m_dt), m_t0(t0), m_illd(i_cols), m_mel(std::move(mel)),
      m_bank(nb.m_swd, nb.m_dt)
{
    // _adopt: append the -mel columns onto the coil EMF rows
    long const n = n_coil;
    if (static_cast<long>(m_illd.size()) != n
        || static_cast<long>(emf_rows.size()) != n
        || static_cast<long>(m_mel.size()) != n * n) {
        throw std::runtime_error("C3Stepper: mel/emf_rows/i_cols size mismatch");
    }
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            nb.Put(emf_rows[i], m_illd[j], -m_mel[i * n + j]);
        }
    }
    if (nb.m_n_row != nb.m_n_unk) {
        throw std::runtime_error("C3Stepper: n_row != n_unk ("
                                 + std::to_string(nb.m_n_row) + " vs "
                                 + std::to_string(nb.m_n_unk) + ")");
    }
    m_rows = std::move(nb.m_rows);
    m_cols = std::move(nb.m_cols);
    m_vals = std::move(nb.m_vals);
    m_coef = std::move(nb.m_rhs_coef);
    m_ref = std::move(nb.m_rhs_ref);
    m_v0_rows = emf_rows;
    m_n_unk = nb.m_n_unk;
    m_x.assign(m_n_unk, 0.0);
    for (auto const& [vc_col, v_chg] : x0_caps) {
        m_x[vc_col] = v_chg;
    }
}

void
C3Stepper::FactorOrReuse ()
{
    // A refactorization of a bit-identical matrix returns bit-identical
    // factors: when the refresh inputs (the four coefficient-bearing swd
    // state arrays) match a cached factorization, reuse it.
    m_use_seq += 1;
    for (std::size_t e = 0; e < m_cache.size(); ++e) {
        auto& ent = m_cache[e];
        if (ent.rd == m_bank.m_rd && ent.tau == m_bank.m_tau
            && ent.rqe == m_bank.m_rqe && ent.cjn == m_bank.m_cjn) {
            ent.last_use = m_use_seq;
            m_active = static_cast<int>(e);
            m_n_reuse += 1;
            return;
        }
    }
    int slot;
    if (static_cast<int>(m_cache.size()) < kFactorCacheSize) {
        m_cache.emplace_back();
        slot = static_cast<int>(m_cache.size()) - 1;
    } else {
        slot = 0;   // evict least recently used
        for (std::size_t e = 1; e < m_cache.size(); ++e) {
            if (m_cache[e].last_use < m_cache[slot].last_use) {
                slot = static_cast<int>(e);
            }
        }
    }
    auto& ent = m_cache[slot];
    ent.lu.Factor(m_n_unk, m_rows, m_cols, m_vals);
    ent.rd = m_bank.m_rd;
    ent.tau = m_bank.m_tau;
    ent.rqe = m_bank.m_rqe;
    ent.cjn = m_bank.m_cjn;
    ent.last_use = m_use_seq;
    m_active = slot;
    m_n_fac += 1;
}

long
C3Stepper::Advance (double t_target, std::vector<double> const* eps)
{
    long const n_before = m_m;
    long const n = static_cast<long>(m_illd.size());
    std::vector<double> deps;
    if (eps != nullptr) {
        if (static_cast<long>(eps->size()) != n) {
            throw std::runtime_error("C3Stepper::Advance: eps size mismatch");
        }
        deps.resize(n);
        for (long i = 0; i < n; ++i) {
            deps[i] = m_dt * (*eps)[i];
        }
    }
    while (m_t0 + (m_m + 1) * m_dt <= t_target + TINY) {
        m_m += 1;
        double const t = m_t0 + m_m * m_dt;
        if (m_bank.Update(t, m_x) || !m_have_lu) {
            m_bank.Refresh(m_vals, m_coef);
            FactorOrReuse();
            m_have_lu = true;
        }
        m_rhs.resize(m_n_unk);
        for (long i = 0; i < m_n_unk; ++i) {
            m_rhs[i] = m_coef[i] * m_x[m_ref[i]];
        }
        for (long i = 0; i < n; ++i) {
            double s = 0.0;
            double const* mrow = m_mel.data() + i * n;
            for (long j = 0; j < n; ++j) {
                s += mrow[j] * m_x[m_illd[j]];
            }
            m_rhs[m_v0_rows[i]] = -s;
            if (eps != nullptr) {
                m_rhs[m_v0_rows[i]] += deps[i];
            }
        }
        m_cache[m_active].lu.Solve(m_rhs, m_xnew);
        m_x.swap(m_xnew);
    }
    return m_m - n_before;
}

std::vector<double>
C3Stepper::Currents () const
{
    std::vector<double> out(m_illd.size());
    for (std::size_t k = 0; k < m_illd.size(); ++k) {
        out[k] = m_x[m_illd[k]];
    }
    return out;
}

C3Stepper::Snapshot
C3Stepper::TakeSnapshot () const
{
    return {m_x, m_m, m_bank.m_stt, m_bank.m_cnt, m_bank.m_rd,
            m_bank.m_tau, m_bank.m_rqe, m_bank.m_cjn};
}

void
C3Stepper::Restore (Snapshot const& s)
{
    m_x = s.x;
    m_m = s.m;
    m_bank.m_stt = s.stt;
    m_bank.m_cnt = s.cnt;
    m_bank.m_rd = s.rd;
    m_bank.m_tau = s.tau;
    m_bank.m_rqe = s.rqe;
    m_bank.m_cjn = s.cjn;
    RefreshAndInvalidate();
}

void
C3Stepper::RefreshAndInvalidate ()
{
    m_bank.Refresh(m_vals, m_coef);
    m_have_lu = false;   // python: self.lu = None (cache may still reuse)
}

} // namespace c3
