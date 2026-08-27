/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "C3BankNetwork.H"

#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

namespace c3 {

namespace {

/** Resolve one yaml coils[] entry by machine name: exact, "coil_"-prefix
 * stripped, or first character dropped (the east/west pack convention
 * "EF09" -> coil "F09" of load_alt). */
YamlNode const&
FindCoilEntry (std::map<std::string, YamlNode const*> const& by_coil,
               std::string const& name)
{
    auto it = by_coil.find(name);
    if (it == by_coil.end() && name.rfind("coil_", 0) == 0) {
        it = by_coil.find(name.substr(5));
    }
    if (it == by_coil.end() && name.size() > 1) {
        it = by_coil.find(name.substr(1));
    }
    if (it == by_coil.end()) {
        throw std::runtime_error("c3 network: no yaml circuit for coil '"
                                 + name + "'");
    }
    return *it->second;
}

Params
ParamsOf (YamlNode const& ent)
{
    Params p;
    for (auto const& [key, node] : ent.At("parameters").m_map) {
        p[key] = node.AsDouble();
    }
    return p;
}

/** SPD check standing in for the reference's eigvalsh assert: an SPD
 * matrix is exactly one with a successful Cholesky factorization. */
void
AssertSpd (std::vector<double> const& m, long n, std::string const& what)
{
    std::vector<double> a = m;
    for (long k = 0; k < n; ++k) {
        double d = a[k * n + k];
        for (long j = 0; j < k; ++j) {
            d -= a[k * n + j] * a[k * n + j];
        }
        if (!(d > 0.0)) {
            throw std::runtime_error("c3 network: " + what
                                     + " must be SPD (Cholesky pivot "
                                     + std::to_string(d) + " at "
                                     + std::to_string(k) + ")");
        }
        double const s = std::sqrt(d);
        a[k * n + k] = s;
        for (long i = k + 1; i < n; ++i) {
            double v = a[i * n + k];
            for (long j = 0; j < k; ++j) {
                v -= a[i * n + j] * a[k * n + j];
            }
            a[i * n + k] = v / s;
        }
    }
}

} // namespace

C3BankNetwork::C3BankNetwork (YamlNode const& doc,
                              std::vector<std::string> const& coil_names,
                              MatrixBlocks const& blocks,
                              std::vector<RingSpec> const& rings,
                              C3Config const& cfg, Logger const& log)
    : m_eps_sign(cfg.eps_sign), m_dt(cfg.dt), m_shift(cfg.shift), m_log(log)
{
    // ---- east circuit list: Define coils + loading-only extra ----------
    std::map<std::string, YamlNode const*> by_coil;
    for (auto const& ent : doc.At("coils").m_seq) {
        by_coil[ent.At("coil").AsString()] = &ent;
    }
    for (auto const& nm : coil_names) {
        m_names_e.push_back(nm.rfind("coil_", 0) == 0 ? nm.substr(5) : nm);
    }
    for (auto const& nm : cfg.extra) {
        m_names_e.push_back(nm);
    }
    m_n_c = static_cast<long>(m_names_e.size());
    std::vector<Params> circuits_e;
    std::vector<double> d_east;
    for (auto const& nm : m_names_e) {
        YamlNode const& ent = FindCoilEntry(by_coil, nm);
        // live circuit must dispatch to its own act card (part_dispatch
        // is a field in the reference shot's schema; the alternate-case extraction predates it)
        if (ent.Has("part_dispatch")) {
            std::string const circ = ent.At("circuit").AsString();
            for (auto const& kv : ent.At("part_dispatch").m_map) {
                if (kv.second.AsString() != circ) {
                    throw std::runtime_error(
                        "c3 network: '" + nm + "' part '" + kv.first
                        + "' dispatches to '" + kv.second.AsString()
                        + "', not its own card '" + circ + "'");
                }
            }
        }
        circuits_e.push_back(ParamsOf(ent));
        d_east.push_back(ent.At("part_turns").m_seq.at(0).AsDouble());
    }

    // ---- rings: resistive plate rings first, R = 0 locks last ----------
    long n_plate_seen = 0;
    for (auto const& rg : rings) {
        m_names_e.push_back(rg.name);
        d_east.push_back(1.0);
        if (rg.lock) { m_n_lock += 1; } else { n_plate_seen += 1; }
    }
    m_n_plate = n_plate_seen;
    m_n_e = static_cast<long>(m_names_e.size());
    long const n = m_n_e;

    // ---- matrix blocks: order contract check ---------------------------
    if (blocks.n != n) {
        throw std::runtime_error("c3 network: matrix file has "
                                 + std::to_string(blocks.n)
                                 + " channels, expected "
                                 + std::to_string(n));
    }
    for (long k = 0; k < n; ++k) {
        if (blocks.names[k] != m_names_e[k]) {
            throw std::runtime_error("c3 network: matrix channel "
                                     + std::to_string(k) + " is '"
                                     + blocks.names[k] + "', expected '"
                                     + m_names_e[k] + "'");
        }
    }

    // ---- mirror fold (build(): east + west twins, C13 its own mirror) --
    if (cfg.mirror) {
        for (long k = 0; k < m_n_c; ++k) {
            if (m_names_e[k] == "C13") { m_kc13 = k; }
        }
        if (m_kc13 < 0) {
            throw std::runtime_error(
                "c3 network: mirror fold needs the shared central coil C13");
        }
        for (long k = 0; k < n; ++k) {
            if (k != m_kc13) { m_twins.push_back(k); }
        }
    }
    long const n_ts_all = static_cast<long>(m_twins.size());
    long const nf = n + n_ts_all;
    std::vector<double> mfull(static_cast<std::size_t>(nf) * nf, 0.0);
    auto const& ee = blocks.ee;
    auto const& ew = blocks.ew;
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            mfull[i * nf + j] = ee[i * n + j];
        }
        for (long j = 0; j < n_ts_all; ++j) {
            mfull[i * nf + (n + j)] = ew[i * n + m_twins[j]];
        }
    }
    for (long i = 0; i < n_ts_all; ++i) {
        for (long j = 0; j < n; ++j) {
            mfull[(n + i) * nf + j] = ew[m_twins[i] * n + j];
        }
        for (long j = 0; j < n_ts_all; ++j) {
            mfull[(n + i) * nf + (n + j)] = ee[m_twins[i] * n + m_twins[j]];
        }
    }
    if (m_kc13 >= 0) {
        for (long i = 0; i < n_ts_all; ++i) {   // W_k <-> E_C13: own mirror
            mfull[(n + i) * nf + m_kc13] = ee[m_twins[i] * n + m_kc13];
            mfull[m_kc13 * nf + (n + i)] = ee[m_kc13 * n + m_twins[i]];
        }
    }
    for (long i = 0; i < nf; ++i) {             // 0.5 (M + M^T)
        for (long j = i + 1; j < nf; ++j) {
            double const s = 0.5 * (mfull[i * nf + j] + mfull[j * nf + i]);
            mfull[i * nf + j] = s;
            mfull[j * nf + i] = s;
        }
    }
    m_d_full = d_east;
    for (long j = 0; j < n_ts_all; ++j) {
        m_d_full.push_back(d_east[m_twins[j]]);
    }
    std::vector<double> mel(static_cast<std::size_t>(nf) * nf);
    for (long i = 0; i < nf; ++i) {
        for (long j = 0; j < nf; ++j) {
            // the reference's M_full * np.outer(d, d): d_i d_j first
            mel[i * nf + j] = mfull[i * nf + j] * (m_d_full[i] * m_d_full[j]);
        }
    }
    AssertSpd(mel, nf, "electrical mutual matrix");

    // ---- step-lattice alignment: sim 0 lands on a whole circuit step ---
    m_n_pre = std::llround((-cfg.shift - cfg.t0_machine) / cfg.dt);
    m_t0_real = -cfg.shift - m_n_pre * cfg.dt;

    // ---- assembly -------------------------------------------------------
    auto add_east = [&] (NetBuilder& nb,
                         std::vector<std::pair<long, double>>& x0, long k) {
        if (k < m_n_c) {
            auto caps = AddCoil(nb, circuits_e[k]);
            x0.insert(x0.end(), caps.begin(), caps.end());
        } else {
            AddPassiveRing(nb, rings[k - m_n_c].r_ohm);
        }
    };

    if (m_n_lock > 0) {
        // two-stage flux-lock machinery (machine-0 between-step swap):
        // stage A = the lock-free system, live from t0_real; stage B =
        // same shared assembly order + lock rows LAST, armed by ArmLocks
        m_n_shared = m_n_c + m_n_plate;
        // twins below n_shared (locks last); empty when mirror is off
        long const n_ts = std::min(m_n_shared - 1, n_ts_all);
        for (long j = n_ts; j < n_ts_all; ++j) {
            if (m_twins[j] != m_n_shared + (j - n_ts)) {
                throw std::runtime_error("c3 network: lock twins not last");
            }
        }
        for (long k = 0; k < m_n_shared; ++k) { m_keep.push_back(k); }
        for (long j = 0; j < n_ts; ++j) { m_keep.push_back(n + j); }
        long const nk = static_cast<long>(m_keep.size());
        std::vector<double> mel_a(static_cast<std::size_t>(nk) * nk);
        for (long i = 0; i < nk; ++i) {
            for (long j = 0; j < nk; ++j) {
                mel_a[i * nk + j] = mel[m_keep[i] * nf + m_keep[j]];
            }
        }
        NetBuilder nb_a(cfg.dt);
        std::vector<std::pair<long, double>> x0_a;
        for (long k = 0; k < m_n_shared; ++k) { add_east(nb_a, x0_a, k); }
        for (long j = 0; j < n_ts; ++j) { add_east(nb_a, x0_a, m_twins[j]); }
        auto const rows_a = nb_a.m_v0_rows;
        auto const cols_a = nb_a.m_illd_cols;
        m_stage_a = std::make_unique<C3Stepper>(
            std::move(nb_a), x0_a, rows_a, cols_a, std::move(mel_a), nk,
            m_t0_real);

        // stage B: SAME shared assembly order (index-identical prefix),
        // lock rings appended LAST (east locks, then west lock twins);
        // registration lists permuted to MEL order through pos_of_mel
        NetBuilder nb_b(cfg.dt);
        std::vector<std::pair<long, double>> x0_b;
        for (long k = 0; k < m_n_shared; ++k) { add_east(nb_b, x0_b, k); }
        for (long j = 0; j < n_ts; ++j) { add_east(nb_b, x0_b, m_twins[j]); }
        for (long k = m_n_shared; k < n; ++k) { add_east(nb_b, x0_b, k); }
        for (long j = n_ts; j < n_ts_all; ++j) {
            add_east(nb_b, x0_b, m_twins[j]);
        }
        std::vector<long> pos_of_mel(nf);
        for (long k = 0; k < m_n_shared; ++k) { pos_of_mel[k] = k; }
        for (long j = 0; j < m_n_lock; ++j) {
            pos_of_mel[m_n_shared + j] = m_n_shared + n_ts + j;
        }
        for (long j = 0; j < n_ts; ++j) {
            pos_of_mel[n + j] = m_n_shared + j;
        }
        for (long j = 0; j < m_n_lock; ++j) {
            pos_of_mel[n + n_ts + j] = m_n_shared + n_ts + m_n_lock + j;
        }
        std::vector<long> rows_b(nf), cols_b(nf);
        for (long i = 0; i < nf; ++i) {
            rows_b[i] = nb_b.m_v0_rows[pos_of_mel[i]];
            cols_b[i] = nb_b.m_illd_cols[pos_of_mel[i]];
        }
        m_stage_b = std::make_unique<C3Stepper>(
            std::move(nb_b), x0_b, rows_b, cols_b, std::move(mel), nf,
            -cfg.shift /* placeholder; set at arming or restore */);
        if (m_stage_b->NUnk() != m_stage_a->NUnk() + 2 * m_n_lock) {
            throw std::runtime_error("c3 network: stage B size mismatch");
        }
        m_live = m_stage_a.get();
        m_lock_phase = 0;
    } else {
        NetBuilder nb(cfg.dt);
        std::vector<std::pair<long, double>> x0;
        for (long k = 0; k < n; ++k) { add_east(nb, x0, k); }
        for (long j = 0; j < n_ts_all; ++j) { add_east(nb, x0, m_twins[j]); }
        auto const rows0 = nb.m_v0_rows;
        auto const cols0 = nb.m_illd_cols;
        m_stage_a = std::make_unique<C3Stepper>(
            std::move(nb), x0, rows0, cols0, std::move(mel), nf, m_t0_real);
        m_live = m_stage_a.get();
        m_lock_phase = 1;
    }
    m_eps.assign(nf, 0.0);

    std::ostringstream os;
    os << "c3 bank source: " << m_n_c << " east";
    if (m_n_plate > 0) { os << " + " << m_n_plate << " plate-ring"; }
    if (m_n_lock > 0) { os << " + " << m_n_lock << " lock-ring"; }
    os << " + " << n_ts_all << " west-twin circuits ("
       << m_live->NUnk() << " unknowns), t0 = " << m_t0_real * 1e6
       << " us (" << m_n_pre << " pre-roll steps at dt = " << cfg.dt
       << " s)";
    m_log(os.str());
}

long
C3BankNetwork::PreRoll ()
{
    long const n_adv = m_live->Advance(-m_shift, nullptr);
    if (n_adv != m_n_pre) {
        throw std::runtime_error("c3 network: pre-roll took "
                                 + std::to_string(n_adv) + " steps, expected "
                                 + std::to_string(m_n_pre));
    }
    return n_adv;
}

void
C3BankNetwork::SetPortEmf (long east_k, double emf_per_turn)
{
    m_eps[east_k] = m_eps_sign * m_d_full[east_k] * emf_per_turn;
}

std::vector<double> const&
C3BankNetwork::MirroredEps ()
{
    long const n = m_n_e;
    for (std::size_t j = 0; j < m_twins.size(); ++j) {
        m_eps[n + j] = m_eps[m_twins[j]];   // eps_W := eps_E
    }
    if (m_lock_phase == 0) {
        m_eps_slice.resize(m_keep.size());
        for (std::size_t i = 0; i < m_keep.size(); ++i) {
            m_eps_slice[i] = m_eps[m_keep[i]];
        }
        return m_eps_slice;
    }
    return m_eps;
}

void
C3BankNetwork::Advance (double t_machine)
{
    auto const& eps = MirroredEps();
    m_live->Advance(t_machine, &eps);
}

double
C3BankNetwork::ChannelCurrent (long east_k) const
{
    if (m_lock_phase == 0 && east_k >= m_n_shared) {
        return 0.0;   // the joint is insulating pre-swap
    }
    return m_live->Current(east_k) * m_d_full[east_k];
}

double
C3BankNetwork::LinkedFlux (long east_k) const
{
    if (m_lock_phase == 0 && east_k >= m_n_shared) {
        return 0.0;
    }
    long const nc = m_live->NCoil();
    double s = 0.0;
    double const* mrow = m_live->Mel().data() + east_k * nc;
    for (long j = 0; j < nc; ++j) {
        s += mrow[j] * m_live->Current(j);
    }
    return s;
}

std::vector<std::pair<std::string, double>>
C3BankNetwork::ArmLocks (double t_machine)
{
    if (m_lock_phase != 0 || m_stage_b == nullptr) {
        throw std::runtime_error("c3 network: ArmLocks with no pending swap");
    }
    C3Stepper& a = *m_stage_a;
    C3Stepper& b = *m_stage_b;
    if (std::fabs(a.T() - t_machine) >= 0.5 * m_dt) {
        throw std::runtime_error(
            "c3 network: swap fired off the scheduled boundary (stepper at "
            + std::to_string(a.T()) + " s, scheduled "
            + std::to_string(t_machine) + " s)");
    }
    b.SetT0(a.T());   // swap-boundary machine time; continuous clock
    b.SetM(0);
    std::vector<double>& xb = b.MutableX();
    std::vector<double> const& xa = a.X();
    for (long i = 0; i < a.NUnk(); ++i) { xb[i] = xa[i]; }
    for (long i = a.NUnk(); i < b.NUnk(); ++i) { xb[i] = 0.0; }
    if (b.Bank().NDev() != a.Bank().NDev()) {
        throw std::runtime_error("c3 network: stage A/B swd banks differ");
    }
    b.Bank().m_stt = a.Bank().m_stt;   // locks add no swd devices
    b.Bank().m_cnt = a.Bank().m_cnt;
    b.Bank().m_rd = a.Bank().m_rd;
    b.Bank().m_tau = a.Bank().m_tau;
    b.Bank().m_rqe = a.Bank().m_rqe;
    b.Bank().m_cjn = a.Bank().m_cjn;
    b.RefreshAndInvalidate();
    m_live = &b;
    m_lock_phase = 1;

    std::vector<std::pair<std::string, double>> locked;
    for (long k = m_n_c + m_n_plate; k < m_n_e; ++k) {
        locked.emplace_back(m_names_e[k], LinkedFlux(k));
    }
    std::ostringstream os;
    os << "c3 flux-lock swap ARMED at machine t = " << a.T() * 1e6
       << " us; locked fluxes [mWb]:";
    std::map<char, double> st_tot;
    for (auto const& [nm, p] : locked) {
        os << " " << nm << "=" << p * 1e3;
        st_tot[nm.size() > 3 ? nm[3] : '?'] += p;
    }
    os << "; station sums [mWb]:";
    for (auto const& [s, v] : st_tot) {
        os << " " << s << "=" << v * 1e3;
    }
    m_log(os.str());
    return locked;
}

void
C3BankNetwork::SelectPhase (int lock_phase, double t0_live)
{
    if (m_stage_b == nullptr) {
        if (lock_phase != 1) {
            throw std::runtime_error(
                "c3 network: lock_phase 0 checkpoint but no lock rings "
                "configured");
        }
    } else if (lock_phase == 1) {
        m_live = m_stage_b.get();
        m_lock_phase = 1;
    } else {
        m_live = m_stage_a.get();
        m_lock_phase = 0;
    }
    m_live->SetT0(t0_live);
}

} // namespace c3
