/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ReferenceCircuit.H"

#include <cmath>
#include <stdexcept>

namespace c3 {

namespace {

double
Get (Params const& p, std::string const& key, double dflt)
{
    auto const it = p.find(key);
    return (it == p.end()) ? dflt : it->second;
}

bool
Has (Params const& p, std::string const& key)
{
    return p.find(key) != p.end();
}

double
Sign (double v)
{
    return (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0);
}

} // namespace

bool
SwdOf (Params const& p, char const* const (&keys)[6], bool diode_default,
       SwdDev& dev)
{
    double const tot0 = diode_default ? 0.0 : BGP;
    double const tnc0 = diode_default ? 0.0 : -1.0;
    dev.rd0 = Get(p, keys[0], 0.0);
    dev.ld = Get(p, keys[1], 0.0);
    dev.tot = Get(p, keys[2], tot0);
    dev.tnc = Get(p, keys[3], tnc0);
    dev.tdc = Get(p, keys[4], BGM);
    dev.trr = Get(p, keys[5], 0.0);
    if (dev.tnc < 0.0) {                     // switch mode: active iff tot given
        return dev.tot != BGP;
    }
    return std::fabs(dev.rd0) > 0.0;         // diode mode: active iff rd0 != 0
}

std::pair<double, double>
RlOf (Params const& p, char const* rkey, char const* lkey)
{
    return {Get(p, rkey, 0.0), Get(p, lkey, 0.0)};
}

// ---------------------------------------------------------------------
// NetBuilder
// ---------------------------------------------------------------------
long
NetBuilder::Unk ()
{
    m_n_unk += 1;
    return m_n_unk - 1;
}

long
NetBuilder::Row ()
{
    m_n_row += 1;
    m_rhs_coef.push_back(0.0);
    m_rhs_ref.push_back(0);
    return m_n_row - 1;
}

long
NetBuilder::Put (long r, long c, double v)
{
    m_rows.push_back(r);
    m_cols.push_back(c);
    m_vals.push_back(v);
    return static_cast<long>(m_vals.size()) - 1;
}

void
NetBuilder::Kcl (std::vector<std::pair<double, long>> const& terms)
{
    long const r = Row();
    for (auto const& [sign, col] : terms) {
        Put(r, col, sign * m_dt);
    }
}

void
NetBuilder::DevRow (long i_dev, long vplus, long vminus, double r, double ll)
{
    long const rw = Row();
    Put(rw, vplus, m_dt);
    if (vminus >= 0) {
        Put(rw, vminus, -m_dt);
    }
    Put(rw, i_dev, m_dt * r + ll);
    m_rhs_coef[rw] = ll;
    m_rhs_ref[rw] = i_dev;
}

void
NetBuilder::SwdRow (long i_dev, long vplus, long vminus, SwdDev const& dev,
                    long vc_col)
{
    bool const rr = dev.tnc >= 0.0 && dev.trr > 0.0;
    long const rw = Row();
    Put(rw, vplus, m_dt);
    if (vminus >= 0) {
        Put(rw, vminus, -m_dt);
    }
    long const tri = Put(rw, i_dev, m_dt * BGR + dev.ld);
    if (vc_col >= 0) {
        Put(rw, vc_col, -m_dt);
    }
    m_rhs_coef[rw] = dev.ld;
    m_rhs_ref[rw] = i_dev;
    SwdReg reg;
    reg.rd0 = dev.rd0;
    reg.ld = dev.ld;
    reg.tot = dev.tot;
    reg.tnc = dev.tnc;
    reg.tdc = dev.tdc;
    reg.trr = dev.trr;
    reg.col = i_dev;
    reg.sense = i_dev;
    reg.tri = tri;
    reg.rr = rr;
    if (rr) {
        // v7.2 reverse-recovery companion (Iqm, Iqe, Vrr + three rows,
        // circ_component_swd.f90 get_linear_model_swd); the conduction
        // sensing then uses Iqe instead of Id
        if (!(dev.tnc > 0.0)) {
            throw std::runtime_error("SwdRow: rr device needs tnc > 0");
        }
        long const iqm = Unk();
        long const iqe = Unk();
        long const vrr = Unk();
        Put(rw, vrr, m_dt);                      // eq1 gains +dt Vrr
        // eq2: 0 = -tau Iqe + tnc tau dIqm/dt + (tnc + tau) Iqm
        long const r2 = Row();
        long const t2m = Put(r2, iqm, m_dt * dev.tnc);   // tau = 0 start
        long const t2e = Put(r2, iqe, 0.0);
        m_rhs_ref[r2] = iqm;                     // coef = tnc tau (state-dep)
        // eq3: 0 = Rqe Iqe - Vrr
        long const r3 = Row();
        long const t3e = Put(r3, iqe, 0.0);      // rqe = 0 start
        Put(r3, vrr, -m_dt);
        // eq4: 0 = Id + Iqm - Iqe - Cjn dVrr/dt
        long const r4 = Row();
        Put(r4, i_dev, m_dt);
        Put(r4, iqm, m_dt);
        Put(r4, iqe, -m_dt);
        long const t4v = Put(r4, vrr, 0.0);      // -cjn, = 0 at start
        m_rhs_ref[r4] = vrr;                     // coef = -cjn (state-dep)
        reg.sense = iqe;
        reg.crr = 1.0e-12;
        reg.r2 = r2;
        reg.r4 = r4;
        reg.t2m = t2m;
        reg.t2e = t2e;
        reg.t3e = t3e;
        reg.t4v = t4v;
    }
    m_swd.push_back(reg);
}

void
NetBuilder::CapRows (long i_c, long vplus, double c, double rc, double lc,
                     long vc_col)
{
    long const rw = Row();
    Put(rw, vplus, m_dt);
    Put(rw, i_c, m_dt * rc + lc);
    Put(rw, vc_col, -m_dt);
    m_rhs_coef[rw] = lc;
    m_rhs_ref[rw] = i_c;
    long const r2 = Row();
    Put(r2, i_c, m_dt);
    Put(r2, vc_col, c);
    m_rhs_coef[r2] = c;
    m_rhs_ref[r2] = vc_col;
}

// ---------------------------------------------------------------------
// circuit assembly
// ---------------------------------------------------------------------
namespace {

//! per-segment namelist keys (SEG_KEYS): l, sw, df, db, c, lf, lx
struct SegKeys
{
    int k;
    char const* l[2];
    char const* sw[6];
    char const* df[6];
    char const* db[6];
    char const* c[4];
    char const* lf[2];
    char const* lx[2];
};

//! diode/segment parameter mapping per import_from_nml: df2/db2=(rd2,rd3),
//! df3/db3=(rd6,rd7), df4/db4=(rd5,rd4), df5/db5=(rd8,rd9)
constexpr SegKeys kSegKeys[] = {
    {2, {"rl2", "l2"}, {"r2", "ls2", "t2", "tn2", "tdc2", "trr2"},
     {"rd2", "ld2", "tot2", "tnc2", "tdcd2", "trrd2"},
     {"rd3", "ld3", "tot3", "tnc3", "tdcd3", "trrd3"},
     {"c2", "rc2", "lc2", "v20"}, {"rlf2", "lf2"}, {"rlx2", "lx2"}},
    {3, {"rl3", "l3"}, {"r3", "ls3", "t3", "tn3", "tdc3", "trr3"},
     {"rd6", "ld6", "tot6", "tnc6", "tdcd6", "trrd6"},
     {"rd7", "ld7", "tot7", "tnc7", "tdcd7", "trrd7"},
     {"c3", "rc3", "lc3", "v30"}, {"rlf3", "lf3"}, {"rlx3", "lx3"}},
    {4, {"rl4", "l4"}, {"r4", "ls4", "t4", "tn4", "tdc4", "trr4"},
     {"rd5", "ld5", "tot5", "tnc5", "tdcd5", "trrd5"},
     {"rd4", "ld4", "tot4", "tnc4", "tdcd4", "trrd4"},
     {"c4", "rc4", "lc4", "v40"}, {"rlf4", "lf4"}, {"rlx4", "lx4"}},
    {5, {"rl5", "l5"}, {"r5", "ls5", "t5", "tn5", "tdc5", "trr5"},
     {"rd8", "ld8", "tot8", "tnc8", "tdcd8", "trrd8"},
     {"rd9", "ld9", "tot9", "tnc9", "tdcd9", "trrd9"},
     {"c5", "rc5", "lc5", "v50"}, {"rlf5", "lf5"}, {"rlx5", "lx5"}},
};

/** Add one switch leg v_f -[lf]- -[sw]- -[lx]- v_b (add_leg); returns
 * (i_lf, i_lx), the top and bottom current columns of the leg. */
std::pair<long, long>
AddLeg (NetBuilder& nb, long v_f, long v_b, std::pair<double, double> lf,
        SwdDev const& sw, std::pair<double, double> lx)
{
    long const i_lf = nb.Unk();
    long const v_fs = nb.Unk();
    long const i_sw = nb.Unk();
    long const v_s = nb.Unk();
    long const i_lx = nb.Unk();
    nb.DevRow(i_lf, v_f, v_fs, lf.first, lf.second);
    nb.Kcl({{-1.0, i_lf}, {+1.0, i_sw}});
    nb.SwdRow(i_sw, v_fs, v_s, sw);
    nb.Kcl({{-1.0, i_sw}, {+1.0, i_lx}});
    nb.DevRow(i_lx, v_s, v_b, lx.first, lx.second);
    return {i_lf, i_lx};
}

} // namespace

std::vector<std::pair<long, double>>
AddCoil (NetBuilder& nb, Params const& p)
{
    std::vector<std::pair<long, double>> caps;   // (vc_col, v0) charges
    long const v0 = nb.Unk();
    long const i_lld = nb.Unk();
    long const vh = nb.Unk();
    long const r_v0 = nb.Row();
    nb.Put(r_v0, v0, nb.m_dt);   // -Mel columns appended globally later
    nb.m_v0_rows.push_back(r_v0);
    nb.m_illd_cols.push_back(i_lld);
    nb.DevRow(i_lld, v0, vh, Get(p, "rld", 0.0), Get(p, "lld", 0.0));
    std::vector<std::pair<double, long>> vh_terms = {{-1.0, i_lld}};

    // segment 1 (cs1 header damper: switch r1/l1 in series with c1)
    SwdDev cs1;
    if (SwdOf(p, {"r1", "l1", "t1", "tn1", "tdc1", "trr1"}, false, cs1)) {
        if (cs1.rd0 < 0.0) {
            throw std::runtime_error(
                "AddCoil: r1 < 0 flux-programming hack not supported");
        }
        long const i1 = nb.Unk();
        long const vc1 = nb.Unk();
        nb.SwdRow(i1, vh, -1, cs1, vc1);
        long const r2 = nb.Row();
        nb.Put(r2, i1, nb.m_dt);
        nb.Put(r2, vc1, Get(p, "c1", 0.0));
        nb.m_rhs_coef[r2] = Get(p, "c1", 0.0);
        nb.m_rhs_ref[r2] = vc1;
        caps.emplace_back(vc1, Get(p, "v10", 0.0));
        vh_terms.emplace_back(+1.0, i1);
    }

    // segments 2..5
    for (auto const& kk : kSegKeys) {
        if (!Has(p, kk.l[1])) {   // lmg inactive -> whole segment dead
            continue;
        }
        if (Has(p, "lu" + std::to_string(kk.k))) {
            throw std::runtime_error(
                "AddCoil: magic-inductor swap not in this deck");
        }
        SwdDev sw, df, db;
        bool const has_sw = SwdOf(p, kk.sw, false, sw);
        bool const has_df = SwdOf(p, kk.df, true, df);
        bool const has_db = SwdOf(p, kk.db, true, db);
        double const cval = Get(p, kk.c[0], 0.0);
        bool const has_c = cval != 0.0;
        bool const leg = has_sw && (has_c || has_db);
        if (!has_df && !leg) {
            continue;
        }
        long const i_l = nb.Unk();
        long const v_kf = nb.Unk();
        auto const rl = RlOf(p, kk.l[0], kk.l[1]);
        nb.DevRow(i_l, vh, v_kf, rl.first, rl.second);
        vh_terms.emplace_back(+1.0, i_l);
        std::vector<std::pair<double, long>> kf_terms = {{-1.0, i_l}};
        if (has_df) {
            long const i_df = nb.Unk();
            nb.SwdRow(i_df, v_kf, -1, df);
            kf_terms.emplace_back(+1.0, i_df);
        }
        if (leg) {
            long const v_kb = nb.Unk();
            auto const [i_lf, i_lx] = AddLeg(nb, v_kf, v_kb,
                                             RlOf(p, kk.lf[0], kk.lf[1]), sw,
                                             RlOf(p, kk.lx[0], kk.lx[1]));
            kf_terms.emplace_back(+1.0, i_lf);
            std::vector<std::pair<double, long>> kb_terms = {{-1.0, i_lx}};
            if (has_db) {
                long const i_db = nb.Unk();
                nb.SwdRow(i_db, v_kb, -1, db);
                kb_terms.emplace_back(+1.0, i_db);
            }
            if (has_c) {
                long const i_c = nb.Unk();
                long const vc = nb.Unk();
                nb.CapRows(i_c, v_kb, cval, Get(p, kk.c[1], 0.0),
                           Get(p, kk.c[2], 0.0), vc);
                caps.emplace_back(vc, Get(p, kk.c[3], 0.0));
                kb_terms.emplace_back(+1.0, i_c);
            }
            nb.Kcl(kb_terms);
        }
        nb.Kcl(kf_terms);
    }

    // segment 6 (three parallel legs sharing v6f / v6b)
    if (Has(p, "l6")) {
        if (Has(p, "lu6")) {
            throw std::runtime_error(
                "AddCoil: magic-inductor swap not in this deck");
        }
        SwdDev sw6, s6a, s6b, df6, db6;
        bool const has_sw6 = SwdOf(
            p, {"r6", "ls6", "t6", "tn6", "tdc6", "trr6"}, true, sw6);
        bool const has_s6a = SwdOf(
            p, {"rs6a", "ls6a", "ts6a", "tns6a", "tdcs6a", "trrs6a"}, true,
            s6a);
        bool const has_s6b = SwdOf(
            p, {"rs6b", "ls6b", "ts6b", "tns6b", "tdcs6b", "trrs6b"}, true,
            s6b);
        bool const has_df6 = SwdOf(
            p, {"rdf6", "ldf6", "totf6", "tncf6", "tdcdf6", "trrdf6"}, true,
            df6);
        bool const has_db6 = SwdOf(
            p, {"rdb6", "ldb6", "totb6", "tncb6", "tdcdb6", "trrdb6"}, true,
            db6);
        bool const has_c6 = Get(p, "c6", 0.0) != 0.0;
        struct Leg
        {
            SwdDev const* s;
            char const* lf[2];
            char const* lx[2];
        };
        std::vector<Leg> legs;
        if (has_sw6) { legs.push_back({&sw6, {"rlf6", "lf6"}, {"rlx6", "lx6"}}); }
        if (has_s6a) { legs.push_back({&s6a, {"rlf6a", "lf6a"}, {"rlx6a", "lx6a"}}); }
        if (has_s6b) { legs.push_back({&s6b, {"rlf6b", "lf6b"}, {"rlx6b", "lx6b"}}); }
        if ((has_c6 || has_db6) && (!legs.empty() || has_df6)) {
            long const i_l6 = nb.Unk();
            long const v6f = nb.Unk();
            auto const rl6 = RlOf(p, "rl6", "l6");
            nb.DevRow(i_l6, vh, v6f, rl6.first, rl6.second);
            vh_terms.emplace_back(+1.0, i_l6);
            std::vector<std::pair<double, long>> f_terms = {{-1.0, i_l6}};
            if (has_df6) {
                long const i_df6 = nb.Unk();
                nb.SwdRow(i_df6, v6f, -1, df6);
                f_terms.emplace_back(+1.0, i_df6);
            }
            if (!legs.empty()) {
                long const v6b = nb.Unk();
                std::vector<std::pair<double, long>> b_terms;
                for (auto const& lg : legs) {
                    auto const [i_lf, i_lx] = AddLeg(
                        nb, v6f, v6b, RlOf(p, lg.lf[0], lg.lf[1]), *lg.s,
                        RlOf(p, lg.lx[0], lg.lx[1]));
                    f_terms.emplace_back(+1.0, i_lf);
                    b_terms.emplace_back(-1.0, i_lx);
                }
                if (has_db6) {
                    long const i_db6 = nb.Unk();
                    nb.SwdRow(i_db6, v6b, -1, db6);
                    b_terms.emplace_back(+1.0, i_db6);
                }
                if (has_c6) {
                    long const i_c6 = nb.Unk();
                    long const vc6 = nb.Unk();
                    nb.CapRows(i_c6, v6b, Get(p, "c6", 0.0),
                               Get(p, "rc6", 0.0), Get(p, "lc6", 0.0), vc6);
                    caps.emplace_back(vc6, Get(p, "v60", 0.0));
                    b_terms.emplace_back(+1.0, i_c6);
                }
                nb.Kcl(b_terms);
            }
            nb.Kcl(f_terms);
        }
    }

    nb.Kcl(vh_terms);   // KCL at vh
    return caps;
}

long
AddPassiveRing (NetBuilder& nb, double r_ohm)
{
    long const i_ring = nb.Unk();
    long const r_emf = nb.Row();
    nb.Put(r_emf, i_ring, -nb.m_dt * r_ohm);
    nb.m_v0_rows.push_back(r_emf);
    nb.m_illd_cols.push_back(i_ring);
    return i_ring;
}

// ---------------------------------------------------------------------
// SwdBank
// ---------------------------------------------------------------------
SwdBank::SwdBank (std::vector<SwdReg> const& reg, double dt) : m_dt(dt)
{
    std::size_t const n = reg.size();
    m_rd0.reserve(n);
    for (auto const& d : reg) {
        m_rd0.push_back(d.rd0);
        m_ld.push_back(d.ld);
        m_tot.push_back(d.tot);
        m_tnc.push_back(d.tnc);
        m_tdc.push_back(d.tdc);
        m_trr.push_back(d.trr);
        m_crr.push_back(d.crr);
        m_sense.push_back(d.sense);
        m_tri.push_back(d.tri);
        m_rr.push_back(d.rr ? 1 : 0);
        m_t2m.push_back(d.t2m);
        m_t2e.push_back(d.t2e);
        m_t3e.push_back(d.t3e);
        m_t4v.push_back(d.t4v);
        m_r2.push_back(d.r2);
        m_r4.push_back(d.r4);
        m_is_sw.push_back(d.tnc < 0.0 ? 1 : 0);
    }
    m_stt.assign(n, -1);
    m_cnt = m_tot;
    m_rd.assign(n, BGR);
    m_tau.assign(n, 0.0);
    m_rqe.assign(n, 0.0);
    m_cjn.assign(n, 0.0);
}

bool
SwdBank::Update (double t, std::vector<double> const& x_old)
{
    // Per-device sequential translation of the numpy mask blocks of
    // update(); each guard is evaluated at the same point relative to the
    // in-place stt/cnt mutations, so cross-mask interactions (a turn_on
    // device also passing through the trailing on_fwd block) reproduce the
    // vectorized reference exactly.
    bool changed = false;
    for (std::size_t i = 0; i < m_rd0.size(); ++i) {
        double const rd_p = m_rd[i];
        double const tau_p = m_tau[i];
        double const rqe_p = m_rqe[i];
        double const cjn_p = m_cjn[i];
        if (m_is_sw[i]) {
            bool const on_abs = m_tot[i] < t;
            m_rd[i] = on_abs ? std::fabs(m_rd0[i]) : BGR;
            if (on_abs && m_stt[i] <= 0) { m_stt[i] = 1; }
        } else {
            bool const armed = (t > m_tdc[i]) && (std::fabs(m_rd0[i]) > 0.0);
            if (!armed) { m_rd[i] = BGR; }
            double const ieff = x_old[m_sense[i]] * Sign(m_rd0[i]);

            bool const off_fwd = armed && m_stt[i] <= 0 && ieff > 0.0;
            if (off_fwd && m_stt[i] < 0) { m_stt[i] = 0; }
            if (off_fwd) { m_cnt[i] -= m_dt; }
            bool const turn_on = off_fwd && m_cnt[i] <= 0.0;
            if (turn_on) {
                m_stt[i] = 1;
                m_rd[i] = std::fabs(m_rd0[i]);
                m_cnt[i] = m_tnc[i] + FRR * m_trr[i];
                m_tau[i] = m_trr[i];
            }
            bool const stay_off = off_fwd && !turn_on;
            if (stay_off) {
                m_rd[i] = BGR;
                m_tau[i] = 0.0;
            }
            bool const off_rev = armed && m_stt[i] <= 0 && ieff <= 0.0;
            if (off_rev) {
                m_cnt[i] = (m_cnt[i] < 0.0) ? m_tot[i] : std::fabs(m_tot[i]);
                m_rd[i] = BGR;
                m_tau[i] = 0.0;
            }
            if (!armed || stay_off || off_rev) {   // off_any
                m_rqe[i] = 0.0;
                m_cjn[i] = 0.0;
            }

            bool const on_rev = armed && m_stt[i] > 0 && ieff < 0.0;
            if (on_rev && m_stt[i] == 1 && m_trr[i] > 0.0) { m_stt[i] = 2; }
            if (on_rev) { m_cnt[i] -= m_dt; }
            bool const turn_off = on_rev && m_cnt[i] <= 0.0;
            if (turn_off) {
                m_stt[i] = 0;
                m_rd[i] = BGR;
                m_rqe[i] = 0.0;
                m_cjn[i] = 0.0;
                m_cnt[i] = std::fabs(m_tot[i]);
                m_tau[i] = 0.0;
            }
            bool const keep = on_rev && !turn_off;
            if (keep) {
                m_rd[i] = std::fabs(m_rd0[i]);
                if (m_stt[i] == 2) {                // in_rr
                    m_rqe[i] = BGR;
                    m_cjn[i] = m_crr[i];
                } else {
                    m_rqe[i] = 0.0;
                    m_cjn[i] = 0.0;
                }
                m_tau[i] = m_trr[i];
            }

            bool const on_fwd = armed && m_stt[i] > 0 && ieff >= 0.0;
            if (on_fwd) {
                if (m_stt[i] == 2) { m_stt[i] = 1; }
                m_cnt[i] = m_tnc[i] + FRR * m_trr[i];
                m_rd[i] = std::fabs(m_rd0[i]);
                m_rqe[i] = 0.0;
                m_cjn[i] = 0.0;
                m_tau[i] = m_trr[i];
            }
        }
        changed = changed || m_rd[i] != rd_p || m_tau[i] != tau_p
            || m_rqe[i] != rqe_p || m_cjn[i] != cjn_p;
    }
    return changed;
}

void
SwdBank::Refresh (std::vector<double>& vals, std::vector<double>& rhs_coef)
    const
{
    for (std::size_t i = 0; i < m_rd0.size(); ++i) {
        vals[m_tri[i]] = m_dt * m_rd[i] + m_ld[i];
        if (m_rr[i]) {
            vals[m_t2m[i]] = m_tnc[i] * m_tau[i]
                + m_dt * (m_tnc[i] + m_tau[i]);
            vals[m_t2e[i]] = -m_dt * m_tau[i];
            vals[m_t3e[i]] = m_dt * m_rqe[i];
            vals[m_t4v[i]] = -m_cjn[i];
            rhs_coef[m_r2[i]] = m_tnc[i] * m_tau[i];
            rhs_coef[m_r4[i]] = -m_cjn[i];
        }
    }
}

} // namespace c3
