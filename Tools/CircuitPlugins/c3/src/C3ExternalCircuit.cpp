/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "C3ExternalCircuit.H"

#include "YamlLite.H"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace c3 {

namespace {

std::string
Hex (double v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%a", v);
    return buf;
}

double
UnHex (std::string const& s)
{
    return std::strtod(s.c_str(), nullptr);
}

void
WriteDoubles (std::ofstream& out, std::string const& key,
              std::vector<double> const& v)
{
    out << key;
    for (double const x : v) {
        out << ' ' << Hex(x);
    }
    out << '\n';
}

std::vector<double>
ReadDoubles (std::ifstream& in, std::string const& key, std::size_t n)
{
    std::string k;
    in >> k;
    if (k != key) {
        throw std::runtime_error("c3 checkpoint: expected '" + key
                                 + "', got '" + k + "'");
    }
    std::vector<double> v(n);
    std::string tok;
    for (auto& x : v) {
        if (!(in >> tok)) {
            throw std::runtime_error("c3 checkpoint: truncated '" + key + "'");
        }
        x = UnHex(tok);
    }
    return v;
}

} // namespace

void
C3ExternalCircuit::Log (std::string const& msg) const
{
    if (m_cfg.verbose) {
        std::fprintf(stderr, "[c3circuit] %s\n", msg.c_str());
        std::fflush(stderr);
    }
}

void
C3ExternalCircuit::Define (std::vector<std::string> const& coil_names,
                           std::vector<amrex::Real> const& i_ref,
                           std::string const& config)
{
    m_cfg = ParseConfig(config);
    m_n_port = static_cast<long>(coil_names.size());
    m_i_ref.assign(i_ref.begin(), i_ref.end());
    if (static_cast<long>(m_i_ref.size()) != m_n_port) {
        throw std::runtime_error("c3 Define: coil_names/i_ref size mismatch");
    }
    YamlNode const doc = ParseYamlFile(m_cfg.yaml_path);
    MatrixBlocks const blocks = ReadMatrixBlocks(m_cfg.matrix_path);
    std::vector<RingSpec> rings;
    if (!m_cfg.rings_path.empty()) {
        rings = ReadRingSpecs(m_cfg.rings_path);
    }
    Logger const log = [this] (std::string const& msg) { Log(msg); };
    m_net = std::make_unique<C3BankNetwork>(doc, coil_names, blocks, rings,
                                            m_cfg, log);
    if (m_cfg.preroll) {
        // the machine's own bias soak: the deck's cards arm their bias
        // switches from glob t0, so the plugin arrives at WarpX t = 0
        // (machine -shift) already rolled
        long const n_pre = m_net->PreRoll();
        std::ostringstream os;
        os << "pre-rolled " << n_pre << " steps to machine "
           << -m_cfg.shift * 1e6 << " us (sim 0), "
           << m_net->Live().NFac() << " factorizations";
        Log(os.str());
    } else {
        Log("pre-roll skipped (restart: ReadCheckpoint supersedes it)");
    }
    m_entry = m_net->Live().TakeSnapshot();
}

void
C3ExternalCircuit::BeginStep (amrex::Real t0, amrex::Real dt)
{
    // machine-0 lock-ring stage swap: between steps only, at the first
    // step boundary t0 > 0 whose machine time is within dt/2 of 0 (the
    // boundary nearest machine 0; the deck's max(1, round(shift/dt)))
    if (m_net->LockSwapPending() && t0 > 0.0
        && t0 - m_cfg.shift >= -0.5 * dt) {
        m_net->ArmLocks(t0 - m_cfg.shift);
    }
    m_entry = m_net->Live().TakeSnapshot();
}

void
C3ExternalCircuit::AdvanceInterval (amrex::Real t0, amrex::Real t1,
                                    std::vector<amrex::Real> const& eps,
                                    bool accept,
                                    std::vector<amrex::Real>& scales)
{
    (void) t0;   // the stepper's whole-step lattice carries the remainder
    if (static_cast<long>(eps.size()) != m_n_port) {
        throw std::runtime_error("c3 AdvanceInterval: eps size mismatch");
    }
    // restore-and-re-advance: every evaluation starts from the entry state
    m_net->Live().Restore(m_entry);
    for (long j = 0; j < m_n_port; ++j) {
        m_net->SetPortEmf(j, eps[j]);   // port j == east index j
    }
    m_net->Advance(t1 - m_cfg.shift);
    scales.resize(m_n_port);
    for (long j = 0; j < m_n_port; ++j) {
        scales[j] = m_net->ChannelCurrent(j) / m_i_ref[j];
    }
    if (accept) {
        m_entry = m_net->Live().TakeSnapshot();   // the latch
    }
}

void
C3ExternalCircuit::FinishStep ()
{
    m_n_steps += 1;
}

void
C3ExternalCircuit::WriteCheckpoint (std::string const& dir) const
{
    std::string const path = dir + "/c3_external_circuit.dat";
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("c3 checkpoint: cannot write '" + path + "'");
    }
    out << "c3-external-circuit-checkpoint v1\n";
    out << "abi " << WARPX_EXTERNAL_CIRCUIT_ABI_VERSION << '\n';
    out << "lock_phase " << m_net->LockPhase() << '\n';
    out << "t0 " << Hex(m_net->Live().T0()) << '\n';
    out << "m " << m_entry.m << '\n';
    out << "n_x " << m_entry.x.size() << '\n';
    WriteDoubles(out, "x", m_entry.x);
    out << "n_swd " << m_entry.cnt.size() << '\n';
    out << "stt";
    for (auto const s : m_entry.stt) {
        out << ' ' << static_cast<int>(s);
    }
    out << '\n';
    WriteDoubles(out, "cnt", m_entry.cnt);
    WriteDoubles(out, "rd", m_entry.rd);
    WriteDoubles(out, "tau", m_entry.tau);
    WriteDoubles(out, "rqe", m_entry.rqe);
    WriteDoubles(out, "cjn", m_entry.cjn);
    Log("checkpoint written to " + path + " (lock_phase "
        + std::to_string(m_net->LockPhase()) + ", m "
        + std::to_string(m_entry.m) + ")");
}

void
C3ExternalCircuit::ReadCheckpoint (std::string const& dir)
{
    std::string const path = dir + "/c3_external_circuit.dat";
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("c3 checkpoint: cannot open '" + path + "'");
    }
    std::string tag, version, key;
    in >> tag >> version;
    if (tag != "c3-external-circuit-checkpoint" || version != "v1") {
        throw std::runtime_error("c3 checkpoint: bad header in '" + path + "'");
    }
    int abi = 0;
    in >> key >> abi;
    if (key != "abi" || abi != WARPX_EXTERNAL_CIRCUIT_ABI_VERSION) {
        throw std::runtime_error("c3 checkpoint: ABI stamp mismatch");
    }
    int lock_phase = 0;
    in >> key >> lock_phase;
    if (key != "lock_phase") {
        throw std::runtime_error("c3 checkpoint: expected lock_phase");
    }
    std::string t0_hex;
    in >> key >> t0_hex;
    if (key != "t0") {
        throw std::runtime_error("c3 checkpoint: expected t0");
    }
    C3Stepper::Snapshot snap;
    in >> key >> snap.m;
    if (key != "m") {
        throw std::runtime_error("c3 checkpoint: expected m");
    }
    std::size_t n_x = 0, n_swd = 0;
    in >> key >> n_x;
    if (key != "n_x") {
        throw std::runtime_error("c3 checkpoint: expected n_x");
    }
    snap.x = ReadDoubles(in, "x", n_x);
    in >> key >> n_swd;
    if (key != "n_swd") {
        throw std::runtime_error("c3 checkpoint: expected n_swd");
    }
    in >> key;
    if (key != "stt") {
        throw std::runtime_error("c3 checkpoint: expected stt");
    }
    snap.stt.resize(n_swd);
    for (auto& s : snap.stt) {
        int v = 0;
        in >> v;
        s = static_cast<std::int8_t>(v);
    }
    snap.cnt = ReadDoubles(in, "cnt", n_swd);
    snap.rd = ReadDoubles(in, "rd", n_swd);
    snap.tau = ReadDoubles(in, "tau", n_swd);
    snap.rqe = ReadDoubles(in, "rqe", n_swd);
    snap.cjn = ReadDoubles(in, "cjn", n_swd);

    m_net->SelectPhase(lock_phase, UnHex(t0_hex));
    if (snap.x.size() != static_cast<std::size_t>(m_net->Live().NUnk())) {
        throw std::runtime_error(
            "c3 checkpoint: state size does not match the "
            + std::string(lock_phase == 1 ? "post" : "pre")
            + "-swap stepper (wrong rings config?)");
    }
    m_net->Live().Restore(snap);
    m_entry = m_net->Live().TakeSnapshot();
    Log("checkpoint restored from " + path + " (lock_phase "
        + std::to_string(lock_phase) + ", m " + std::to_string(snap.m) + ")");
}

} // namespace c3
