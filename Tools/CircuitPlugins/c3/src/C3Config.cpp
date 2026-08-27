/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "C3Config.H"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace c3 {

namespace {

std::string
Trim (std::string const& s)
{
    std::size_t const b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) { return {}; }
    std::size_t const e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

double
ToDouble (std::string const& key, std::string const& v)
{
    char* end = nullptr;
    double const x = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0') {
        throw std::runtime_error("c3 config: non-numeric value for '" + key
                                 + "': '" + v + "'");
    }
    return x;
}

bool
ToBool (std::string const& key, std::string const& v)
{
    if (v == "0" || v == "false" || v == "off") { return false; }
    if (v == "1" || v == "true" || v == "on") { return true; }
    throw std::runtime_error("c3 config: non-boolean value for '" + key
                             + "': '" + v + "'");
}

//! '+'-separated list (inline mode reserves ',' for pair separation)
std::vector<std::string>
ToList (std::string const& v)
{
    std::vector<std::string> out;
    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, '+')) {
        tok = Trim(tok);
        if (!tok.empty()) { out.push_back(tok); }
    }
    return out;
}

void
Apply (C3Config& cfg, std::string const& key, std::string const& value)
{
    if (key == "yaml") { cfg.yaml_path = value; }
    else if (key == "matrix") { cfg.matrix_path = value; }
    else if (key == "rings") { cfg.rings_path = value; }
    else if (key == "dt") { cfg.dt = ToDouble(key, value); }
    else if (key == "t0") { cfg.t0_machine = ToDouble(key, value); }
    else if (key == "shift") { cfg.shift = ToDouble(key, value); }
    else if (key == "extra") { cfg.extra = ToList(value); }
    else if (key == "eps_sign") { cfg.eps_sign = ToDouble(key, value); }
    else if (key == "mirror") { cfg.mirror = ToBool(key, value); }
    else if (key == "preroll") { cfg.preroll = ToBool(key, value); }
    else if (key == "verbose") { cfg.verbose = ToBool(key, value); }
    else {
        throw std::runtime_error("c3 config: unknown key '" + key + "'");
    }
}

void
ApplyPair (C3Config& cfg, std::string const& pair)
{
    std::size_t const eq = pair.find('=');
    if (eq == std::string::npos) {
        throw std::runtime_error("c3 config: expected key=value, got '"
                                 + pair + "'");
    }
    Apply(cfg, Trim(pair.substr(0, eq)), Trim(pair.substr(eq + 1)));
}

} // namespace

C3Config
ParseConfig (std::string const& config_string)
{
    C3Config cfg;
    std::string const s = Trim(config_string);
    if (s.empty()) {
        throw std::runtime_error("c3 config: empty circuit.plugin_config");
    }
    if (s.find('=') == std::string::npos) {
        // a path to an INI-style file of "key = value" lines
        std::ifstream in(s);
        if (!in) {
            throw std::runtime_error("c3 config: cannot open '" + s + "'");
        }
        std::string line;
        while (std::getline(in, line)) {
            line = Trim(line);
            if (line.empty() || line[0] == '#') { continue; }
            ApplyPair(cfg, line);
        }
    } else {
        std::stringstream ss(s);
        std::string pair;
        while (std::getline(ss, pair, ',')) {
            pair = Trim(pair);
            if (!pair.empty()) { ApplyPair(cfg, pair); }
        }
    }
    if (cfg.yaml_path.empty() || cfg.matrix_path.empty()) {
        throw std::runtime_error("c3 config: 'yaml' and 'matrix' are required");
    }
    return cfg;
}

std::vector<RingSpec>
ReadRingSpecs (std::string const& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("c3 rings: cannot open '" + path + "'");
    }
    std::vector<RingSpec> out;
    std::string line;
    bool seen_lock = false;
    while (std::getline(in, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') { continue; }
        std::stringstream ss(line);
        RingSpec rg;
        int lock = 0;
        if (!(ss >> rg.name >> rg.r_ohm >> lock)) {
            throw std::runtime_error("c3 rings: bad line '" + line + "'");
        }
        rg.lock = lock != 0;
        if (seen_lock && !rg.lock) {
            // the ring_meta/mel ordering contract: resistive plate rings
            // first, R = 0 flux-lock rings last
            throw std::runtime_error(
                "c3 rings: plate rings must precede lock rings");
        }
        seen_lock = seen_lock || rg.lock;
        if (rg.lock && rg.r_ohm != 0.0) {
            throw std::runtime_error("c3 rings: lock ring '" + rg.name
                                     + "' must have R_ohm = 0");
        }
        out.push_back(rg);
    }
    return out;
}

MatrixBlocks
ReadMatrixBlocks (std::string const& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("c3 matrix: cannot open '" + path + "'");
    }
    std::string tag, version;
    in >> tag >> version;
    if (tag != "c3-mel-blocks" || version != "v1") {
        throw std::runtime_error("c3 matrix: bad header in '" + path + "'");
    }
    MatrixBlocks mb;
    std::string key;
    in >> key >> mb.n;
    if (key != "n" || mb.n <= 0) {
        throw std::runtime_error("c3 matrix: bad size line");
    }
    mb.names.resize(mb.n);
    for (long i = 0; i < mb.n; ++i) {
        in >> mb.names[i];
    }
    auto read_block = [&] (std::string const& want,
                           std::vector<double>& block) {
        in >> key;
        if (key != want) {
            throw std::runtime_error("c3 matrix: expected block '" + want
                                     + "', got '" + key + "'");
        }
        block.resize(static_cast<std::size_t>(mb.n) * mb.n);
        std::string tok;
        for (auto& v : block) {
            if (!(in >> tok)) {
                throw std::runtime_error("c3 matrix: truncated block '"
                                         + want + "'");
            }
            v = std::strtod(tok.c_str(), nullptr);   // hexfloat-exact
        }
    };
    read_block("M_EE", mb.ee);
    read_block("M_EW", mb.ew);
    return mb;
}

} // namespace c3
