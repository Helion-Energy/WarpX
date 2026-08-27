/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef C3_YAML_LITE_H_
#define C3_YAML_LITE_H_

#include <string>
#include <utility>
#include <vector>

namespace c3 {

/** Minimal reader for the machine-generated c3 circuit-yaml schema
 * (make_circuit_yaml output: ref_shot_circuit.yaml, alt_case_circuit.yaml).
 *
 * Supported subset: block mappings and block sequences with two-space
 * indentation, "- key: value" inline-first sequence items, plain / quoted
 * scalars, and plain-scalar continuation lines. This is NOT a general YAML
 * parser -- the validation harness cross-checks it end to end against
 * PyYAML through the trajectory comparison.
 */
class YamlNode
{
public:
    enum class Kind { Scalar, Map, Seq };

    Kind m_kind = Kind::Scalar;
    std::string m_scalar;                                  //!< Kind::Scalar
    std::vector<std::pair<std::string, YamlNode>> m_map;   //!< Kind::Map (ordered)
    std::vector<YamlNode> m_seq;                           //!< Kind::Seq

    bool Has (std::string const& key) const;
    /** Map lookup; throws std::runtime_error if absent. */
    YamlNode const& At (std::string const& key) const;
    /** Scalar as double (strtod, full-string); throws on non-numeric. */
    double AsDouble () const;
    std::string const& AsString () const { return m_scalar; }
    bool IsNull () const;
};

/** Parse a whole file; throws std::runtime_error on I/O or shape errors. */
YamlNode ParseYamlFile (std::string const& path);

} // namespace c3

#endif // C3_YAML_LITE_H_
