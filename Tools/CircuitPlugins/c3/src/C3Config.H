/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef C3_C3_CONFIG_H_
#define C3_C3_CONFIG_H_

#include <string>
#include <vector>

namespace c3 {

//! one passive-ring spec (rings file line: name R_ohm lock)
struct RingSpec
{
    std::string name;
    double r_ohm = 0.0;
    bool lock = false;
};

/** Amp-turn mutual-inductance blocks of the east half (matrix file): the
 * exporter's post-symmetrization, post-ring-override M_EE and M_EW, in
 * hexfloat so both the plugin and the Python reference consume
 * bit-identical inputs. Row/column order = east order (Define coils,
 * extra circuits, rings). */
struct MatrixBlocks
{
    long n = 0;
    std::vector<std::string> names;
    std::vector<double> ee;   //!< (n, n) row-major [H]
    std::vector<double> ew;   //!< (n, n) row-major [H] (all-zero if no mirror)
};

/** Parsed circuit.plugin_config. The free-form string is either
 * "key=value,key=value,..." inline pairs or a path to an INI-style file of
 * "key = value" lines (# comments); see the README for the schema. */
struct C3Config
{
    std::string yaml_path;                       //!< yaml = (required)
    std::string matrix_path;                     //!< matrix = (required)
    std::string rings_path;                      //!< rings = (optional)
    double dt = 5.0e-8;                          //!< dt = circuit BE step [s]
    double t0_machine = -200.0e-6;               //!< t0 = machine glob t0 [s]
    double shift = 2.5e-6;                       //!< shift = sim - machine [s]
    std::vector<std::string> extra = {"D03", "D04"};   //!< extra = A+B+...
    double eps_sign = 1.0;                       //!< eps_sign =
    bool mirror = true;                          //!< mirror = 0|1
    bool preroll = true;                         //!< preroll = 0|1
    bool verbose = true;                         //!< verbose = 0|1
};

C3Config ParseConfig (std::string const& config_string);

std::vector<RingSpec> ReadRingSpecs (std::string const& path);

MatrixBlocks ReadMatrixBlocks (std::string const& path);

} // namespace c3

#endif // C3_C3_CONFIG_H_
