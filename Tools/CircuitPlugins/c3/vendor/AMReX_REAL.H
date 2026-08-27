/* VENDORED SHIM -- stands in for AMReX's Src/Base/AMReX_REAL.H.
 *
 * The vendored ExternalCircuit.H includes <AMReX_REAL.H> for the
 * amrex::Real typedef only (the ABI is otherwise AMReX-free). The real
 * header depends on the generated AMReX_Config.H, so a self-contained
 * plugin cannot use it; this shim provides the one typedef the contract
 * needs, fixed to double.
 *
 * PRECISION CONTRACT: the plugin is double-precision throughout and the
 * ABI version stamp does NOT encode the Real width. Loading this plugin
 * into a WarpX built with -DWarpX_PRECISION=SINGLE (amrex::Real = float)
 * is undefined behavior; the same-toolchain/same-precision discipline of
 * the circuit-coupling lane is a build-time responsibility.
 */
#ifndef AMREX_REAL_H_
#define AMREX_REAL_H_

using amrex_real = double;

namespace amrex {
    using Real = amrex_real;
}

#endif // AMREX_REAL_H_
