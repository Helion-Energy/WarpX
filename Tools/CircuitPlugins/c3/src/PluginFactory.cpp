/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "C3ExternalCircuit.H"

#include "ExternalCircuit.H"

/* The two symbols the WarpX plugin loader resolves after dlopen
 * (RTLD_LOCAL): the ABI stamp and the factory. Kept in their own
 * translation unit so the core library carries no exported C symbols. */
extern "C"
{

int
warpx_external_circuit_abi_version ()
{
    return WARPX_EXTERNAL_CIRCUIT_ABI_VERSION;
}

ExternalCircuit*
warpx_create_external_circuit ()
{
    return new c3::C3ExternalCircuit();
}

} // extern "C"
