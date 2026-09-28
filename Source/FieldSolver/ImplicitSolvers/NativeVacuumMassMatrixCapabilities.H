/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_VACUUM_MASS_MATRIX_CAPABILITIES_H_
#define WARPX_NATIVE_VACUUM_MASS_MATRIX_CAPABILITIES_H_
#include <cstdint>
class WarpX;
class ThetaImplicitHybrid;
// Immutable simulation-lifetime selection. Only collective Theta startup may
// bind Unbound to Off/On; replacing a solver cannot change that selection.
enum class NativePrivatePMCMassMatrixSelection : unsigned char { Unbound, Off, On };
NativePrivatePMCMassMatrixSelection NativePrivatePMCMassMatrixSelectionOf(WarpX const&) noexcept;
std::uint64_t NativePrivatePMCMassMatrixOwnerGeneration(WarpX const&) noexcept;
// Pure local metadata query. A caller must globally AND the result before a
// private action. It never allocates field/device storage, copies fields or performs a collective.
ThetaImplicitHybrid const* NativePrivatePMCMassMatrixOwnerLocal(WarpX&);
// Qualified source-free RZ vacuum response. This does not replace the native
// tensor's particle/shape/geometry checks or admit circuit/source feedback.
bool NativeVacuumMassMatrixSupported(WarpX&);
// Private PMC current+density qualification, owned by the actual initialized
// Theta solver. Uniform validated Off returns locally; otherwise this performs
// one metadata AND. It never accepts a raw flag
// alone and never compares process-local owner addresses between ranks. This
// is not a cached particle/linearization lease or a source/circuit capability.
ThetaImplicitHybrid const* NativePrivatePMCMassMatrixOwner(WarpX&);
#endif
