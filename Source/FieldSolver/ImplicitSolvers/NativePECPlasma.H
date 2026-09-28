/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_PEC_PLASMA_H_
#define WARPX_NATIVE_PEC_PLASMA_H_
#include <ablastr/fields/MultiFabRegister.H>
#include <AMReX_MultiFab.H>
#include <optional>
class WarpX;
namespace warpx::darwin {
class NativeCoilField;
inline constexpr char PECWallCurrentName[]="native_PEC_wall_endpoint_fp";
inline constexpr char PECWallStageName[]="native_PEC_wall_stage_fp";
inline constexpr char PECPlasmaRateName[]="native_PEC_plasma_rate_fp";
inline constexpr char PECPressureHallName[]="native_PEC_pressure_hall_fp";
bool NativePECPlasmaEnabled();
void ValidateNativePECPlasma(WarpX&);
// Private native Yee scratch only: homogeneous electric/axis/periodic images.
void CompleteNativeElectricGatherImages(WarpX&,ablastr::fields::VectorField const&);
// First driven qualification scope: actual normal B on every wall footprint is zero.
bool NativeDrivenPECNormalFieldSupported(WarpX&);
void CombineNativePECTangential(WarpX&,ablastr::fields::VectorField const& out,
    ablastr::fields::VectorField const& first,amrex::Real first_scale,
    ablastr::fields::VectorField const& second,amrex::Real second_scale);
void AllocateNativePECPlasma(WarpX&);
// Rebuild C-Dbar from this trial. Only the native-defined grow1 band is read.
void RebuildNativePECStageCurrent(WarpX&,amrex::Real dt,
    NativeCoilField* coil=nullptr,std::optional<amrex::Real> stage_time=std::nullopt);
// Eliminate the tangential plasma constitutive row, update stage Jp and W.
// Kappa is the SAME nodal mass coefficient built by the native inertia path.
bool PrepareNativePECStage(WarpX&,amrex::Real dt,amrex::MultiFab const& kappa,
    ablastr::fields::VectorField const& old_ion,
    ablastr::fields::VectorField const& endpoint_ion,
    ablastr::fields::VectorField const& stage_plasma);
void CompleteNativePECStageRate(WarpX&,ablastr::fields::VectorField const&);
void RotateNativePECWall(WarpX&);
// Exact support primitive. No normal component or unconstrained row changes.
void SetNativePECTangential(WarpX&,ablastr::fields::VectorField const& destination,
    ablastr::fields::VectorField const* source=nullptr,amrex::Real scale=0.);
void ValidateNativePECWallState(WarpX&);
// After an explicitly owned fixed-density event commits independent Je/D/A/B/U:
// publish its wall reaction and refresh only the Je mirrors. Caller retains
// full field/particle/RNG rollback until endpoint recovery succeeds. This is
// not an Evolve event dispatch or permission to open generic event guards.
void PublishNativePECEventWall(WarpX&,ablastr::fields::ConstVectorField const&);
}
#endif
