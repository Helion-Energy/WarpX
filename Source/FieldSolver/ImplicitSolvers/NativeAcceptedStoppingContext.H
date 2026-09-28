/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_ACCEPTED_STOPPING_CONTEXT_H_
#define WARPX_NATIVE_ACCEPTED_STOPPING_CONTEXT_H_

#include "Particles/Gather/FieldGather.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <cmath>
#include <cstdint>

namespace warpx::thermal {

struct AcceptedStoppingOptions {
    amrex::Real number_density_floor = 0.; // physical m^-3, raw eligibility
    amrex::Real reference_number_density = 1.; // native inertia n0_ref, m^-3
    amrex::Real model_electron_mass = PhysConst::m_e;
    amrex::IntVect ghosts{3};
    std::array<FieldBoundaryType, AMREX_SPACEDIM> field_lo{}, field_hi{};
    std::array<ParticleBoundaryType, AMREX_SPACEDIM> particle_lo{}, particle_hi{};
    int levels = 1;
    int azimuthal_modes = 1;
    int shape_order = 3;
    bool embedded_boundary = false;
    bool moving_window = false;
    bool galilean = false;
    bool filter = false;
    bool current_centering = false;
    bool single_precision_communications = false;
    bool galerkin_interpolation = false;
};

/** Named accepted-endpoint binding, never an interval Jp-Ji reconstruction.
 * Prepare verifies pointers against the const native registry. The producer
 * still owns the physical epoch/value provenance of those named fields.
 */
struct AcceptedStoppingBinding {
    ablastr::fields::ConstVectorField electron_current{};
    amrex::MultiFab const* raw_charge = nullptr;
    amrex::MultiFab const* pedestal = nullptr;
    int charge_component = 0;
    amrex::Real endpoint_time = 0.;
    std::uint64_t endpoint_epoch = 0;
};

struct StoppingGatherResult {
    amrex::GpuArray<amrex::ParticleReal, 3> cartesian{};
    amrex::GpuArray<amrex::ParticleReal, 3> collision_frame{};
    bool valid = false;
};

/** Device-copyable view of the ACTUAL native stopping gather. The B slots are
 * read-only and discarded exactly as in HybridElectronStopping. Caller keeps
 * the supplied B fields and this context alive and immutable during gathers.
 */
struct AcceptedStoppingGather {
    amrex::GpuArray<amrex::Array4<amrex::Real const>, 3> velocity{}, magnetic{};
    amrex::GpuArray<amrex::IndexType, 3> velocity_type{}, magnetic_type{};
    amrex::XDim3 inverse{}, origin{};
    amrex::Dim3 lower{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> tile_lo{}, tile_hi{};
    bool ready = false;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    StoppingGatherResult operator() (amrex::ParticleReal xp, amrex::ParticleReal yp,
                                     amrex::ParticleReal zp) const
    {
        StoppingGatherResult result;
        if (!ready || !std::isfinite(xp) || !std::isfinite(yp) || !std::isfinite(zp)) {
            return result;
        }
#if defined(WARPX_DIM_RZ)
        auto const radius = std::sqrt(xp*xp + yp*yp);
        if (radius < tile_lo[0] || radius > tile_hi[0] ||
            zp < tile_lo[1] || zp > tile_hi[1]) { return result; }
#elif defined(WARPX_DIM_3D)
        if (xp < tile_lo[0] || xp > tile_hi[0] || yp < tile_lo[1] ||
            yp > tile_hi[1] || zp < tile_lo[2] || zp > tile_hi[2]) { return result; }
#else
        return result;
#endif
        amrex::ParticleReal vx=0., vy=0., vz=0., bx=0., by=0., bz=0.;
        doGatherShapeN(xp, yp, zp, vx, vy, vz, bx, by, bz,
            velocity[0], velocity[1], velocity[2], magnetic[0], magnetic[1], magnetic[2],
            velocity_type[0], velocity_type[1], velocity_type[2],
            magnetic_type[0], magnetic_type[1], magnetic_type[2],
            inverse, origin, lower, 1, 3, false);
        auto const speed2 = vx*vx + vy*vy + vz*vz;
        if (!std::isfinite(speed2) || speed2 >= PhysConst::c*PhysConst::c) { return result; }
        result.cartesian = {vx, vy, vz};
#if defined(WARPX_DIM_RZ)
        auto const co = radius > 0. ? xp/radius : 1.;
        auto const si = radius > 0. ? yp/radius : 0.;
        result.collision_frame = {vx*co + vy*si, -vx*si + vy*co, vz};
#else
        result.collision_frame = result.cartesian;
#endif
        result.valid = true;
        return result;
    }
};

/** Pure accepted-state prerequisite only. No stopping rate, momentum kick,
 * heat transfer, event publication or event capability is changed here.
 * Only private scratch is written. Failed preparation clears readiness and
 * private output. The accepted registry, particles and RNG are never mutated.
 */
class NativeAcceptedStoppingContext {
public:
    static constexpr char const* AcceptedCurrentName = "diagnostic_Je_endpoint_fp";
    NativeAcceptedStoppingContext (amrex::Geometry const&, amrex::BoxArray const& cells,
        amrex::DistributionMapping const&, AcceptedStoppingOptions const&);

    static AcceptedStoppingBinding Bind (ablastr::fields::MultiFabRegister const&,
        bool pedestal_active, amrex::Real endpoint_time, std::uint64_t endpoint_epoch);
    bool Prepare (ablastr::fields::MultiFabRegister const&, AcceptedStoppingBinding const&);
    void Invalidate () { Reject("invalidated"); }
    bool Valid () const { return m_valid; }
    char const* Failure () const { return m_failure; }
    std::uint64_t Generation () const { return m_generation; }
    amrex::Geometry const& Geometry () const { return m_geometry; }
    amrex::BoxArray const& Cells () const { return m_cells; }
    amrex::DistributionMapping const& Distribution () const { return m_distribution; }
    AcceptedStoppingOptions const& Options () const { return m_options; }
    ablastr::fields::ConstVectorField Velocity () const;
    amrex::MultiFab const& Kappa () const { return m_kappa; }
    AcceptedStoppingBinding const& Binding () const { return m_binding; }
    AcceptedStoppingGather GatherView (amrex::MFIter const&,
        ablastr::fields::ConstVectorField const& magnetic) const;

private:
    bool Supported () const;
    bool Reject (char const*);
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells, m_nodes;
    amrex::DistributionMapping m_distribution;
    AcceptedStoppingOptions m_options;
    amrex::MultiFab m_kappa;
    std::array<amrex::MultiFab, 3> m_velocity;
    AcceptedStoppingBinding m_binding;
    std::uint64_t m_generation = 0;
    bool m_valid = false;
    char const* m_failure = "unprepared";
};
} // namespace warpx::thermal
#endif
