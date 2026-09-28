/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_ENDPOINT_CURRENT_RESPONSE_H_
#define WARPX_NATIVE_ENDPOINT_CURRENT_RESPONSE_H_

#include <ablastr/fields/MultiFabRegister.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GpuMemory.H>
#include <array>
#include <cstdint>

class WarpX;
class ThetaImplicitHybrid;
namespace warpx::particles
{

/** Independent instantaneous endpoint and native local Boris response.
 * The legacy frozen-B/gamma algebra remains available for its explicit oracle.
 * The production correlated route selects local_boris_response: proper-momentum
 * and relativistic velocity derivatives in E and B, at held gather weights.
 * It rebuilds raw interval E bands AFTER the independent PC reduction and owns
 * separate nine-band interval-B and endpoint-B tensors. No symmetry is assumed.
 * Only Freeze allocates; Apply composes native current volume/BC/sum operations.
 * Interval current and continuity density remain owned by ImplicitSolver.
 * The represented endpoint and correlated increment have independent bases.
 * Initial native envelope: single-level RZ/order3/MC, positive fixed-charge
 * physical particles, periodic z, no source-dependent push or suborbits.
 * The separate private PMC branch additionally binds the actual Theta owner
 * and its live current/density/material capability; it is not a public scope.
 * Pure directional kernels also support 3D; this stored tensor does not yet.
 */
class NativeEndpointCurrentResponse
{
  public:
    using View = ablastr::fields::VectorField;
    void Invalidate () noexcept
    {
        m_valid = false;
        ++m_epoch;
    }
    [[nodiscard]] bool IsValid () const noexcept
    {
        return m_valid;
    }
    [[nodiscard]] std::uint64_t Epoch () const noexcept
    {
        return m_epoch;
    }
    [[nodiscard]] int Width () const noexcept
    {
        return m_width;
    }
    [[nodiscard]] std::uint64_t MatrixBytes () const noexcept
    {
        return m_matrix_bytes;
    }
    [[nodiscard]] std::uint64_t ScratchBytes () const noexcept
    {
        return m_scratch_bytes;
    }
    [[nodiscard]] amrex::MultiFab const &Matrix (int row, int col) const
    {
        return m_matrix[3 * row + col];
    }

    [[nodiscard]] bool LocalBorisResponse () const noexcept { return m_local_boris_response; }
    [[nodiscard]] amrex::MultiFab const &MagneticMatrix (int row, int col) const
    {
        return m_endpoint_magnetic[3 * row + col];
    }
    [[nodiscard]] amrex::MultiFab const &IntervalMagneticMatrix (int row, int col) const
    {
        return m_interval_magnetic[3 * row + col];
    }
    // local_boris_response requires completed independent MassMatrices_PC
    // assembly: it intentionally replaces the raw electric Jacobian bands.
    bool Freeze(WarpX &, amrex::Real dt, int interval_width, View const &gather_base,
                View const &old_current, View const &endpoint_current,
                View const &current_increment, bool local_boris_response = false);
    bool Apply (WarpX &, amrex::Real dt, View const &gather_trial, View const &old_current,
                View const &endpoint_current, View const &current_increment);
    // Add the magnetic response to the RAW interval current before its one
    // native inverse-volume scaling, guard sum and current boundary images.
    bool ApplyIntervalMagnetic(WarpX &, amrex::Real dt, View const &magnetic_trial,
                               View const &raw_interval_current);
    bool ApplyElectromagneticDirection(WarpX &, amrex::Real dt, View const &electric_direction,
                                       View const &magnetic_direction,
                                       View const &current_direction);
    // Independent directional oracle: no subtraction of a large base field.
    bool ApplyDirection (WarpX &, amrex::Real dt, View const &direction,
                         View const &current_direction);
    // Internal kernel entry is public for CUDA extended-lambda instantiation.
    bool Multiply(WarpX &, amrex::Real dt, View const &, bool subtract_base,
                  View const *magnetic_direction = nullptr);

  private:
    bool ContextCurrentLocal(WarpX&,amrex::Real dt) const;
    bool CollectiveContextRequired(WarpX const&) const noexcept;
    // Uniform dispatch is captured at Freeze and survives Invalidate. An
    // unbound legacy caller stays conservative; validated Off adds no MPI.
    bool m_private_pmc_collective_required = true;
    std::uint64_t m_private_pmc_owner_generation = 0;
    // Keep the binding on invalidation: rank-local invalidation must still
    // enter the same collective decline path as the other ranks. Only a fresh
    // collective Freeze may replace it; the response epoch still owns reuse.
    ThetaImplicitHybrid const* m_private_pmc_owner = nullptr;
    amrex::Gpu::DeviceScalar<int> m_invalid{0};
    std::array<amrex::MultiFab, 9> m_matrix, m_endpoint_magnetic, m_interval_magnetic;
    std::array<amrex::MultiFab, 3> m_magnetic_base;
    std::array<amrex::MultiFab, 3> m_gather_base, m_old, m_endpoint, m_increment, m_direction;
    bool m_valid = false, m_local_boris_response = false;
    int m_width = 0, m_interval_width = 0, m_step = -1;
    amrex::Real m_dt = 0., m_time = 0.;
    std::uint64_t m_epoch = 0, m_matrix_bytes = 0, m_scratch_bytes = 0;
};
} // namespace warpx::particles
#endif
