/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_VACUUM_PARTICLE_SUPPORT_H_
#define WARPX_NATIVE_VACUUM_PARTICLE_SUPPORT_H_
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GpuContainers.H>
#include <array>
#include <string>
#include <vector>
class WarpX;
namespace warpx::darwin {
/** Attempt-owned, cancellation-independent support certificate. No particle,
 * field, deposit or classification is changed. The caller owns the actual
 * Trial -> native finisher -> Finished transaction and its receipt. */
class NativeVacuumParticleSupport {
public:
    enum class Phase { StageTrial, FinishedEndpoint };
    struct Report {
        // checked particles, invalid operands, interval V, density Z, point V,
        // latent V, stale records, uncovered allocation, branch failure.
        std::array<amrex::Long,9> counts{};
        bool valid=false;
    };
    bool Prepare(WarpX&,amrex::Array<amrex::iMultiFab const*,3> const& vacuum,
                 amrex::MultiFab const& raw_support,std::string& error,
                 bool allow_axial_reflection=false);
    Report Check(WarpX&,Phase);
    void Invalidate() noexcept { m_ready=false; }
    bool Ready() const noexcept { return m_ready; }
    // Pure integer support operation, public for independent one-hot native
    // image controls. Input/output components must have distinct storage.
    static bool FillPullback(WarpX&,amrex::iMultiFab&);
    amrex::iMultiFab const& CurrentMask(int c) const { return m_v[c]; }
    amrex::iMultiFab const& DensityMask() const { return m_z; }
private:
    bool m_ready=false,m_axial_reflection=false;
    bool m_adjacent=false; // captured collective policy; a changed selector invalidates Check
    int m_step=0;
    amrex::Real m_time=0.;
    amrex::Array<amrex::iMultiFab,3> m_v;
    amrex::iMultiFab m_z;
    std::vector<std::string> m_species;
    std::vector<std::size_t> m_layout;
    amrex::Gpu::DeviceVector<amrex::Long> m_counts;
};
}
#endif
