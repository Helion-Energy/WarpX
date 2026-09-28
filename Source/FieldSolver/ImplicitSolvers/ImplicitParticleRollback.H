/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_PARTICLE_ROLLBACK_H_
#define WARPX_IMPLICIT_PARTICLE_ROLLBACK_H_
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Particle.H>
#include <AMReX_Reduce.H>
#include <array>
#include <cstddef>
#include <cstdint>

namespace warpx::implicit {
// Existing noncommunicating implicit-step attributes. Missing Cartesian
// components are null in reduced geometries. They remain owned by the tile.
struct ParticleStartView {
    std::array<const amrex::ParticleReal*, 3> position{};
    std::array<const amrex::ParticleReal*, 3> momentum{};
};

// Only information absent from x_n/u_n is copied: original native curvilinear
// coordinates, identity/order, and the optional solver suborbit counter.
// This is a pre-boundary/pre-redistribution rollback, not a particle
// checkpoint.
class ParticleRollbackTile {
  public:
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    static constexpr int native_coordinates = 2;
#elif defined(WARPX_DIM_RSPHERE)
    static constexpr int native_coordinates = 3;
#else
    static constexpr int native_coordinates = 0;
#endif
    template <class GetPosition>
    void
    Capture (amrex::Long count, const std::uint64_t* identities,
             GetPosition const& get_position, ParticleStartView start,
             const int* suborbits) {
        m_count = count;
        m_identity_storage = identities;
        m_start = start;
        m_has_suborbits = suborbits != nullptr;
        m_id.resize(count);
        m_native.resize(native_coordinates * count);
        m_suborbits.resize(suborbits ? count : 0);
        auto* id = m_id.data();
        auto* native = m_native.data();
        auto* nsub = m_suborbits.data();
        amrex::ParallelFor(count,
                           [identities, id, native, nsub, suborbits,
                            get_position] AMREX_GPU_DEVICE(amrex::Long i) {
                               id[i] = identities[i];
                               if constexpr (native_coordinates > 0) {
                                   amrex::ParticleReal a, b, c;
                                   get_position.AsStored(i, a, b, c);
                                   native[native_coordinates * i] = a;
                                   native[native_coordinates * i + 1] = b;
                                   if constexpr (native_coordinates == 3) {
                                       native[3 * i + 2] = c;
                                   }
                               }
                               if (suborbits) {
                                   nsub[i] = suborbits[i];
                               }
                           });
    }
    // Does not mutate particles. The caller must validate EVERY tile/rank
    // before restoring any tile, so a failed guard cannot partially roll back.
    bool
    Matches (amrex::Long count, const std::uint64_t* identities,
             ParticleStartView start, const int* suborbits) const {
        if (count != m_count || identities != m_identity_storage ||
            start.position != m_start.position ||
            start.momentum != m_start.momentum ||
            (suborbits != nullptr) != m_has_suborbits) {
            return false;
        }
        if (count == 0) {
            return true;
        }
        auto const* id = m_id.data();
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<int> data(op);
        op.eval(count, data,
                [=] AMREX_GPU_DEVICE(amrex::Long i) -> amrex::GpuTuple<int> {
                    return {id[i] != identities[i]};
                });
        return amrex::get<0>(data.value()) == 0;
    }
    template <class SetPosition>
    void
    Restore (SetPosition const& set_position,
             std::array<amrex::ParticleReal*, 3> momentum,
             int* suborbits) const {
        auto const start = m_start;
        auto const* native = m_native.data();
        auto const* nsub = m_suborbits.data();
        amrex::ParallelFor(
            m_count, [start, native, nsub, set_position, momentum,
                      suborbits] AMREX_GPU_DEVICE(amrex::Long i) {
                if constexpr (native_coordinates > 0) {
                    auto const z =
                        native_coordinates == 3
                            ? native[3 * i + 2]
                            : (start.position[2] ? start.position[2][i]
                                                 : amrex::ParticleReal(0));
                    set_position.AsStored(i, native[native_coordinates * i],
                                          native[native_coordinates * i + 1],
                                          z);
                } else {
                    set_position(i,
                                 start.position[0] ? start.position[0][i]
                                                   : amrex::ParticleReal(0),
                                 start.position[1] ? start.position[1][i]
                                                   : amrex::ParticleReal(0),
                                 start.position[2] ? start.position[2][i]
                                                   : amrex::ParticleReal(0));
                }
                for (int d = 0; d < 3; ++d) {
                    momentum[d][i] = start.momentum[d][i];
                }
                if (suborbits) {
                    suborbits[i] = nsub[i];
                }
            });
    }
    static constexpr std::size_t
    BytesPerParticle (bool has_suborbits) {
        return sizeof(std::uint64_t) +
               native_coordinates * sizeof(amrex::ParticleReal) +
               (has_suborbits ? sizeof(int) : 0);
    }

  private:
    amrex::Long m_count = 0;
    const std::uint64_t* m_identity_storage = nullptr;
    ParticleStartView m_start;
    bool m_has_suborbits = false;
    amrex::Gpu::DeviceVector<std::uint64_t> m_id;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_native;
    amrex::Gpu::DeviceVector<int> m_suborbits;
};
} // namespace warpx::implicit
#endif
