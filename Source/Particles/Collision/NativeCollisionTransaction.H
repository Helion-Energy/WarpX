/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_COLLISION_TRANSACTION_H_
#define WARPX_NATIVE_COLLISION_TRANSACTION_H_

#include <cstdint>
#include <memory>

class WarpX;

namespace warpx::particles {

/** Snapshot for the complete native DT collision handler, including its RZ
 * frame rotations. This does not enable collision/event support in an implicit
 * solver. RZ/3D, double particle precision, and host or CUDA/HIP RNG are required.
 * Fixed geometry, species and attribute schema are required. Stopping,
 * Coulomb, QED, callbacks, boundary tallies and arbitrary field/source mutation
 * are outside this contract. Calls are collective, quiescent, and outside OMP
 * regions. No full-particle host round trip is used.
 *
 * Restore consumes the snapshot and swaps complete particle maps. Every raw
 * particle view becomes invalid. The caller supplies a mandatory invalidation
 * nonthrowing hook for its borrowed views/caches, invoked once after the all-rank
 * preflight. The hook must only invalidate private caches; it must not mutate
 * particles, fields, RNG, IDs, configuration, or perform collectives.
 * Native implicit MM caches are invalidated and the old fixed-topology particle
 * rollback is discarded by Restore. It must never be restored after this swap.
 */
class NativeCollisionTransaction
{
public:
    enum class Status {
        Success = 0,
        AlreadyActive,
        Inactive,
        UnsupportedScope,
        ConfigurationChanged,
        MissingInvalidationHook
    };
    struct Memory {
        std::uint64_t particles = 0;
        std::uint64_t tiles = 0;
        std::uint64_t particle_bytes = 0;
        std::uint64_t particle_capacity_bytes = 0;
        std::uint64_t host_rng_bytes = 0;
        std::uint64_t device_rng_bytes = 0;
        std::uint64_t cost_bytes = 0;
    };

    struct InvalidationHook {
        void (*invalidate)(void*) noexcept = nullptr;
        void* context = nullptr;
    };

    NativeCollisionTransaction ();
    ~NativeCollisionTransaction ();
    NativeCollisionTransaction (NativeCollisionTransaction const&) = delete;
    NativeCollisionTransaction& operator= (NativeCollisionTransaction const&) = delete;

    [[nodiscard]] Status Capture (WarpX& simulation);
    [[nodiscard]] Status CanRestore (WarpX& simulation) const;
    [[nodiscard]] Status Restore (WarpX& simulation,
                                  InvalidationHook invalidate_borrowed_views);
    /** Discard only; does not publish state or validate post-event caches. */
    void Commit ();
    [[nodiscard]] bool Active () const;
    /** Per-rank counts; excludes C++ map nodes and allocator bookkeeping. */
    [[nodiscard]] Memory LocalMemory () const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace warpx::particles
#endif
