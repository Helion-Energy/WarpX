/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_SOURCE_CHECK_H_
#define WARPX_HYBRID_SOURCE_CHECK_H_

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuMemory.H>
#include <memory>

/** Release-safe source validation. AMReX device assertions may be disabled by
 * NDEBUG even for ALWAYS_ASSERT. Invalid lanes return without changing state;
 * a device error flag is copied back and triggers a host abort before the next
 * source phase. No allocation is made when the corresponding guard is off.
 */
class HybridSourceCheck
{
public:
    enum Code { OhmicState=1, OhmicSpecies, OhmicCurrent, OhmicSource,
                ThermalSpecies, ThermalMoments, ThermalRate,
                StoppingThermodynamics, StoppingRate, StoppingParticle, StoppingResult };

    explicit HybridSourceCheck (bool enabled)
    {
        if (enabled) m_error=std::make_unique<amrex::Gpu::DeviceScalar<int>>(0);
    }
    int* data () const { return m_error ? m_error->dataPtr() : nullptr; }
    void Finish () const { if (m_error) Abort(m_error->dataValue()); }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static bool
    Require (bool valid, Code code, int* error)
    {
        if (!valid) {
            AMREX_IF_ON_DEVICE((amrex::Gpu::Atomic::Max(error,static_cast<int>(code));))
            AMREX_IF_ON_HOST((amrex::ignore_unused(error); Abort(code);))
        }
        return valid;
    }

private:
    std::unique_ptr<amrex::Gpu::DeviceScalar<int>> m_error;
    static void Abort (int code)
    {
        switch (code) {
        case 0: return;
        case OhmicState: amrex::Abort("source_guard: invalid eligible Ohmic state"); break;
        case OhmicSpecies: amrex::Abort("source_guard: invalid eligible species density"); break;
        case OhmicCurrent: amrex::Abort("source_guard: invalid eligible Ohmic current"); break;
        case OhmicSource: amrex::Abort("source_guard: invalid eligible Ohmic source"); break;
        case ThermalSpecies: amrex::Abort("source_guard: invalid eligible species normalization"); break;
        case ThermalMoments: amrex::Abort("source_guard: invalid eligible relaxation moments"); break;
        case ThermalRate: amrex::Abort("source_guard: invalid eligible relaxation rate"); break;
        case StoppingThermodynamics:
            amrex::Abort("source_guard: invalid eligible stopping thermodynamics"); break;
        case StoppingRate: amrex::Abort("source_guard: invalid eligible stopping rate"); break;
        case StoppingParticle:
            amrex::Abort("source_guard: invalid eligible stopping velocity or particle"); break;
        case StoppingResult: amrex::Abort("source_guard: invalid eligible stopping result"); break;
        default: amrex::Abort("source_guard: unknown invalid eligible source state");
        }
    }
};
#endif
