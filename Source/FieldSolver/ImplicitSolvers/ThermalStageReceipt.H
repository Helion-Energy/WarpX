/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_THERMAL_STAGE_RECEIPT_H_
#define WARPX_THERMAL_STAGE_RECEIPT_H_
#include <cstdint>
#include <limits>
#include <memory>
namespace warpx::thermal {
class EulerianThermalStage;
// Owned by DTA across Stage replacement. A reader retains the allocation, so
// destruction/reconstruction at the same Stage or DTA address cannot be ABA.
// Only local identities are compared; action validity is reduced by its owner.
class ThermalStageLifetime {
    friend class EulerianThermalStage;
    std::uint64_t m_instance=0,m_evaluation=0;
    bool m_complete=false,m_exhausted=false;
    std::uint64_t NewInstance() noexcept {
        m_complete=false;
        if(m_exhausted || m_instance==std::numeric_limits<std::uint64_t>::max()) {
            m_exhausted=true;return 0;
        }
        return ++m_instance;
    }
    bool Invalidate(std::uint64_t instance) noexcept {
        if(instance!=m_instance || !instance || m_exhausted)return false;
        m_complete=false;
        if(m_evaluation==std::numeric_limits<std::uint64_t>::max()) {
            m_exhausted=true;return false;
        }
        ++m_evaluation;return true;
    }
public:
    bool Current(std::uint64_t instance,std::uint64_t evaluation) const noexcept {
        return !m_exhausted && m_complete && instance && evaluation &&
            instance==m_instance && evaluation==m_evaluation;
    }
};
struct ThermalStageReceipt {
    std::shared_ptr<ThermalStageLifetime const> owner;
    std::uint64_t instance=0,evaluation=0;
    bool Current() const noexcept {
        return owner && owner->Current(instance,evaluation);
    }
    bool SameInstance(ThermalStageReceipt const& other) const noexcept {
        return owner && owner==other.owner && instance && instance==other.instance;
    }
};
}
#endif
