/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_CIRCUIT_REJECTION_H_
#define WARPX_NATIVE_CIRCUIT_REJECTION_H_
#include "ExternalCircuitRejection.h"
#include <string>
namespace warpx::circuit {
// Plugin-owned in-memory snapshot lifetime. No ABI2 method is synthesized.
// The containing coupler owns its external-field/metadata snapshot separately.
class NativeCircuitRejection {
  public:
    NativeCircuitRejection () = default;
    NativeCircuitRejection (const NativeCircuitRejection&) = delete;
    NativeCircuitRejection& operator=(const NativeCircuitRejection&) = delete;
    ~NativeCircuitRejection () { Release(); }
    static bool
    Supports (const WarpxCircuitRejectionApiV1* api) {
        return api && api->struct_bytes >= sizeof(*api) &&
               api->api_version == WARPX_CIRCUIT_REJECTION_API_V1 &&
               (api->capabilities & WARPX_CIRCUIT_REJECTION_PRE_ACCEPT_V1) &&
               api->snapshot && api->cancel && api->release;
    }
    bool
    Snapshot (void* instance, const WarpxCircuitRejectionApiV1* api) {
        if (m_state != State::Idle || !instance || !Supports(api))
            return false;
        char error[512]{};
        uint64_t token = 0;
        int32_t status = WARPX_REJECTION_INTERNAL_ERROR;
        try {
            status = api->snapshot(instance, &token, error, sizeof(error));
        } catch (...) {
            m_state = State::Poisoned;
        }
        error[sizeof(error) - 1] = '\0';
        m_error = error;
        if (status != WARPX_REJECTION_OK || token == 0) {
            if (status == WARPX_REJECTION_OK && token == 0)
                m_state = State::Poisoned;
            if (token) {
                // Nonconforming provider: contain its outstanding allocation
                // but do not advertise a retryable transaction.
                try {
                    api->release(instance, token);
                } catch (...) {
                }
                m_state = State::Poisoned;
            }
            return false;
        }
        m_api = api;
        m_instance = instance;
        m_token = token;
        m_state = State::Prepared;
        return true;
    }
    bool
    Cancel () {
        if (!Ready())
            return false;
        char error[512]{};
        int32_t status = WARPX_REJECTION_INTERNAL_ERROR;
        try {
            status = m_api->cancel(m_instance, m_token, error, sizeof(error));
        } catch (...) {
        }
        error[sizeof(error) - 1] = '\0';
        m_error = error;
        if (status != WARPX_REJECTION_OK) {
            m_state = State::Poisoned;
            return false;
        }
        m_token = 0;
        m_state = State::Idle;
        return true;
    }
    // Must precede any engine accept=true, even if it later throws.
    bool
    MakeIrreversible () {
        if (!Ready())
            return false;
        m_state = State::Irreversible;
        if (!Release()) {
            m_state = State::Poisoned;
            return false;
        }
        return true;
    }
    bool
    FinishAccepted () {
        if (m_state != State::Irreversible)
            return false;
        m_state = State::Idle;
        return true;
    }
    bool
    Ready () const {
        return m_state == State::Prepared;
    }
    bool
    Irreversible () const {
        return m_state == State::Irreversible;
    }
    bool
    Idle () const {
        return m_state == State::Idle;
    }
    bool
    Failed () const {
        return m_state == State::Poisoned;
    }
    const std::string&
    Error () const {
        return m_error;
    }

  private:
    bool
    Release () noexcept {
        if (!m_token)
            return true;
        auto token = m_token;
        m_token = 0;
        try {
            m_api->release(m_instance, token);
        } catch (...) {
            return false;
        }
        return true;
    }
    enum class State { Idle, Prepared, Irreversible, Poisoned };
    State m_state = State::Idle;
    const WarpxCircuitRejectionApiV1* m_api = nullptr;
    void* m_instance = nullptr;
    uint64_t m_token = 0;
    std::string m_error;
};
} // namespace warpx::circuit
#endif
