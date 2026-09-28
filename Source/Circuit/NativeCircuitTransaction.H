/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_CIRCUIT_TRANSACTION_H_
#define WARPX_NATIVE_CIRCUIT_TRANSACTION_H_
#include "ExternalCircuitTransaction.h"
#include <array>
#include <cstring>
namespace warpx::circuit {
// Explicit optional post-accept restoration. This owner never calls ABI2
// Begin/Finish as cancellation and never uses the PRE_ACCEPT V1 release path.
class NativeCircuitTransaction {
public:
    NativeCircuitTransaction() = default;
    NativeCircuitTransaction(NativeCircuitTransaction const&) = delete;
    NativeCircuitTransaction& operator=(NativeCircuitTransaction const&) = delete;
    ~NativeCircuitTransaction() {
        if(m_token && m_api && m_instance) {try{(void)m_api->cancel(m_instance,m_token,nullptr,0);}catch(...) {}}
    }
    static bool Supports(WarpxCircuitTransactionApiV1 const* api) {
        return api && api->struct_bytes>=sizeof(*api) && api->api_version==WARPX_CIRCUIT_TRANSACTION_API_V1 &&
            (api->capabilities&WARPX_CIRCUIT_TRANSACTION_POST_ACCEPT_RESTORE_V1) &&
            api->snapshot && api->query && api->cancel && api->finalize;
    }
    bool Snapshot(void* instance,WarpxCircuitTransactionApiV1 const* api) {
        if(!Idle() || !instance || !Supports(api))return false;
        uint64_t token=0;int32_t status=WARPX_TRANSACTION_INTERNAL_ERROR;m_error.fill(0);
        try {status=api->snapshot(instance,&token,m_error.data(),m_error.size());}
        catch(...) {m_failed=true;}
        m_error.back()='\0';
        if(status!=WARPX_TRANSACTION_OK || !token) {
            if(token) {try{(void)api->cancel(instance,token,nullptr,0);}catch(...){} m_failed=true;}
            if(status==WARPX_TRANSACTION_OK)m_failed=true;
            return false;
        }
        m_api=api;m_instance=instance;m_token=token;return true;
    }
    bool Query(WarpxCircuitTransactionViewV1& view) const noexcept {
        view={};view.struct_bytes=sizeof(view);
        if(!m_token || m_failed)return false;
        int32_t status=WARPX_TRANSACTION_INTERNAL_ERROR;
        try {status=m_api->query(m_instance,m_token,&view,nullptr,0);}catch(...) {return false;}
        return status==WARPX_TRANSACTION_OK && view.struct_bytes==sizeof(view) && view.token==m_token &&
            view.can_cancel==1 && view.can_finalize<=1 &&
            view.phase>=WARPX_TRANSACTION_SNAPSHOTTED && view.phase<=WARPX_TRANSACTION_FAILED_CALL &&
            view.can_finalize==(view.phase==WARPX_TRANSACTION_ACCEPTED_CLOSED);
    }
    bool CanCancel() const noexcept {WarpxCircuitTransactionViewV1 view{};return Query(view);}
    bool CanFinalize() const noexcept {WarpxCircuitTransactionViewV1 view{};return Query(view)&&view.can_finalize;}
    bool Cancel() noexcept {return Finish(false);}
    bool Finalize() noexcept {return Finish(true);}
    bool Idle() const noexcept {return !m_token&&!m_failed;}
    bool Active() const noexcept {return m_token!=0;}
    bool Failed() const noexcept {return m_failed;}
    char const* Error() const noexcept {return m_error.data();}
private:
    bool Finish(bool finalize) noexcept {
        if(!(finalize?CanFinalize():CanCancel()))return false;
        m_error.fill(0);int32_t status=WARPX_TRANSACTION_INTERNAL_ERROR;
        try {status=finalize?m_api->finalize(m_instance,m_token,m_error.data(),m_error.size()):
            m_api->cancel(m_instance,m_token,m_error.data(),m_error.size());}catch(...) {}
        m_error.back()='\0';
        if(status!=WARPX_TRANSACTION_OK){m_failed=true;return false;}
        m_token=0;return true;
    }
    WarpxCircuitTransactionApiV1 const* m_api=nullptr;
    void* m_instance=nullptr;
    uint64_t m_token=0;
    bool m_failed=false;
    std::array<char,512> m_error{};
};
} // namespace warpx::circuit
#endif
