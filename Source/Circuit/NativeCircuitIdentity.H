/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_CIRCUIT_IDENTITY_H_
#define WARPX_NATIVE_CIRCUIT_IDENTITY_H_
#include "ExternalCircuitIdentity.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
namespace warpx::circuit {
// Exact length-delimited model metadata, not a path or a digest of filenames.
class IdentityWriter {
public:
    IdentityWriter() { m_out.imbue(std::locale::classic()); }
    void Text(std::string const& value) { m_out<<value.size()<<':'<<value<<'\n'; }
    void Number(double value) {
        if(!std::isfinite(value)) { throw std::runtime_error("nonfinite native circuit definition"); }
        m_out<<std::hexfloat<<value<<'\n';
    }
    template<class Integer> void IntegerValue(Integer value) { m_out<<value<<'\n'; }
    std::string Value() const { return m_out.str(); }
private:
    std::ostringstream m_out;
};
inline std::string QueryNativeCircuitIdentity(void* plugin,WarpxCircuitIdentityApiV1 const* api) {
    if(!api || api->struct_bytes!=sizeof(*api) || api->api_version!=WARPX_CIRCUIT_IDENTITY_API_V1 ||
       api->flags!=WARPX_CIRCUIT_IDENTITY_IMMUTABLE_MODEL_V1 || !api->query) {
        throw std::runtime_error("protected native circuit restart requires immutable identity API v1");
    }
    WarpxCircuitIdentityViewV1 view{};view.struct_bytes=sizeof(view);char error[1024]{};
    auto const status=api->query(plugin,&view,error,sizeof(error));
    if(status!=0) { throw std::runtime_error(std::string("native identity provider: ")+error); }
    if(view.struct_bytes!=sizeof(view) || !view.data || view.data_bytes==0 ||
       view.data_bytes>std::string().max_size()) {
        throw std::runtime_error("invalid immutable native circuit identity view");
    }
    return {view.data,static_cast<std::size_t>(view.data_bytes)};
}
}
#endif
