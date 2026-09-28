/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#include "DeviceCircuitRate.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX.H>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>

namespace warpx::circuit {
namespace {
bool Finite(double const* values, std::size_t count) {
    if(count && !values) return false;
    for(std::size_t i=0;i<count;++i) if(!std::isfinite(values[i])) return false;
    return true;
}
bool Finite(std::vector<double> const& values) { return Finite(values.data(),values.size()); }
void Upload(amrex::Gpu::DeviceVector<double>& to, std::vector<double> const& from) {
    to.resize(from.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,from.begin(),from.end(),to.begin());
}
// Dimensionless Schur block; scaled row pivoting is performed on independent
// copies. No native circuit factor or accepted state is modified.
struct Factor {
    int n;
    std::vector<double> a, row_scale;
    std::vector<int> swaps;
    bool Define(std::vector<double> const& matrix, int size) {
        n=size; a=matrix; row_scale.assign(n,0.); swaps.resize(n);
        for(int i=0;i<n;++i) {
            for(int j=0;j<n;++j) row_scale[i]=std::max(row_scale[i],std::abs(a[i*n+j]));
            if(!(row_scale[i]>0.)) return false;
            for(int j=0;j<n;++j) a[i*n+j]/=row_scale[i];
        }
        for(int k=0;k<n;++k) {
            int p=k;
            for(int i=k+1;i<n;++i) if(std::abs(a[i*n+k])>std::abs(a[p*n+k])) p=i;
            if(a[p*n+k]==0. || !std::isfinite(a[p*n+k])) return false;
            swaps[k]=p;
            if(p!=k) for(int j=0;j<n;++j) std::swap(a[p*n+j],a[k*n+j]);
            for(int i=k+1;i<n;++i) {
                a[i*n+k]/=a[k*n+k];
                for(int j=k+1;j<n;++j) a[i*n+j]-=a[i*n+k]*a[k*n+j];
            }
        }
        return Finite(a);
    }
    bool Solve(std::vector<double> const& rhs,std::vector<double>& x) const {
        x=rhs;
        for(int i=0;i<n;++i) x[i]/=row_scale[i];
        for(int k=0;k<n;++k) if(swaps[k]!=k) std::swap(x[k],x[swaps[k]]);
        for(int i=0;i<n;++i) for(int j=0;j<i;++j) x[i]-=a[i*n+j]*x[j];
        for(int i=n-1;i>=0;--i) {
            for(int j=i+1;j<n;++j) x[i]-=a[i*n+j]*x[j];
            x[i]/=a[i*n+i];
        }
        return Finite(x);
    }
};
AMREX_GPU_HOST_DEVICE void DeviceRequire(bool valid) {
    if(valid) return;
    AMREX_IF_ON_DEVICE((printf("Invalid instantaneous device circuit rate arithmetic\n");))
#if defined(__CUDA_ARCH__)
    __trap();
#elif defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#else
    amrex::Abort("Invalid instantaneous device circuit rate arithmetic");
#endif
}
AMREX_GPU_HOST_DEVICE double AddProduct(double sum,double a,double b) {
    volatile double p=a*b;
    return sum+p;
}
} // namespace

bool DeviceCircuitRate::Prepare(WarpxCircuitRateViewV1 const& p,double time,
    std::vector<int> const& port_fields,std::vector<double> const& scales,
    std::vector<double> const& iref,std::vector<double> const& norm,
    std::vector<int> const& measured,std::vector<double> const& h,
    std::vector<double> const& q,std::vector<double> const& fixed,std::string& error,
    WarpxCircuitRateClockViewV1 const* clock)
{
    m_ready=false; m_np=m_nf=0;
    auto fail=[&](char const* message){error=message;return false;};
    auto const np=port_fields.size(),nf=scales.size();
    auto const cap=static_cast<std::size_t>(std::numeric_limits<int>::max());
    if(np==0 || nf==0 || np>cap || nf>cap || np>cap/np || nf>cap/np)
        return fail("Invalid rate block dimensions");
    if(p.struct_bytes!=sizeof(p) || p.scalar_kind!=WARPX_CIRCUIT_RATE_F64 ||
       p.coefficient_policy!=WARPX_CIRCUIT_RATE_FROZEN_ACCEPTED_COEFFICIENTS_V1 ||
       p.reserved!=0 || p.token==0 || p.n_port!=np)
        return fail("Unsupported rate packet contract");
    if(!std::isfinite(time) || p.requested_time_sim!=time ||
       !std::isfinite(p.state_time_sim) || !std::isfinite(p.lattice_dt) || p.lattice_dt<=0.)
        return fail("Invalid accepted rate clock");
    double const eps=std::numeric_limits<double>::epsilon();
    if (clock) {
        auto const& c = *clock;
        if (c.struct_bytes != sizeof(c) || c.scalar_kind != WARPX_CIRCUIT_RATE_CLOCK_F64 ||
            (c.clock_policy != WARPX_CIRCUIT_RATE_CLOCK_NATIVE_OPERAND_ENCLOSURE_V1 &&
             c.clock_policy != WARPX_CIRCUIT_RATE_CLOCK_NATIVE_LATTICE_PROVENANCE_V1) ||
            c.reserved != 0 || c.token != p.token || c.step_index < 0 ||
            c.step_index == std::numeric_limits<std::int64_t>::max() ||
            c.requested_time_sim != time || c.state_time_sim != p.state_time_sim ||
            c.lattice_dt != p.lattice_dt)
            return fail("Rate clock certificate is not bound to this packet");
        double const values[] = {c.time_origin_machine, c.time_shift,
            c.requested_time_machine, c.state_time_machine, c.sync_error_bound,
            c.neighbor_separation};
        if (!Finite(values, 6) || c.sync_error_bound < 0. || c.neighbor_separation <= 0.)
            return fail("Invalid or ambiguous rate clock certificate");
        // Reproduce the provider's separated native clock expressions. This
        // verifies operand/representation binding; the trusted optional API
        // supplies the arithmetic enclosure and adjacent-neighborhood proof.
        // Policy2 also bounds native origin construction and positive integer-
        // multiple caller-clock accumulation; all packet/operand checks below
        // remain identical. Fractional or ambiguous lattice points still reject.
        auto lattice = [&](std::int64_t step) {
            volatile double index = static_cast<double>(step);
            volatile double product = index*c.lattice_dt;
            return c.time_origin_machine + product;
        };
        double const state = lattice(c.step_index);
        double const before = lattice(c.step_index-1);
        double const after = lattice(c.step_index+1);
        if (!std::isfinite(before) || !std::isfinite(after) ||
            c.requested_time_machine != time-c.time_shift ||
            c.state_time_machine != state || c.state_time_sim != state+c.time_shift)
            return fail("Rate clock machine operands do not match the certificate");
        long double const gap = std::min(static_cast<long double>(state)-before,
                                        static_cast<long double>(after)-state);
        long double const difference = std::abs(
            static_cast<long double>(c.requested_time_machine)-state);
        // Necessary independent bounds: the certified neighborhood cannot
        // reach a neighboring represented lattice point, and its claimed gap
        // cannot exceed either spacing minus this point's own radius.
        if (!(gap > c.sync_error_bound) || difference > c.sync_error_bound ||
            static_cast<long double>(c.neighbor_separation) > gap-c.sync_error_bound)
            return fail("Rate clock enclosure is inconsistent or not uniquely synchronized");
    } else {
        double const clock_bound=16.*eps*std::max(std::abs(time),std::abs(p.state_time_sim));
        if(std::abs(time-p.state_time_sim)>clock_bound)
            return fail("Endpoint rate requires a synchronized accepted circuit lattice");
    }
    return PrepareResponse(p.current,p.p0,p.g,p.i_ref,port_fields,scales,iref,norm,
        measured,h,q,fixed,error);
}
bool DeviceCircuitRate::PrepareImpulse(WarpxCircuitImpulseViewV1 const& p,
    std::uint64_t transaction,double time,std::uint32_t phase,
    std::vector<int> const& port_fields,std::vector<double> const& scales,
    std::vector<double> const& iref,std::vector<double> const& norm,
    std::vector<int> const& measured,std::vector<double> const& h,
    std::vector<double> const& q,std::string& error)
{
    m_ready=false;m_np=m_nf=0;
    auto fail=[&](char const* message){error=message;return false;};
    if(p.struct_bytes!=sizeof(p) || p.scalar_kind!=WARPX_CIRCUIT_IMPULSE_F64 ||
       p.reserved!=0 || !p.token || !transaction || p.transaction_token!=transaction ||
       p.phase!=phase || (phase!=WARPX_IMPULSE_PRE_FIELD && phase!=WARPX_IMPULSE_POST_FIELD) ||
       !std::isfinite(time) || p.time_sim!=time || p.n_port!=1 || p.n_state!=5 ||
       port_fields.size()!=1 || !Finite(p.state,5) || !Finite(p.delta_x,5) || !Finite(p.zeta,5))
        return fail("Unsupported or stale retained source-impulse packet");
    // Finite algebraic state and time-integrated algebraic voltages have
    // different dimensions. Only the provider can publish their full state.
    // This device block consumes only its independently validated port map.
    std::vector<double> const zero_offset(port_fields.size(),0.);
    std::vector<double> const zero_drive(scales.size(),0.);
    return PrepareResponse(p.current,zero_offset.data(),p.g,p.i_ref,
        port_fields,scales,iref,norm,measured,h,q,zero_drive,error);
}
bool DeviceCircuitRate::PrepareResponse(double const* current,double const* p0,
    double const* g,double const* packet_reference,
    std::vector<int> const& port_fields,std::vector<double> const& scales,
    std::vector<double> const& iref,std::vector<double> const& norm,
    std::vector<int> const& measured,std::vector<double> const& h,
    std::vector<double> const& q,std::vector<double> const& fixed,std::string& error)
{
    m_ready=false;m_np=m_nf=0;
    auto fail=[&](char const* message){error=message;return false;};
    auto const np=port_fields.size(),nf=scales.size();
    auto const cap=static_cast<std::size_t>(std::numeric_limits<int>::max());
    if(np==0 || nf==0 || np>cap || nf>cap || np>cap/np || nf>cap/np)
        return fail("Invalid response block dimensions");
    double const eps=std::numeric_limits<double>::epsilon();
    if(iref.size()!=np || norm.size()!=np || measured.size()!=np ||
       h.size()!=np*nf || q.size()!=np*nf || fixed.size()!=nf ||
       !Finite(current,np) || !Finite(p0,np) || !Finite(g,np*np) || !Finite(packet_reference,np))
        return fail("Invalid rate preparation shape or packet values");
    for(auto const* values:{&scales,&iref,&norm,&h,&q,&fixed})
        if(!Finite(*values)) return fail("Nonfinite rate preparation values");
    std::vector<int> field_port(nf,-1);
    for(std::size_t i=0;i<np;++i) {
        int const f=port_fields[i];
        if(f<0 || f>=static_cast<int>(nf) || field_port[f]!=-1 ||
           packet_reference[i]!=iref[i] || iref[i]==0. || norm[i]==0. ||
           (measured[i]!=0 && measured[i]!=1) || fixed[f]!=0.)
            return fail("Invalid rate port mapping, reference or prescribed rate");
        double const scale=current[i]/iref[i];
        if(!std::isfinite(scale) || std::abs(scale-scales[f])>
           256.*eps*std::max({1.,std::abs(scale),std::abs(scales[f])}))
            return fail("Rate packet current disagrees with accepted field scale");
        field_port[f]=static_cast<int>(i);
    }
    // y = r + R eps, eps = T (g + (H-Q)(fixed+S y)).
    std::vector<double> k(np*nf),t(np),rt(np*np),l(np*np,0.),rhs0(np),offset=fixed;
    for(std::size_t i=0;i<np;++i) {
        t[i]=measured[i]?1./norm[i]:0.;
        for(std::size_t f=0;f<nf;++f) k[i*nf+f]=h[i*nf+f]-q[i*nf+f];
    }
    for(std::size_t i=0;i<np;++i) for(std::size_t j=0;j<np;++j)
        rt[i*np+j]=(g[i*np+j]/iref[i])*t[j];
    for(std::size_t i=0;i<np;++i) {
        long double value=static_cast<long double>(p0[i])/iref[i];
        for(std::size_t j=0;j<np;++j) for(std::size_t f=0;f<nf;++f)
            value+=static_cast<long double>(rt[i*np+j])*k[j*nf+f]*fixed[f];
        rhs0[i]=static_cast<double>(value);
        for(std::size_t c=0;c<np;++c) {
            long double v=i==c?1.L:0.L;
            for(std::size_t j=0;j<np;++j) v-=static_cast<long double>(rt[i*np+j])*k[j*nf+port_fields[c]];
            l[i*np+c]=static_cast<double>(v);
        }
    }
    if(!Finite(k)||!Finite(t)||!Finite(rt)||!Finite(rhs0)||!Finite(l))
        return fail("Rate Schur block assembly overflow");
    Factor factor;
    if(!factor.Define(l,static_cast<int>(np))) return fail("Singular instantaneous circuit/field response");
    double const gamma=512.*static_cast<double>(np)*eps;
    auto checked_solve=[&](std::vector<double> const& rhs,std::vector<double>& solution){
        if(!factor.Solve(rhs,solution)) return false;
        for(std::size_t i=0;i<np;++i) {
            long double defect=-rhs[i],scale=std::abs(rhs[i]);
            for(std::size_t j=0;j<np;++j) {
                auto term=static_cast<long double>(l[i*np+j])*solution[j];
                defect+=term;scale+=std::abs(term);
            }
            if(std::abs(defect)>gamma*scale) return false;
        }
        return true;
    };
    std::vector<double> column(np),solution,inv(np*np),response(nf*np,0.);
    for(std::size_t j=0;j<np;++j) {
        std::fill(column.begin(),column.end(),0.);column[j]=1.;
        if(!checked_solve(column,solution))return fail("Unresolved circuit response inverse");
        for(std::size_t i=0;i<np;++i)inv[i*np+j]=solution[i];
    }
    long double matrix_norm=0.,inverse_norm=0.,formation_norm=0.;
    for(std::size_t i=0;i<np;++i) {
        long double an=0.,bn=0.;
        for(std::size_t j=0;j<np;++j){an+=std::abs(l[i*np+j]);bn+=std::abs(inv[i*np+j]);}
        matrix_norm=std::max(matrix_norm,an);inverse_norm=std::max(inverse_norm,bn);
        long double formation=1.;
        for(std::size_t c=0;c<np;++c)for(std::size_t j=0;j<np;++j)
            formation+=std::abs(static_cast<long double>(rt[i*np+j])*k[j*nf+port_fields[c]]);
        formation_norm=std::max(formation_norm,formation);
    }
    if(!(std::max(matrix_norm,formation_norm)*inverse_norm<1.L/(1024.*np*eps)))
        return fail("Ill-conditioned instantaneous circuit/field response");
    if(!checked_solve(rhs0,solution))return fail("Circuit rate offset residual exceeded bound");
    for(std::size_t i=0;i<np;++i)offset[port_fields[i]]=solution[i];
    for(std::size_t j=0;j<np;++j) {
        for(std::size_t i=0;i<np;++i)column[i]=rt[i*np+j];
        if(!checked_solve(column,solution))return fail("Circuit rate response residual exceeded bound");
        for(std::size_t i=0;i<np;++i)response[port_fields[i]*np+j]=solution[i];
    }
    Upload(m_offset,offset);Upload(m_response,response);Upload(m_k,k);Upload(m_t,t);
    m_rates.resize(nf);m_emf.resize(np);m_np=static_cast<int>(np);m_nf=static_cast<int>(nf);
    // Outputs are intentionally not readable as accepted fields until Apply.
    m_ready=true;error.clear();return true;
}
void DeviceCircuitRate::Apply(amrex::Gpu::DeviceVector<double> const& input,bool affine)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_ready && input.size()==static_cast<std::size_t>(m_np),
        "Instantaneous device circuit response is unprepared or has wrong input shape");
    // Accessors expose const views of our output storage. Such a view must not
    // be used as the input: rows are written while later rows still read g.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(input.data()!=m_rates.data() &&
        input.data()!=m_emf.data(),
        "Instantaneous device circuit response input aliases output storage");
    int const np=m_np,nf=m_nf;
    auto const* g=input.data();auto const* offset=m_offset.data();auto const* response=m_response.data();
    auto const* k=m_k.data();auto const* t=m_t.data();auto* rates=m_rates.data();auto* emf=m_emf.data();
    amrex::ParallelFor(1,[=] AMREX_GPU_DEVICE(int) noexcept {
        for(int i=0;i<np;++i)DeviceRequire(amrex::Math::isfinite(g[i]));
        for(int f=0;f<nf;++f) {
            double value=affine?offset[f]:0.;
            for(int i=0;i<np;++i)value=AddProduct(value,response[f*np+i],g[i]);
            DeviceRequire(amrex::Math::isfinite(value));rates[f]=value;
        }
        for(int i=0;i<np;++i) {
            double value=g[i];
            for(int f=0;f<nf;++f)value=AddProduct(value,k[i*nf+f],rates[f]);
            emf[i]=t[i]*value;DeviceRequire(amrex::Math::isfinite(emf[i]));
        }
    });
}
} // namespace warpx::circuit
