/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EXPECTED_ION_EXCHANGE_SOURCE_K_H_
#define WARPX_EXPECTED_ION_EXCHANGE_SOURCE_K_H_
#include "AcceptedIonExchange_K.H"
#include <AMReX_GpuQualifiers.H>
#include <cmath>
namespace warpx::thermal {
enum class ExpectedIonEnergyMode { Unspecified, RelativisticQuadrature, NonrelativisticBounded };
struct ExpectedIonEnergyOptions {
    ExpectedIonEnergyMode mode = ExpectedIonEnergyMode::Unspecified;
    // Per physical ion; relative tolerance scales the absolute work components.
    amrex::Real absolute_tolerance_joule = 1.e-30;
    amrex::Real relative_component_tolerance = 1.e-10;
};
struct ExpectedIonEnergy {
    bool valid = false;
    amrex::Real relaxation = 0, redirect = 0;
    amrex::Real nr_relaxation = 0, nr_redirect = 0, nr_drift_work = 0;
    // Exact NR-minus-relativistic conditional-total-work interval. The bound
    // uses convexity and the Gaussian fourth moment; it is not sampling noise.
    amrex::Real convention_lower = 0, convention_upper = 0;
    amrex::Real quadrature_error_estimate = 0;
};
// Positive halves of standard-normal Gaussian quadrature. Generated from
// physicists' Hermite roots by z=sqrt(2)*x, w=w_H/sqrt(pi).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
GaussianNoiseCoefficient32 (amrex::Real mean_over_c, amrex::Real sigma_over_c) {
    using R = amrex::Real;
    amrex::GpuArray<R, 16> const node{0.27554641923027584,0.82728490377976516,1.3809801992721442,1.9380049059257174,2.4998404151873954,3.0681351690131216,3.6447812498808334,4.2320211099954097,4.8326046132444889,5.4500332736234292,6.0889643090769878,6.7559308305407049,7.4607557541215197,8.2197287653822464,9.0643992107024065,10.077422674229467};
    amrex::GpuArray<R, 16> const weight{0.21170556988047937,0.1565389937575985,0.085344808272080672,0.034109847726092095,0.0099034617023205963,0.0020620510513078838,0.00030255702581706253,3.0559803060896349e-05,2.0596221039534304e-06,8.8812907131058625e-08,2.312518412074231e-09,3.3475012398012285e-11,2.3780648557777909e-13,6.7552902236701287e-16,5.2084495919608421e-19,4.1246074890182323e-23};
    R const y = mean_over_c, g = std::sqrt(1+y*y);
    R answer = 0;
    for (int j = 0; j < 16; ++j) {
        R const z = node[j], t = sigma_over_c*z;
        R const gp = std::sqrt(1+(y+t)*(y+t));
        R const gm = std::sqrt(1+(y-t)*(y-t));
        // Factored paired integral for [E K(mu+sigma G)-K(mu)]/(m sigma^2).
        // It remains finite at mean=0 and sigma=0 and avoids subtracting K's.
        R const radial = 2*(1+(2*(1-y*y)+t*t)/(gp*gm+g*g))/(gp+gm+2*g);
        answer += weight[j]*z*z*(radial+4/(gp+gm));
    }
    return answer;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
GaussianNoiseCoefficient64 (amrex::Real mean_over_c, amrex::Real sigma_over_c) {
    using R = amrex::Real;
    amrex::GpuArray<R, 32> const node{0.19558891056727554,0.58688282330529251,0.97852590898502922,1.3707539637008053,1.7638073769428102,2.1579331167626106,2.5533868709840828,2.9504354015399628,3.3493591797524949,3.7504553852256111,4.1540413713409086,4.5604587281440505,4.9700781116024269,5.3833050611646138,5.8005871018292101,6.2224225326264557,6.6493714563417212,7.0820698308048415,7.5212476617873092,7.9677529819412216,8.422584092328588,8.8869339058381804,9.3622525462523072,9.8503384578821507,10.353475921269004,10.874651988398705,11.417918056820673,11.989036605128156,12.596752480605721,13.255649357397287,13.994049908876471,14.886186143339454};
    amrex::GpuArray<R, 32> const weight{0.15310831636189676,0.13145323131750097,0.09686336389597526,0.061213638510676481,0.033140485961927986,0.015347721995577658,0.0060684460158915033,0.0020438258387848809,0.00058468608306959987,0.0001416023883413604,2.8919958244145686e-05,4.9583797210632406e-06,7.0994246219066156e-07,8.4376410475409834e-08,8.2660844214792879e-09,6.6214234109507312e-10,4.2964262489852221e-11,2.2337268555257646e-12,9.1869287873296525e-14,2.94429314699771e-15,7.2221535735243866e-17,1.3269088554685335e-18,1.7784691911122742e-20,1.6829001120429461e-22,1.0785651103547654e-24,4.4355444204707863e-27,1.0883801541458748e-29,1.4384921208585243e-32,8.7866356793107015e-36,1.9301704298296966e-39,9.4769631900428472e-44,3.1231879651075858e-49};
    R const y = mean_over_c, g = std::sqrt(1+y*y);
    R answer = 0;
    for (int j = 0; j < 32; ++j) {
        R const z = node[j], t = sigma_over_c*z;
        R const gp = std::sqrt(1+(y+t)*(y+t));
        R const gm = std::sqrt(1+(y-t)*(y-t));
        // Factored paired integral for [E K(mu+sigma G)-K(mu)]/(m sigma^2).
        // It remains finite at mean=0 and sigma=0 and avoids subtracting K's.
        R const radial = 2*(1+(2*(1-y*y)+t*t)/(gp*gm+g*g))/(gp+gm+2*g);
        answer += weight[j]*z*z*(radial+4/(gp+gm));
    }
    return answer;
}
// Pure conditional expectation of the EXACT accepted proper-u Gaussian map.
// No RNG. Drift is cylindrical; u is Cartesian; output is J per physical ion.
// Nonrelativistic mode is explicitly approximate to native relativistic K and
// refuses when its proven convention bound exceeds the requested tolerance.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ExpectedIonEnergy
ExpectedIonExchangeWork (
    amrex::GpuArray<amrex::ParticleReal, 3> const& u,
    amrex::GpuArray<amrex::ParticleReal, 3> const& cylindrical_drift,
    amrex::ParticleReal theta, amrex::ParticleReal nu,
    amrex::ParticleReal temperature, amrex::ParticleReal redirect_energy,
    amrex::ParticleReal mass, amrex::Real dt, ExpectedIonEnergyOptions options) {
    using R = amrex::Real;
    using P = amrex::ParticleReal;
    ExpectedIonEnergy out;
    if ((options.mode != ExpectedIonEnergyMode::RelativisticQuadrature
         && options.mode != ExpectedIonEnergyMode::NonrelativisticBounded)
        || !std::isfinite(options.absolute_tolerance_joule) || options.absolute_tolerance_joule < 0
        || !std::isfinite(options.relative_component_tolerance) || options.relative_component_tolerance < 0
        || !std::isfinite(dt)
        || dt <= 0 || !std::isfinite(nu) || nu < 0 || !std::isfinite(temperature)
        || temperature < 0 || !std::isfinite(redirect_energy) || redirect_energy < 0
        || !std::isfinite(mass) || mass <= 0 || !std::isfinite(theta)) return out;
    P const cs = std::cos(theta), sn = std::sin(theta);
    amrex::GpuArray<P, 3> const drift{
        cylindrical_drift[0]*cs-cylindrical_drift[1]*sn,
        cylindrical_drift[0]*sn+cylindrical_drift[1]*cs, cylindrical_drift[2]};
    P const x = nu*dt, drag = -std::expm1(-x);
    P const thermal = -PhysConst::kb*temperature*std::expm1(-2*x);
    // Match native variance rounding before evaluating the Gaussian integral.
    P const vr = thermal/mass, vt = (thermal+redirect_energy)/mass;
    R s0 = 0, sm = 0, ds = 0;
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(u[d]) || !std::isfinite(drift[d])) return out;
        P const delta = -drag*(u[d]-drift[d]), mean = u[d]+delta;
        s0 += R(u[d])*u[d]; sm += R(mean)*mean;
        ds += R(delta)*(2*R(u[d])+delta);
        out.nr_drift_work += R(mass)*drift[d]*delta;
    }
    R const c2 = PhysConst::c*PhysConst::c;
    R const g0 = std::sqrt(1+s0/c2), gm = std::sqrt(1+sm/c2);
    R const deterministic_nr = .5*R(mass)*ds;
    R const deterministic_rel = R(mass)*ds/(gm+g0);
    out.nr_relaxation = deterministic_nr+1.5*R(mass)*vr;
    out.nr_redirect = 1.5*R(mass)*(vt-vr);
    // D(u)=K_NR(u)-K_rel(u). Both D and m|u|^4/(8c^2)-D are convex.
    R const mean_convention = .5*R(mass)*ds*
        ((sm/c2)/(gm+1)+(s0/c2)/(g0+1))/(gm+g0);
    R const noise_convention_bound = R(mass)*R(vt)*(10*sm+15*R(vt))/(8*c2);
    out.convention_lower = mean_convention;
    out.convention_upper = mean_convention+noise_convention_bound;
    R const scale = std::abs(deterministic_nr)+1.5*R(mass)*vt;
    R const tolerance = options.absolute_tolerance_joule+options.relative_component_tolerance*scale;
    if (!std::isfinite(vr) || !std::isfinite(vt) || vr < 0 || vt < vr
        || !std::isfinite(scale) || !std::isfinite(tolerance) || tolerance < 0
        || !std::isfinite(out.convention_lower) || !std::isfinite(out.convention_upper)) return out;
    if (options.mode == ExpectedIonEnergyMode::NonrelativisticBounded) {
        out.relaxation = out.nr_relaxation; out.redirect = out.nr_redirect;
        out.valid = amrex::max(std::abs(out.convention_lower),std::abs(out.convention_upper)) <= tolerance;
        return out;
    }
    R const y = std::sqrt(sm/c2);
    R nr32 = 0, nr64 = 0, nt32 = 0, nt64 = 0;
    if (vr > 0) {
        R const sigma = std::sqrt(R(vr)/c2);
        nr32 = R(mass)*vr*GaussianNoiseCoefficient32(y,sigma);
        nr64 = R(mass)*vr*GaussianNoiseCoefficient64(y,sigma);
    }
    if (vt == vr) { nt32 = nr32; nt64 = nr64; }
    else if (vt > 0) {
        R const sigma = std::sqrt(R(vt)/c2);
        nt32 = R(mass)*vt*GaussianNoiseCoefficient32(y,sigma);
        nt64 = R(mass)*vt*GaussianNoiseCoefficient64(y,sigma);
    }
    out.relaxation = deterministic_rel+nr64;
    out.redirect = nt64-nr64;
    out.quadrature_error_estimate = 2*std::abs(nr64-nr32)+std::abs(nt64-nt32);
    out.valid = std::isfinite(out.relaxation) && std::isfinite(out.redirect)
        && out.redirect >= 0 && out.quadrature_error_estimate <= tolerance;
    return out;
}
} // namespace warpx::thermal
#endif
