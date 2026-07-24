/* Copyright 2026 The WarpX Community
 *
 * Unit test for the RTZ azimuthal band-limit filter
 * (Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/AzimuthalFilter.*).
 *
 * On a theta- and z-split BoxArray (so the full-ring gather machinery is
 * exercised), for several Yee staggerings, checks that ApplyFilter
 *   (A) keeps every azimuthal mode m <= m_max(r) EXACTLY and kills every
 *       mode m > m_max(r) to machine precision (per-ring Fourier projection),
 *   (B) is the identity on rings with m_max >= NT/2,
 *   (C) is idempotent (P^2 = P),
 *   (D) preserves each ring's average (m=0) on a pseudo-random field,
 *   (E) fills theta-periodic ghost cells consistently with the filtered
 *       valid data.
 *
 * Returns a non-zero exit code if any check fails (so CTest reports failure).
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/AzimuthalFilter.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParReduce.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

using amrex::Real;

namespace {
    int g_failures = 0;
    int g_checks = 0;

    void check (const std::string& name, double measured, double tol)
    {
        ++g_checks;
        const bool ok = (measured <= tol);
        if (!ok) { ++g_failures; }
        std::printf("  [%s] %-52s measured=%.3e tol=%.1e\n",
                    ok ? "PASS" : "FAIL", name.c_str(), measured, tol);
    }

    constexpr int NR = 16;
    constexpr int NT = 16;
    constexpr int NZ = 8;
    constexpr Real RMAX = 0.2;   // dr = 0.0125 -> rings 0..2 filtered (cell)
    AMREX_GPU_HOST_DEVICE
    Real spectrum_value (Real theta, Real zfac, int m_keep)
    {
        // Test spectrum: amplitudes and phases per azimuthal mode number.
        // Function-local so the constants are usable in device code.
        constexpr int NMODE = 5;
        constexpr int MODES[NMODE] = {0, 1, 3, 5, 7};
        constexpr Real AMPL[NMODE] = {1.0, 0.8, 0.6, 0.4, 0.3};
        constexpr Real PHASE[NMODE] = {0.3, 1.1, 2.0, 0.7, 1.9};
        Real acc = 0.0;
        for (int n = 0; n < NMODE; ++n) {
            if (MODES[n] <= m_keep) {
                acc += AMPL[n]*std::cos(MODES[n]*theta + PHASE[n]);
            }
        }
        return acc*zfac;
    }

    AMREX_GPU_HOST_DEVICE
    int ring_m_max (Real r, Real dr, Real alpha)
    {
        const int m = static_cast<int>(std::floor(alpha*MathConst::pi*r/dr));
        return (m < 1) ? 1 : m;
    }

    // deterministic pseudo-random in [-1, 1)
    AMREX_GPU_HOST_DEVICE
    Real hash_val (int i, int j, int k)
    {
        const Real s = std::sin(12.9898*i + 78.233*j + 37.719*k + 1.0)*43758.5453;
        return 2.0*(s - std::floor(s)) - 1.0;
    }
}

void run_tests ()
{
    using namespace amrex;

    const Box domain(IntVect(0, 0, 0), IntVect(NR - 1, NT - 1, NZ - 1));
    RealBox rb({0.0, -MathConst::pi, 0.0}, {RMAX, MathConst::pi, 1.0});
    // theta periodic; z NOT periodic (matches the RTZ formation BCs -- for
    // periodic nodal directions WarpX keeps the duplicated boundary planes
    // synced, which a raw test fill would violate)
    const std::array<int, 3> is_per = {0, 1, 0};
    Geometry geom(domain, rb, CoordSys::cartesian, is_per);

    // theta- and z-split boxes: the full-ring gather is genuinely exercised
    BoxArray ba(domain);
    ba.maxSize(IntVect(NR, NT/4, NZ/2));
    DistributionMapping dm{ba};

    const Real dr = geom.CellSize(0);
    const Real dth = geom.CellSize(1);
    const Real alpha = 1.0;

    AzimuthalFilter filter;
    filter.Init(geom, alpha);

    // Staggerings: rho-like nodal, cell-centered, Bt-like, Br-like
    const std::array<IntVect, 4> staggerings = {
        IntVect(1, 1, 1), IntVect(0, 0, 0), IntVect(0, 1, 0), IntVect(1, 0, 0)};
    const std::array<std::string, 4> stag_names = {
        "nodal(rho)", "cell", "theta-node(Bt-like)", "r-node(Br-like)"};

    for (int s = 0; s < 4; ++s) {
        const IntVect iv = staggerings[s];
        const Real r_off = (iv[0] == 1) ? 0.0 : 0.5;
        const Real th_off = (iv[1] == 1) ? 0.0 : 0.5;

        MultiFab mf(amrex::convert(ba, iv), dm, 1, 2);
        mf.setVal(0.0);

        // ---- (A) exact keep/kill of the mode spectrum ----
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const Box vb = mfi.validbox();
            Array4<Real> const& a = mf.array(mfi);
            ParallelFor(vb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const Real theta = -MathConst::pi + (j + th_off)*dth;
                const Real zfac = 1.0 + 0.1*k;
                a(i, j, k) = spectrum_value(theta, zfac, /*m_keep=*/999);
            });
        }
        filter.ApplyFilter(mf);

        auto max_err_vs = [&](int mode_cap_flag) {
            ReduceOps<ReduceOpMax> rop;
            ReduceData<Real> rdata(rop);
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                const Box vb = mfi.validbox();
                Array4<Real const> const& a = mf.const_array(mfi);
                rop.eval(vb, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    -> GpuTuple<Real>
                {
                    const Real r = (i + r_off)*dr;
                    const int m_keep = (mode_cap_flag == 0)
                        ? 999
                        : ((ring_m_max(r, dr, alpha) >= NT/2)
                               ? 999 : ring_m_max(r, dr, alpha));
                    // theta-nodal grids: skip the duplicated j = NT plane in
                    // the comparison (it mirrors j = 0 by construction)
                    const Real theta = -MathConst::pi + (j + th_off)*dth;
                    const Real zfac = 1.0 + 0.1*k;
                    return { std::abs(a(i, j, k)
                                      - spectrum_value(theta, zfac, m_keep)) };
                });
            }
            Real v = amrex::get<0>(rdata.value(rop));
            ParallelDescriptor::ReduceRealMax(v);
            return v;
        };
        check(stag_names[s] + ": keep/kill exactness", max_err_vs(1), 1.0e-11);

        // ---- (C) idempotence ----
        MultiFab mf2(mf.boxArray(), dm, 1, 0);
        MultiFab::Copy(mf2, mf, 0, 0, 1, 0);
        filter.ApplyFilter(mf);
        MultiFab::Subtract(mf2, mf, 0, 0, 1, 0);
        check(stag_names[s] + ": idempotence (P^2 = P)", mf2.norm0(), 1.0e-12);

        // ---- (E) theta-periodic ghost fill on filtered rings ----
        {
            ReduceOps<ReduceOpMax> rop;
            ReduceData<Real> rdata(rop);
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                const Box vb = mfi.validbox();
                Box gb = vb;
                gb.grow(1, 1);   // one theta ghost layer
                Array4<Real const> const& a = mf.const_array(mfi);
                const int jper = NT;   // theta period in index space
                rop.eval(gb, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    -> GpuTuple<Real>
                {
                    const Real r = (i + r_off)*dr;
                    if (vb.contains(i, j, k)
                        || ring_m_max(r, dr, alpha) >= NT/2) { return {0.0}; }
                    const Real theta = -MathConst::pi + (j + th_off)*dth;
                    const Real zfac = 1.0 + 0.1*k;
                    const int m_keep = ring_m_max(r, dr, alpha);
                    amrex::ignore_unused(jper);
                    return { std::abs(a(i, j, k)
                                      - spectrum_value(theta, zfac, m_keep)) };
                });
            }
            Real v = amrex::get<0>(rdata.value(rop));
            ParallelDescriptor::ReduceRealMax(v);
            check(stag_names[s] + ": theta ghost fill", v, 1.0e-11);
        }

        // ---- (D) ring-average conservation on a pseudo-random field ----
        {
            MultiFab mfr(amrex::convert(ba, iv), dm, 1, 0);
            for (MFIter mfi(mfr); mfi.isValid(); ++mfi) {
                const Box vb = mfi.validbox();
                Array4<Real> const& a = mfr.array(mfi);
                ParallelFor(vb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    a(i, j, k) = hash_val(i, j, k);
                });
            }
            // ring sums before/after over the unique theta samples
            const bool th_nodal = (iv[1] == 1);
            auto ring_sum = [&](MultiFab const& m, int ring, int kk) {
                ReduceOps<ReduceOpSum> rop;
                ReduceData<Real> rdata(rop);
                for (MFIter mfi(m); mfi.isValid(); ++mfi) {
                    const Box vb = mfi.validbox();
                    // count each unique theta sample once: for theta-nodal
                    // staggerings adjacent boxes share their boundary plane
                    // (and j = NT duplicates j = 0), so exclude each box's
                    // top theta plane from its own count
                    const int j_excl = th_nodal ? vb.bigEnd(1)
                                                : vb.bigEnd(1) + 1;
                    Array4<Real const> const& a = m.const_array(mfi);
                    rop.eval(vb, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        -> GpuTuple<Real>
                    {
                        return { (i == ring && k == kk && j < j_excl)
                                     ? a(i, j, k) : 0.0 };
                    });
                }
                Real v = amrex::get<0>(rdata.value(rop));
                ParallelDescriptor::ReduceRealSum(v);
                return v;
            };
            const Real before0 = ring_sum(mfr, 0, 1);
            const Real before2 = ring_sum(mfr, 2, 3);
            filter.ApplyFilter(mfr, 0, 1, false);
            const Real after0 = ring_sum(mfr, 0, 1);
            const Real after2 = ring_sum(mfr, 2, 3);
            check(stag_names[s] + ": ring-average conservation",
                  std::abs(before0 - after0) + std::abs(before2 - after2),
                  1.0e-11);
        }
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        run_tests();
    }
    std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
    const int failures = g_failures;
    amrex::Finalize();
    return (failures == 0) ? 0 : 1;
}
