/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

/* Standalone unit test of the continuum ring kernels in
 * ablastr/coils/RingKernel.H: complete elliptic integrals and the kernel
 * bracket against 50-digit reference values (mpmath), the series/elliptic
 * seam, the loop fields against the same references, on-axis limits,
 * reciprocity, finite-difference consistency of (B_r, B_z) with psi, and
 * the degenerate-argument guards. Host-only, runs in milliseconds. */

#include <ablastr/coils/RingKernel.H>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace
{
    int s_failures = 0;

    void check_rel (const char* label, const double value, const double ref,
                    const double rtol)
    {
        const double denom = std::abs(ref) > 0.0 ? std::abs(ref) : 1.0;
        const double rel = std::abs(value - ref) / denom;
        if (!(rel <= rtol)) {
            std::printf("FAIL %s: value %.17e ref %.17e rel %.3e (rtol %.1e)\n",
                        label, value, ref, rel, rtol);
            ++s_failures;
        }
    }

    void check_true (const char* label, const bool ok)
    {
        if (!ok) {
            std::printf("FAIL %s\n", label);
            ++s_failures;
        }
    }
}

int main ()
{
    using namespace ablastr::coils;

    // --- K(m), E(m), f(m) vs mpmath (50 digits), spanning both bracket
    // branches and the near-singular limit ---------------------------------
    struct KEFRef { double m, K, E, f; };
    const KEFRef kef[] = {
        { 1e-12, 1.57079632679528932, 1.57079632679450392, 1.9634954084950934e-25 },
        { 1e-8, 1.57079633072188746, 1.57079632286790579, 1.96349542321983645e-17 },
        { 1e-4, 1.57083559891215224, 1.57075705615038529, 1.96364268215505335e-9 },
        { 0.01, 1.57474556151735595, 1.56686194202166829, 1.97833762017633704e-5 },
        { 0.05, 1.59100345379079218, 1.55097335178047233, 5.10031331100095774e-4 },
        { 0.0999, 1.61239715123544994, 1.53079847791411826, 2.11887123424190872e-3 },
        { 0.1, 1.6124413487202194, 1.5307576368977632, 2.1232887728904517e-3 },
        { 0.1001, 1.61248555170592855, 1.53071679420221858, 2.12771128165649902e-3 },
        { 0.3, 1.71388944817879106, 1.44536306441266526, 2.28859330786142814e-2 },
        { 0.5, 1.85407467730137192, 1.3506438810476755, 7.98242538567068726e-2 },
        { 0.7, 2.07536313529246914, 1.24167056794582275, 2.14630939988564385e-1 },
        { 0.9, 2.57809211334817319, 1.10477473270407333, 6.26351859274843855e-1 },
        { 0.99, 3.69563736298987468, 1.01599354502522394, 1.70060664656932555 },
        // near-1 reference rows are computed from the double-rounded m
        // (K diverges logarithmically, so the representation of 1-m matters)
        { 0.999999, 8.2940514636010622, 1.00000389702617217, 6.29405196360018171 },
        { 0.999999999999, 15.2018159800701203, 1.00000000000735075, 13.2018159800706203 },
    };
    for (const auto& t : kef) {
        double K, E;
        CompleteEllipticIntegrals(t.m, K, E);
        check_rel("K(m)", K, t.K, 1.0e-13);
        check_rel("E(m)", E, t.E, 1.0e-13);
        // On the direct branch, f = (2-m)K - 2E cancels ~3 digits near the
        // branch point, amplifying the double-precision K/E roundoff to the
        // 1e-12..1e-11 relative scale -- the very reason the series branch
        // exists below m = 0.1.
        const double rtol_f = (t.m <= 0.1) ? 1.0e-14 : 2.0e-11;
        check_rel("f(m)", KernelBracket(t.m), t.f, rtol_f);
    }

    // --- series/elliptic seam continuity at the branch point (limited by
    // the direct branch's cancellation, see above) --------------------------
    {
        const double below = KernelBracket(0.1 * (1.0 - 1.0e-12));
        const double above = KernelBracket(0.1 * (1.0 + 1.0e-12));
        check_rel("bracket seam", below, above, 2.0e-11);
    }

    // --- loop fields vs mpmath (unit current) ------------------------------
    struct LoopRef { double r, z, Rc, zc, G0, A, Br, Bz; };
    const LoopRef loops[] = {
        { 0.05, 0.03, 0.25, 0.0, 3.11810104713065699e-9, 6.23620209426131399e-8, 9.38989669617194785e-8, 2.52976923874903681e-6 },
        { 0.2, -0.1, 0.25, 0.0, 3.95957379703425747e-8, 1.97978689851712874e-7, -1.47290317390954989e-6, 1.69168055343079147e-6 },
        { 0.3, 0.02, 0.25, 0.0, 9.47681954280672832e-8, 3.15893984760224277e-7, 1.20695296696043434e-6, -2.19921214245194043e-6 },
        { 0.24, 0.001, 0.25, 0.0, 1.60416597692705717e-7, 6.68402490386273819e-7, 2.01536854715378055e-6, 2.19702577916394839e-5 },
        { 1.5, 2.0, 0.25, 0.1, 3.08845847906909428e-9, 2.05897231937939618e-9, 2.00175385497062157e-9, 1.18588460798536174e-9 },
        { 0.001, 0.0005, 0.25, 0.0, 1.25663706132281802e-12, 1.25663706132281802e-9, 3.01598926623624646e-11, 2.51328920213956628e-6 },
    };
    for (const auto& t : loops) {
        check_rel("G0", RingPsiPerAmp(t.r, t.z, t.Rc, t.zc), t.G0, 1.0e-13);
        check_rel("A_theta", LoopATheta(t.r, t.z, t.Rc, t.zc), t.A, 1.0e-13);
        // the -K + (...)E bracket of B_r cancels near the axis (small m),
        // amplifying roundoff to the 1e-12..1e-11 relative scale
        check_rel("B_r", LoopBr(t.r, t.z, t.Rc, t.zc), t.Br, 1.0e-10);
        check_rel("B_z", LoopBz(t.r, t.z, t.Rc, t.zc), t.Bz, 1.0e-10);
    }

    // --- on-axis limits -----------------------------------------------------
    check_true("A_theta axis", LoopATheta(0.0, 0.1, 0.25, 0.0) == 0.0);
    check_true("B_r axis", LoopBr(0.0, 0.1, 0.25, 0.0) == 0.0);
    check_rel("B_z axis 1", LoopBz(0.0, 0.05, 0.25, 0.0), 2.36968080538670025e-6, 1.0e-14);
    check_rel("B_z axis 2", LoopBz(0.0, 1.0, 0.25, 0.1), 4.81849854583829316e-8, 1.0e-14);

    // --- reciprocity of the ring kernel ------------------------------------
    {
        const double ab = RingPsiPerAmp(0.17, 0.21, 0.29, -0.04);
        const double ba = RingPsiPerAmp(0.29, -0.04, 0.17, 0.21);
        check_rel("reciprocity", ab, ba, 1.0e-15);
    }

    // --- finite-difference consistency: B_r = -(1/r) dpsi/dz,
    //     B_z = (1/r) dpsi/dr at a well-conditioned point ------------------
    {
        const double r = 0.18, z = 0.07, Rc = 0.25, zc = 0.0;
        const double h = 1.0e-6;
        const double dpsi_dz = (RingPsiPerAmp(r, z + h, Rc, zc)
                                - RingPsiPerAmp(r, z - h, Rc, zc)) / (2.0 * h);
        const double dpsi_dr = (RingPsiPerAmp(r + h, z, Rc, zc)
                                - RingPsiPerAmp(r - h, z, Rc, zc)) / (2.0 * h);
        check_rel("B_r vs -dpsi/dz / r", LoopBr(r, z, Rc, zc), -dpsi_dz / r, 1.0e-8);
        check_rel("B_z vs dpsi/dr / r", LoopBz(r, z, Rc, zc), dpsi_dr / r, 1.0e-8);
    }

    // --- far-field dipole scaling: Bz on axis ~ z^-3 -----------------------
    {
        const double b1 = LoopBz(0.0, 10.0, 0.25, 0.0);
        const double b2 = LoopBz(0.0, 20.0, 0.25, 0.0);
        check_rel("far-field z^-3", b1 / b2, 8.0, 1.0e-3);
    }

    // --- guards and the filament divergence --------------------------------
    check_true("rb <= 0 guard", RingPsiPerAmp(0.0, 0.0, 0.25, 0.0) == 0.0);
    check_true("rs <= 0 guard", RingPsiPerAmp(0.25, 0.0, 0.0, 0.0) == 0.0);
    check_true("filament inf", std::isinf(RingPsiPerAmp(0.25, 0.0, 0.25, 0.0)));

    // --- mu0 convention: classical 4e-7*pi exactly --------------------------
    check_rel("mu0_ring", mu0_ring, 4.0e-7 * 3.14159265358979323846, 0.0);

    if (s_failures > 0) {
        std::printf("coils_unit_kernel: %d FAILURES\n", s_failures);
        return EXIT_FAILURE;
    }
    std::printf("coils_unit_kernel: all checks passed\n");
    return EXIT_SUCCESS;
}
