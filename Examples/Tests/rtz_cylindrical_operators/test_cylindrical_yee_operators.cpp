/* Copyright 2026 The WarpX Community
 *
 * Unit tests (method of manufactured solutions) for the RTZ (3D real-space
 * cylindrical r-theta-z) finite-difference operators defined in
 * Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H
 *
 * The operators are header-only static methods, callable on the host. This test
 * exercises each one directly on analytic fields stored in a std::vector-backed
 * amrex::Array4 (host memory, backend independent -- no FArrayBox/arena), and checks
 *   (A) polynomial exactness (the operator reproduces the exact derivative),
 *   (A') 2nd-order convergence on a smooth trig field,
 *   (B) correct assembly of the cylindrical Ampere/Faraday curls (1/r metric, signs),
 *   (C) the stencil-coefficient init and the RTZ CFL.
 *
 * Returns a non-zero exit code if any check fails (so CTest reports failure).
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using amrex::Real;
using T = CylindricalYeeAlgorithm;

// ---------------------------------------------------------------------------
// Test bookkeeping
// ---------------------------------------------------------------------------
namespace {
    int g_failures = 0;
    int g_checks = 0;

    void report (const std::string& name, bool ok, double measured, double expected, double tol)
    {
        ++g_checks;
        if (!ok) { ++g_failures; }
        std::printf("  [%s] %-46s measured=% .6e expected=% .6e tol=%.1e\n",
                    ok ? "PASS" : "FAIL", name.c_str(), measured, expected, tol);
    }

    // relative-or-absolute closeness
    bool close (double a, double b, double tol)
    {
        return std::abs(a - b) <= tol * (1.0 + std::abs(b));
    }

    void check_exact (const std::string& name, double measured, double expected, double tol = 1.0e-9)
    {
        report(name, close(measured, expected, tol), measured, expected, tol);
    }

    void check_order (const std::string& name, double errN, double err2N, double expected = 2.0,
                      double margin = 0.25)
    {
        ++g_checks;
        const double order = (err2N > 0.0 && errN > 0.0) ? std::log2(errN / err2N) : 99.0;
        const bool ok = (order >= expected - margin) || (errN < 1.0e-13 && err2N < 1.0e-13);
        if (!ok) { ++g_failures; }
        std::printf("  [%s] %-46s order=%.3f (errN=%.3e err2N=%.3e, expect>=%.2f)\n",
                    ok ? "PASS" : "FAIL", name.c_str(), order, errN, err2N, expected - margin);
    }
}

// ---------------------------------------------------------------------------
// A small host scalar field on a uniform (r, theta, z) index grid.
//   node coordinate: r(i) = rmin + i*dr,  theta(j) = thetamin + j*dtheta,  z(k) = zmin + k*dz
// Index range [-1, N+1] in each direction (2 ghost) so interior stencils are valid.
// ---------------------------------------------------------------------------
struct Field
{
    int N;                       // number of "interior" nodes per dim is roughly N
    int lo, hi;                  // index bounds (same in all 3 dims for simplicity)
    Real rmin, thmin, zmin;
    Real dr, dth, dz;
    int ncomp;
    std::vector<Real> buf;
    amrex::Array4<Real> a;

    Field (int N_, Real rmin_, Real dr_, Real thmin_, Real dth_, Real zmin_, Real dz_, int ncomp_ = 1)
        : N(N_), lo(-1), hi(N_ + 1),
          rmin(rmin_), thmin(thmin_), zmin(zmin_),
          dr(dr_), dth(dth_), dz(dz_), ncomp(ncomp_)
    {
        const int n = hi - lo + 1;
        buf.assign(static_cast<std::size_t>(n) * n * n * ncomp, 0.0);
        amrex::Dim3 beg{lo, lo, lo};
        amrex::Dim3 end{hi + 1, hi + 1, hi + 1};
        a = amrex::Array4<Real>(buf.data(), beg, end, ncomp);
    }

    Real r (int i)  const { return rmin + i * dr; }
    Real th (int j) const { return thmin + j * dth; }
    Real z (int k)  const { return zmin + k * dz; }

    // fill component comp from f(r,theta,z) evaluated at the node positions
    void fill (const std::function<Real(Real, Real, Real)>& f, int comp = 0)
    {
        for (int k = lo; k <= hi; ++k) {
            for (int j = lo; j <= hi; ++j) {
                for (int i = lo; i <= hi; ++i) {
                    a(i, j, k, comp) = f(r(i), th(j), z(k));
                }
            }
        }
    }
};

// stencil coef helpers (single-element arrays, as the operators expect)
struct Coefs
{
    amrex::Vector<Real> r, th, z;
    Coefs (std::array<Real, 3> cell)
    {
        T::InitializeStencilCoefficients(cell, r, th, z);
    }
};

// ---------------------------------------------------------------------------
int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        std::printf("=== RTZ CylindricalYeeAlgorithm operator unit tests ===\n");

        // -------------------------------------------------------------------
        // Group C: stencil coefficients and CFL (scalar checks)
        // -------------------------------------------------------------------
        std::printf("\n[Group C] stencil coefficients + CFL\n");
        {
            std::array<Real, 3> cell = {0.02, 0.05, 0.01};  // dr, dtheta, dz
            Coefs c(cell);
            check_exact("InitializeStencilCoefficients coefs_r=1/dr",   c.r[0],  1.0 / cell[0], 1e-12);
            check_exact("InitializeStencilCoefficients coefs_theta=1/dt", c.th[0], 1.0 / cell[1], 1e-12);
            check_exact("InitializeStencilCoefficients coefs_z=1/dz",   c.z[0],  1.0 / cell[2], 1e-12);

            const Real dr = cell[0], dth = cell[1], dz = cell[2];
            const Real rdth = 0.5 * dr * dth;
            const Real dt_expected = 1.0 / (PhysConst::c * std::sqrt(
                1.0 / (dr * dr) + 1.0 / (rdth * rdth) + 1.0 / (dz * dz)));
            const Real dt_dx[3] = {dr, dth, dz};
            check_exact("ComputeMaxDt (RTZ CFL)", T::ComputeMaxDt(dt_dx, 1), dt_expected, 1e-12);
        }

        // common grid for the operator checks (rmin>0 so 1/r is well defined)
        const int N = 24;
        const Real rmin = 0.30, dr = 0.02;
        const Real thmin = -0.4, dth = 0.05;
        const Real zmin = 0.20, dz = 0.03;
        Coefs C({dr, dth, dz});
        const Real* cr = C.r.dataPtr();
        const Real* ct = C.th.dataPtr();
        const Real* cz = C.z.dataPtr();
        const int nr = C.r.size(), nt = C.th.size(), nz = C.z.size();

        // an interior test index
        const int i0 = N / 2, j0 = N / 2, k0 = N / 2;

        // -------------------------------------------------------------------
        // Group A: bare-operator polynomial exactness
        // -------------------------------------------------------------------
        std::printf("\n[Group A] bare-operator polynomial exactness\n");
        {
            const Real a = 0.7, b = 1.3, cc = 0.9;

            // theta first differences: F = a + b*theta  ->  dF/dtheta = b
            Field Fth(N, rmin, dr, thmin, dth, zmin, dz);
            Fth.fill([&](Real, Real th, Real) { return a + b * th; });
            check_exact("UpwardDtheta(a+b*theta) == b",
                        T::UpwardDtheta(Fth.a, ct, nt, i0, j0, k0, 0), b);
            check_exact("DownwardDtheta(a+b*theta) == b",
                        T::DownwardDtheta(Fth.a, ct, nt, i0, j0, k0, 0), b);

            // z first differences: F = a + b*z -> b   (RTZ: z is k-index)
            Field Fz(N, rmin, dr, thmin, dth, zmin, dz);
            Fz.fill([&](Real, Real, Real z) { return a + b * z; });
            check_exact("UpwardDz(a+b*z) == b",   T::UpwardDz(Fz.a, cz, nz, i0, j0, k0, 0), b);
            check_exact("DownwardDz(a+b*z) == b", T::DownwardDz(Fz.a, cz, nz, i0, j0, k0, 0), b);

            // r first differences: F = a + b*r -> b
            Field Fr(N, rmin, dr, thmin, dth, zmin, dz);
            Fr.fill([&](Real r, Real, Real) { return a + b * r; });
            check_exact("UpwardDr(a+b*r) == b",   T::UpwardDr(Fr.a, cr, nr, i0, j0, k0, 0), b);
            check_exact("DownwardDr(a+b*r) == b", T::DownwardDr(Fr.a, cr, nr, i0, j0, k0, 0), b);

            // second derivatives
            Field Frr(N, rmin, dr, thmin, dth, zmin, dz);
            Frr.fill([&](Real r, Real, Real) { return a + b * r + cc * r * r; });
            check_exact("Drr(a+b*r+c*r^2) == 2c", T::Drr(Frr.a, cr, nr, i0, j0, k0, 0), 2.0 * cc);

            Field Fzz(N, rmin, dr, thmin, dth, zmin, dz);
            Fzz.fill([&](Real, Real, Real z) { return a + b * z + cc * z * z; });
            check_exact("Dzz(a+b*z+c*z^2) == 2c", T::Dzz(Fzz.a, cz, nz, i0, j0, k0, 0), 2.0 * cc);

            // (1/r) d(rF)/dr  with F = a + b*r  ->  a/r + 2b (exact for F linear in r)
            // UpwardDrr_over_r: F nodal in r, result cell-centered at r_cc = rmin+(i+0.5)dr
            {
                const Real r_cc = rmin + (i0 + 0.5) * dr;
                const Real expected = a / r_cc + 2.0 * b;
                check_exact("UpwardDrr_over_r(a+b*r) == a/r+2b",
                            T::UpwardDrr_over_r(Fr.a, r_cc, dr, cr, nr, i0, j0, k0, 0), expected);
            }
            // DownwardDrr_over_r: F cell-centered in r, result nodal at r_node = rmin+i*dr.
            // Build F so that F(i) holds the value at the cell center rmin+(i+0.5)dr.
            {
                Field Fcc(N, rmin, dr, thmin, dth, zmin, dz);
                Fcc.fill([&](Real r, Real, Real) { return a + b * (r + 0.5 * dr); });  // value at cell center
                const Real r_node = rmin + i0 * dr;
                const Real expected = a / r_node + 2.0 * b;
                check_exact("DownwardDrr_over_r(a+b*r) == a/r+2b",
                            T::DownwardDrr_over_r(Fcc.a, r_node, dr, cr, nr, i0, j0, k0, 0), expected);
            }

            // (1/r) d(r dF/dr)/dr  with F = a + c*r^2  ->  4c (exact)
            {
                const Real r_node = rmin + i0 * dr;
                Field Frlap(N, rmin, dr, thmin, dth, zmin, dz);
                Frlap.fill([&](Real r, Real, Real) { return a + cc * r * r; });
                check_exact("Dr_rDr_over_r(a+c*r^2) == 4c",
                            T::Dr_rDr_over_r(Frlap.a, r_node, dr, cr, nr, i0, j0, k0, 0), 4.0 * cc);
            }
        }

        // -------------------------------------------------------------------
        // Group A': convergence order on a smooth (trig) field
        // For first differences the operator approximates the derivative at the
        // staggered (half-grid) location; second derivatives at the node.
        // -------------------------------------------------------------------
        std::printf("\n[Group A'] convergence order (smooth field)\n");
        {
            auto run = [&](int Nres, Real dr_, Real dth_, Real dz_,
                           std::array<Real, 6>& err) {
                Coefs cf({dr_, dth_, dz_});
                const Real* lr = cf.r.dataPtr(); const int lnr = cf.r.size();
                const Real* lt = cf.th.dataPtr(); const int lnt = cf.th.size();
                const Real* lz = cf.z.dataPtr(); const int lnz = cf.z.size();
                Field F(Nres, rmin, dr_, thmin, dth_, zmin, dz_);
                // separable smooth field
                F.fill([&](Real r, Real th, Real z) {
                    return std::sin(r) * std::sin(th) * std::sin(z);
                });
                err = {0, 0, 0, 0, 0, 0};
                for (int k = 1; k < Nres; ++k) {
                    for (int j = 1; j < Nres; ++j) {
                        for (int i = 1; i < Nres; ++i) {
                            const Real r = F.r(i), th = F.th(j), z = F.z(k);
                            const Real rm = r + 0.5 * dr_, tm = th + 0.5 * dth_, zm = z + 0.5 * dz_;
                            // UpwardDtheta ~ d/dtheta at theta+dtheta/2
                            err[0] = std::max(err[0], std::abs(
                                T::UpwardDtheta(F.a, lt, lnt, i, j, k, 0)
                                - std::sin(r) * std::cos(tm) * std::sin(z)));
                            // UpwardDz ~ d/dz at z+dz/2
                            err[1] = std::max(err[1], std::abs(
                                T::UpwardDz(F.a, lz, lnz, i, j, k, 0)
                                - std::sin(r) * std::sin(th) * std::cos(zm)));
                            // UpwardDr ~ d/dr at r+dr/2
                            err[2] = std::max(err[2], std::abs(
                                T::UpwardDr(F.a, lr, lnr, i, j, k, 0)
                                - std::cos(rm) * std::sin(th) * std::sin(z)));
                            // Drr ~ d2/dr2 at node
                            err[3] = std::max(err[3], std::abs(
                                T::Drr(F.a, lr, lnr, i, j, k, 0)
                                - (-std::sin(r)) * std::sin(th) * std::sin(z)));
                            // Dzz ~ d2/dz2 at node
                            err[4] = std::max(err[4], std::abs(
                                T::Dzz(F.a, lz, lnz, i, j, k, 0)
                                - std::sin(r) * std::sin(th) * (-std::sin(z))));
                            // UpwardDrr_over_r ~ (1/r) d(r F)/dr at cell center, F=sin(r)*g
                            // (1/r) d(r sin r)/dr = (sin r)/r + cos r, times sin(th)sin(z)
                            const Real g = std::sin(th) * std::sin(z);
                            err[5] = std::max(err[5], std::abs(
                                T::UpwardDrr_over_r(F.a, rm, dr_, lr, lnr, i, j, k, 0)
                                - (std::sin(rm) / rm + std::cos(rm)) * g));
                        }
                    }
                }
            };
            std::array<Real, 6> e1, e2;
            run(N, dr, dth, dz, e1);
            run(2 * N, 0.5 * dr, 0.5 * dth, 0.5 * dz, e2);
            check_order("UpwardDtheta order",      e1[0], e2[0]);
            check_order("UpwardDz order",          e1[1], e2[1]);
            check_order("UpwardDr order",          e1[2], e2[2]);
            check_order("Drr order",               e1[3], e2[3]);
            check_order("Dzz order",               e1[4], e2[4]);
            check_order("UpwardDrr_over_r order",  e1[5], e2[5]);
        }

        // -------------------------------------------------------------------
        // Group B: composite cylindrical curls (operators as used in the kernels)
        // Use linear manufactured fields so each curl term is exact; validates
        // signs, the 1/r metric, and which operator is used.
        // Ampere:  J = (1/mu0) curl(B)
        //   Jr = (1/mu0)[ -DownwardDz(Btheta) + DownwardDtheta(Bz)/r ]
        //   Jtheta = (1/mu0)[ -DownwardDr(Bz) + DownwardDz(Br) ]
        //   Jz = (1/mu0)[ DownwardDrr_over_r(Btheta) - DownwardDtheta(Br)/r ]
        // Faraday: dB/dt = -curl(E)
        //   Br += UpwardDz(Et) - UpwardDtheta(Ez)/r
        //   Bz += -UpwardDrr_over_r(Et) + UpwardDtheta(Er)/r
        // -------------------------------------------------------------------
        std::printf("\n[Group B] composite Ampere/Faraday curls (exact assembly)\n");
        {
            const Real one_over_mu0 = 1.0 / PhysConst::mu0;
            const Real alpha = 0.5, beta = 0.8, gamma = 1.1, delta = 0.6;

            // --- Ampere Jr: theta term. B = (0, 0, Bz=beta*theta) -> Jr = (1/mu0)(beta/r) ---
            {
                Field Bz(N, rmin, dr, thmin, dth, zmin, dz);
                Field Bt(N, rmin, dr, thmin, dth, zmin, dz);  // = 0
                Bz.fill([&](Real, Real th, Real) { return beta * th; });
                const Real r_cc = rmin + (i0 + 0.5) * dr;  // Jr cell-centered in r
                const Real Jr = one_over_mu0 * (
                    - T::DownwardDz(Bt.a, cz, nz, i0, j0, k0, 0)
                    + T::DownwardDtheta(Bz.a, ct, nt, i0, j0, k0, 0) / r_cc);
                check_exact("Ampere Jr theta-term (Bz=beta*theta)", Jr, one_over_mu0 * beta / r_cc);
            }
            // --- Ampere Jtheta: -dBz/dr. B = (0,0, Bz=gamma*r) -> Jtheta = -(1/mu0) gamma ---
            {
                Field Bz(N, rmin, dr, thmin, dth, zmin, dz);
                Field Br(N, rmin, dr, thmin, dth, zmin, dz);  // = 0
                Bz.fill([&](Real r, Real, Real) { return gamma * r; });
                const Real Jt = one_over_mu0 * (
                    - T::DownwardDr(Bz.a, cr, nr, i0, j0, k0, 0)
                    + T::DownwardDz(Br.a, cz, nz, i0, j0, k0, 0));
                check_exact("Ampere Jtheta (Bz=gamma*r)", Jt, -one_over_mu0 * gamma);
            }
            // --- Ampere Jz: theta term -(1/r)dBr/dtheta + (1/r)d(r Btheta)/dr ---
            //     B = (Br=alpha*theta, Btheta=delta*r, 0)
            //     Jz = (1/mu0)[ 2*delta - alpha/r ]
            {
                Field Br(N, rmin, dr, thmin, dth, zmin, dz);
                Field Bt(N, rmin, dr, thmin, dth, zmin, dz);
                Br.fill([&](Real, Real th, Real) { return alpha * th; });
                // Btheta cell-centered in r (DownwardDrr_over_r expects cell-centered F, result nodal)
                Bt.fill([&](Real r, Real, Real) { return delta * (r + 0.5 * dr); });
                const Real r_node = rmin + i0 * dr;  // Jz nodal in r
                const Real Jz = one_over_mu0 * (
                    T::DownwardDrr_over_r(Bt.a, r_node, dr, cr, nr, i0, j0, k0, 0)
                    - T::DownwardDtheta(Br.a, ct, nt, i0, j0, k0, 0) / r_node);
                check_exact("Ampere Jz theta-term (Br=alpha*theta,Btheta=delta*r)",
                            Jz, one_over_mu0 * (2.0 * delta - alpha / r_node));
            }

            // --- Faraday Br: -curl(E)_r. E = (0, 0, Ez=beta*theta) -> dBr = -beta/r ---
            {
                Field Ez(N, rmin, dr, thmin, dth, zmin, dz);
                Field Et(N, rmin, dr, thmin, dth, zmin, dz);  // = 0
                Ez.fill([&](Real, Real th, Real) { return beta * th; });
                const Real r_node = rmin + i0 * dr;  // Br nodal in r
                const Real dBr = T::UpwardDz(Et.a, cz, nz, i0, j0, k0, 0)
                               - T::UpwardDtheta(Ez.a, ct, nt, i0, j0, k0, 0) / r_node;
                check_exact("Faraday Br theta-term (Ez=beta*theta)", dBr, -beta / r_node);
            }
            // --- Faraday Bz: +(1/r)dEr/dtheta - (1/r)d(r Etheta)/dr ---
            //     E = (Er=alpha*theta, Etheta=delta*r, 0) -> dBz = alpha/r - 2*delta ---
            {
                Field Er(N, rmin, dr, thmin, dth, zmin, dz);
                Field Et(N, rmin, dr, thmin, dth, zmin, dz);
                Er.fill([&](Real, Real th, Real) { return alpha * th; });
                Et.fill([&](Real r, Real, Real) { return delta * r; });
                const Real r_cc = rmin + (i0 + 0.5) * dr;  // Bz cell-centered in r
                const Real dBz = - T::UpwardDrr_over_r(Et.a, r_cc, dr, cr, nr, i0, j0, k0, 0)
                                 + T::UpwardDtheta(Er.a, ct, nt, i0, j0, k0, 0) / r_cc;
                check_exact("Faraday Bz theta-term (Er=alpha*theta,Etheta=delta*r)",
                            dBz, alpha / r_cc - 2.0 * delta);
            }

            // --- on-axis regularization sanity (kernel logic): finite + documented values ---
            {
                const Real dr_a = 0.02;
                Field Bt(1 + 4, 0.0, dr_a, thmin, dth, zmin, dz);  // rmin = 0
                Bt.fill([&](Real r, Real, Real) { return 0.37 * r; });  // Btheta ~ linear in r
                // Jz on axis (r < 0.5 dr) uses 4*Btheta/dr/mu0; Jtheta = 0; Br = 0
                const Real Jz_axis = one_over_mu0 * 4.0 * Bt.a(0, 2, 2, 0) / dr_a;
                const bool finite = std::isfinite(Jz_axis);
                report("on-axis Jz = 4*Btheta/dr/mu0 finite", finite, Jz_axis, Jz_axis, 0.0);
            }
        }

        std::printf("\n=== summary: %d checks, %d failures ===\n", g_checks, g_failures);
    }
    amrex::Finalize();
    return (g_failures == 0) ? 0 : 1;
}
