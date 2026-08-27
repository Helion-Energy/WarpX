/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

/* Unit test of the DISCRETE (Yee-convention) loop-inductance layer in
 * Circuit/Coils/LoopInductance.H, freezing the convention:
 *  - the reference pin of the mesh-regularized self-inductance of a single
 *    loop (R = 0.25 m on a 224x512 grid, r_max = 0.35 m, z in [-2, 2] m):
 *    L = 1.4596352993e-6 H, with the exact 0.25 z-interpolation weight and
 *    the 160-cell staircase disk;
 *  - the discrete-curl exactness identity (uniform solenoid);
 *  - n_turns^2 scaling;
 *  - the ~ln(1/dr) growth of the discrete self-inductance under refinement
 *    (an operator property of the convention, not a bug);
 *  - first-order convergence of the discrete mutual inductance to the
 *    continuum (Maxwell) coaxial mutual 2*pi*G0. */

#include <Circuit/Coils/LoopInductance.H>

#include <algorithm>
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
    using namespace warpx::circuit;

    const double R_coil = 0.25;   // m
    const double Z_coil = 0.0;    // m

    // --- the discrete-L reference pin ---------------------------------------
    {
        const LoopGridRZ grid{224, 512, 0.35, -2.0, 2.0};
        DiskFluxInfo info;
        const double L = DiscreteSelfInductance(R_coil, Z_coil, 1.0, grid, &info);
        check_rel("L_disc pin", L, 1.4596352993e-6, 1.0e-6);
        check_true("z-weight exactly 0.25", info.z_weight == 0.25);
        check_true("160-cell disk", info.n_disk_cells == 160);
        std::printf("L_disc(224x512) = %.10e H (%d disk cells, w = %.3f)\n",
                    L, info.n_disk_cells, info.z_weight);
    }

    // --- discrete-curl exactness: A = 0.5*B0*r  =>  B_z == B0 --------------
    {
        const LoopGridRZ grid{56, 64, 0.35, -0.5, 0.5};
        const double B0 = 0.37;
        std::vector<double> a(static_cast<std::size_t>(grid.nr + 1) * (grid.nz + 1));
        for (int i = 0; i <= grid.nr; ++i) {
            for (int j = 0; j <= grid.nz; ++j) {
                a[static_cast<std::size_t>(i) * (grid.nz + 1) + j] =
                    0.5 * B0 * i * grid.dr();
            }
        }
        const auto bz = CurlABz(a, grid);
        double max_rel = 0.0;
        for (const double b : bz) {
            max_rel = std::max(max_rel, std::abs(b - B0) / B0);
        }
        // exact identity up to the rounding of r_hi^2 - r_lo^2, which
        // accumulates as ~eps * i / 2 across the radial index
        std::printf("solenoid curl identity max rel = %.3e\n", max_rel);
        check_true("solenoid curl exactness <= 1e-13", max_rel <= 1.0e-13);
    }

    // --- n_turns^2 scaling ---------------------------------------------------
    {
        const LoopGridRZ grid{112, 256, 0.35, -2.0, 2.0};
        const double L1 = DiscreteSelfInductance(R_coil, Z_coil, 1.0, grid);
        const double L3 = DiscreteSelfInductance(R_coil, Z_coil, 3.0, grid);
        check_rel("n_turns^2 scaling", L3, 9.0 * L1, 1.0e-14);
    }

    // --- ln(1/dr) growth of the discrete self-inductance --------------------
    // Doubling the resolution adds ~ mu0 * R * c1 * ln(2) with c1 ~ 0.99
    // (the fitted divergence slope of the convention).
    {
        const LoopGridRZ coarse{112, 256, 0.35, -2.0, 2.0};
        const LoopGridRZ fine{224, 512, 0.35, -2.0, 2.0};
        const double L_c = DiscreteSelfInductance(R_coil, Z_coil, 1.0, coarse);
        const double L_f = DiscreteSelfInductance(R_coil, Z_coil, 1.0, fine);
        check_true("L grows under refinement", L_f > L_c);
        const double c1 = (L_f - L_c)
            / (ablastr::coils::mu0_ring * R_coil * std::log(2.0));
        check_rel("log-divergence slope", c1, 0.99, 0.1);
        std::printf("two-point divergence slope c1 = %.4f\n", c1);
    }

    // --- discrete mutual -> Maxwell mutual at first order -------------------
    {
        const double R2 = 0.25, Z2 = 0.1;
        const double M_maxwell = 2.0 * ablastr::coils::pi_ring
            * ablastr::coils::RingPsiPerAmp(R_coil, Z_coil, R2, Z2);

        auto sym_mutual = [&](const LoopGridRZ& grid) {
            const double m_ij = DiscreteMutualInductance(
                R_coil, Z_coil, 1.0, R2, Z2, 1.0, grid);
            const double m_ji = DiscreteMutualInductance(
                R2, Z2, 1.0, R_coil, Z_coil, 1.0, grid);
            check_rel("mutual asymmetry", m_ij, m_ji, 5.0e-2);
            return 0.5 * (m_ij + m_ji);
        };

        const double err_c = std::abs(
            sym_mutual(LoopGridRZ{112, 256, 0.35, -2.0, 2.0}) - M_maxwell);
        const double err_m = std::abs(
            sym_mutual(LoopGridRZ{224, 512, 0.35, -2.0, 2.0}) - M_maxwell);
        const double err_f = std::abs(
            sym_mutual(LoopGridRZ{448, 1024, 0.35, -2.0, 2.0}) - M_maxwell);
        const double ratio_cm = err_c / err_m;
        const double ratio_mf = err_m / err_f;
        std::printf("yee-mutual error ratios per refinement = %.3f, %.3f "
                    "(M_maxwell = %.6e H)\n", ratio_cm, ratio_mf, M_maxwell);
        // The offset/staircase policies are first order in the mesh
        // spacing; the pre-asymptotic coarse pair may overshoot, the finer
        // pair must sit near a ratio of 2.
        check_true("mutual converging", ratio_cm > 1.2);
        check_true("mutual first order on the fine pair",
                   ratio_mf > 1.5 && ratio_mf < 2.6);
    }

    if (s_failures > 0) {
        std::printf("coils_unit_L_disc: %d FAILURES\n", s_failures);
        return EXIT_FAILURE;
    }
    std::printf("coils_unit_L_disc: all checks passed\n");
    return EXIT_SUCCESS;
}
