/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_LOOPINDUCTANCE_H_
#define WARPX_CIRCUIT_LOOPINDUCTANCE_H_

#include "YeeLoopKernel.H"

#include <cmath>
#include <vector>

/** \file LoopInductance.H
 *
 * Host-side, setup-time evaluation of the DISCRETE (mesh-realized) loop
 * inductances on the global RZ index space: the quarter-offset filament's
 * Yee-sampled A_theta, its conservative discrete curl, the staircase disk
 * flux, and the resulting self/mutual inductances. Deliberately independent
 * of the box decomposition (plain arrays over the global grid) so the
 * result is identical for any rank count.
 *
 * The discrete self-inductance is RESOLUTION DEPENDENT (grows ~ln(1/dr)
 * under refinement) by construction: it is the flux the mesh actually
 * links, and therefore the value a circuit port coupled to fields on this
 * mesh must use ("discrete-L" rule). Never mix it with continuum
 * inductances in one matrix. The disk-flux rules (strict r_cell < r_disk
 * cell test, unclamped linear z interpolation) are shared verbatim with
 * the runtime flux probes -- probe and port self-inductance must use the
 * identical rule or the coupled system's passivity is violated.
 */
namespace warpx::circuit
{
    /** Global RZ Yee grid extents (axis at r = 0). */
    struct LoopGridRZ
    {
        int nr = 0;            //!< number of cells in r
        int nz = 0;            //!< number of cells in z
        double r_max = 0.0;    //!< outer radius [m] (r_lo = 0)
        double z_min = 0.0;    //!< lower z [m]
        double z_max = 0.0;    //!< upper z [m]

        [[nodiscard]] double dr () const { return r_max / nr; }
        [[nodiscard]] double dz () const { return (z_max - z_min) / nz; }
    };

    /** Diagnostics of a staircase disk-flux evaluation (the quantities the
     * discrete-convention reference pins assert). */
    struct DiskFluxInfo
    {
        double flux = 0.0;     //!< full-loop flux [Wb]
        int n_disk_cells = 0;  //!< cells passing the strict r_cell < r_disk test
        double z_weight = 0.0; //!< linear interpolation weight toward j0+1
    };

    /** Yee-sampled A_theta [T m] per unit current of the quarter-offset
     * filament of a coil at (coil_r, coil_z), on the (node r, node z) mesh;
     * row-major a[i*(nz+1) + j], exactly 0 on the axis. */
    inline std::vector<double>
    SampleLoopAThetaNodes (const double coil_r, const double coil_z,
                           const LoopGridRZ& grid)
    {
        double r_off, z_off;
        QuarterOffset(coil_r, coil_z, grid.dr(), grid.dz(), r_off, z_off);
        std::vector<double> a(static_cast<std::size_t>(grid.nr + 1) * (grid.nz + 1));
        for (int i = 0; i <= grid.nr; ++i) {
            const double r = i * grid.dr();
            for (int j = 0; j <= grid.nz; ++j) {
                const double z = grid.z_min + j * grid.dz();
                a[static_cast<std::size_t>(i) * (grid.nz + 1) + j] =
                    YeeLoopATheta(r, z, r_off, z_off);
            }
        }
        return a;
    }

    /** Conservative discrete RZ Yee curl, z component, of nodal A_theta:
     *     B_z(i+1/2, j) = (r_{i+1} A[i+1,j] - r_i A[i,j]) / (r_{i+1/2} dr),
     * on (cell r, node z), row-major bz[i*(nz+1) + j]. Exactness identity:
     * A = 0.5*B0*r gives B_z == B0 to roundoff because
     * r_{i+1}^2 - r_i^2 = 2 r_{i+1/2} dr. */
    inline std::vector<double>
    CurlABz (const std::vector<double>& a, const LoopGridRZ& grid)
    {
        std::vector<double> bz(static_cast<std::size_t>(grid.nr) * (grid.nz + 1));
        for (int i = 0; i < grid.nr; ++i) {
            const double r_lo = i * grid.dr();
            const double r_hi = (i + 1) * grid.dr();
            const double r_c = (i + 0.5) * grid.dr();
            for (int j = 0; j <= grid.nz; ++j) {
                const double a_lo = a[static_cast<std::size_t>(i) * (grid.nz + 1) + j];
                const double a_hi = a[static_cast<std::size_t>(i + 1) * (grid.nz + 1) + j];
                bz[static_cast<std::size_t>(i) * (grid.nz + 1) + j] =
                    (r_hi * a_hi - r_lo * a_lo) / (r_c * grid.dr());
            }
        }
        return bz;
    }

    /** Staircase disk flux of a (cell r, node z) B_z field through the
     * circle (r_disk, z_disk):
     *     Phi = 2 pi sum_{r_cell < r_disk} B_z(r_cell, z_disk) r_cell dr,
     * with B_z linearly interpolated in z between the two bracketing node
     * planes (weight NOT clamped: z_disk outside the mesh extrapolates,
     * deliberately) and a STRICT cell-center radius test (the quarter
     * offset guarantees no r_cell ever ties r_disk). The single 2*pi of
     * the flux normalization lives here. */
    inline DiskFluxInfo
    DiskFluxBz (const std::vector<double>& bz, const LoopGridRZ& grid,
                const double r_disk, const double z_disk)
    {
        DiskFluxInfo info;
        int j0 = static_cast<int>(std::floor((z_disk - grid.z_min) / grid.dz()));
        if (j0 < 0) { j0 = 0; }
        if (j0 > grid.nz - 1) { j0 = grid.nz - 1; }
        const double z_j0 = grid.z_min + j0 * grid.dz();
        const double w = (z_disk - z_j0) / grid.dz();
        info.z_weight = w;
        double flux = 0.0;
        for (int i = 0; i < grid.nr; ++i) {
            const double r_c = (i + 0.5) * grid.dr();
            if (!(r_c < r_disk)) { continue; }
            ++info.n_disk_cells;
            const double b_lo = bz[static_cast<std::size_t>(i) * (grid.nz + 1) + j0];
            const double b_hi = bz[static_cast<std::size_t>(i) * (grid.nz + 1) + j0 + 1];
            flux += ((1.0 - w) * b_lo + w * b_hi) * r_c;
        }
        info.flux = 2.0 * ablastr::coils::pi_ring * flux * grid.dr();
        return info;
    }

    /** Discrete (mesh-regularized) self-inductance of a coil: the disk flux
     * of its own unit-current Yee-sampled field through its own offset
     * circle, times n_turns^2 [H]. Resolution dependent by construction. */
    inline double
    DiscreteSelfInductance (const double coil_r, const double coil_z,
                            const double n_turns, const LoopGridRZ& grid,
                            DiskFluxInfo* info = nullptr)
    {
        double r_off, z_off;
        QuarterOffset(coil_r, coil_z, grid.dr(), grid.dz(), r_off, z_off);
        const auto a = SampleLoopAThetaNodes(coil_r, coil_z, grid);
        const auto bz = CurlABz(a, grid);
        const DiskFluxInfo f = DiskFluxBz(bz, grid, r_off, z_off);
        if (info != nullptr) { *info = f; }
        return f.flux * n_turns * n_turns;
    }

    /** Discrete mutual inductance: the disk flux of the SOURCE coil's
     * unit-current Yee-sampled field through the TARGET coil's offset
     * circle, times the two turn counts [H]. Not exactly symmetric (the
     * offset and staircase policies are first order in the mesh spacing);
     * callers build both orientations and symmetrize, recording the
     * asymmetry. */
    inline double
    DiscreteMutualInductance (const double src_r, const double src_z,
                              const double src_turns,
                              const double tgt_r, const double tgt_z,
                              const double tgt_turns,
                              const LoopGridRZ& grid)
    {
        const auto a = SampleLoopAThetaNodes(src_r, src_z, grid);
        const auto bz = CurlABz(a, grid);
        double r_off, z_off;
        QuarterOffset(tgt_r, tgt_z, grid.dr(), grid.dz(), r_off, z_off);
        const DiskFluxInfo f = DiskFluxBz(bz, grid, r_off, z_off);
        return f.flux * src_turns * tgt_turns;
    }

} // namespace warpx::circuit

#endif // WARPX_CIRCUIT_LOOPINDUCTANCE_H_
