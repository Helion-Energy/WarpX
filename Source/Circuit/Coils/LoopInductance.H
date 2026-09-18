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

    // ----------------------------------------------------------------------
    // 3D Cartesian: the same discrete-L rule on the 3D Yee mesh.
    //
    // The 3D unit field (CoilFieldSolver.cpp, 3D branch) samples the loop
    // A of the filament at its NOMINAL position -- the quarter offset is
    // an RZ-mesh policy -- as A_x = -A_theta y/rho, A_y = A_theta x/rho on
    // the Yee edge staggerings, and the hybrid solver's order-2 Yee curl
    // (CartesianYeeAlgorithm) makes B_z on (cell x, cell y, node z). The
    // discrete self-inductance is the staircase disk flux of that B_z
    // through the coil's nominal circle (STRICT squared cell-center radius
    // test x_c^2 + y_c^2 < r^2, unclamped linear z interpolation between
    // the bracketing node planes) -- the rule the 3D DiskFluxLinkage probe
    // uses verbatim, so probe and port share one convention (passivity).
    // Only the two bracketing node planes are ever sampled: B_z on a node
    // plane needs A_x, A_y on that plane only, so the table costs
    // O(n_coils^2 nx ny) kernel evaluations, not a volume. Like the RZ
    // table, a circle reaching outside the domain is truncated to the
    // domain, and the value is resolution dependent by construction.
    // ----------------------------------------------------------------------

    /** Global 3D Yee grid extents (Cartesian; the machine axis is the line
     * x = y = 0 in absolute coordinates, as in the unit-field painting). */
    struct LoopGrid3D
    {
        int nx = 0;             //!< number of cells in x
        int ny = 0;             //!< number of cells in y
        int nz = 0;             //!< number of cells in z
        double x_min = 0.0;     //!< lower x [m]
        double y_min = 0.0;     //!< lower y [m]
        double z_min = 0.0;     //!< lower z [m]
        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
    };

    /** The two z node planes bracketing z_disk and the (unclamped) linear
     * weight toward the upper one: k0 = floor((z - z_min)/dz) clamped to
     * [0, nz - 1], w = (z - z_k0)/dz -- the RZ disk rule in z. */
    inline void
    DiskPlanes3D (const LoopGrid3D& grid, const double z_disk,
                  int& k0, double& w)
    {
        k0 = static_cast<int>(std::floor((z_disk - grid.z_min) / grid.dz));
        if (k0 < 0) { k0 = 0; }
        if (k0 > grid.nz - 1) { k0 = grid.nz - 1; }
        w = (z_disk - (grid.z_min + k0 * grid.dz)) / grid.dz;
    }

    /** Yee-sampled A_x, A_y [T m] per unit current of the loop at its
     * nominal (coil_r, coil_z) on the node plane k (z = z_min + k dz), in
     * the exact arithmetic of the 3D unit-field painting:
     *   ax[i*(ny+1) + j] at (x_{i+1/2}, y_j),   i in [0, nx), j in [0, ny]
     *   ay[i*ny + j]     at (x_i, y_{j+1/2}),   i in [0, nx], j in [0, ny)
     * with A = 0 on the axis (rho <= 0). */
    inline void
    SampleLoopAPlane3D (const double coil_r, const double coil_z,
                        const LoopGrid3D& grid, const int k,
                        std::vector<double>& ax, std::vector<double>& ay)
    {
        const double z = grid.z_min + (k + 0.0) * grid.dz;
        ax.assign(static_cast<std::size_t>(grid.nx) * (grid.ny + 1), 0.0);
        ay.assign(static_cast<std::size_t>(grid.nx + 1) * grid.ny, 0.0);
        for (int i = 0; i < grid.nx; ++i) {
            const double x = grid.x_min + (i + 0.5) * grid.dx;
            for (int j = 0; j <= grid.ny; ++j) {
                const double y = grid.y_min + (j + 0.0) * grid.dy;
                const double rho = std::sqrt(x * x + y * y);
                if (rho <= 0.0) { continue; }
                const double a_theta = YeeLoopATheta(rho, z, coil_r, coil_z);
                ax[static_cast<std::size_t>(i) * (grid.ny + 1) + j] =
                    -a_theta * y / rho;
            }
        }
        for (int i = 0; i <= grid.nx; ++i) {
            const double x = grid.x_min + (i + 0.0) * grid.dx;
            for (int j = 0; j < grid.ny; ++j) {
                const double y = grid.y_min + (j + 0.5) * grid.dy;
                const double rho = std::sqrt(x * x + y * y);
                if (rho <= 0.0) { continue; }
                const double a_theta = YeeLoopATheta(rho, z, coil_r, coil_z);
                ay[static_cast<std::size_t>(i) * grid.ny + j] =
                    a_theta * x / rho;
            }
        }
    }

    /** Order-2 Yee curl, z component, of the plane samples (the
     * CartesianYeeAlgorithm stencil of the hybrid solver's ComputeCurlA:
     * B_z = -UpwardDy(A_x) + UpwardDx(A_y)), on (cell x, cell y):
     *   bz[i*ny + j] = -(ax[i, j+1] - ax[i, j]) / dy + (ay[i+1, j] - ay[i, j]) / dx. */
    inline std::vector<double>
    CurlABzPlane3D (const std::vector<double>& ax, const std::vector<double>& ay,
                    const LoopGrid3D& grid)
    {
        const double inv_dx = 1.0 / grid.dx;
        const double inv_dy = 1.0 / grid.dy;
        std::vector<double> bz(static_cast<std::size_t>(grid.nx) * grid.ny);
        for (int i = 0; i < grid.nx; ++i) {
            for (int j = 0; j < grid.ny; ++j) {
                const double ax_lo = ax[static_cast<std::size_t>(i) * (grid.ny + 1) + j];
                const double ax_hi = ax[static_cast<std::size_t>(i) * (grid.ny + 1) + j + 1];
                const double ay_lo = ay[static_cast<std::size_t>(i) * grid.ny + j];
                const double ay_hi = ay[static_cast<std::size_t>(i + 1) * grid.ny + j];
                bz[static_cast<std::size_t>(i) * grid.ny + j] =
                    -((ax_hi - ax_lo) * inv_dy) + ((ay_hi - ay_lo) * inv_dx);
            }
        }
        return bz;
    }

    /** Staircase disk flux through the circle (r_disk, z_disk) of a B_z
     * given on its two bracketing node planes (k0, k0 + 1) with the linear
     * weight w toward the upper one:
     *     Phi = dx dy Sum_{x_c^2 + y_c^2 < r_disk^2} ((1 - w) bz0 + w bz1),
     * STRICT squared cell-center test (the identical expression as the 3D
     * disk probe; a tie is excluded in both). */
    inline DiskFluxInfo
    DiskFluxBzPlanes3D (const std::vector<double>& bz0,
                        const std::vector<double>& bz1, const double w,
                        const LoopGrid3D& grid, const double r_disk)
    {
        DiskFluxInfo info;
        info.z_weight = w;
        const double r2 = r_disk * r_disk;
        double flux = 0.0;
        for (int i = 0; i < grid.nx; ++i) {
            const double x_c = grid.x_min + (i + 0.5) * grid.dx;
            for (int j = 0; j < grid.ny; ++j) {
                const double y_c = grid.y_min + (j + 0.5) * grid.dy;
                if (!(x_c * x_c + y_c * y_c < r2)) { continue; }
                ++info.n_disk_cells;
                const std::size_t idx = static_cast<std::size_t>(i) * grid.ny + j;
                flux += (1.0 - w) * bz0[idx] + w * bz1[idx];
            }
        }
        info.flux = flux * grid.dx * grid.dy;
        return info;
    }

    /** Disk flux of the SOURCE loop's unit-current Yee field through the
     * TARGET circle (tgt_r, tgt_z), 3D rule [Wb/A]. */
    inline DiskFluxInfo
    LoopDiskFlux3D (const double src_r, const double src_z,
                    const double tgt_r, const double tgt_z,
                    const LoopGrid3D& grid)
    {
        int k0;
        double w;
        DiskPlanes3D(grid, tgt_z, k0, w);
        std::vector<double> ax, ay;
        SampleLoopAPlane3D(src_r, src_z, grid, k0, ax, ay);
        const std::vector<double> bz0 = CurlABzPlane3D(ax, ay, grid);
        SampleLoopAPlane3D(src_r, src_z, grid, k0 + 1, ax, ay);
        const std::vector<double> bz1 = CurlABzPlane3D(ax, ay, grid);
        return DiskFluxBzPlanes3D(bz0, bz1, w, grid, tgt_r);
    }

    /** Discrete (mesh-regularized) 3D self-inductance of a coil: the disk
     * flux of its own unit-current Yee-sampled field through its own
     * nominal circle, times n_turns^2 [H]. Resolution dependent by
     * construction; truncated to the domain for a circle reaching outside. */
    inline double
    DiscreteSelfInductance3D (const double coil_r, const double coil_z,
                              const double n_turns, const LoopGrid3D& grid,
                              DiskFluxInfo* info = nullptr)
    {
        const DiskFluxInfo f = LoopDiskFlux3D(coil_r, coil_z, coil_r, coil_z, grid);
        if (info != nullptr) { *info = f; }
        return f.flux * n_turns * n_turns;
    }

    /** Discrete 3D mutual inductance: the disk flux of the SOURCE coil's
     * unit-current Yee-sampled field through the TARGET coil's nominal
     * circle, times the two turn counts [H]. Not exactly symmetric (the
     * staircase is first order in the mesh spacing); callers build both
     * orientations and symmetrize, recording the asymmetry. */
    inline double
    DiscreteMutualInductance3D (const double src_r, const double src_z,
                                const double src_turns,
                                const double tgt_r, const double tgt_z,
                                const double tgt_turns,
                                const LoopGrid3D& grid)
    {
        const DiskFluxInfo f = LoopDiskFlux3D(src_r, src_z, tgt_r, tgt_z, grid);
        return f.flux * src_turns * tgt_turns;
    }

} // namespace warpx::circuit

#endif // WARPX_CIRCUIT_LOOPINDUCTANCE_H_
