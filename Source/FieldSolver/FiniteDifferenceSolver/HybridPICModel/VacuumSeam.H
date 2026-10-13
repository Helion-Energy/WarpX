/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_VACUUM_SEAM_H_
#define WARPX_VACUUM_SEAM_H_

#include <AMReX_Array4.H>
#include <AMReX_Enum.H>
#include <AMReX_GpuContainers.H>
#include <limits>

/** Sampling of the density that decides the vacuum-seam treatment of the
 *  generalized Ohm's law: the holmstrom vacuum branch when
 *  hybrid_pic_model.holmstrom_vacuum_region is on and the density-floor
 *  selection of the guarded Hall term otherwise. Node and Cell are
 *  vacuum-favoring and supported in Cartesian geometry only. */
AMREX_ENUM(VacuumSeamSwitchMode,
    Edge, /**< legacy per-component edge average; the E components of a cell can take inconsistent branches at the plasma/vacuum seam */
    Node, /**< minimum over the component's edge-endpoint nodes */
    Cell  /**< minimum over the component's adjacent cells of the node-averaged density: a single decision for all three components of an index */
);

namespace warpx::hybrid
{
    /** Endpoint-min nodal decision density for vacuum_seam_switch_mode =
     *  "node": the minimum nodal rho over the staggered component's edge
     *  endpoints (the same nodes the legacy edge average reads).
     *  Vacuum-favoring: the unguarded Hall branch runs only where every
     *  endpoint is above the floor. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NodalSwitchRho (
        amrex::Array4<amrex::Real const> const& rho,
        amrex::GpuArray<int,3> const& stag,
        int i, int j, int k)
    {
        const int ni = (stag[0] == 0) ? 2 : 1;
        const int nj = (stag[1] == 0) ? 2 : 1;
        const int nk = (stag[2] == 0) ? 2 : 1;
        amrex::Real r = rho(i, j, k);
        for (int kk = 0; kk < nk; ++kk) {
        for (int jj = 0; jj < nj; ++jj) {
        for (int ii = 0; ii < ni; ++ii) {
            r = amrex::min(r, rho(i+ii, j+jj, k+kk));
        }}}
        return r;
    }

    /** Adjacent-cell decision density for vacuum_seam_switch_mode = "cell":
     *  the minimum over the component's adjacent cells of the node-averaged
     *  rho -- one piecewise-constant-per-cell decision for all three E
     *  components, reflection- and C4-equivariant, vacuum-favoring like
     *  "node". */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real CellSwitchRho (
        amrex::Array4<amrex::Real const> const& rho,
        amrex::GpuArray<int,3> const& stag,
        int i, int j, int k)
    {
        using namespace amrex::literals;
        int off_lo[3] = {0, 0, 0};
        int span[3] = {1, 1, 1};
        int const ic[3] = {i, j, k};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            off_lo[d] = (stag[d] == 1) ? -1 : 0;
            span[d] = 2;
        }
        // Running minimum over the adjacent cells: max() is the identity
        // element for min (no cell average can exceed it), not an overflow
        // guard. The outer loops visit every cell that touches this E
        // component (two per direction where the component is nodal, one
        // where it is cell-centered); the inner loops form each cell's
        // node-averaged rho (sum/cnt, cnt = 2^AMREX_SPACEDIM); the result is
        // the minimum of those cell averages -- vacuum-favoring.
        amrex::Real rho_cell_min = std::numeric_limits<amrex::Real>::max();
        for (int ok = off_lo[2]; ok <= 0; ++ok) {
        for (int oj = off_lo[1]; oj <= 0; ++oj) {
        for (int oi = off_lo[0]; oi <= 0; ++oi) {
            amrex::Real sum = 0.0_rt;
            int cnt = 0;
            for (int kk = 0; kk < span[2]; ++kk) {
            for (int jj = 0; jj < span[1]; ++jj) {
            for (int ii = 0; ii < span[0]; ++ii) {
                sum += rho(ic[0]+oi+ii, ic[1]+oj+jj, ic[2]+ok+kk);
                cnt += 1;
            }}}
            rho_cell_min = amrex::min(rho_cell_min, sum/static_cast<amrex::Real>(cnt));
        }}}
        return rho_cell_min;
    }
}
#endif
