/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ImplicitMHDWallMask.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace amrex::literals;

namespace
{
    /** Read the wall polyline CSV ("z, r" rows, optional header). The
     * path is resolved against AMREX_INPUTS_FILE_PREFIX (the ctest
     * convention for the inputs file itself) when it does not open
     * as given. */
    void ReadWallPolyline (const std::string& file,
                           std::vector<double>& z_points,
                           std::vector<double>& r_points)
    {
        std::string path = file;
        if (!std::ifstream(path).good()) {
            const char* prefix = std::getenv("AMREX_INPUTS_FILE_PREFIX");
            if (prefix != nullptr) { path = std::string(prefix) + file; }
        }
        std::ifstream in(path);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(in.good(),
            "implicit_mhd.wall_polyline_file: cannot open '" + file + "'");

        std::string line;
        bool first_content_line = true;
        while (std::getline(in, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream ls(line);
            double z = 0.0;
            double r = 0.0;
            if (!(ls >> z >> r)) {
                // allow a single non-numeric header row ("z, r") and
                // blank/comment lines
                std::istringstream probe(line);
                std::string token;
                const bool has_content = static_cast<bool>(probe >> token);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    !has_content || first_content_line || token[0] == '#',
                    "implicit_mhd.wall_polyline_file: malformed row '" + line +
                        "' in '" + path + "'");
                if (has_content) { first_content_line = false; }
                continue;
            }
            first_content_line = false;
            z_points.push_back(z);
            r_points.push_back(r);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(z_points.size() >= 2,
            "implicit_mhd.wall_polyline_file: need at least two (z, r) points in '"
                + path + "'");
        for (std::size_t p = 1; p < z_points.size(); ++p) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(z_points[p] >= z_points[p - 1],
                "implicit_mhd.wall_polyline_file: z must be non-decreasing");
        }
    }

    /** Wall radius at axial position z: piecewise-linear interpolation of
     * the polyline, clamped to its axial range (constant extrapolation
     * beyond the ends). Zero-length (duplicate-z) segments take the
     * smaller radius: at a vertical wall face the whole radial jump is
     * conductor, so the conservative single-valued reduction is min. */
    double WallRadiusAt (const std::vector<double>& z_points,
                         const std::vector<double>& r_points, double z)
    {
        const std::size_t n = z_points.size();
        if (z <= z_points.front()) { return r_points.front(); }
        if (z >= z_points.back()) { return r_points.back(); }
        for (std::size_t p = 0; p + 1 < n; ++p) {
            if (z > z_points[p + 1]) { continue; }
            const double dz_seg = z_points[p + 1] - z_points[p];
            if (dz_seg <= 0.0) {
                return std::min(r_points[p], r_points[p + 1]);
            }
            const double f = (z - z_points[p]) / dz_seg;
            return r_points[p] + f * (r_points[p + 1] - r_points[p]);
        }
        return r_points.back();
    }
}

void ImplicitMHDWallMask::Define (const amrex::Geometry& geom,
                                  const amrex::IntVect& ngrow)
{
    const amrex::ParmParse pp("implicit_mhd");
    std::string wall_model = "none";
    pp.query("wall_model", wall_model);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        wall_model == "none" || wall_model == "pec" ||
            wall_model == "pec_response",
        "implicit_mhd.wall_model must be 'none', 'pec' or 'pec_response'");

    // Thermal wall boundary at the stair-step fluid interface (see the
    // class comment): parsed BEFORE the early return so a thermal BC
    // without a wall model is a loud input error, never a silent no-op.
    std::string thermal_bc = "none";
    pp.query("wall_thermal_bc", thermal_bc);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        thermal_bc == "none" || thermal_bc == "zero_flux" ||
            thermal_bc == "temperature",
        "implicit_mhd.wall_thermal_bc must be 'none', 'zero_flux' or "
        "'temperature'");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        thermal_bc == "none" || wall_model != "none",
        "implicit_mhd.wall_thermal_bc requires implicit_mhd.wall_model = "
        "pec or pec_response (the thermal wall lives on the shaped-wall "
        "mask)");
    const bool has_wall_temperature = utils::parser::queryWithParser(
        pp, "wall_temperature", m_wall_temperature);
    if (thermal_bc == "temperature") {
        m_thermal_bc = ThermalBC::temperature;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            has_wall_temperature && m_wall_temperature > 0.0,
            "implicit_mhd.wall_thermal_bc = temperature requires a "
            "positive implicit_mhd.wall_temperature (in eV)");
    } else {
        m_thermal_bc = (thermal_bc == "zero_flux") ? ThermalBC::zero_flux
                                                   : ThermalBC::none;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !has_wall_temperature,
            "implicit_mhd.wall_temperature requires "
            "implicit_mhd.wall_thermal_bc = temperature");
    }

    if (wall_model == "none") {
        m_active = false;
        return;
    }
    m_total_field = (wall_model == "pec");
    if (!m_total_field) {
        // pec_response pins the PLASMA-RESPONSE field only: the prescribed
        // drive passes through the wall as-is because the fitted waveforms
        // already embed the machine's wall response (the run32 one-way
        // coupling contract). Without the split external registers there
        // is no plasma/external split and the mode is meaningless.
        const amrex::ParmParse pp_hybrid("hybrid_pic_model");
        bool add_external_fields = false;
        pp_hybrid.query("add_external_fields", add_external_fields);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(add_external_fields,
            "implicit_mhd.wall_model = pec_response requires the split "
            "external fields (hybrid_pic_model.add_external_fields = 1): "
            "the mode pins the plasma-response field only, which is not "
            "defined without the plasma/external split. Use wall_model = "
            "pec for the total-field conductor.");
    }

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(geom, ngrow);
    WARPX_ABORT_WITH_MESSAGE(
        "implicit_mhd.wall_model = pec requires cylindrical RZ geometry "
        "(the wall is a revolved poloidal polyline)");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!geom.isPeriodic(1),
        "implicit_mhd.wall_model = pec requires non-periodic z (the shaped "
        "wall is not an axially periodic structure)");

    std::string polyline_file;
    pp.query("wall_polyline_file", polyline_file);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!polyline_file.empty(),
        "implicit_mhd.wall_model = pec requires implicit_mhd.wall_polyline_file "
        "(CSV of 'z, r' rows)");

    std::vector<double> z_points;
    std::vector<double> r_points;
    ReadWallPolyline(polyline_file, z_points, r_points);

    const amrex::Box& domain = geom.Domain();
    m_nz = domain.length(1);
    m_ng = ngrow[1];
    const int nr = domain.length(0);
    const double plo_r = geom.ProbLo(0);
    const double plo_z = geom.ProbLo(1);
    const double dr = geom.CellSize(0);
    const double dz = geom.CellSize(1);
    // On-or-outside with a deterministic sliver of tolerance: a DOF within
    // one thousandth of a cell of the polyline counts as ON it (absorbs
    // the floating-point noise of coordinates computed as i * dr when the
    // polyline is grid-aligned; physically far below the stair-step
    // resolution).
    const double tol_r = 1.0e-3 * dr;
    constexpr int never_masked = std::numeric_limits<int>::max();
    const int imax = nr + ngrow[0] + 2;

    if (geom.ProbLo(1) - m_ng * dz < z_points.front() - 0.5 * dz ||
        geom.ProbHi(1) + m_ng * dz > z_points.back() + 0.5 * dz) {
        amrex::Print() << "ImplicitMHDWallMask: note: the domain (incl. ghosts) "
                          "extends beyond the wall polyline's axial range ["
                       << z_points.front() << ", " << z_points.back()
                       << "]; the wall radius is continued constantly there.\n";
    }

    // First masked radial index at axial position z, per radial staggering
    const auto first_masked = [&] (double z, bool nodal_r) {
        const double rw = WallRadiusAt(z_points, r_points, z) - tol_r;
        for (int i = 0; i <= imax; ++i) {
            const double r = plo_r + (nodal_r ? i : i + 0.5) * dr;
            if (r >= rw) { return i; }
        }
        return never_masked;
    };

    const int n_nodal = m_nz + 2 * m_ng + 1;  // j in [-ng, nz + ng]
    const int n_cell = m_nz + 2 * m_ng;       // j in [-ng, nz - 1 + ng]
    amrex::Vector<int> er_h(n_nodal);
    amrex::Vector<int> et_h(n_nodal);
    amrex::Vector<int> ez_h(n_cell);
    amrex::Vector<int> cc_h(n_cell);
    long masked_corner_rows = 0;
    int min_first_masked = never_masked;
    for (int jj = 0; jj < n_nodal; ++jj) {
        const double z_node = plo_z + (jj - m_ng) * dz;
        er_h[jj] = first_masked(z_node, false);  // E_r: r-cell, z-node
        et_h[jj] = first_masked(z_node, true);   // E_theta: r-node, z-node
        if (jj >= m_ng && jj <= m_ng + m_nz && et_h[jj] <= nr) {
            masked_corner_rows += nr + 1 - et_h[jj];
        }
        min_first_masked =
            std::min({min_first_masked, er_h[jj], et_h[jj]});
    }
    for (int jj = 0; jj < n_cell; ++jj) {
        const double z_cell = plo_z + (jj - m_ng + 0.5) * dz;
        ez_h[jj] = first_masked(z_cell, true);   // E_z: r-node, z-cell
        cc_h[jj] = first_masked(z_cell, false);  // fluid: r-cell, z-cell
        min_first_masked =
            std::min({min_first_masked, ez_h[jj], cc_h[jj]});
    }

    // Axis clearance guard: the r = 0 reflecting boundary is the m = 0
    // parity ghost fill plus the axis-corner EMF parities, which assume
    // plasma at the axis. A polyline pinching to r = 0 would put masked
    // conductor rows in direct conflict with those fills (and swallow
    // whole radial rows of the thermal wall), so it is a loud input
    // error rather than a silent mis-composition.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(min_first_masked >= 1,
        "implicit_mhd.wall_model: the wall polyline reaches the r = 0 "
        "axis (first masked radial index 0); the shaped wall must stay "
        "clear of the axis parity boundary");

    // --- Seam-guard tables (see the class comment): first radial index
    // per axial row at which the Hall/electron-inertia/hyper-resistive
    // Ohm stencils reach into the masked band. Footprint of the widest
    // chains (the nodal-inertia advection and the hyper
    // Laplacian-of-current both read currents whose own curl-B cells sit
    // one more cell out): TWO cells of radial reach outward, and the
    // axial cell windows [j-2, j+1] around a z-node row and [j-2, j+2]
    // around a z-cell row. Derived purely from the cell-centered masked
    // table (geometry-only, so JFNK probes see constant structure), then
    // clipped by the family's own masked table so first_guarded <=
    // first_masked everywhere (masked rows are trivially seam-adjacent).
    constexpr int guard_radial_reach = 2;
    const auto cc_masked_at = [&] (const int jc) {
        // Clamp to the ghost-extended cell table (constant continuation,
        // like the polyline itself).
        const int clamped = std::clamp(jc, -m_ng, m_nz - 1 + m_ng);
        return cc_h[clamped + m_ng];
    };
    const auto guard_from_cells = [&] (const int jlo, const int jhi) {
        int first_cc = never_masked;
        for (int jc = jlo; jc <= jhi; ++jc) {
            first_cc = std::min(first_cc, cc_masked_at(jc));
        }
        return (first_cc == never_masked)
                   ? never_masked
                   : first_cc - guard_radial_reach;
    };
    amrex::Vector<int> er_g(n_nodal);
    amrex::Vector<int> et_g(n_nodal);
    amrex::Vector<int> ez_g(n_cell);
    for (int jj = 0; jj < n_nodal; ++jj) {
        const int j = jj - m_ng;
        const int guard = guard_from_cells(j - 2, j + 1);
        er_g[jj] = std::min(er_h[jj], guard);
        et_g[jj] = std::min(et_h[jj], guard);
    }
    for (int jj = 0; jj < n_cell; ++jj) {
        const int j = jj - m_ng;
        ez_g[jj] = std::min(ez_h[jj], guard_from_cells(j - 2, j + 2));
    }
    // Live seam-guarded rows per family in the valid domain (banner
    // reporting): rows i with first_guarded <= i < first_masked. E_r
    // rows are r-cells 0..nr-1; E_theta and E_z rows are r-nodes 0..nr.
    m_seam_guarded_rows = {0, 0, 0};
    for (int jj = m_ng; jj <= m_ng + m_nz; ++jj) {
        m_seam_guarded_rows[0] +=
            std::max(0, std::min(er_h[jj], nr) - std::max(er_g[jj], 0));
        m_seam_guarded_rows[1] +=
            std::max(0, std::min(et_h[jj], nr + 1) - std::max(et_g[jj], 0));
    }
    for (int jj = m_ng; jj < m_ng + m_nz; ++jj) {
        m_seam_guarded_rows[2] +=
            std::max(0, std::min(ez_h[jj], nr + 1) - std::max(ez_g[jj], 0));
    }

    // Managed storage: read by device kernels (residual projection, seam
    // guard, Chebyshev/banded stencil emission, wall thermal BC) and by
    // the direct solver's host-side sparse-row assembly through the same
    // pointers.
    m_first_masked_er.resize(n_nodal);
    m_first_masked_et.resize(n_nodal);
    m_first_masked_ez.resize(n_cell);
    m_first_masked_cc.resize(n_cell);
    m_first_guarded_er.resize(n_nodal);
    m_first_guarded_et.resize(n_nodal);
    m_first_guarded_ez.resize(n_cell);
    std::copy(er_h.begin(), er_h.end(), m_first_masked_er.begin());
    std::copy(et_h.begin(), et_h.end(), m_first_masked_et.begin());
    std::copy(ez_h.begin(), ez_h.end(), m_first_masked_ez.begin());
    std::copy(cc_h.begin(), cc_h.end(), m_first_masked_cc.begin());
    std::copy(er_g.begin(), er_g.end(), m_first_guarded_er.begin());
    std::copy(et_g.begin(), et_g.end(), m_first_guarded_et.begin());
    std::copy(ez_g.begin(), ez_g.end(), m_first_guarded_ez.begin());

    // Active BEFORE the banner: ThermalBCName()/GetThermalBC() gate on
    // m_active, so printing first reports "thermal BC none" for an
    // engaged mode (caught by the RR12 boot-banner gate).
    m_active = true;

    const auto [rw_min, rw_max] =
        std::minmax_element(r_points.begin(), r_points.end());
    amrex::Print() << "ImplicitMHDWallMask: stair-step conducting wall ("
                   << wall_model << ") active from '"
                   << polyline_file << "' (" << z_points.size()
                   << " polyline points, r in [" << *rw_min << ", " << *rw_max
                   << "], z in [" << z_points.front() << ", "
                   << z_points.back() << "]); " << masked_corner_rows
                   << " masked corner (E_theta) rows in the valid domain; "
                   << "axis clearance " << min_first_masked
                   << " cells; thermal BC " << ThermalBCName();
    if (m_thermal_bc == ThermalBC::temperature) {
        amrex::Print() << " (T_wall = " << m_wall_temperature << " eV)";
    }
    amrex::Print() << "\n";
#endif
}

const char* ImplicitMHDWallMask::ThermalBCName () const
{
    if (!m_active || m_thermal_bc == ThermalBC::none) { return "none"; }
    return (m_thermal_bc == ThermalBC::temperature) ? "temperature"
                                                    : "zero_flux";
}

const int* ImplicitMHDWallMask::FirstMaskedCellCentered () const
{
    if (GetThermalBC() == ThermalBC::none) { return nullptr; }
    return m_first_masked_cc.data() + m_ng;
}

warpx::mhd_pc::WallMaskView ImplicitMHDWallMask::View () const
{
    warpx::mhd_pc::WallMaskView view;
    view.active = m_active;
    if (m_active) {
        view.first_masked_er = m_first_masked_er.data() + m_ng;
        view.first_masked_et = m_first_masked_et.data() + m_ng;
        view.first_masked_ez = m_first_masked_ez.data() + m_ng;
        view.first_guarded_er = m_first_guarded_er.data() + m_ng;
        view.first_guarded_et = m_first_guarded_et.data() + m_ng;
        view.first_guarded_ez = m_first_guarded_ez.data() + m_ng;
    }
    return view;
}

void ImplicitMHDWallMask::ProjectElectricField (
    ablastr::fields::VectorField const& efield,
    ablastr::fields::VectorField const* efield_external,
    const amrex::Geometry& geom) const
{
#if defined(WARPX_DIM_RZ)
    if (!m_active) { return; }

    const warpx::mhd_pc::WallMaskView view = View();
    const amrex::Array<const int*, 3> first_masked = {
        view.first_masked_er, view.first_masked_et, view.first_masked_ez};

    for (int d = 0; d < 3; ++d) {
        amrex::MultiFab& mf = *efield[d];
        // pec (total-field contract): masked plasma E = -E_ext, so the
        // total tangential E vanishes. pec_response (plasma-response
        // contract, run32 EB parity): masked plasma E = 0, the external
        // drive is transparent -- the external register is not read.
        const amrex::MultiFab* const ext =
            (m_total_field && efield_external != nullptr)
                ? (*efield_external)[d] : nullptr;

        // Cover the domain ghosts as far as the mask tables and (when
        // present) the external register reach: the split external E
        // carries trusted domain-ghost values (analytic fill + exact
        // ghost curls). No growth below the axis (no wall there and no
        // trusted parity data).
        amrex::IntVect ngrow = mf.nGrowVect();
        if (ext != nullptr) { ngrow = amrex::min(ngrow, ext->nGrowVect()); }
        ngrow[1] = std::min(ngrow[1], m_ng);
        amrex::Box grow_region = geom.Domain();
        grow_region.grow(ngrow);
        if (geom.ProbLo(0) == 0.0_rt) {
            grow_region.setSmall(0, geom.Domain().smallEnd(0));
        }
        const amrex::Box allowed = amrex::convert(grow_region, mf.ixType());
        const int* const AMREX_RESTRICT first = first_masked[d];
        const bool has_ext = (ext != nullptr);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.growntilebox(ngrow) & allowed;
            if (box.isEmpty()) { continue; }
            amrex::Array4<amrex::Real> const& e_arr = mf.array(mfi);
            amrex::Array4<amrex::Real const> const e_ext =
                has_ext ? ext->const_array(mfi)
                        : amrex::Array4<amrex::Real const>{};
            amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (i >= first[j]) {
                        e_arr(i, j, k) =
                            has_ext ? -e_ext(i, j, k) : 0.0_rt;
                    }
                });
        }
    }
#else
    amrex::ignore_unused(efield, efield_external, geom);
#endif
}
