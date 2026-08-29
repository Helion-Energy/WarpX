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
            wall_model == "pec_response" || wall_model == "dielectric",
        "implicit_mhd.wall_model must be 'none', 'pec', 'pec_response' or "
        "'dielectric'");

    // Consumed BEFORE the early return so a deck that switches
    // wall_model off (e.g. a CTest twin overriding to none on the CLI)
    // does not trip the unused-ParmParse finalize check on the polyline
    // path; validated below only when a wall is active.
    std::string polyline_file;
    pp.query("wall_polyline_file", polyline_file);

    // Wall-band resistivity override (see the class comment): parsed
    // before the early return so an override without a wall model is a
    // loud input error, never a silent no-op.
    const bool has_band_eta_override = utils::parser::queryWithParser(
        pp, "wall_band_eta_override", m_band_eta_override);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_band_eta_override >= 0.0,
        "implicit_mhd.wall_band_eta_override must be non-negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !has_band_eta_override || m_band_eta_override == 0.0 ||
            wall_model != "none",
        "implicit_mhd.wall_band_eta_override requires an active "
        "implicit_mhd.wall_model (the override acts on the masked wall "
        "band)");

    // Field-side freeze of the masked band (work item B, see the class
    // comment): parsed before the early return so a freeze without a
    // wall model is a loud input error, never a silent no-op.
    pp.query("wall_field_freeze", m_field_freeze);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_field_freeze || wall_model != "none",
        "implicit_mhd.wall_field_freeze requires an active "
        "implicit_mhd.wall_model (the freeze acts on the masked wall "
        "band's magnetic-field faces)");

    // Thermal wall boundary at the stair-step fluid interface (see the
    // class comment): parsed BEFORE the early return so a thermal BC
    // without a wall model is a loud input error, never a silent no-op.
    std::string thermal_bc = "none";
    pp.query("wall_thermal_bc", thermal_bc);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        thermal_bc == "none" || thermal_bc == "zero_flux" ||
            thermal_bc == "temperature" || thermal_bc == "dirichlet",
        "implicit_mhd.wall_thermal_bc must be 'none', 'zero_flux', "
        "'temperature' or 'dirichlet'");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        thermal_bc == "none" || wall_model != "none",
        "implicit_mhd.wall_thermal_bc requires implicit_mhd.wall_model = "
        "pec, pec_response or dielectric (the thermal wall lives on the "
        "shaped-wall mask)");
    const bool has_wall_temperature = utils::parser::queryWithParser(
        pp, "wall_temperature", m_wall_temperature);
    if (thermal_bc == "temperature" || thermal_bc == "dirichlet") {
        m_thermal_bc = (thermal_bc == "dirichlet") ? ThermalBC::dirichlet
                                                   : ThermalBC::temperature;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            has_wall_temperature && m_wall_temperature > 0.0,
            "implicit_mhd.wall_thermal_bc = temperature/dirichlet requires "
            "a positive implicit_mhd.wall_temperature (in eV)");
    } else {
        m_thermal_bc = (thermal_bc == "zero_flux") ? ThermalBC::zero_flux
                                                   : ThermalBC::none;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !has_wall_temperature,
            "implicit_mhd.wall_temperature requires "
            "implicit_mhd.wall_thermal_bc = temperature or dirichlet");
    }

    if (wall_model == "none") {
        m_active = false;
        return;
    }
    m_total_field = (wall_model == "pec");
    m_dielectric = (wall_model == "dielectric");
    // The standoff's ENTIRE wall action is the fluid contract (the field
    // side is deliberately transparent), so without a thermal wall the
    // mode would be a complete silent no-op -- a loud input error
    // instead. The conductor modes stay legal without it (they are
    // electromagnetic walls first).
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_dielectric || m_thermal_bc != ThermalBC::none,
        "implicit_mhd.wall_model = dielectric requires an active "
        "implicit_mhd.wall_thermal_bc (zero_flux or temperature): the "
        "EM-transparent standoff acts on the fluid only");
    if (!m_total_field) {
        // pec_response pins the PLASMA-RESPONSE field only: the prescribed
        // drive passes through the wall as-is because the fitted waveforms
        // already embed the machine's wall response (the run32 one-way
        // coupling contract). The dielectric standoff likewise acts on the
        // response side of the split (its fluid contract absorbs the
        // plasma while the prescribed drive supplies the programmed
        // fields). Without the split external registers there is no
        // plasma/external split and either mode is meaningless.
        const amrex::ParmParse pp_hybrid("hybrid_pic_model");
        bool add_external_fields = false;
        pp_hybrid.query("add_external_fields", add_external_fields);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(add_external_fields,
            "implicit_mhd.wall_model = " + wall_model + " requires the "
            "split external fields (hybrid_pic_model.add_external_fields "
            "= 1): the mode acts on the plasma-response side of the "
            "plasma/external split, which is not defined without it. Use "
            "wall_model = pec for the total-field conductor.");
    }

#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(geom, ngrow);
    WARPX_ABORT_WITH_MESSAGE(
        "implicit_mhd.wall_model requires cylindrical RZ geometry "
        "(the wall is a revolved poloidal polyline)");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!geom.isPeriodic(1),
        "implicit_mhd.wall_model requires non-periodic z (the shaped "
        "wall is not an axially periodic structure)");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!polyline_file.empty(),
        "implicit_mhd.wall_model requires implicit_mhd.wall_polyline_file "
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
    // --- Band-interior tables (wall_band_eta_override, see the class
    // comment): first radial index per axial row at which EVERY
    // cell-centered cell of the location's density/temperature
    // interpolation neighborhood is masked -- E locations on the stair
    // interface (bounding at least one live cell) are excluded by
    // construction, so flooring eta there never touches the live-side
    // field evolution. Derived purely from the cell-centered masked
    // table (geometry-static). Per staggering: an E_r z-face (r-cell i,
    // z-node j) reads cells (i, j-1) and (i, j); an E_theta corner
    // (r-node i, z-node j) reads the four cells around it; an E_z
    // r-face (r-node i, z-cell j) reads cells (i-1, j) and (i, j).
    amrex::Vector<int> er_b(n_nodal);
    amrex::Vector<int> et_b(n_nodal);
    amrex::Vector<int> ez_b(n_cell);
    const auto nodal_shift = [&] (const int first_cc) {
        return (first_cc == never_masked) ? never_masked : first_cc + 1;
    };
    for (int jj = 0; jj < n_nodal; ++jj) {
        const int j = jj - m_ng;
        const int both = std::max(cc_masked_at(j - 1), cc_masked_at(j));
        er_b[jj] = both;
        et_b[jj] = nodal_shift(both);
    }
    for (int jj = 0; jj < n_cell; ++jj) {
        ez_b[jj] = nodal_shift(cc_h[jj]);
    }
    // --- Field-freeze tables (wall_field_freeze, see the class comment):
    // first FROZEN radial index per axial row -- a magnetic-field face is
    // frozen iff ALL curl-E edges of its Faraday update are masked, so a
    // face with even one live edge stays live (the last live Faraday row
    // closes against the surface E). INT_MAX-safe max composition like
    // the band tables. Per staggering: Br (r-node i, z-cell j) reads
    // E_theta (i, j) and (i, j+1); Bt (r-cell i, z-cell j) reads E_r
    // (i, j), (i, j+1) and the E_z nodes (i, j), (i+1, j) -- both nodes
    // masked iff i >= first_masked_ez[j]; Bz (r-cell i, z-node j) reads
    // the E_theta nodes (i, j) and (i+1, j) -- both masked iff
    // i >= first_masked_et[j]. Built in every active wall_model
    // (geometry-static); consumed only when the freeze is requested.
    amrex::Vector<int> br_f(n_cell);
    amrex::Vector<int> bt_f(n_cell);
    amrex::Vector<int> bz_f(n_nodal);
    for (int jj = 0; jj < n_cell; ++jj) {
        br_f[jj] = std::max(et_h[jj], et_h[jj + 1]);
        bt_f[jj] = std::max({er_h[jj], er_h[jj + 1], ez_h[jj]});
    }
    for (int jj = 0; jj < n_nodal; ++jj) {
        bz_f[jj] = et_h[jj];
    }
    // Frozen faces per family in the valid domain (banner reporting):
    // Br rows are r-nodes 0..nr; Bt and Bz rows are r-cells 0..nr-1.
    m_frozen_faces = {0, 0, 0};
    for (int jj = m_ng; jj < m_ng + m_nz; ++jj) {
        m_frozen_faces[0] +=
            std::max(0, nr + 1 - std::max(br_f[jj], 0));
        m_frozen_faces[1] += std::max(0, nr - std::max(bt_f[jj], 0));
    }
    for (int jj = m_ng; jj <= m_ng + m_nz; ++jj) {
        m_frozen_faces[2] += std::max(0, nr - std::max(bz_f[jj], 0));
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
    m_first_band_er.resize(n_nodal);
    m_first_band_et.resize(n_nodal);
    m_first_band_ez.resize(n_cell);
    m_first_frozen_br.resize(n_cell);
    m_first_frozen_bt.resize(n_cell);
    m_first_frozen_bz.resize(n_nodal);
    std::copy(er_h.begin(), er_h.end(), m_first_masked_er.begin());
    std::copy(et_h.begin(), et_h.end(), m_first_masked_et.begin());
    std::copy(ez_h.begin(), ez_h.end(), m_first_masked_ez.begin());
    std::copy(cc_h.begin(), cc_h.end(), m_first_masked_cc.begin());
    std::copy(er_g.begin(), er_g.end(), m_first_guarded_er.begin());
    std::copy(et_g.begin(), et_g.end(), m_first_guarded_et.begin());
    std::copy(ez_g.begin(), ez_g.end(), m_first_guarded_ez.begin());
    std::copy(er_b.begin(), er_b.end(), m_first_band_er.begin());
    std::copy(et_b.begin(), et_b.end(), m_first_band_et.begin());
    std::copy(ez_b.begin(), ez_b.end(), m_first_band_ez.begin());
    std::copy(br_f.begin(), br_f.end(), m_first_frozen_br.begin());
    std::copy(bt_f.begin(), bt_f.end(), m_first_frozen_bt.begin());
    std::copy(bz_f.begin(), bz_f.end(), m_first_frozen_bz.begin());

    // Active BEFORE the banner: ThermalBCName()/GetThermalBC() gate on
    // m_active, so printing first reports "thermal BC none" for an
    // engaged mode (caught by the RR12 boot-banner gate).
    m_active = true;

    const auto [rw_min, rw_max] =
        std::minmax_element(r_points.begin(), r_points.end());
    if (m_dielectric) {
        amrex::Print() << "ImplicitMHDWallMask: stair-step shaped wall "
                          "dielectric (EM-transparent standoff; fluid "
                          "contract = pec_response) active from '";
    } else {
        amrex::Print() << "ImplicitMHDWallMask: stair-step conducting wall ("
                       << wall_model << ") active from '";
    }
    amrex::Print() << polyline_file << "' (" << z_points.size()
                   << " polyline points, r in [" << *rw_min << ", " << *rw_max
                   << "], z in [" << z_points.front() << ", "
                   << z_points.back() << "]); " << masked_corner_rows
                   << " masked corner (E_theta) rows in the valid domain; "
                   << "axis clearance " << min_first_masked
                   << " cells; thermal BC " << ThermalBCName();
    if (m_thermal_bc == ThermalBC::temperature ||
        m_thermal_bc == ThermalBC::dirichlet) {
        amrex::Print() << " (T_wall = " << m_wall_temperature << " eV)";
    }
    if (m_band_eta_override > 0.0) {
        amrex::Print() << "; band eta override " << m_band_eta_override
                       << " ohm m (constant at band-interior E rows)";
    }
    if (m_field_freeze) {
        amrex::Print() << "; field freeze active (exterior evolved-B "
                          "identity rows: "
                       << m_frozen_faces[0] << " B_r, " << m_frozen_faces[1]
                       << " B_theta, " << m_frozen_faces[2]
                       << " B_z faces in the valid domain)";
    }
    amrex::Print() << "\n";
#endif
}

const char* ImplicitMHDWallMask::ThermalBCName () const
{
    if (!m_active || m_thermal_bc == ThermalBC::none) { return "none"; }
    if (m_thermal_bc == ThermalBC::dirichlet) { return "dirichlet"; }
    return (m_thermal_bc == ThermalBC::temperature) ? "temperature"
                                                    : "zero_flux";
}

const int* ImplicitMHDWallMask::FirstMaskedCellCentered () const
{
    // Non-null in EVERY active wall_model (the table is geometry-static
    // and always built): the thermal-BC consumers gate themselves on
    // GetThermalBC(), while the wall-row viscosity mask needs the
    // contour with or without a thermal wall.
    if (!m_active) { return nullptr; }
    return m_first_masked_cc.data() + m_ng;
}

warpx::mhd_pc::WallMaskView ImplicitMHDWallMask::View () const
{
    warpx::mhd_pc::WallMaskView view;
    // Dielectric standoff: the band imposes no field constraint, so the
    // preconditioner's stencil emission drops no rows and the residual's
    // seam guard never engages -- the view reports no wall at all,
    // keeping residual and PC exact twins of the wall_model = none field
    // operator. (The FLUID contract does not read this view: it goes
    // through FirstMaskedCellCentered().)
    view.active = m_active && !m_dielectric;
    if (view.active) {
        view.first_masked_er = m_first_masked_er.data() + m_ng;
        view.first_masked_et = m_first_masked_et.data() + m_ng;
        view.first_masked_ez = m_first_masked_ez.data() + m_ng;
        view.first_guarded_er = m_first_guarded_er.data() + m_ng;
        view.first_guarded_et = m_first_guarded_et.data() + m_ng;
        view.first_guarded_ez = m_first_guarded_ez.data() + m_ng;
    }
    return view;
}

WallBandEtaOverrideView ImplicitMHDWallMask::BandEtaOverrideView () const
{
    WallBandEtaOverrideView view;
    // Active in every wall_model: under the conductor contracts the
    // overridden band-interior rows are subsequently pinned by the
    // projection (the override is redundant but harmless); under the
    // dielectric standoff it is the band's electrical-hygiene contract
    // (see the class comment).
    view.active = m_active && (m_band_eta_override > 0.0);
    if (view.active) {
        view.eta_override = m_band_eta_override;
        if (m_dielectric) {
            // Dielectric standoff: the override covers ALL masked E
            // rows INCLUDING the interface node on the contour. The
            // interface row's composed (plasma-keyed) eta is the
            // conductor-mode convention; on an EM-transparent
            // dielectric it turns the contour node into a
            // near-superconducting ring whenever plasma abuts the
            // wall (the nodal rho average has no wall exclusion, the
            // vacuum boost never engages, eta -> Spitzer at the
            // wall-cooled edge): E_theta ~ 0 on the contour freezes
            // the response flux through the wall -- a PEC in
            // dielectric clothing. Measured signature: bias Bz
            // collapsed to ~0 in the last interior cell while the
            // bulk carried the full soak value, driving an outward
            // B^2/2mu0 edge force and a self-sustaining scrape-off
            // inflow. A dielectric surface carries no current: the
            // constant override on the contour rows restores flux
            // transparency (the residual and the PC edge fills
            // consume this same view, so they stay exact twins).
            view.first_band_er = m_first_masked_er.data() + m_ng;
            view.first_band_et = m_first_masked_et.data() + m_ng;
            view.first_band_ez = m_first_masked_ez.data() + m_ng;
        } else {
            view.first_band_er = m_first_band_er.data() + m_ng;
            view.first_band_et = m_first_band_et.data() + m_ng;
            view.first_band_ez = m_first_band_ez.data() + m_ng;
        }
        view.first_band_cc = m_first_masked_cc.data() + m_ng;
    }
    return view;
}

warpx::mhd_pc::WallFieldFreezeView ImplicitMHDWallMask::FieldFreezeView () const
{
    warpx::mhd_pc::WallFieldFreezeView view;
    // Active only on explicit request (implicit_mhd.wall_field_freeze):
    // the default keeps every wall_model bit-identical to the pre-freeze
    // field advance. The tables themselves exist in every active mode
    // (geometry-static).
    view.active = m_active && m_field_freeze;
    if (view.active) {
        view.first_frozen_br = m_first_frozen_br.data() + m_ng;
        view.first_frozen_bt = m_first_frozen_bt.data() + m_ng;
        view.first_frozen_bz = m_first_frozen_bz.data() + m_ng;
    }
    return view;
}

warpx::mhd_pc::WallFluidFreezeView ImplicitMHDWallMask::FluidFreezeView () const
{
    warpx::mhd_pc::WallFluidFreezeView view;
    // Active with the residual's wall_live gate exactly: any thermal wall
    // BC freezes the masked band's fluid rows to identities, in every
    // wall_model (the dielectric standoff keeps the field-side View()
    // inactive but its fluid contract still freezes the band).
    view.active = (GetThermalBC() != ThermalBC::none);
    if (view.active) {
        view.first_masked_cc = m_first_masked_cc.data() + m_ng;
        view.z_lo = -m_ng;
        view.z_hi = m_nz - 1 + m_ng;
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
    // Dielectric standoff: an insulator imposes no condition on the
    // evolved fields (the PEC_Insulator insulator-side convention:
    // tangential fields are left unchanged unless a value is prescribed,
    // and the normal B stays evolved -- nothing is prescribed here), so
    // the plasma-response E lives everywhere and the projection performs
    // NOTHING. The band's frozen floor-density dust plus the vacuum-keyed
    // resistivity already keep the region current-free in the response
    // solve: the physically correct insulator interior.
    if (m_dielectric) { return; }

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
