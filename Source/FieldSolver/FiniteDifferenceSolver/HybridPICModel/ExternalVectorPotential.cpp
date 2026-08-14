/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ExternalVectorPotential.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Initialization/DivCleaner/ProjectionDivCleaner.H"
#include "Fields.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <string>

using namespace amrex;
using namespace warpx::fields;

ExternalVectorPotential::ExternalVectorPotential ()
{
    ReadParameters();
}

void
ExternalVectorPotential::ReadParameters ()
{
    const ParmParse pp_ext_A("external_vector_potential");

    pp_ext_A.query("do_diva_cleaning", m_do_clean_divA);

    pp_ext_A.queryarr("fields", m_field_names);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_field_names.empty(),
        "No external field names defined in external_vector_potential.fields");

    m_nFields = static_cast<int>(m_field_names.size());

    // Resize vectors and set defaults
    m_Ax_ext_grid_function.resize(m_nFields);
    m_Ay_ext_grid_function.resize(m_nFields);
    m_Az_ext_grid_function.resize(m_nFields);
    for (std::string & field : m_Ax_ext_grid_function) { field = "0.0"; }
    for (std::string & field : m_Ay_ext_grid_function) { field = "0.0"; }
    for (std::string & field : m_Az_ext_grid_function) { field = "0.0"; }

    m_A_external_parser.resize(m_nFields);
    m_A_external.resize(m_nFields);

    m_A_ext_time_function.resize(m_nFields);
    for (std::string & field_time : m_A_ext_time_function) { field_time = "1.0"; }

    m_A_external_time_parser.resize(m_nFields);
    m_A_time_scale.resize(m_nFields);

    m_read_A_from_file.resize(m_nFields);
    m_external_file_path.resize(m_nFields);
    for (std::string & file_name : m_external_file_path) { file_name = ""; }

    m_use_python_scale.resize(m_nFields, false);
    m_python_scale.resize(m_nFields);

    for (int i = 0; i < m_nFields; ++i) {
        bool read_from_file = false;
        utils::parser::queryWithParser(pp_ext_A,
            (m_field_names[i]+".read_from_file").c_str(), read_from_file);
        m_read_A_from_file[i] = read_from_file;

        bool python_scale = false;
        utils::parser::queryWithParser(pp_ext_A,
            (m_field_names[i]+".python_scale").c_str(), python_scale);
        m_use_python_scale[i] = python_scale;
        if (python_scale) {
            m_any_python_scale = true;
            // Scale held constant at initial_scale until the first
            // SetScale call (e.g. a coil at its pre-ramp current).
            amrex::Real initial_scale = 1.0_rt;
            utils::parser::queryWithParser(pp_ext_A,
                (m_field_names[i]+".initial_scale").c_str(), initial_scale);
            m_python_scale[i].s_old = initial_scale;
            m_python_scale[i].s_new = initial_scale;
        }

        if (m_read_A_from_file[i]) {
            pp_ext_A.query(m_field_names[i]+".path", m_external_file_path[i]);
        } else {
            pp_ext_A.query(m_field_names[i]+".Ax_external_grid_function(x,y,z)",
                m_Ax_ext_grid_function[i]);
            pp_ext_A.query(m_field_names[i]+".Ay_external_grid_function(x,y,z)",
                m_Ay_ext_grid_function[i]);
            pp_ext_A.query(m_field_names[i]+".Az_external_grid_function(x,y,z)",
                m_Az_ext_grid_function[i]);
        }

        pp_ext_A.query(m_field_names[i]+".A_time_external_function(t)",
            m_A_ext_time_function[i]);
    }
}

void
ExternalVectorPotential::AllocateLevelMFs (
    ablastr::fields::MultiFabRegister & fields,
    int lev, const BoxArray& ba, const DistributionMapping& dm,
    const int ncomps,
    const IntVect& ngEB,
    const IntVect& Ex_nodal_flag,
    const IntVect& Ey_nodal_flag,
    const IntVect& Ez_nodal_flag,
    const IntVect& Bx_nodal_flag,
    const IntVect& By_nodal_flag,
    const IntVect& Bz_nodal_flag)
{
    using ablastr::fields::Direction;
    for (std::string const & field_name : m_field_names) {
        const std::string Aext_field = field_name + std::string{"_Aext"};
        fields.alloc_init(Aext_field, Direction{0},
            lev, amrex::convert(ba, Ex_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(Aext_field, Direction{1},
            lev, amrex::convert(ba, Ey_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(Aext_field, Direction{2},
            lev, amrex::convert(ba, Ez_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);

        const std::string curlAext_field = field_name + std::string{"_curlAext"};
        fields.alloc_init(curlAext_field, Direction{0},
            lev, amrex::convert(ba, Bx_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(curlAext_field, Direction{1},
            lev, amrex::convert(ba, By_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(curlAext_field, Direction{2},
            lev, amrex::convert(ba, Bz_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
    }
    fields.alloc_init(FieldType::hybrid_E_fp_external, Direction{0},
        lev, amrex::convert(ba, Ex_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_E_fp_external, Direction{1},
        lev, amrex::convert(ba, Ey_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_E_fp_external, Direction{2},
        lev, amrex::convert(ba, Ez_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_B_fp_external, Direction{0},
        lev, amrex::convert(ba, Bx_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_B_fp_external, Direction{1},
        lev, amrex::convert(ba, By_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_B_fp_external, Direction{2},
        lev, amrex::convert(ba, Bz_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);

    // Cached partial sums of the parser-driven fields, used by the
    // CircuitOnly refresh mode to avoid re-accumulating every field when
    // only the SetScale-driven scales changed. Only needed when at least
    // one field is SetScale-driven.
    if (m_any_python_scale) {
        const amrex::IntVect E_flags[3] = {Ex_nodal_flag, Ey_nodal_flag, Ez_nodal_flag};
        const amrex::IntVect B_flags[3] = {Bx_nodal_flag, By_nodal_flag, Bz_nodal_flag};
        for (int idir = 0; idir < 3; ++idir) {
            fields.alloc_init("hybrid_E_fp_external_static", Direction{idir},
                lev, amrex::convert(ba, E_flags[idir]),
                dm, ncomps, ngEB, 0.0_rt);
            fields.alloc_init("hybrid_B_fp_external_static", Direction{idir},
                lev, amrex::convert(ba, B_flags[idir]),
                dm, ncomps, ngEB, 0.0_rt);
        }
    }
}

void
ExternalVectorPotential::InitData ()
{
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

    for (int i = 0; i < m_nFields; ++i) {

        const std::string Aext_field = m_field_names[i] + std::string{"_Aext"};

        if (m_read_A_from_file[i]) {
            // Read A fields from file
            for (auto lev = 0; lev <= warpx.finestLevel(); ++lev) {
#if defined(WARPX_DIM_RZ)
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{0}, lev),
                    "A", "r");
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{1}, lev),
                    "A", "t");
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{2}, lev),
                    "A", "z");
#else
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{0}, lev),
                    "A", "x");
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{1}, lev),
                    "A", "y");
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{2}, lev),
                    "A", "z");
#endif
            }
        } else {
            // Initialize the A fields from expression
            m_A_external_parser[i][0] = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(m_Ax_ext_grid_function[i],{"x","y","z","t"}));
            m_A_external_parser[i][1] = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(m_Ay_ext_grid_function[i],{"x","y","z","t"}));
            m_A_external_parser[i][2] = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(m_Az_ext_grid_function[i],{"x","y","z","t"}));
            m_A_external[i][0] = m_A_external_parser[i][0]->compile<4>();
            m_A_external[i][1] = m_A_external_parser[i][1]->compile<4>();
            m_A_external[i][2] = m_A_external_parser[i][2]->compile<4>();

            // check if the external current parsers depend on time
            for (int idim=0; idim<3; idim++) {
                const std::set<std::string> A_ext_symbols = m_A_external_parser[i][idim]->symbols();
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!A_ext_symbols.contains("t"),
                    "Externally Applied Vector potential time variation must be set with A_time_external_function(t)");
            }

            // Initialize data onto grid
            for (auto lev = 0; lev <= warpx.finestLevel(); ++lev) {
                warpx.ComputeExternalFieldOnGridUsingParser(
                    Aext_field,
                    m_A_external[i][0],
                    m_A_external[i][1],
                    m_A_external[i][2],
                    lev, PatchType::fine,
                    warpx.GetEBUpdateEFlag(),
                    false);
            }
            // NOTE: Fill Boundary is not done here since non-periodic A fields can lead to periodic E/B fields
            // This requires valid definitions of the vector potential in the ghost cells.
        }

        amrex::Gpu::streamSynchronize();

        if (m_do_clean_divA) {
            warpx::initialization::ProjectionDivCleaner dc(Aext_field, true);
            dc.setSourceFromField();
            dc.solve();
            dc.correctField();
            amrex::Print() << Utils::TextMsg::Info( "Finished Projection A-Field divergence cleaner.");
        }

        CalculateExternalCurlA(m_field_names[i]);

        // Generate parser for time function
        m_A_external_time_parser[i] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_A_ext_time_function[i],{"t",}));
        m_A_time_scale[i] = m_A_external_time_parser[i]->compile<1>();
    }

    UpdateHybridExternalFields(warpx.gett_new(0), warpx.getdt(0));
}


amrex::Real
ExternalVectorPotential::TimeScale (const int i, const amrex::Real t) const
{
    if (m_use_python_scale[i]) {
        const ScaleSegment& segment = m_python_scale[i];
        const amrex::Real slope =
            (segment.t_new > segment.t_old)
                ? (segment.s_new - segment.s_old) /
                      (segment.t_new - segment.t_old)
                : 0.0_rt;
        return segment.s_old + slope * (t - segment.t_old);
    }
    return m_A_time_scale[i](t);
}

void
ExternalVectorPotential::SetScale (const std::string& coil_name,
                                   const amrex::Real s_old,
                                   const amrex::Real s_new,
                                   const amrex::Real t_old,
                                   const amrex::Real t_new)
{
    for (int i = 0; i < m_nFields; ++i) {
        if (m_field_names[i] == coil_name) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_use_python_scale[i],
                "SetScale called for external field '" + coil_name +
                    "' which was not declared with external_vector_potential." +
                    coil_name + ".python_scale = 1");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                t_new >= t_old,
                "SetScale segment for '" + coil_name + "' must have t_new >= t_old");
            m_python_scale[i] = {s_old, s_new, t_old, t_new};
            return;
        }
    }
    WARPX_ABORT_WITH_MESSAGE(
        "SetScale: unknown external field '" + coil_name + "'");
}

amrex::Real
ExternalVectorPotential::GetScale (const std::string& coil_name,
                                   const amrex::Real t) const
{
    for (int i = 0; i < m_nFields; ++i) {
        if (m_field_names[i] == coil_name) {
            return TimeScale(i, t);
        }
    }
    WARPX_ABORT_WITH_MESSAGE(
        "GetScale: unknown external field '" + coil_name + "'");
    return 0.0_rt;
}

void
ExternalVectorPotential::CalculateExternalCurlA ()
{
    for (const auto& fname : m_field_names) {
        CalculateExternalCurlA(fname);
    }
}

void
ExternalVectorPotential::CalculateExternalCurlA (const std::string& coil_name)
{
    using ablastr::fields::Direction;
    auto & warpx = WarpX::GetInstance();

    // Compute the curl of the reference A field (unscaled by time function)
    const std::string Aext_field = coil_name + std::string{"_Aext"};
    const std::string curlAext_field = coil_name + std::string{"_curlAext"};

    ablastr::fields::MultiLevelVectorField A_ext =
        warpx.m_fields.get_mr_levels_alldirs(Aext_field, warpx.finestLevel());
    ablastr::fields::MultiLevelVectorField curlA_ext =
        warpx.m_fields.get_mr_levels_alldirs(curlAext_field, warpx.finestLevel());

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        // Compute the curl on the boxes grown into the domain ghosts as
        // well: A carries trusted ghost values (the analytic parser fill
        // and the file reader both cover the full grown box), so the ghost
        // curls are exact. FillBoundary below cannot reach non-periodic
        // domain ghosts -- without the growth, e.g. the RZ r_max ring of
        // the external B field would keep its allocation value (zero) and
        // inject a spurious drive-scaled field jump into every wall
        // stencil that reads it.
        amrex::IntVect ngrow = A_ext[lev][0]->nGrowVect();
        for (int idir = 0; idir < 3; ++idir) {
            ngrow = amrex::min(ngrow, A_ext[lev][idir]->nGrowVect());
            ngrow = amrex::min(ngrow, curlA_ext[lev][idir]->nGrowVect());
        }
        warpx.get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
            curlA_ext[lev],
            A_ext[lev],
            warpx.GetEBUpdateBFlag()[lev],
            lev,
            ngrow);

        for (int idir = 0; idir < 3; ++idir) {
            warpx.m_fields.get(curlAext_field, Direction{idir}, lev)->
                FillBoundary(warpx.Geom(lev).periodicity());
        }
    }
}

AMREX_FORCE_INLINE
void
ExternalVectorPotential::AddExternalFieldFromVectorPotential (
    ablastr::fields::VectorField const& dstField,
    amrex::Real scale_factor,
    ablastr::fields::VectorField const& srcField,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
    int lev)
{
    auto& warpx = WarpX::GetInstance();

    // Accumulate over the boxes grown into the domain ghosts as well: the
    // source ghosts are trusted there (the analytic A fill and its exact
    // ghost curls), which is what fills e.g. the RZ r_max ring of the
    // external E/B fields that wall-adjacent stencils read; FillBoundary
    // cannot reach non-periodic domain ghosts. No growth below the axis
    // in curvilinear geometries (no trusted parity data there), and none
    // when embedded boundaries are enabled (the eb_update flags are not
    // guaranteed to carry matching ghost data). There is no
    // double-counting of the += on shared box faces: MFIter::tilebox(nodal,
    // ngrow) grows only the tiles touching the valid-box edge, ghost
    // regions are per-FAB storage, and the nodal points shared between
    // neighboring FABs receive the same contribution in each FAB's own
    // array.
    amrex::IntVect ngrow = dstField[0]->nGrowVect();
    for (int idir = 0; idir < 3; ++idir) {
        ngrow = amrex::min(ngrow, dstField[idir]->nGrowVect());
        ngrow = amrex::min(ngrow, srcField[idir]->nGrowVect());
    }
    if (EB::enabled()) {
        ngrow = amrex::IntVect(0);
    }
    amrex::Box grow_region = warpx.Geom(lev).Domain();
    grow_region.grow(ngrow);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    if (warpx.Geom(lev).ProbLo(0) == 0.0_rt) {
        grow_region.setSmall(0, warpx.Geom(lev).Domain().smallEnd(0));
    }
#endif
    const amrex::Box allowed_x = amrex::convert(grow_region, dstField[0]->ixType());
    const amrex::Box allowed_y = amrex::convert(grow_region, dstField[1]->ixType());
    const amrex::Box allowed_z = amrex::convert(grow_region, dstField[2]->ixType());

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*dstField[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Extract field data for this grid/tile
        Array4<Real> const& Fx = dstField[0]->array(mfi);
        Array4<Real> const& Fy = dstField[1]->array(mfi);
        Array4<Real> const& Fz = dstField[2]->array(mfi);

        Array4<const Real> const& Sx = srcField[0]->const_array(mfi);
        Array4<const Real> const& Sy = srcField[1]->const_array(mfi);
        Array4<const Real> const& Sz = srcField[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        amrex::Array4<int> update_Fx_arr, update_Fy_arr, update_Fz_arr;
        if (EB::enabled()) {
            update_Fx_arr = eb_update[0]->array(mfi);
            update_Fy_arr = eb_update[1]->array(mfi);
            update_Fz_arr = eb_update[2]->array(mfi);
        }

        // Extract tileboxes for which to loop (grown into the domain
        // ghosts where allowed)
        Box const tbx = mfi.tilebox(dstField[0]->ixType().toIntVect(), ngrow) & allowed_x;
        Box const tby = mfi.tilebox(dstField[1]->ixType().toIntVect(), ngrow) & allowed_y;
        Box const tbz = mfi.tilebox(dstField[2]->ixType().toIntVect(), ngrow) & allowed_z;

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Fx_arr && update_Fx_arr(i, j, k) == 0) { return; }

                Fx(i,j,k) += scale_factor * Sx(i,j,k);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Fy_arr && update_Fy_arr(i, j, k) == 0) { return; }

                Fy(i,j,k) += scale_factor * Sy(i,j,k);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Fz_arr && update_Fz_arr(i, j, k) == 0) { return; }

                Fz(i,j,k) += scale_factor * Sz(i,j,k);
            }
        );
    }
}

void
ExternalVectorPotential::UpdateHybridExternalFields (const amrex::Real t, const amrex::Real dt,
                                                     const RefreshMode mode)
{
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

    ablastr::fields::MultiLevelVectorField B_ext =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_B_fp_external, warpx.finestLevel());
    ablastr::fields::MultiLevelVectorField E_ext =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_E_fp_external, warpx.finestLevel());

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mode == RefreshMode::All || (m_any_python_scale && m_static_cache_valid),
        "UpdateHybridExternalFields(CircuitOnly) requires at least one "
        "SetScale-driven field and a preceding All-mode refresh");

    ablastr::fields::MultiLevelVectorField B_static, E_static;
    if (m_any_python_scale) {
        B_static = warpx.m_fields.get_mr_levels_alldirs("hybrid_B_fp_external_static", warpx.finestLevel());
        E_static = warpx.m_fields.get_mr_levels_alldirs("hybrid_E_fp_external_static", warpx.finestLevel());
    }

    // Per-field accumulation into a destination register pair. The scale of
    // a SetScale-driven field follows its live piecewise-linear segment
    // (E carries the exact constant slope -- the discrete Faraday partner of
    // the linear B-scale); parser-driven fields keep the compiled time
    // function with a centered finite difference for the E scale.
    auto accumulate_field = [&](int i,
                                ablastr::fields::MultiLevelVectorField& dst_E,
                                ablastr::fields::MultiLevelVectorField& dst_B)
    {
        const std::string Aext_field = m_field_names[i] + std::string{"_Aext"};
        const std::string curlAext_field = m_field_names[i] + std::string{"_curlAext"};

        amrex::Real scale_factor_B;
        amrex::Real scale_factor_E;
        if (m_use_python_scale[i]) {
            const ScaleSegment& segment = m_python_scale[i];
            const amrex::Real slope =
                (segment.t_new > segment.t_old)
                    ? (segment.s_new - segment.s_old) /
                          (segment.t_new - segment.t_old)
                    : 0.0_rt;
            scale_factor_B = segment.s_old + slope * (t - segment.t_old);
            scale_factor_E = -slope;

            // A missed per-interval push silently ramps the drive along the
            // stale segment's extrapolation -- warn when the refresh time
            // has run well past the segment.
            if (segment.t_new > segment.t_old &&
                t > segment.t_new + 0.5_rt * (segment.t_new - segment.t_old)) {
                ablastr::warn_manager::WMRecordWarning(
                    "ExternalVectorPotential",
                    "External field '" + m_field_names[i] + "' is being "
                    "refreshed at t = " + std::to_string(t) + ", well past "
                    "its scale segment's t_new = " +
                    std::to_string(segment.t_new) + "; the drive is "
                    "extrapolating a stale segment (missed SetScale push?)",
                    ablastr::warn_manager::WarnPriority::high);
            }
        } else {
            // Get B-field Scaling Factor
            scale_factor_B = m_A_time_scale[i](t);

            // Get dA/dt scaling factor based on time centered FD around t
            const amrex::Real sf_l = m_A_time_scale[i](t-0.5_rt*dt);
            const amrex::Real sf_r = m_A_time_scale[i](t+0.5_rt*dt);
            scale_factor_E = -(sf_r - sf_l)/dt;
        }

        ablastr::fields::MultiLevelVectorField A_ext =
            warpx.m_fields.get_mr_levels_alldirs(Aext_field, warpx.finestLevel());
        ablastr::fields::MultiLevelVectorField curlA_ext =
            warpx.m_fields.get_mr_levels_alldirs(curlAext_field, warpx.finestLevel());

        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            AddExternalFieldFromVectorPotential(dst_E[lev], scale_factor_E, A_ext[lev], warpx.GetEBUpdateEFlag()[lev], lev);
            AddExternalFieldFromVectorPotential(dst_B[lev], scale_factor_B, curlA_ext[lev], warpx.GetEBUpdateBFlag()[lev], lev);
        }
    };

    auto copy_register = [&](ablastr::fields::MultiLevelVectorField& dst,
                             ablastr::fields::MultiLevelVectorField const& src)
    {
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            for (int idir = 0; idir < 3; ++idir) {
                auto& dst_mf = *dst[lev][Direction{idir}];
                auto const& src_mf = *src[lev][Direction{idir}];
                amrex::MultiFab::Copy(dst_mf, src_mf, 0, 0, dst_mf.nComp(),
                    amrex::min(dst_mf.nGrowVect(), src_mf.nGrowVect()));
            }
        }
    };

    if (mode == RefreshMode::All) {
        if (m_any_python_scale) {
            // Accumulate the parser-driven fields into the static cache,
            // seed the totals from it, then add the SetScale-driven fields.
            for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
                for (int idir = 0; idir < 3; ++idir) {
                    B_static[lev][Direction{idir}]->setVal(0.0_rt);
                    E_static[lev][Direction{idir}]->setVal(0.0_rt);
                }
            }
            for (int i = 0; i < m_nFields; ++i) {
                if (!m_use_python_scale[i]) { accumulate_field(i, E_static, B_static); }
            }
            m_static_cache_valid = true;
            copy_register(E_ext, E_static);
            copy_register(B_ext, B_static);
        } else {
            for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
                for (int idir = 0; idir < 3; ++idir) {
                    B_ext[lev][Direction{idir}]->setVal(0.0_rt);
                    E_ext[lev][Direction{idir}]->setVal(0.0_rt);
                }
            }
            for (int i = 0; i < m_nFields; ++i) {
                accumulate_field(i, E_ext, B_ext);
            }
        }
    } else {
        // CircuitOnly: the parser-driven fields stay frozen at their cached
        // sums (exactly as they are frozen between the step's scheduled
        // All refreshes).
        copy_register(E_ext, E_static);
        copy_register(B_ext, B_static);
    }

    if (m_any_python_scale) {
        for (int i = 0; i < m_nFields; ++i) {
            if (m_use_python_scale[i]) { accumulate_field(i, E_ext, B_ext); }
        }
    }

    // One ghost exchange per component after all accumulation (the
    // accumulation itself never reads ghosts, so exchanging once at the end
    // is exact and avoids the per-field FillBoundary cost).
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idir = 0; idir < 3; ++idir) {
            E_ext[lev][Direction{idir}]->FillBoundary(warpx.Geom(lev).periodicity());
            B_ext[lev][Direction{idir}]->FillBoundary(warpx.Geom(lev).periodicity());
        }
    }
    amrex::Gpu::streamSynchronize();
}
