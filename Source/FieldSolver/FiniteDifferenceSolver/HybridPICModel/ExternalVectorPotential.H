/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EXTERNAL_VECTOR_POTENTIAL_H_
#define WARPX_EXTERNAL_VECTOR_POTENTIAL_H_

#include "Fields.H"

#include "Utils/WarpXAlgorithmSelection.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <optional>

/**
 * \brief This class contains the parameters needed to evaluate a
 * time varying external vector potential, leading to external E/B
 * fields to be applied in Hybrid Solver. This class is used to break up
 * the passed in fields into a spatial and time dependent solution.
 *
 * Eventually this can be used in a list to control independent external
 * fields with different time profiles.
 *
 */
class ExternalVectorPotential
{
protected:
    int m_nFields;

    std::vector<std::string> m_field_names;

    std::vector<std::string> m_Ax_ext_grid_function;
    std::vector<std::string> m_Ay_ext_grid_function;
    std::vector<std::string> m_Az_ext_grid_function;
    std::vector<std::array< std::unique_ptr<amrex::Parser>, 3>> m_A_external_parser;
    std::vector<std::array< amrex::ParserExecutor<4>, 3>> m_A_external;

    std::vector<std::string> m_A_ext_time_function;
    std::vector<std::unique_ptr<amrex::Parser>> m_A_external_time_parser;
    std::vector<amrex::ParserExecutor<1>> m_A_time_scale;

    std::vector<bool> m_read_A_from_file;
    std::vector<std::string> m_external_file_path;

    // Clean div A
    bool m_do_clean_divA = true;

    /** Piecewise-linear scale segment for externally driven (e.g.
     * circuit-coupled) fields: the field scale is s_old at t_old and s_new
     * at t_new, interpolated linearly in between and extrapolated linearly
     * beyond t_new (so a per-step push contract is required; a missed push
     * silently ramps the drive, which UpdateHybridExternalFields warns
     * about). The E-field scale is the exact constant slope
     * -(s_new - s_old)/(t_new - t_old), which is the discrete Faraday
     * partner of the linear B-scale (no finite-difference approximation).
     */
    struct ScaleSegment {
        amrex::Real s_old = 1.0;
        amrex::Real s_new = 1.0;
        amrex::Real t_old = 0.0;
        amrex::Real t_new = 0.0;
    };

    /** Per-field flag: scale comes from SetScale instead of the compiled
     * A_time_external_function parser. Set by <name>.python_scale. */
    std::vector<bool> m_use_python_scale;
    std::vector<ScaleSegment> m_python_scale;

public:

    /** Number of independent external vector-potential fields. */
    [[nodiscard]] int nFields () const { return m_nFields; }

    /** Registry base name of external field i (fields "<name>_Aext"). */
    [[nodiscard]] const std::string& FieldName (int i) const { return m_field_names[i]; }

    /** Scale of external field i at time t: the live piecewise-linear
     * segment for SetScale-driven fields, otherwise the compiled
     * A_time_external_function parser. */
    [[nodiscard]] amrex::Real TimeScale (int i, amrex::Real t) const;

    /** Whether external field i is scale-driven through SetScale. */
    [[nodiscard]] bool UsesPythonScale (int i) const { return m_use_python_scale[i]; }

    /** Set the piecewise-linear scale segment of a scale-driven field for
     * the upcoming coupling interval. Must be called before the interval's
     * field refresh (e.g. from a "beforestep" callback or the circuit
     * coupler); aborts if the field was not declared with
     * <name>.python_scale = 1 or the name is unknown.
     */
    void SetScale (const std::string& coil_name, amrex::Real s_old,
                   amrex::Real s_new, amrex::Real t_old, amrex::Real t_new);

    /** Current scale of a named field at time t (parser or segment). */
    [[nodiscard]] amrex::Real GetScale (const std::string& coil_name,
                                        amrex::Real t) const;

protected:

    /** Whether any field is SetScale-driven (fixed after ReadParameters). */
    bool m_any_python_scale = false;

    /** Whether the parser-driven ("static within a coupling interval")
     * partial sums cached in the <...>_static registry fields are current.
     * Set by an All refresh, consumed by CircuitOnly refreshes. */
    bool m_static_cache_valid = false;

public:

    // Default Constructor
    ExternalVectorPotential ();

    void ReadParameters ();

    void AllocateLevelMFs (
        ablastr::fields::MultiFabRegister & fields,
        int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
        int ncomps,
        const amrex::IntVect& ngEB,
        const amrex::IntVect& Ex_nodal_flag,
        const amrex::IntVect& Ey_nodal_flag,
        const amrex::IntVect& Ez_nodal_flag,
        const amrex::IntVect& Bx_nodal_flag,
        const amrex::IntVect& By_nodal_flag,
        const amrex::IntVect& Bz_nodal_flag
    );

    void InitData ();

    void CalculateExternalCurlA ();
    void CalculateExternalCurlA (const std::string& coil_name);

    AMREX_FORCE_INLINE
    void AddExternalFieldFromVectorPotential (
        ablastr::fields::VectorField const& dstField,
        amrex::Real scale_factor,
        ablastr::fields::VectorField const& srcField,
        std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update,
        int lev);

    /** Which fields to re-accumulate in UpdateHybridExternalFields.
     *
     * All: every field; when SetScale-driven fields exist, the sums of the
     * parser-driven fields are cached to the <...>_static registry fields
     * as a side effect. CircuitOnly: reuse those cached sums (the
     * parser-driven fields stay frozen at their last All refresh, exactly
     * as they are frozen between the step's scheduled refreshes) and
     * re-accumulate only the SetScale-driven fields at their live
     * segments -- the cheap per-substep refresh for circuit-coupled
     * drives. Both modes end with a single FillBoundary per component.
     */
    enum class RefreshMode { All, CircuitOnly };

    void UpdateHybridExternalFields (
        amrex::Real t,
        amrex::Real dt,
        RefreshMode mode = RefreshMode::All
    );
};

#endif //WARPX_TIME_DEPENDENT_VECTOR_POTENTIAL_H_
