/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Prabhat Kumar (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EXTERNAL_FIELD_SOURCE_H_
#define WARPX_EXTERNAL_FIELD_SOURCE_H_

#include "Fields.H"

#include "Utils/WarpXAlgorithmSelection.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Initialization/ExternalField.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <optional>

class ExternalFieldSource
{
protected:

    // Initialization Type for External E and B excitation on grid
    static std::string B_excitation_grid_s;
    static std::string E_excitation_grid_s;

    static int ApplyExcitationInPML;

    // Function for B excitation (x,y,z,t) on the grid
    static std::string str_Bx_excitation_grid_function;
    static std::string str_By_excitation_grid_function;
    static std::string str_Bz_excitation_grid_function;
    // Function for E excitation (x,y,z,t) on the grid
    static std::string str_Ex_excitation_grid_function;
    static std::string str_Ey_excitation_grid_function;
    static std::string str_Ez_excitation_grid_function;

    // Flag for type of excitation (hard/soft)
    static std::string str_Ex_excitation_flag_function;
    static std::string str_Ey_excitation_flag_function;
    static std::string str_Ez_excitation_flag_function;
    static std::string str_Bx_excitation_flag_function;
    static std::string str_By_excitation_flag_function;
    static std::string str_Bz_excitation_flag_function;

    // Parser for B_external excitation on the grid
    std::unique_ptr<amrex::Parser> Bxfield_xt_grid_parser;
    std::unique_ptr<amrex::Parser> Byfield_xt_grid_parser;
    std::unique_ptr<amrex::Parser> Bzfield_xt_grid_parser;
    // Parser for E_external excitation on the grid
    std::unique_ptr<amrex::Parser> Exfield_xt_grid_parser;
    std::unique_ptr<amrex::Parser> Eyfield_xt_grid_parser;
    std::unique_ptr<amrex::Parser> Ezfield_xt_grid_parser;

    std::unique_ptr<amrex::Parser> Exfield_flag_parser;
    std::unique_ptr<amrex::Parser> Eyfield_flag_parser;
    std::unique_ptr<amrex::Parser> Ezfield_flag_parser;
    std::unique_ptr<amrex::Parser> Bxfield_flag_parser;
    std::unique_ptr<amrex::Parser> Byfield_flag_parser;
    std::unique_ptr<amrex::Parser> Bzfield_flag_parser;

public:

    // Default Constructor
    ExternalFieldSource ();

        /** \brief Adds the contribution of user-defined external field-excitation
     *   to Efield, Bfield, and also Hfield if `USE_LLG=TRUE`.
     *   Two types of excitation are supported, namely, hard source and soft source.
     *   An excitation is called hard source if the field value is set exactly equal
     *   to the excitation (similar to a Dirichlet boundary) (field = excitation).
     *   An excitation is called soft source if the field value is updated by adding
     *   to it the contribution from the externally applied excitation (field += excitation).
     *   To ensure that the excitation is appropriately accounted for, a flag parser for
     *   each component of the field must be specified.
     *   The flag can have three values :
     *       If flag_type == 0, field is not updated with external excitation
     *       If flag_type == 1, field is set equal to the
     *                          external excitation (aka hard source)
     *       If flag_type == 2, field is updated to add the contribution from
     *                          external excitation (aka soft source)
     *
     *   \param[in] mfx, mfy, mfz : The field component Multifabs to be updated with
     *                              the external excitation.
     *   \param[in] xfield_parser : external excitation for xcomponent of the field
     *   \param[in] yfield_parser : external excitation for ycomponent of the field
     *   \param[in] zfield_parser : external excitation for zcomponent of the field
     *   \param[in] xflag_parser  : Type xfield excitation (hard source=0/soft source=1)
     *   \param[in] yflag_parser  : Type yfield excitation (hard source=0/soft source=1)
     *   \param[in] zflag_parser  : Type zfield excitation (hard source=0/soft source=1)
     *   \param[in] lev           : level on which the excitation is applied.
     */
    void ApplyExternalFieldExcitationOnGrid (int const externalfieldtype, DtType a_dt_type = DtType::Full);
    void ApplyExternalFieldExcitationOnGrid ( amrex::MultiFab *mfx,
         amrex::MultiFab *mfy, amrex::MultiFab *mfz,
         amrex::ParserExecutor<4> const& xfield_parser,
         amrex::ParserExecutor<4> const& yfield_parser,
         amrex::ParserExecutor<4> const& zfield_parser,
         amrex::ParserExecutor<3> const& xflag_parser,
         amrex::ParserExecutor<3> const& yflag_parser,
         amrex::ParserExecutor<3> const& zflag_parser, const int lev,
         DtType a_dt_type );
    /** Parse field excitation functions and flags*/
    void ReadExcitationParser ();

};

#endif //WARPX_EXTERNAL_FIELD_SOURCE_H_
