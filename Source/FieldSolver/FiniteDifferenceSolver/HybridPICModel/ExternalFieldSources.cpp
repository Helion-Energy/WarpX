/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Prabhat Kumar (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ExternalFieldSources.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Fields.H"
#include "WarpX.H"
#include "BoundaryConditions/PML.H"
#include "Evolve/WarpXDtType.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/Parser/ParserUtils.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>

#include <ablastr/fields/MultiFabRegister.H>

using namespace amrex;
using namespace warpx::fields;

ExternalFieldSource::ExternalFieldSource ()
{
    ReadExcitationParser();
}

void
ExternalFieldSource::ReadExcitationParser ()
{
    ParmParse pp_warpx("warpx");

    // Query for type of external space-time (xt) varying excitation
    pp_warpx.query("B_excitation_on_grid_style", B_excitation_grid_s);
    std::transform(B_excitation_grid_s.begin(),
               B_excitation_grid_s.end(),
               B_excitation_grid_s.begin(),
               ::tolower);

    pp_warpx.query("E_excitation_on_grid_style", E_excitation_grid_s);
    std::transform(E_excitation_grid_s.begin(),
                   E_excitation_grid_s.end(),
                   E_excitation_grid_s.begin(),
                   ::tolower);

    if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
        // if E excitation type is set to parser then the corresponding
        // source type (hard=1, soft=2) must be specified for all components
        // using the flag function. Note that a flag value of 0 will not update
        // the field with the excitation.
        utils::parser::Store_parserString(pp_warpx, "Ex_excitation_flag_function(x,y,z)",
                                str_Ex_excitation_flag_function);
        utils::parser::Store_parserString(pp_warpx, "Ey_excitation_flag_function(x,y,z)",
                                str_Ey_excitation_flag_function);
        utils::parser::Store_parserString(pp_warpx, "Ez_excitation_flag_function(x,y,z)",
                                str_Ez_excitation_flag_function);
        Exfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ex_excitation_flag_function,{"x","y","z"}));
        Eyfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ey_excitation_flag_function,{"x","y","z"}));
        Ezfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ez_excitation_flag_function,{"x","y","z"}));

        pp_warpx.query("Apply_E_excitation_in_pml_region", ApplyExcitationInPML);
    }
    if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
        // if B excitation type is set to parser then the corresponding
        // source type (hard=1, soft=2) must be specified for all components
        // using the flag function. Note that a flag value of 0 will not update
        // the field with the excitation.
        utils::parser::Store_parserString(pp_warpx, "Bx_excitation_flag_function(x,y,z)",
                                str_Bx_excitation_flag_function);
        utils::parser::Store_parserString(pp_warpx, "By_excitation_flag_function(x,y,z)",
                                str_By_excitation_flag_function);
        utils::parser::Store_parserString(pp_warpx, "Bz_excitation_flag_function(x,y,z)",
                                str_Bz_excitation_flag_function);
        Bxfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Bx_excitation_flag_function,{"x","y","z"}));
        Byfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_By_excitation_flag_function,{"x","y","z"}));
        Bzfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Bz_excitation_flag_function,{"x","y","z"}));
    }

    // make parser for the external B-excitation in space-time
    if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
#ifdef WARPX_DIM_RZ
       amrex::Abort("E and B parser for external fields does not work with RZ -- TO DO");
#endif
       utils::parser::Store_parserString(pp_warpx, "Bx_excitation_grid_function(x,y,z,t)",
                                                    str_Bx_excitation_grid_function);
       utils::parser::Store_parserString(pp_warpx, "By_excitation_grid_function(x,y,z,t)",
                                                    str_By_excitation_grid_function);
       utils::parser::Store_parserString(pp_warpx, "Bz_excitation_grid_function(x,y,z,t)",
                                                    str_Bz_excitation_grid_function);
       Bxfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Bx_excitation_grid_function,{"x","y","z","t"}));
       Byfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_By_excitation_grid_function,{"x","y","z","t"}));
       Bzfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Bz_excitation_grid_function,{"x","y","z","t"}));
    }

    // make parser for the external E-excitation in space-time
    if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
#ifdef WARPX_DIM_RZ
       amrex::Abort("E and B parser for external fields does not work with RZ -- TO DO");
#endif
       utils::parser::Store_parserString(pp_warpx, "Ex_excitation_grid_function(x,y,z,t)",
                                                    str_Ex_excitation_grid_function);
       utils::parser::Store_parserString(pp_warpx, "Ey_excitation_grid_function(x,y,z,t)",
                                                    str_Ey_excitation_grid_function);
       utils::parser::Store_parserString(pp_warpx, "Ez_excitation_grid_function(x,y,z,t)",
                                                    str_Ez_excitation_grid_function);
       Exfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ex_excitation_grid_function,{"x","y","z","t"}));
       Eyfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ey_excitation_grid_function,{"x","y","z","t"}));
       Ezfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ez_excitation_grid_function,{"x","y","z","t"}));
    }
}

void
ExternalFieldSource::ApplyExternalFieldExcitationOnGrid (int const externalfieldtype, DtType a_dt_type)
{
    auto& warpx = WarpX::GetInstance();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::EfieldExternal) {
            if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
                ApplyExternalFieldExcitationOnGrid(Efield_fp[lev][0].get(),
                                                   Efield_fp[lev][1].get(),
                                                   Efield_fp[lev][2].get(),
                                                   Exfield_xt_grid_parser->compile<4>(),
                                                   Eyfield_xt_grid_parser->compile<4>(),
                                                   Ezfield_xt_grid_parser->compile<4>(),
                                                   Exfield_flag_parser->compile<3>(),
                                                   Eyfield_flag_parser->compile<3>(),
                                                   Ezfield_flag_parser->compile<3>(),
                                                   lev, a_dt_type );
            }
        }
        // The excitation, especially when used to set an internal PEC, will be extended
        // to the PML region with user-defined parser.
        // As clarified in the documentation, it is important that the parser is valid in the pml region
        if (WarpX::isAnyBoundaryPML() and externalfieldtype == ExternalFieldType::EfieldExternalPML) {
            if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
                    ApplyExternalFieldExcitationOnGrid(pml[lev]->GetE_fp(0),
                                                       pml[lev]->GetE_fp(1),
                                                       pml[lev]->GetE_fp(2),
                                                       Exfield_xt_grid_parser->compile<4>(),
                                                       Eyfield_xt_grid_parser->compile<4>(),
                                                       Ezfield_xt_grid_parser->compile<4>(),
                                                       Exfield_flag_parser->compile<3>(),
                                                       Eyfield_flag_parser->compile<3>(),
                                                       Ezfield_flag_parser->compile<3>(),
                                                       lev, a_dt_type );
            }
        }
        if (externalfieldtype == ExternalFieldType::AllExternal || externalfieldtype == ExternalFieldType::BfieldExternal) {
            if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
                ApplyExternalFieldExcitationOnGrid(Bfield_fp[lev][0].get(),
                                                   Bfield_fp[lev][1].get(),
                                                   Bfield_fp[lev][2].get(),
                                                   Bxfield_xt_grid_parser->compile<4>(),
                                                   Byfield_xt_grid_parser->compile<4>(),
                                                   Bzfield_xt_grid_parser->compile<4>(),
                                                   Bxfield_flag_parser->compile<3>(),
                                                   Byfield_flag_parser->compile<3>(),
                                                   Bzfield_flag_parser->compile<3>(),
                                                   lev, a_dt_type );
            }
        }
    } // for loop over level
}


