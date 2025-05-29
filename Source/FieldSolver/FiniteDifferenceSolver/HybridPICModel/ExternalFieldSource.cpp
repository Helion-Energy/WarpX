/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Prabhat Kumar (Helion Energy Inc)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"
#include "Fields.H"
#include "ExternalFieldSource.H"
#include "BoundaryConditions/PML.H"
#include "Evolve/WarpXDtType.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/Parser/ParserUtils.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>

using namespace amrex;

// Define all static member variables declared in the header
std::string ExternalFieldSource::B_excitation_grid_s = "";
std::string ExternalFieldSource::E_excitation_grid_s = "";
int ExternalFieldSource::ApplyExcitationInPML = 0;
int ExternalFieldSource::excite_E_source = 0;
int ExternalFieldSource::excite_B_source = 0;
int ExternalFieldSource::excite_all = 0;

std::string ExternalFieldSource::str_Bx_excitation_grid_function = "";
std::string ExternalFieldSource::str_By_excitation_grid_function = "";
std::string ExternalFieldSource::str_Bz_excitation_grid_function = "";

std::string ExternalFieldSource::str_Ex_excitation_grid_function = "";
std::string ExternalFieldSource::str_Ey_excitation_grid_function = "";
std::string ExternalFieldSource::str_Ez_excitation_grid_function = "";

std::string ExternalFieldSource::str_Ex_excitation_flag_function = "";
std::string ExternalFieldSource::str_Ey_excitation_flag_function = "";
std::string ExternalFieldSource::str_Ez_excitation_flag_function = "";

std::string ExternalFieldSource::str_Bx_excitation_flag_function = "";
std::string ExternalFieldSource::str_By_excitation_flag_function = "";
std::string ExternalFieldSource::str_Bz_excitation_flag_function = "";


ExternalFieldSource::ExternalFieldSource ()
{
	ReadExcitationParser();
}

void
ExternalFieldSource::ApplyExternalFieldExcitationOnGrid (DtType a_dt_type,  bool apply_E, bool apply_B)
{
        const int finest_level = 0;
    	for (int lev = 0; lev <= finest_level; ++lev) {
        
	using warpx::fields::FieldType;
         
	if ((apply_E && (excite_E_source || excite_all)) &&
             E_excitation_grid_s == "parse_e_excitation_grid_function")
        {
        //if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
            ApplyExternalFieldExcitationOnGrid(FieldType::Efield_fp,
                                               Exfield_xt_grid_parser->compile<4>(),
                                               Eyfield_xt_grid_parser->compile<4>(),
                                               Ezfield_xt_grid_parser->compile<4>(),
                                               Exfield_flag_parser->compile<3>(),
                                               Eyfield_flag_parser->compile<3>(),
                                               Ezfield_flag_parser->compile<3>(),
                                               lev, a_dt_type );
        }

	if ((apply_B && (excite_B_source || excite_all)) &&
             B_excitation_grid_s == "parse_b_excitation_grid_function")
        {
        //if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
            ApplyExternalFieldExcitationOnGrid(FieldType::Bfield_fp,
                                               Bxfield_xt_grid_parser->compile<4>(),
                                               Byfield_xt_grid_parser->compile<4>(),
                                               Bzfield_xt_grid_parser->compile<4>(),
                                               Bxfield_flag_parser->compile<3>(),
                                               Byfield_flag_parser->compile<3>(),
                                               Bzfield_flag_parser->compile<3>(),
                                               lev, a_dt_type );
        }
    } // for loop over level
}

void
ExternalFieldSource::ApplyExternalFieldExcitationOnGrid (
       warpx::fields::FieldType field,
       amrex::ParserExecutor<4> const& xfield_parser,
       amrex::ParserExecutor<4> const& yfield_parser,
       amrex::ParserExecutor<4> const& zfield_parser,
       amrex::ParserExecutor<3> const& xflag_parser,
       amrex::ParserExecutor<3> const& yflag_parser,
       amrex::ParserExecutor<3> const& zflag_parser, 
       const int lev, DtType a_dt_type )
{


    auto &warpx = WarpX::GetInstance();
    auto const &geom = warpx.Geom(lev);

    using ablastr::fields::Direction;
    amrex::MultiFab* mfx = warpx.m_fields.get(field, Direction{0}, lev);
    amrex::MultiFab* mfy = warpx.m_fields.get(field, Direction{1}, lev);
    amrex::MultiFab* mfz = warpx.m_fields.get(field, Direction{2}, lev);

    // This function adds the contribution from an external excitation to the fields.
    // A flag is used to determine the type of excitation.
    // If flag == 1, it is a hard source and the field = excitation
    // If flag == 2, if is a soft source and the field += excitation
    // If flag == 0, the excitation parser is not computed and the field is unchanged.
    // If flag is not 0, or 1, or 2, the code will Abort!

    // Gpu vector to store Ex-Bz staggering
    GpuArray<int, AMREX_SPACEDIM> mfx_stag, mfy_stag, mfz_stag;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        mfx_stag[idim] = mfx->ixType()[idim];
        mfy_stag[idim] = mfy->ixType()[idim];
        mfz_stag[idim] = mfz->ixType()[idim];
    }

    amrex::Real t = warpx.gett_new(lev);
    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();

    amrex::IntVect x_nodal_flag = mfx->ixType().toIntVect();
    amrex::IntVect y_nodal_flag = mfy->ixType().toIntVect();
    amrex::IntVect z_nodal_flag = mfz->ixType().toIntVect();
    // For each multifab, apply excitation to ncomponents
    // If not split pml fields, the excitation is applied to the regular Efield used in Maxwell's eq.
    // If pml field, then the excitation is applied to all the split field components.
    const int nComp_x = mfx->nComp();
    const int nComp_y = mfy->nComp();
    const int nComp_z = mfz->nComp();
    // Multiplication factor for field parser depending on dt_type
    // If Full, then 1 (default), if FirstHalf or SecondHalf then 0.5
    int dt_type_flag = 0;
    if (a_dt_type == DtType::FirstHalf or a_dt_type == DtType::SecondHalf ) {
        dt_type_flag = 1;
    }

//    amrex::Print() << "ApplyExternalFieldExcitationOnGrid is called. field = " << static_cast<int>(field) << ", DtType = " << dt_type_flag <<"\n"; 

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Extract field data for this grid/tile
        amrex::Array4<amrex::Real> const& Fx = mfx->array(mfi);
        amrex::Array4<amrex::Real> const& Fy = mfy->array(mfi);
        amrex::Array4<amrex::Real> const& Fz = mfz->array(mfi);

        const amrex::Box& tbx = mfi.tilebox( x_nodal_flag, mfx->nGrowVect() );
        const amrex::Box& tby = mfi.tilebox( y_nodal_flag, mfy->nGrowVect() );
        const amrex::Box& tbz = mfi.tilebox( z_nodal_flag, mfz->nGrowVect() );

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, nComp_x,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfx_stag.data(),
                                                  problo.data(), dx.data(), x, y, z);
                auto flag_type = xflag_parser(x,y,z);
/*
		if (flag_type > 0){
			//amrex::Print() << "(x,y,z) = (" << x << ", " << y << ", " << z << "). flag type = " << flag_type << ", Fx(i,j,k,n) =  " << Fx(i,j,k,n) << ", xfield_parser(x,y,z,t) = " << xfield_parser(x,y,z,t) << "\n";
			amrex::Print() << "field" << static_cast<int>(field) <<"(" << i << ", " << j << ", " << k << "). flag type = " << flag_type << ", Fx(i,j,k,n) =  " << Fx(i,j,k,n) << ", xfield_parser(x,y,z,t) = " << xfield_parser(x,y,z,t) << "\n";
		}
*/
		amrex::Real dt_type_factor = 1._rt;
                // For soft source and FirstHalf/SecondHalf evolve
                // the excitation is split with a prefector of 0.5
                if (flag_type == 2._rt and dt_type_flag == 1) {
                    dt_type_factor = 0.5_rt;
                }
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fx(i, j, k, n) = Fx(i,j,k,n)*(flag_type-1.0_rt)
                                   + dt_type_factor * xfield_parser(x,y,z,t);
                }
            },
            tby, nComp_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfy_stag.data(),
                                                  problo.data(), dx.data(), x, y, z);
                auto flag_type = yflag_parser(x,y,z);
                amrex::Real dt_type_factor = 1._rt;
                // For soft source and FirstHalf/SecondHalf evolve
                // the excitation is split with a prefector of 0.5
                if (flag_type == 2._rt and dt_type_flag == 1) {
                    dt_type_factor = 0.5_rt;
                }
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fy(i, j, k, n) = Fy(i,j,k,n)*(flag_type-1.0_rt)
                                   + dt_type_factor * yfield_parser(x,y,z,t);
                }
            },
            tbz, nComp_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfz_stag.data(),
                                                  problo.data(), dx.data(), x, y, z);
                auto flag_type = zflag_parser(x,y,z);
                amrex::Real dt_type_factor = 1._rt;
                // For soft source and FirstHalf/SecondHalf evolve
                // the excitation is split with a prefector of 0.5
                if (flag_type == 2._rt and dt_type_flag == 1) {
                    dt_type_factor = 0.5_rt;
                }
                if (flag_type != 0._rt && flag_type != 1._rt && flag_type != 2._rt) {
                    amrex::Abort("flag type for excitation must be 0, or 1, or 2!");
                } else if ( flag_type > 0._rt ) {
                    Fz(i, j, k,n) = Fz(i,j,k,n)*(flag_type-1.0_rt)
                                  + dt_type_factor * zfield_parser(x,y,z,t);
                }
            }
        );
    }
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

    utils::parser::queryWithParser(pp_warpx, "excite_E_source", excite_E_source);
    utils::parser::queryWithParser(pp_warpx, "excite_B_source", excite_B_source);
    utils::parser::queryWithParser(pp_warpx, "excite_all", excite_all);

    amrex::Print() << "excite_E_source = " << excite_E_source << ", excite_B_source = " << excite_B_source << ", excite_all = " << excite_all << "\n";

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
