/* Copyright 2021 Lorenzo Giacomel
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpX.H"

#include "EmbeddedBoundary/Enabled.H"
#ifdef AMREX_USE_EB
#  include "Fields.H"
#  include "Utils/Parser/ParserUtils.H"
#  include "Utils/TextMsg.H"

#   include <AMReX_BLProfiler.H>
#   include <AMReX_BoxArray.H>
#   include <AMReX_Config.H>
#   include <AMReX_EB2.H>
#   include <AMReX_EB2_GeometryShop.H>
#   include <AMReX_EB2_IF_Base.H>
#   include <AMReX_EB_utils.H>
#   include <AMReX_GpuQualifiers.H>
#   include <AMReX_ParmParse.H>
#   include <AMReX_REAL.H>
#   include <AMReX_SPACE.H>

#  include <algorithm>
#  include <cstdlib>
#  include <string>

using namespace ablastr::fields;

#endif

#ifdef AMREX_USE_EB
namespace {
    class ParserIF
        : public amrex::GPUable
    {
    public:
        explicit
        ParserIF (const amrex::ParserExecutor<3>& a_parser)
            : m_parser(a_parser)
            {}

        ParserIF (const ParserIF& rhs) noexcept = default;
        ParserIF (ParserIF&& rhs) noexcept = default;
        ParserIF& operator= (const ParserIF& rhs) = delete;
        ParserIF& operator= (ParserIF&& rhs) = delete;

        ~ParserIF() = default;

        AMREX_GPU_HOST_DEVICE inline
        amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                             amrex::Real z)) const noexcept {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            return m_parser(x,amrex::Real(0.0),y);
#else
            return m_parser(x,y,z);
#endif
        }

        inline amrex::Real operator() (const amrex::RealArray& p) const noexcept {
            return this->operator()(AMREX_D_DECL(p[0],p[1],p[2]));
        }

    private:
        amrex::ParserExecutor<3> m_parser; //! function parser with three arguments (x,y,z)
    };
}
#endif

void
WarpX::InitEB ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("InitEB only works when EBs are enabled at runtime");
    }

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("EBs only implemented in 2D and 3D");
#endif

#ifdef AMREX_USE_EB
    BL_PROFILE("InitEB");

    const amrex::ParmParse pp_warpx("warpx");
    pp_warpx.query("build_eb_data_per_level", m_build_eb_data_per_level);
    std::string impf;
    pp_warpx.query("eb_implicit_function", impf);
    if (! impf.empty()) {
        auto eb_if_parser = utils::parser::makeParser(impf, {"x", "y", "z"});
        ParserIF const pif(eb_if_parser.compile<3>());
        auto gshop = amrex::EB2::makeShop(pif, eb_if_parser);
         // The last argument of amrex::EB2::Build is the maximum coarsening level
         // to which amrex should try to coarsen the EB.  It will stop after coarsening
         // as much as it can, if it cannot coarsen to that level.  Here we use a big
         // number (e.g., maxLevel()+20) for multigrid solvers.  Because the coarse
         // level has only 1/8 of the cells on the fine level, the memory usage should
         // not be an issue.
        if (m_build_eb_data_per_level) {
            // Build the EB data independently at each mesh-refinement level's own resolution.
            for (int ilev = 0; ilev <= maxLevel(); ++ilev) {
                amrex::EB2::Build(gshop, Geom(ilev), 0, 20,
                                 std::max(4, eb_distance_guard_cells));
                m_eb_index_space.push_back(&(amrex::EB2::IndexSpace::top()));
            }
        } else {
            // Build the EB data once at the finest level and coarsen it for coarser levels.
            amrex::EB2::Build(gshop, Geom(maxLevel()), maxLevel(), maxLevel()+20,
                             std::max(4, eb_distance_guard_cells));
        }
    } else {
        amrex::ParmParse pp_eb2("eb2");
        if (!pp_eb2.contains("geom_type")) {
            std::string const geom_type = "all_regular";
            pp_eb2.add("geom_type", geom_type); // use all_regular by default
        }
        // See the comment above on amrex::EB2::Build for the hard-wired number 20.
        if (m_build_eb_data_per_level) {
            // Build the EB data independently at each mesh-refinement level's own resolution.
            for (int ilev = 0; ilev <= maxLevel(); ++ilev) {
                amrex::EB2::Build(Geom(ilev), 0, 20, std::max(4, eb_distance_guard_cells));
                m_eb_index_space.push_back(&(amrex::EB2::IndexSpace::top()));
            }
        } else {
            // Build the EB data once at the finest level and coarsen it for coarser levels.
            amrex::EB2::Build(Geom(maxLevel()), maxLevel(), maxLevel()+20,
                             std::max(4, eb_distance_guard_cells));
        }
    }
#endif
}

void
WarpX::ComputeDistanceToEB ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeDistanceToEB only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeDistanceToEB");
    using warpx::fields::FieldType;
    for (int lev=0; lev<=maxLevel(); lev++) {
        auto const* eb_index_space = GetEBIndexSpace(lev);
        const amrex::EB2::Level& eb_level = eb_index_space->getLevel(Geom(lev));
        auto const eb_fact = fieldEBFactory(lev);
        if (eb_boundary_type == EmbeddedBoundaryType::Insulating) {
            auto const dx = Geom(lev).CellSizeArray();
            amrex::Real const dx_min = amrex::min(AMREX_D_DECL(dx[0], dx[1], dx[2]));
            amrex::Real const dx_max = amrex::max(AMREX_D_DECL(dx[0], dx[1], dx[2]));
            amrex::Real const roof = dx_min *
                (eb_fact.getMultiEBCellFlagFab().nGrow()+1);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(eb_standoff_cells*dx_max < roof,
                "Insulating EB standoff reaches the AMReX signed-distance ceiling; "
                "increase boundary.eb_distance_guard_cells before collecting particles");
            amrex::Print() << "Insulating EB distance support: level=" << lev
                           << " standoff_m=" << eb_standoff_cells*dx_max
                           << " distance_ceiling_m=" << roof
                           << " geometry_guards=" << eb_fact.getMultiEBCellFlagFab().nGrow()
                           << "\n";
        }
        amrex::FillSignedDistance(*m_fields.get(FieldType::distance_to_eb, lev), eb_level, eb_fact, 1);
#if defined(WARPX_DIM_3D)
        if (eb_footprint_cylinder_radius > 0.0) {
            // Certify every cell intersecting the interior cylinder is regular.
            auto const dx = Geom(lev).CellSizeArray();
            auto const plo = Geom(lev).ProbLoArray();
            auto const radius2 = eb_footprint_cylinder_radius*eb_footprint_cylinder_radius;
            amrex::Gpu::DeviceScalar<int> bad(0);
            auto* failure = bad.dataPtr();
            auto const& flag_mf = eb_fact.getMultiEBCellFlagFab();
            for (amrex::MFIter mfi(flag_mf); mfi.isValid(); ++mfi) {
                auto const flag = flag_mf.const_array(mfi);
                amrex::For(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const xl = plo[0]+i*dx[0], xh = xl+dx[0];
                    auto const yl = plo[1]+j*dx[1], yh = yl+dx[1];
                    auto const x = xl > 0.0 ? xl : (xh < 0.0 ? -xh : 0.0);
                    auto const y = yl > 0.0 ? yl : (yh < 0.0 ? -yh : 0.0);
                    if (x*x+y*y < radius2 && !flag(i,j,k).isRegular()) {
                        amrex::Gpu::Atomic::AddNoRet(failure,1);
                    }
                });
            }
            int failures = bad.dataValue();
            amrex::ParallelDescriptor::ReduceIntSum(failures);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(failures == 0,
                "Requested insulating footprint cylinder intersects a cut or covered EB cell");
        }
#endif
    }
#endif
}
