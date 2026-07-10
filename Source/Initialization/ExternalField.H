/* Copyright 2023 Luca Fedeli
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EXTERNAL_FIELD_H_
#define WARPX_EXTERNAL_FIELD_H_

#include "ExternalField_fwd.H"

#include <ablastr/math/LinearInterpolation.H>

#include <AMReX_Array.H>
#include <AMReX_BaseFab.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_RealVect.H>
#include <AMReX_TableData.H>

#include <functional>
#include <memory>
#include <string>

enum class ExternalFieldType
{
    default_zero,
    constant,
    parse_ext_grid_function,
    read_from_file,
    load_from_python
};

/**
 * \brief Struct to store data related to external electromagnetic fields
 * (flags, field values, and field parsers)
 */
struct ExternalFieldParams
{

    /**
    * \brief The constructor reads and stores the parameters related to the external fields.
    * "pp_warpx" must point at the "warpx" parameter group in the inputfile.
    */
    explicit ExternalFieldParams(const amrex::ParmParse& pp_warpx);

    //! Initial electric field on the grid
    amrex::GpuArray<amrex::Real,3> E_external_grid = {0,0,0};
    //! Initial magnetic field on the grid
    amrex::GpuArray<amrex::Real,3> B_external_grid = {0,0,0};

    //! Initialization type for external magnetic field on the grid
    ExternalFieldType B_ext_grid_type = ExternalFieldType::default_zero;
    //! Initialization type for external electric field on the grid
    ExternalFieldType E_ext_grid_type = ExternalFieldType::default_zero;

    //! User-defined parser to initialize x-component of the magnetic field on the grid
    std::unique_ptr<amrex::Parser> Bxfield_parser;
    //! User-defined parser to initialize y-component of the magnetic field on the grid
    std::unique_ptr<amrex::Parser> Byfield_parser;
    //! User-defined parser to initialize z-component of the magnetic field on the grid
    std::unique_ptr<amrex::Parser> Bzfield_parser;
    //! User-defined parser to initialize x-component of the electric field on the grid
    std::unique_ptr<amrex::Parser> Exfield_parser;
    //! User-defined parser to initialize y-component of the electric field on the grid
    std::unique_ptr<amrex::Parser> Eyfield_parser;
    //! User-defined parser to initialize z-component of the electric field on the grid
    std::unique_ptr<amrex::Parser> Ezfield_parser;

     //! Path of the file where external fields are stored
    std::string external_fields_path;
};

/**
 * \brief Companion class for ExternalFieldReader.
 *
 * This is a lightweight functor that can be used in kernels.
 */
struct ExternalFieldView
{
    //! Quadratic B-spline weights for the three lattice points nearest
    //! to a target at in-cell offset t in [-1/2, 1/2].
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void quadratic_weights (amrex::Real t, amrex::Real w[3])
    {
        w[0] = amrex::Real(0.5)*(amrex::Real(0.5)-t)*(amrex::Real(0.5)-t);
        w[1] = amrex::Real(0.75) - t*t;
        w[2] = amrex::Real(0.5)*(amrex::Real(0.5)+t)*(amrex::Real(0.5)+t);
    }

    //! Return value for given position. Linear interpolation is
    //! performed by default. With interp_order = 2, a quadratic
    //! B-spline (momentum-conserving style) stencil is used in every
    //! direction: its leading error is the isotropic (dx^2/8) Laplacian
    //! of the field, independent of the in-cell offset, so resampling
    //! staggered target points from nodal file data does not imprint an
    //! axis-aligned (m=4) anisotropy. Note that order 2 is not
    //! interpolatory: it smooths the data even where the target point
    //! coincides with a file lattice point. This returns zero if no
    //! data are available; points too close to the data edge for the
    //! quadratic stencil fall back to linear interpolation.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::RealVect const& pos) const
    {
#if defined(AMREX_USE_GPU)
        AMREX_IF_ON_HOST((amrex::Abort("For GPU builds, ExternalFieldView only works on device. We could make it work on host if necessary.");))
#endif

        if (interp_order == 2) {
            // Nearest lattice point and in-cell offset in [-1/2, 1/2]
            AMREX_D_TERM(const amrex::Real g0 = (pos[0]-offset[0])/dx[0];,
                         const amrex::Real g1 = (pos[1]-offset[1])/dx[1];,
                         const amrex::Real g2 = (pos[2]-offset[2])/dx[2]);
            AMREX_D_TERM(const auto j0 = static_cast<int>(std::floor(g0 + amrex::Real(0.5)));,
                         const auto j1 = static_cast<int>(std::floor(g1 + amrex::Real(0.5)));,
                         const auto j2 = static_cast<int>(std::floor(g2 + amrex::Real(0.5))));
#if (AMREX_SPACEDIM == 1)
            const bool has_stencil = (j0-1 >= table.begin && j0+1 < table.end);
#else
            const bool has_stencil =
                AMREX_D_TERM(   (j0-1 >= table.begin[0] && j0+1 < table.end[0]),
                             && (j1-1 >= table.begin[1] && j1+1 < table.end[1]),
                             && (j2-1 >= table.begin[2] && j2+1 < table.end[2]));
#endif
            if (has_stencil) {
                amrex::Real w0[3];
                quadratic_weights(g0 - j0, w0);
#if (AMREX_SPACEDIM >= 2)
                amrex::Real w1[3];
                quadratic_weights(g1 - j1, w1);
#endif
#if (AMREX_SPACEDIM == 3)
                amrex::Real w2[3];
                quadratic_weights(g2 - j2, w2);
#endif
                double sum = 0.0;
#if (AMREX_SPACEDIM == 1)
                for (int a = 0; a < 3; ++a) {
                    sum += double(w0[a]) * table(j0-1+a);
                }
#elif (AMREX_SPACEDIM == 2)
                for (int b = 0; b < 3; ++b) {
                    for (int a = 0; a < 3; ++a) {
                        sum += double(w0[a]) * double(w1[b]) * table(j0-1+a, j1-1+b);
                    }
                }
#elif (AMREX_SPACEDIM == 3)
                for (int c = 0; c < 3; ++c) {
                    for (int b = 0; b < 3; ++b) {
                        for (int a = 0; a < 3; ++a) {
                            sum += double(w0[a]) * double(w1[b]) * double(w2[c])
                                * table(j0-1+a, j1-1+b, j2-1+c);
                        }
                    }
                }
#endif
                return static_cast<amrex::Real>(sum);
            }
            // Fall through to linear interpolation near the data edge.
        }

        // Get index of the external field array
        AMREX_D_TERM(const auto i0 = static_cast<int>(std::floor( (pos[0]-offset[0])/dx[0] ));,
                     const auto i1 = static_cast<int>(std::floor( (pos[1]-offset[1])/dx[1] ));,
                     const auto i2 = static_cast<int>(std::floor( (pos[2]-offset[2])/dx[2] )));

        // Get coordinates of external grid point
        AMREX_D_TERM(amrex::Real const xx0 = offset[0] + i0 * dx[0];,
                     amrex::Real const xx1 = offset[1] + i1 * dx[1];,
                     amrex::Real const xx2 = offset[2] + i2 * dx[2]);

#if (AMREX_SPACEDIM == 1)

        if (i0 < table.begin || i0 >= table.end-1) // This covers empty Table.
        {
            AMREX_ALWAYS_ASSERT(i0 < 0 || i0 >= global_size[0]-1);
            return 0;
        } else {
            return static_cast<amrex::Real>(
                ablastr::math::linear_interp<double>(
                    xx0, xx0+dx[0], table(i0), table(i0+1), pos[0]));
        }

#elif (AMREX_SPACEDIM == 2)

        if (i0 < table.begin[0] || i0 >= table.end[0]-1 ||
            i1 < table.begin[1] || i1 >= table.end[1]-1)
        {
            AMREX_ALWAYS_ASSERT(i0 < 0 || i0 >= global_size[0]-1 ||
                                i1 < 0 || i1 >= global_size[1]-1);
            return 0;
        } else {
            return static_cast<amrex::Real>(
                ablastr::math::bilinear_interp<double>(
                    xx0, xx0+dx[0], xx1, xx1+dx[1],
                    table(i0  ,i1  ),
                    table(i0  ,i1+1),
                    table(i0+1,i1  ),
                    table(i0+1,i1+1),
                    pos[0], pos[1]));
        }

#elif (AMREX_SPACEDIM == 3)

        if (i0 < table.begin[0] || i0 >= table.end[0]-1 ||
            i1 < table.begin[1] || i1 >= table.end[1]-1 ||
            i2 < table.begin[2] || i2 >= table.end[2]-1)
        {
            AMREX_ALWAYS_ASSERT(i0 < 0 || i0 >= global_size[0]-1 ||
                                i1 < 0 || i1 >= global_size[1]-1 ||
                                i2 < 0 || i2 >= global_size[2]-1);
            return 0;
        } else {
            return static_cast<amrex::Real>(
                ablastr::math::trilinear_interp<double>(
                    xx0, xx0+dx[0], xx1, xx1+dx[1], xx2, xx2+dx[2],
                    table(i0  ,i1  ,i2  ),
                    table(i0  ,i1  ,i2+1),
                    table(i0  ,i1+1,i2  ),
                    table(i0  ,i1+1,i2+1),
                    table(i0+1,i1  ,i2  ),
                    table(i0+1,i1  ,i2+1),
                    table(i0+1,i1+1,i2  ),
                    table(i0+1,i1+1,i2+1),
                    pos[0], pos[1], pos[2]));
        }
#endif
    }

    amrex::RealVect dx;     //! mesh spacing of the data
    amrex::RealVect offset; //! data's physical location at (0,0,0)
    int interp_order = 1;   //! 1: n-linear, 2: quadratic B-spline (MC-style)
#if (AMREX_SPACEDIM == 1)
    amrex::Table1D<double> table;
#elif (AMREX_SPACEDIM == 2)
    amrex::Table2D<double> table;
#elif (AMREX_SPACEDIM == 3)
    amrex::Table3D<double> table; //! data
#endif
    amrex::IntVect global_size; //! global size of the data
};

/**
 * OpenPMD field data reader
 *
 * This reads an OpenPMD file for a given field. Then we can get a
 * lightweight object `ExternalFieldView` that can be use in kernels. This
 * class can be used for the initialization of particles and fields, and
 * during moving window.
 */
class ExternalFieldReader
{
public:
    /**
     * \brief Constructor
     *
     * \param[in] read_fields_from_path OpenPMD file name
     * \param[in] F_name      field name
     * \param[in] F_component field component name
     * \param[in] problo      lower corner of WarpX problem domain
     * \param[in] pdx         cell size of WarpX
     * \param[in] dombox      WarpX domain box
     * \param[in] distributed read OpenPMD data distributely
     */
    ExternalFieldReader (std::string read_fields_from_path,
                         std::string F_name, std::string F_component,
                         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& pdx,
                         amrex::Box const& dombox, bool distributed);

    //! Return lightweight view that can be used in kernels
    [[nodiscard]] ExternalFieldView getView (int li) const;

    //! Return lightweight view that can be used in kernels
    [[nodiscard]] ExternalFieldView getView () const;

    /**
     * \brief This must be called before calling `getView` to prepare for
     * loading distributed data. This is usually used during the
     * initialization of particles and fields.
     *
     * \param[in] grids    BoxArray of destionation ParticleContainer/MultiFab
     * \param[in] dmap     DistributionMapping of destionation ParticleContainer/MultiFab
     * \param[in] ngrow    number of ghost cells in destination MultiFab
     * \param[in] get_zlab optional function for transforming z-coordinates into lab frame
     */
    void prepare (amrex::BoxArray const& grids,
                  amrex::DistributionMapping const& dmap,
                  amrex::IntVect const& ngrow,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab = nullptr);

    /**
     * \brief This needs to be called before loading data for moving window.
     *
     * \param[in] pbox        target domain
     * \param[in] moving_dir  moving window direction
     * \param[in] moving_sign positive or negative direction
     * \param[in] get_zlab    optional function for transforming z-coordinates into lab frame
     */
    void prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab = nullptr);

    //! Is the data loaded distributedly?
    [[nodiscard]] bool distributed () const { return m_distributed; }

private:

    //! Read data within this box
    void load_data (amrex::RealBox const& pbox);
    //! Used by getView to make ExternalFieldView
    [[nodiscard]] ExternalFieldView make_view (amrex::BaseFab<double> const& fab) const;
    //! During the moving window stage, we cache the data for
    //! performance. This function makes a Box for caching.
    void make_cache_box (amrex::RealBox const& pbox, int moving_dir, int moving_sign);

    std::string m_file;      //! OpenMPD file name
    std::string m_name;      //! Field name
    std::string m_component; //! Component name
    int m_interp_order = 1;  //! interpolation order used by the views (1 or 2)
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_problo; //! WarpX's initial prob lo
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_probdx; //! WarpX's cell size
    amrex::Box m_dombox;                                  //! WarpX's initial domain
    bool m_distributed = false;   //! Is the data loaded distributedly?
    bool m_moving_window = false; //! In moving window stage?
    amrex::RealVect m_dx;     // Field data's dx
    amrex::RealVect m_offset; // Field data's physical location at (0,0,0)
    amrex::IntVect m_size;    // Field data's size (i.e., number of elements)
    amrex::RealBox m_domain;  // Field data's total physical domain
    amrex::RealBox m_cache_domain; // Current cache's domain
    std::shared_ptr<double> m_FC_data_cpu; //! buffer in cpu memory for loading data
    amrex::BaseFab<double> m_fab; //! data container for loaded data
    amrex::FabArray<amrex::BaseFab<double>> m_mf; //! non-owning container for communication purpose
};

#endif //WARPX_EXTERNAL_FIELD_H_
