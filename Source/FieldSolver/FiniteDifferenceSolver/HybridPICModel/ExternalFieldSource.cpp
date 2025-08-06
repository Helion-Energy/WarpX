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
#include <ablastr/coarsen/sample.H>

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

// Circuit coupling static members
int ExternalFieldSource::m_circuit_coupling_interval = 1;

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

    // Simplified: All ports are soft sources (flag > 0 means add excitation)
    // No hard sources, no flag validation needed - just add excitation where flag > 0

    // Gpu vector to store field staggering
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

    const int nComp_x = mfx->nComp();
    const int nComp_y = mfy->nComp();
    const int nComp_z = mfz->nComp();

    // Multiplication factor for FirstHalf/SecondHalf time stepping
    amrex::Real dt_type_factor = 1.0_rt;
    if (a_dt_type == DtType::FirstHalf or a_dt_type == DtType::SecondHalf) {
        dt_type_factor = 0.5_rt;
    }

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

        // Apply B-field excitation from circuit coupling
        // Add excitation where ports exist (flag > 0)
        amrex::ParallelFor(
            tbx, nComp_x,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfx_stag.data(),
                                                 problo.data(), dx.data(), x, y, z);
                auto flag_type = xflag_parser(x,y,z);
        
                if (flag_type > 0._rt) {
                    amrex::Real field_value = 0.0_rt;
                    if (m_circuit_coupling_enabled) {
                        // Current flows through center at (0,0) in z-direction
                        constexpr amrex::Real current_x = 0.0_rt;
                        constexpr amrex::Real current_y = 0.0_rt;
                        constexpr amrex::Real mu0_over_2pi = 2.0e-7_rt; // μ₀/(2π)
        
                        amrex::Real dx_from_current = x - current_x;
                        amrex::Real dy_from_current = y - current_y;
                        amrex::Real r_squared = dx_from_current*dx_from_current + dy_from_current*dy_from_current;
        
                        if (r_squared > 1e-12_rt) { // Avoid division by zero
                            // Bx = -μ₀I*y/(2π*r²) for current in +z direction
                            field_value = -mu0_over_2pi * m_circuit_current * dy_from_current / r_squared;
                            //field_value = xfield_parser(x,y,z,t);
                        }
                    } else {
                        field_value = xfield_parser(x,y,z,t);
                    }
                    Fx(i, j, k, n) += dt_type_factor * field_value;
                }
            },
            tby, nComp_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfy_stag.data(),
                                                 problo.data(), dx.data(), x, y, z);
                auto flag_type = yflag_parser(x,y,z);
        
                if (flag_type > 0._rt) {
                    amrex::Real field_value = 0.0_rt;
                    if (m_circuit_coupling_enabled) {
                        // Current flows through center at (0,0) in z-direction
                        constexpr amrex::Real current_x = 0.0_rt;
                        constexpr amrex::Real current_y = 0.0_rt;
                        constexpr amrex::Real mu0_over_2pi = 2.0e-7_rt; // μ₀/(2π)
        
                        amrex::Real dx_from_current = x - current_x;
                        amrex::Real dy_from_current = y - current_y;
                        amrex::Real r_squared = dx_from_current*dx_from_current + dy_from_current*dy_from_current;
        
                        if (r_squared > 1e-12_rt) { // Avoid division by zero
                            // By = +μ₀I*x/(2π*r²) for current in +z direction
                            field_value = mu0_over_2pi * m_circuit_current * dx_from_current / r_squared;
                            //field_value = yfield_parser(x,y,z,t);
                        }
                    } else {
                        field_value = yfield_parser(x,y,z,t);
                    }
                    Fy(i, j, k, n) += dt_type_factor * field_value;
                }
            },
            tbz, nComp_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                amrex::Real x, y, z;
                WarpXUtilAlgo::getCellCoordinates(i, j, k, mfz_stag.data(),
                                                 problo.data(), dx.data(), x, y, z);
                auto flag_type = zflag_parser(x,y,z);
        
                if (flag_type > 0._rt) {
                    amrex::Real field_value = 0.0_rt;
                    if (m_circuit_coupling_enabled) {
                        // For current in z-direction, Bz = 0 (no field along current direction)
                        field_value = 0.0_rt;
                    } else {
                        field_value = zfield_parser(x,y,z,t);
                    }
                    Fz(i, j, k, n) += dt_type_factor * field_value;
                }
            }
        );
    }
}



void ExternalFieldSource::InitializePorts(int lev)
{
    if (m_ports_initialized) return;

    if (m_circuit_coupling_enabled) {
        m_unique_ports = GetUniquePortIDs(lev);

        // Convert set to vector for consistent ordering
        m_port_ids.clear();
        for (int port_id : m_unique_ports) {
            if (port_id > 0) { // Skip flag value 0 (no excitation)
                m_port_ids.push_back(port_id);
            }
        }

        // Initialize voltage storage
        m_port_voltages.resize(m_port_ids.size(), 0.0);

        m_ports_initialized = true;

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Auto-detected " << m_port_ids.size()
                          << " ports for circuit coupling\n";
            PrintPortInfo();
        }
    }
}

std::set<int> ExternalFieldSource::GetUniquePortIDs(int lev)
{
    auto &warpx = WarpX::GetInstance();
    auto const &geom = warpx.Geom(lev);

    std::set<int> unique_ports;

    // Only proceed if we have the necessary parsers
    if (!Exfield_flag_parser || !Eyfield_flag_parser || !Ezfield_flag_parser) {
        return unique_ports;
    }

    auto ex_flag_parser = Exfield_flag_parser->compile<3>();
    auto ey_flag_parser = Eyfield_flag_parser->compile<3>();
    auto ez_flag_parser = Ezfield_flag_parser->compile<3>();

    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();

    // Sample the domain to find unique port IDs
    const auto& domain = geom.Domain();

    // Use stride = 1 to ensure we don't miss any ports
    for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
        for (int j = domain.smallEnd(1); j <= domain.bigEnd(1); ++j) {
            for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); ++i) {
                amrex::Real x, y, z;
                // Use cell-centered coordinates for sampling
                x = problo[0] + (i + 0.5) * dx[0];
                y = problo[1] + (j + 0.5) * dx[1];
                z = problo[2] + (k + 0.5) * dx[2];

                int ex_flag = static_cast<int>(ex_flag_parser(x, y, z));
                int ey_flag = static_cast<int>(ey_flag_parser(x, y, z));
                int ez_flag = static_cast<int>(ez_flag_parser(x, y, z));

                if (ex_flag > 0) unique_ports.insert(ex_flag);
                if (ey_flag > 0) unique_ports.insert(ey_flag);
                if (ez_flag > 0) unique_ports.insert(ez_flag);
            }
        }
    }

    // Simple MPI reduction to find maximum port ID
    int max_port_local = 0;
    if (!unique_ports.empty()) {
        max_port_local = *unique_ports.rbegin();
    }

    int max_port_global = max_port_local;
    amrex::ParallelDescriptor::ReduceIntMax(max_port_global);

    // Check which ports actually exist by testing each ID
    std::set<int> global_unique_ports;
    for (int port_id = 1; port_id <= max_port_global; ++port_id) {
        int port_exists_local = (unique_ports.count(port_id) > 0) ? 1 : 0;
        int port_exists_global = port_exists_local;
        amrex::ParallelDescriptor::ReduceIntMax(port_exists_global);

        if (port_exists_global > 0) {
            global_unique_ports.insert(port_id);
        }
    }

    return global_unique_ports;
}

void ExternalFieldSource::CalculatePortVoltages(int lev)
{
    if (!m_circuit_coupling_enabled) return;

    // Initialize ports if not done yet
    if (!m_ports_initialized) {
        InitializePorts(lev);
    }

    // Clear voltage map
    m_port_voltage_map.clear();

    // Calculate voltage for each port
    for (size_t i = 0; i < m_port_ids.size(); ++i) {
        int port_id = m_port_ids[i];
        amrex::Real voltage = CalculateVoltageForPort(port_id, lev);
        m_port_voltages[i] = voltage;
        m_port_voltage_map[port_id] = voltage;
    }
}
/*
amrex::Real ExternalFieldSource::CalculateVoltageForPort(int port_id, int lev)
{
    auto &warpx = WarpX::GetInstance();
    auto const &geom = warpx.Geom(lev);

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    amrex::MultiFab* Ex = warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
    amrex::MultiFab* Ey = warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, lev);

    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    amrex::Real t = warpx.gett_new(lev);

    // Cell-centered index type
    const amrex::GpuArray<int,3> cc{0,0,0};
    // Coarsening ratio (no coarsening)
    const amrex::GpuArray<int,3> cr{1,1,1};

    // --- Hardcoded coil geometry ---
    constexpr amrex::Real coil_radius = 0.01_rt;  // 1 cm radius
    constexpr amrex::Real coil_x_center = 0.0_rt;
    constexpr amrex::Real coil_y_center = 0.0_rt;

    // Voltage accumulator
    amrex::Real total_voltage = 0.0_rt;

    // Get staggering info for Ex and Ey
    GpuArray<int, AMREX_SPACEDIM> ex_stag, ey_stag;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        ex_stag[idim] = Ex->ixType()[idim];
        ey_stag[idim] = Ey->ixType()[idim];
    }

    amrex::IntVect ex_nodal_flag = Ex->ixType().toIntVect();
    amrex::IntVect ey_nodal_flag = Ey->ixType().toIntVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        auto const& ex_arr = Ex->array(mfi);
        auto const& ey_arr = Ey->array(mfi);

        // Use a common box that works for both Ex and Ey
        const amrex::Box& box = mfi.tilebox(); // Cell-centered box

        amrex::Gpu::DeviceScalar<amrex::Real> local_voltage(0.0_rt);

        amrex::ParallelFor(box, [=, &local_voltage] AMREX_GPU_DEVICE (int i, int j, int k) {

            amrex::Real x, y, z;
            // Use cell-centered coordinates for consistent geometry
            WarpXUtilAlgo::getCellCoordinates(i, j, k, cc.data(), problo.data(), dx.data(), x, y, z);

            // Only integrate near z = 0 plane
            //if (std::abs(z) > 2.0 * dx[2]) return;

            // Distance from coil center
            amrex::Real dx_c = x - coil_x_center;
            amrex::Real dy_c = y - coil_y_center;
            amrex::Real r = std::sqrt(dx_c*dx_c + dy_c*dy_c);

            // Select cells near the coil contour
            if (std::abs(r - coil_radius) < 0.5 * std::max(dx[0], dx[1])) {
                amrex::Real phi = std::atan2(dy_c, dx_c);

                // Only integrate over 180° arc (right half of circle) to avoid cancellation
                if (phi >= -M_PI/2.0 && phi <= M_PI/2.0) {
                    // Interpolate both fields to cell center for consistency
                    const amrex::Real Ex_cc = ablastr::coarsen::sample::Interp(ex_arr, ex_stag, cc, cr, i, j, k, 0);
                    const amrex::Real Ey_cc = ablastr::coarsen::sample::Interp(ey_arr, ey_stag, cc, cr, i, j, k, 0);

                    // Tangential E-field component
                    amrex::Real E_tan = Ex_cc * (-std::sin(phi)) + Ey_cc * std::cos(phi);

                    amrex::Real dl = std::max(dx[0], dx[1]);
                    amrex::Gpu::Atomic::Add(local_voltage.dataPtr(), -E_tan * dl);
                }
            }
        });

        total_voltage += local_voltage.dataValue();
    }

    // Reduce across MPI
    amrex::ParallelDescriptor::ReduceRealSum(total_voltage);

    return 2.0*total_voltage;
}
*/

amrex::Real ExternalFieldSource::CalculateVoltageForPort(int port_id, int lev)
{
    // Coil / port geometry
    constexpr amrex::Real coil_radius = 0.01_rt;
    constexpr amrex::Real coil_z_plane = 0.01_rt; // Plane to sample Ex
    constexpr int Npts = 50; // Number of integration samples

    // Access WarpX fields
    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;
    amrex::MultiFab* Ex = warpx.m_fields.get(warpx::fields::FieldType::Efield_fp, Direction{0}, lev);

    const auto geom = warpx.Geom(lev);
    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();

    // Staggering info for interpolation
    amrex::GpuArray<int, AMREX_SPACEDIM> ex_stag;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        ex_stag[idim] = Ex->ixType()[idim];
    }
    amrex::GpuArray<int, 3> cc = {AMREX_D_DECL(0, 0, 0)};
    amrex::GpuArray<int, 3> cr = {AMREX_D_DECL(1, 1, 1)};

    amrex::Real gap_voltage = 0.0_rt;

    // Integration endpoints along x-axis
    const amrex::Real x1 = -coil_radius;
    const amrex::Real x2 =  coil_radius;
    const amrex::Real y_fixed = 0.0_rt;
    const amrex::Real z_fixed = coil_z_plane;

    const amrex::Real dx_line = (x2 - x1) / (Npts - 1);

    // Loop over MFIter tiles
    for (MFIter mfi(*Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const& ex_arr = Ex->array(mfi);

        amrex::Real V_local = 0.0_rt;

        for (int n = 0; n < Npts; ++n) {
            amrex::Real xn = x1 + n * dx_line;
            amrex::Real yn = y_fixed;
            amrex::Real zn = z_fixed;

            // Convert to cell indices
            int i = static_cast<int>((xn - problo[0]) / dx[0]);
            int j = static_cast<int>((yn - problo[1]) / dx[1]);
            int k = static_cast<int>((zn - problo[2]) / dx[2]);

            // Only sample if inside valid tile
            if (mfi.validbox().contains(amrex::IntVect(AMREX_D_DECL(i,j,k)))) {
                amrex::Real Ex_val = ablastr::coarsen::sample::Interp(
                    ex_arr, ex_stag, cc, cr, i, j, k, 0);
                V_local += Ex_val;
            }
        }

        // Convert average E to voltage (negative line integral)
        gap_voltage += - (V_local / static_cast<amrex::Real>(Npts)) * (x2 - x1);
    }

    // Reduce across MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(gap_voltage);

    return gap_voltage;
}


amrex::Real ExternalFieldSource::GetVoltageForPort(int port_id) const
{
    auto it = m_port_voltage_map.find(port_id);
    if (it != m_port_voltage_map.end()) {
        return it->second;
    }
    return 0.0;
}

void ExternalFieldSource::UpdateExcitationsFromCircuitCurrents(const std::vector<amrex::Real>& currents)
{
    if (currents.size() != m_port_ids.size()) {
        amrex::Abort("Number of currents must match number of ports!");
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Updating field excitations from circuit currents:\n";
        for (size_t i = 0; i < m_port_ids.size(); ++i) {
            amrex::Print() << "  Port " << m_port_ids[i]
                          << ": Current = " << currents[i] << " A\n";
        }
    }
}

void ExternalFieldSource::UpdateBExcitationFromCurrent(amrex::Real current)
{
    m_circuit_current = current;
    m_circuit_coupling_enabled = true;
}

void ExternalFieldSource::PrintPortInfo() const
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Circuit coupling port information:\n";
        amrex::Print() << "  Number of ports: " << m_port_ids.size() << "\n";
        amrex::Print() << "  Port IDs: ";
        for (size_t i = 0; i < m_port_ids.size(); ++i) {
            amrex::Print() << m_port_ids[i];
            if (i < m_port_ids.size() - 1) {
                amrex::Print() << ", ";
            }
        }
        amrex::Print() << "\n";

        if (!m_port_voltages.empty() && m_port_voltages.size() == m_port_ids.size()) {
            amrex::Print() << "  Current voltages:\n";
            for (size_t i = 0; i < m_port_ids.size(); ++i) {
                amrex::Print() << "    Port " << m_port_ids[i]
                              << ": " << m_port_voltages[i] << " V\n";
            }
        }

        amrex::Print() << "  Circuit coupling enabled: "
                      << (m_circuit_coupling_enabled ? "Yes" : "No") << "\n";
    }
}

// ========================================================================
// LC Circuit Implementation
// ========================================================================

LCCircuit::LCCircuit(amrex::Real L, amrex::Real C, amrex::Real V0, amrex::Real I0)
    : m_L(L), m_C(C), m_V0(V0), m_I0(I0), 
      m_voltage(V0), m_current(I0), m_external_voltage(0.0), m_time(0.0)
{
    if (L <= 0.0 || C <= 0.0) {
        amrex::Abort("LC circuit: Inductance and Capacitance must be positive!");
    }
    
    if (amrex::ParallelDescriptor::IOProcessor()) {
        PrintInfo();
    }
}

void LCCircuit::UpdateTimeStep(amrex::Real dt)
{
    // LC circuit differential equations:
    // L * dI/dt + V = V_ext  =>  dI/dt = (V_ext - V) / L
    // C * dV/dt = I          =>  dV/dt = I / C
    //
    // State vector: [V, I]
    // Derivatives: [I/C, (V_ext - V)/L]

    // Current state
    amrex::Real V = m_voltage;
    amrex::Real I = m_current;

    // RK4 implementation
    // k1 = f(t, y)
    amrex::Real dV_dt_k1 = I / m_C;
    amrex::Real dI_dt_k1 = (m_external_voltage - V) / m_L;

    // k2 = f(t + dt/2, y + k1*dt/2)
    amrex::Real V_k2 = V + 0.5 * dt * dV_dt_k1;
    amrex::Real I_k2 = I + 0.5 * dt * dI_dt_k1;
    amrex::Real dV_dt_k2 = I_k2 / m_C;
    amrex::Real dI_dt_k2 = (m_external_voltage - V_k2) / m_L;

    // k3 = f(t + dt/2, y + k2*dt/2)
    amrex::Real V_k3 = V + 0.5 * dt * dV_dt_k2;
    amrex::Real I_k3 = I + 0.5 * dt * dI_dt_k2;
    amrex::Real dV_dt_k3 = I_k3 / m_C;
    amrex::Real dI_dt_k3 = (m_external_voltage - V_k3) / m_L;

    // k4 = f(t + dt, y + k3*dt)
    amrex::Real V_k4 = V + dt * dV_dt_k3;
    amrex::Real I_k4 = I + dt * dI_dt_k3;
    amrex::Real dV_dt_k4 = I_k4 / m_C;
    amrex::Real dI_dt_k4 = (m_external_voltage - V_k4) / m_L;

    // Final update: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    m_voltage = V + (dt / 6.0) * (dV_dt_k1 + 2.0 * dV_dt_k2 + 2.0 * dV_dt_k3 + dV_dt_k4);
    m_current = I + (dt / 6.0) * (dI_dt_k1 + 2.0 * dI_dt_k2 + 2.0 * dI_dt_k3 + dI_dt_k4);

    m_time += dt;
}

void LCCircuit::Reset()
{
    m_voltage = m_V0;
    m_current = m_I0;
    m_time = 0.0;
    m_external_voltage = 0.0;
}

void LCCircuit::PrintInfo() const
{
    amrex::Print() << "LC Circuit Parameters:\n";
    amrex::Print() << "  Inductance L = " << m_L << " H\n";
    amrex::Print() << "  Capacitance C = " << m_C << " F\n";
    amrex::Print() << "  Initial Voltage V0 = " << m_V0 << " V\n";
    amrex::Print() << "  Initial Current I0 = " << m_I0 << " A\n";
    amrex::Print() << "  Resonant Frequency = " << GetResonantFrequency() << " Hz\n";
    amrex::Print() << "  Resonant Period = " << GetResonantPeriod() << " s\n";
    amrex::Print() << "  Current State: V = " << m_voltage << " V, I = " << m_current << " A\n";
}

// ========================================================================
// ExternalFieldSource LC Circuit Methods
// ========================================================================

void ExternalFieldSource::InitializeLCCircuit(amrex::Real L, amrex::Real C, amrex::Real V0)
{
    m_lc_circuit = std::make_unique<LCCircuit>(L, C, V0);
    
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "LC Circuit initialized for circuit coupling\n";
    }
}

void ExternalFieldSource::UpdateLCCircuit(amrex::Real dt)
{
    if (m_lc_circuit) {
        m_lc_circuit->UpdateTimeStep(dt);
    }
}

amrex::Real ExternalFieldSource::GetLCVoltage() const
{
    return m_lc_circuit ? m_lc_circuit->GetVoltage() : 0.0;
}

amrex::Real ExternalFieldSource::GetLCCurrent() const
{
    return m_lc_circuit ? m_lc_circuit->GetCurrent() : 0.0;
}

void
ExternalFieldSource::ReadExcitationParser ()
{

    ParmParse pp_external_source("external_source");

    // Query for type of external space-time (xt) varying excitation
    pp_external_source.query("B_excitation_on_grid_style", B_excitation_grid_s);
    std::transform(B_excitation_grid_s.begin(),
               B_excitation_grid_s.end(),
               B_excitation_grid_s.begin(),
               ::tolower);

    pp_external_source.query("E_excitation_on_grid_style", E_excitation_grid_s);
    std::transform(E_excitation_grid_s.begin(),
                   E_excitation_grid_s.end(),
                   E_excitation_grid_s.begin(),
                   ::tolower);

    utils::parser::queryWithParser(pp_external_source, "excite_E_source", excite_E_source);
    utils::parser::queryWithParser(pp_external_source, "excite_B_source", excite_B_source);
    utils::parser::queryWithParser(pp_external_source, "excite_all", excite_all);

    amrex::Print() << "excite_E_source = " << excite_E_source << ", excite_B_source = " << excite_B_source << ", excite_all = " << excite_all << "\n";

    if (E_excitation_grid_s == "parse_e_excitation_grid_function") {
        // if E excitation type is set to parser then the corresponding
        // source type (hard=1, soft=2) must be specified for all components
        // using the flag function. Note that a flag value of 0 will not update
        // the field with the excitation.
        utils::parser::Store_parserString(pp_external_source, "Ex_excitation_flag_function(x,y,z)",
                                str_Ex_excitation_flag_function);
        utils::parser::Store_parserString(pp_external_source, "Ey_excitation_flag_function(x,y,z)",
                                str_Ey_excitation_flag_function);
        utils::parser::Store_parserString(pp_external_source, "Ez_excitation_flag_function(x,y,z)",
                                str_Ez_excitation_flag_function);
        Exfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ex_excitation_flag_function,{"x","y","z"}));
        Eyfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ey_excitation_flag_function,{"x","y","z"}));
        Ezfield_flag_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ez_excitation_flag_function,{"x","y","z"}));

        pp_external_source.query("Apply_E_excitation_in_pml_region", ApplyExcitationInPML);
    }
    if (B_excitation_grid_s == "parse_b_excitation_grid_function") {
        // if B excitation type is set to parser then the corresponding
        // source type (hard=1, soft=2) must be specified for all components
        // using the flag function. Note that a flag value of 0 will not update
        // the field with the excitation.
        utils::parser::Store_parserString(pp_external_source, "Bx_excitation_flag_function(x,y,z)",
                                str_Bx_excitation_flag_function);
        utils::parser::Store_parserString(pp_external_source, "By_excitation_flag_function(x,y,z)",
                                str_By_excitation_flag_function);
        utils::parser::Store_parserString(pp_external_source, "Bz_excitation_flag_function(x,y,z)",
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
       utils::parser::Store_parserString(pp_external_source, "Bx_excitation_grid_function(x,y,z,t)",
                                                    str_Bx_excitation_grid_function);
       utils::parser::Store_parserString(pp_external_source, "By_excitation_grid_function(x,y,z,t)",
                                                    str_By_excitation_grid_function);
       utils::parser::Store_parserString(pp_external_source, "Bz_excitation_grid_function(x,y,z,t)",
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
       utils::parser::Store_parserString(pp_external_source, "Ex_excitation_grid_function(x,y,z,t)",
                                                    str_Ex_excitation_grid_function);
       utils::parser::Store_parserString(pp_external_source, "Ey_excitation_grid_function(x,y,z,t)",
                                                    str_Ey_excitation_grid_function);
       utils::parser::Store_parserString(pp_external_source, "Ez_excitation_grid_function(x,y,z,t)",
                                                    str_Ez_excitation_grid_function);
       Exfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ex_excitation_grid_function,{"x","y","z","t"}));
       Eyfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ey_excitation_grid_function,{"x","y","z","t"}));
       Ezfield_xt_grid_parser = std::make_unique<amrex::Parser>(
                   utils::parser::makeParser(str_Ez_excitation_grid_function,{"x","y","z","t"}));
    }


    // Circuit coupling parameters
    amrex::ParmParse pp_circuit("circuit");
    pp_circuit.query("enable_coupling", m_circuit_coupling_enabled);
    pp_circuit.query("coupling_interval", m_circuit_coupling_interval);

    if (m_circuit_coupling_enabled) {
        amrex::Print() << "Circuit coupling enabled\n";
        amrex::Print() << "Ports will be auto-detected from flag functions\n";
    }

    int enable_lc_test = 0;
    pp_circuit.query("enable_lc_test", enable_lc_test);

    if (enable_lc_test) {
        amrex::Real lc_inductance = 1.0e-6;     // Default 1 μH
        amrex::Real lc_capacitance = 1.0e-9;    // Default 1 nF
        amrex::Real lc_initial_voltage = 10.0;  // Default 10 V

        pp_circuit.query("lc_inductance", lc_inductance);
        pp_circuit.query("lc_capacitance", lc_capacitance);
        pp_circuit.query("lc_initial_voltage", lc_initial_voltage);

        InitializeLCCircuit(lc_inductance, lc_capacitance, lc_initial_voltage);
    }
}
