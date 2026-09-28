/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>

#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

int
main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    {
        using ablastr::fields::Direction;
        amrex::ParmParse pp("test");
        std::string kind = "Efield_fp";
        std::string output = "legacy.dat";
        int max_grid = 4;
        int periodic = 0;
        pp.query("kind", kind);
        pp.query("output", output);
        pp.query("max_grid_size", max_grid);
        pp.query("periodic", periodic);
        WarpX sim;
        const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
        const amrex::RealBox physical({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
        amrex::Array<int, AMREX_SPACEDIM> periods{};
        periods[AMREX_SPACEDIM - 1] = periodic;
#ifdef WARPX_DIM_RZ
        constexpr int coord = 1;
#else
        constexpr int coord = 0;
#endif
        sim.geometry = amrex::Geometry(domain, &physical, coord, periods.data());
        amrex::BoxArray cells(domain);
        cells.maxSize(max_grid);
        const amrex::DistributionMapping dm(cells);
        if (kind == "phi_fp") {
            sim.m_fields.alloc_init(kind, 0, amrex::convert(cells, amrex::IntVect(1)), dm, 1,
                                    amrex::IntVect(1), 0.0);
        } else {
            for (int n = 0; n < 3; ++n) {
                amrex::IntVect centering(1);
#ifdef WARPX_DIM_RZ
                if (n == 0) {
                    centering[0] = 0;
                }
                if (n == 2) {
                    centering[1] = 0;
                }
#else
                centering[n] = 0;
#endif
                sim.m_fields.alloc_init(kind, Direction{n}, 0, amrex::convert(cells, centering), dm,
                                        1, amrex::IntVect(1), 0.0);
            }
        }
        WarpXSolverVec x, y;
        x.Define(&sim, kind == "phi_fp" ? "none" : kind, kind == "phi_fp" ? kind : "none");
        y.Define(x);
        std::vector<amrex::Real> host(x.nDOF_local());
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i] = 1.125 + amrex::Real(i % 47) * 0.00137;
        }
        amrex::Gpu::DeviceVector<amrex::Real> values(host.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), values.begin());
        x.copyFrom(values.data());
        y.Copy(x);
        y.scale(1.03125);
        x.increment(y, -0.0625);
        y.linComb(0.28125, x, 0.625, y);
        x += y;
        x -= y;
        const auto norm = x.dotProduct(x);
        x.copyTo(values.data());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(), values.end(), host.begin());
        std::ofstream out(output + ".rank" + std::to_string(amrex::ParallelDescriptor::MyProc()));
        out << x.nDOF_global() << ' ' << x.nDOF_local() << '\n' << std::hex;
        out << std::bit_cast<std::uint64_t>(norm) << '\n';
        for (auto value : host) {
            out << std::bit_cast<std::uint64_t>(value) << '\n';
        }
    }
    amrex::Finalize();
}
