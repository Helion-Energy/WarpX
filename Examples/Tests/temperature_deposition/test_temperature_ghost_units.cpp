/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/Deposition/VarianceAccumulationBuffer.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <AMReX_MFParallelFor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_OpenMP.H>
#include <AMReX_VisMF.H>

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>

using namespace amrex::literals;

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    int status = 0;
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        amrex::ParmParse test("ghost_units");
        bool particles = false, profile = false, inject_nan = false;
        test.query("particles", particles);
        test.query("profile", profile);
        test.query("inject_nan", inject_nan);
        bool const apply_filter = WarpX::use_filter;
        if (particles) {
            // Use the supported native entry point, including its full-array reset.
            w.GetPartContainer().DepositTemperatures(w.m_fields, 0._rt);
            bool finite = true;
            for (auto const& name : w.GetPartContainer().GetSpeciesNames()) {
                auto const t = w.m_fields.get_alldirs("T_" + name, 0);
                for (int c = 0; c < 3; ++c) {
                    finite = !t[c]->contains_nan() && !t[c]->contains_inf() && finite;
                    amrex::VisMF::Write(*t[c], name + "_T" + std::to_string(c));
                }
            }
            amrex::Print() << "GHOST_PARTICLES {\"native_deposition_complete\":true,\"all_fabs_finite\":" << (finite?"true":"false") << "}\n";
            status = finite ? 0 : 2;
        } else {
            amrex::Vector<int> guards(AMREX_SPACEDIM), passes(AMREX_SPACEDIM);
            test.getarr("guards", guards);
            amrex::ParmParse pp("warpx");
            pp.getarr("filter_npass_each_dir", passes);
            amrex::IntVect ng;
            amrex::GpuArray<int, AMREX_SPACEDIM> np{}, periodic{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                ng[d] = guards[d]; np[d] = apply_filter ? passes[d] : 0;
                periodic[d] = w.Geom(0).isPeriodic(d);
                AMREX_ALWAYS_ASSERT(ng[d] >= np[d]);
            }
            auto const dx = w.Geom(0).CellSizeArray();
            auto const lo = w.Geom(0).ProbLoArray();
            auto const hi = w.Geom(0).ProbHiArray();
            std::array<amrex::MultiFab, 3> temperature;
            ablastr::fields::MultiLevelVectorField tv(1);
            auto const native = w.m_fields.get_alldirs("current_fp", 0);
            constexpr amrex::Real norm = 0.0002421747868140273_rt;
            for (int c = 0; c < 3; ++c) {
                temperature[c].define(native[c]->boxArray(), native[c]->DistributionMap(), 1, ng);
                tv[0][c] = &temperature[c];
                auto const type = temperature[c].ixType().toIntVect();
                amrex::GpuArray<int, AMREX_SPACEDIM> nodal{};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { nodal[d] = type[d]; }
                auto const arrays = temperature[c].arrays();
                amrex::ParallelFor(temperature[c], ng,
                    [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
                        int const cell[3] = {i,j,k};
                        amrex::Real value = 2.5e6_rt * (1._rt + .2_rt*c);
                        if (profile) {
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                auto const x = (lo[d] + (cell[d] + .5_rt*(1-nodal[d]))*dx[d] - lo[d])/(hi[d]-lo[d]);
                                value += 1.e5_rt * (d+1) * (periodic[d] ? std::cos(2._rt*MathConst::pi*x) : x*x);
                            }
                        }
                        arrays[box](i,j,k) = value/norm;
                    });
            }
            amrex::Gpu::streamSynchronize();
            warpx::particles::deposition::VarianceAccumulationBuffer buffer(tv, "unit_fixture");
            buffer.ConvertVarianceToTemperatureAndFilter(tv, norm, apply_filter);
            if (inject_nan) {
                // An outer physical ghost lies outside the supported-error region.
                // Only the explicit complete-FAB finiteness gate can reject it.
                auto const arrays = temperature[0].arrays();
                amrex::ParallelFor(temperature[0], ng,
                    [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
                        if (i == -ng[0] && j == -ng[1]) {
                            arrays[box](i,j,k) = std::numeric_limits<amrex::Real>::quiet_NaN();
                        }
                    });
                amrex::Gpu::streamSynchronize();
            }
            bool finite = true;
            int tiles = 0;
            for (amrex::MFIter mfi(temperature[0], true); mfi.isValid(); ++mfi) { ++tiles; }
            amrex::ParallelDescriptor::ReduceIntSum(tiles);
            amrex::Real max_valid = 0._rt, max_supported = 0._rt;
            for (int c = 0; c < 3; ++c) {
                finite = !temperature[c].contains_nan() && !temperature[c].contains_inf() && finite;
                auto const type = temperature[c].ixType().toIntVect();
                amrex::GpuArray<int, AMREX_SPACEDIM> nodal{};
                amrex::IntVect supported = ng;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    nodal[d] = type[d]; supported[d] -= np[d];
                }
                amrex::MultiFab error(temperature[c].boxArray(),temperature[c].DistributionMap(),1,ng);
                error.setVal(0._rt);
                auto const tt = temperature[c].const_arrays();
                auto const ee = error.arrays();
                amrex::ParallelFor(error, supported,
                    [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
                        int const cell[3] = {i,j,k};
                        amrex::Real expected = 2.5e6_rt * (1._rt + .2_rt*c);
                        if (profile) {
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                auto const x = (lo[d] + (cell[d] + .5_rt*(1-nodal[d]))*dx[d] - lo[d])/(hi[d]-lo[d]);
                                auto const h = dx[d]/(hi[d]-lo[d]);
                                auto const term = periodic[d] ? std::cos(2._rt*MathConst::pi*x)*std::pow(.5_rt+.5_rt*std::cos(2._rt*MathConst::pi*h),np[d]) : x*x+.5_rt*np[d]*h*h;
                                expected += 1.e5_rt*(d+1)*term;
                            }
                        }
                        ee[box](i,j,k) = tt[box](i,j,k)-expected;
                    });
                amrex::Gpu::streamSynchronize();
                max_valid = std::max(max_valid,error.norminf(0,0));
                max_supported = std::max(max_supported,error.norm0(0,1,ng));
                amrex::VisMF::Write(temperature[c],"T"+std::to_string(c));
            }
            constexpr amrex::Real bound = 256*std::numeric_limits<amrex::Real>::epsilon()*1.e7_rt;
            bool const pass = finite && max_valid <= bound && max_supported <= bound;
            amrex::Print() << std::setprecision(17)
                << "GHOST_UNITS {\"max_valid_error_K\":" << max_valid
                << ",\"max_supported_ghost_error_K\":" << max_supported
                << ",\"all_fabs_finite\":" << (finite?"true":"false")
                << ",\"apply_filter\":" << (apply_filter?"true":"false")
                << ",\"tiles\":" << tiles << ",\"omp_threads\":" << amrex::OpenMP::get_max_threads()
                << ",\"bound_K\":" << bound << ",\"passed\":" << (pass?"true":"false") << "}\n";
            status = pass ? 0 : 2;
        }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
    return status;
}
