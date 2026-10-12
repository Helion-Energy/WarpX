/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Fields.H"
#include "Diagnostics/MultiDiagnostics.H"
#include "Particles/PhysicalParticleContainer.H"
#include <AMReX_Reduce.H>
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"
#include <AMReX_EBFabFactory.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <cmath>
#include <exception>
#include <string>

int main(int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        amrex::ParmParse pp("standoff_test");
        bool reject = false;
        pp.query("reject", reject);
        amrex::Real expected_roof = 0;
        utils::parser::getWithParser(pp, "expected_roof", expected_roof);
        bool restart_test = false;
        pp.query("restart", restart_test);
        amrex::Long restart_count = 0, restart_removed = 0;
        for (int pass = 0; pass < (restart_test ? 2 : 1); ++pass) {
        auto& w = WarpX::GetInstance();
        bool caught = false;
        try { w.InitData(); }
        catch (std::exception const& e) {
            if (!reject || std::string(e.what()).find("signed-distance ceiling") == std::string::npos) {
                throw;
            }
            caught = true;
        }
        if (reject) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(caught, "Oversize band must fail before collection");
            amrex::Print() << "STANDOFF_REJECTION_PASS\n";
        } else {
            auto& phi = *w.m_fields.get(warpx::fields::FieldType::distance_to_eb, 0);
            auto const dx = w.Geom(0).CellSizeArray();
            auto const plo = w.Geom(0).ProbLoArray();
            amrex::Real const roof = amrex::min(dx[0], amrex::min(dx[1], dx[2])) *
                (w.fieldEBFactory(0).getMultiEBCellFlagFab().nGrow()+1);
            AMREX_ALWAYS_ASSERT(std::abs(roof-expected_roof) < 1.e-12);
            AMREX_ALWAYS_ASSERT(std::abs(phi.max(0)-roof) < 1.e-12);
            auto& pc = w.GetPartContainer().GetParticleContainer(0);
            AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() > 0);
            if (pass == 1) {
                AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == restart_count-restart_removed);
                amrex::Print() << "STANDOFF_RESTART_COLLECTION_PASS\n";
            }
            amrex::ReduceOps<amrex::ReduceOpMax> reduce;
            amrex::ReduceData<amrex::Real> data(reduce);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
                auto const a = phi.const_array(mfi);
                reduce.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        amrex::Real const x = plo[0]+i*dx[0], y = plo[1]+j*dx[1];
                        amrex::Real const exact = 0.2-std::sqrt(x*x+y*y);
                        if (std::abs(exact) > roof-2*dx[0]) { return {0.0}; }
                        return {std::abs(a(i,j,k)-exact)};
                    });
            }
            amrex::Real error = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceRealMax(error);
            amrex::Print() << "STANDOFF_DISTANCE_ERROR max_m=" << error << "\n";
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error < 1.e-4,
                "Facet distance error exceeds the 0.1 mm clearance allowance");
            bool split = false;
            pp.query("split", split);
            if (split) {
                auto& physical = dynamic_cast<PhysicalParticleContainer&>(pc);
                amrex::MultiFab rho(amrex::convert(pc.ParticleBoxArray(0),
                    amrex::IntVect::TheNodeVector()), pc.ParticleDistributionMap(0), 1, 1);
                rho.setVal(1.0);
                auto const count = pc.TotalNumberOfParticles();
                // Exact weights make the sum independent of serial CPU vs GPU
                // reduction order (millions of unequal floating sums are noisy).
                for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                    auto const a = pti.GetParticleTile().getParticleTileData();
                    amrex::ParallelFor(pti.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
                        a.m_rdata[PIdx::w][i] = 30.0;
                        a.m_rdata[PIdx::ux][i] = 1024.0;
                        a.m_rdata[PIdx::uy][i] = 2048.0;
                        a.m_rdata[PIdx::uz][i] = 4096.0;
                    });
                }
                auto const weight = pc.sumParticleWeight(false);
                physical.SplitDepletedBand(rho, 0.5, 1.5, 0);
                pc.Redistribute();
                AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == 30*count);
                AMREX_ALWAYS_ASSERT(std::abs(pc.sumParticleWeight(false)-weight) < 1.e-12*weight);
                // Every child has the exact parent velocity and 1/30 weight,
                // proving momentum and relativistic energy conservation without
                // an order-sensitive six-million-term energy sum.
                amrex::Gpu::DeviceScalar<int> mismatches(0);
                auto* wrong = mismatches.dataPtr();
                for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                    auto const a = pti.GetParticleTile().getConstParticleTileData();
                    amrex::For(pti.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
                        if (a.m_rdata[PIdx::w][i] != 1.0 ||
                            a.m_rdata[PIdx::ux][i] != 1024.0 ||
                            a.m_rdata[PIdx::uy][i] != 2048.0 ||
                            a.m_rdata[PIdx::uz][i] != 4096.0) {
                            amrex::Gpu::Atomic::AddNoRet(wrong, 1);
                        }
                    });
                }
                int bad = mismatches.dataValue();
                amrex::ParallelDescriptor::ReduceIntSum(bad);
                AMREX_ALWAYS_ASSERT(bad == 0);
                amrex::Print() << "STANDOFF_SPLIT_CONSERVATION_PASS\n";
            }
            int advance = 0;
            pp.query("advance", advance);
            if (advance > 0) {
                auto const before = w.GetPartContainer().GetEBCollectedCharge("ions");
                w.Evolve(advance);
                AMREX_ALWAYS_ASSERT(w.GetPartContainer().GetEBCollectedCharge("ions") > before);
            }
            bool reject_footprint = false;
            pp.query("reject_footprint", reject_footprint);
            if (reject_footprint) {
                for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                    auto const data = pti.GetParticleTile().getParticleTileData();
                    amrex::ParallelFor(pti.numParticles(), [=] AMREX_GPU_DEVICE(int i) {
                        data.m_rdata[PIdx::x][i] = 0.194;
                        data.m_rdata[PIdx::y][i] = 0.0;
                    });
                }
                pc.Redistribute();
                bool failed = false;
                try {
                    pc.DepositCurrent(w.m_fields.get_mr_levels_alldirs(
                        warpx::fields::FieldType::current_fp, 0), w.getdt(0), -0.5*w.getdt(0));
                } catch (std::exception const& e) {
                    if (std::string(e.what()).find("footprint leaves") == std::string::npos) { throw; }
                    failed = true;
                }
                int failures = failed ? 1 : 0;
                amrex::ParallelDescriptor::ReduceIntSum(failures);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(failures > 0,
                    "Release builds must reject an unsafe particle/current footprint");
                amrex::Print() << "STANDOFF_FOOTPRINT_REJECTION_PASS\n";
            }
            if (WarpX::eb_boundary_type == EmbeddedBoundaryType::Insulating && !reject_footprint) {
                AMREX_ALWAYS_ASSERT(w.GetPartContainer().GetEBCollectedCharge("ions") > 0);
                amrex::Real const threshold = WarpX::eb_standoff_cells *
                    amrex::max(dx[0], amrex::max(dx[1], dx[2]));
                amrex::Gpu::DeviceScalar<int> failures(0);
                auto* failed = failures.dataPtr();
                for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                    auto const data = pti.GetParticleTile().getConstParticleTileData();
                    auto const field = phi.const_array(pti);
                    amrex::For(pti.numParticles(), [=] AMREX_GPU_DEVICE(int p) {
                        amrex::Real const x = (data.m_rdata[PIdx::x][p]-plo[0])/dx[0];
                        amrex::Real const y = (data.m_rdata[PIdx::y][p]-plo[1])/dx[1];
                        amrex::Real const z = (data.m_rdata[PIdx::z][p]-plo[2])/dx[2];
                        int const i = static_cast<int>(std::floor(x));
                        int const j = static_cast<int>(std::floor(y));
                        int const k = static_cast<int>(std::floor(z));
                        amrex::Real distance = 0;
                        for (int a=0; a<2; ++a) {
                            for (int b=0; b<2; ++b) {
                                for (int c=0; c<2; ++c) {
                                    distance += field(i+a,j+b,k+c) *
                                        (a ? x-i : 1-x+i) * (b ? y-j : 1-y+j) *
                                        (c ? z-k : 1-z+k);
                                }
                            }
                        }
                        if (distance < threshold-1.e-12) {
                            amrex::Gpu::Atomic::AddNoRet(failed, 1);
                        }
                    });
                }
                int errors = failures.dataValue();
                amrex::ParallelDescriptor::ReduceIntSum(errors);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(errors == 0,
                    "Initial insulating collection left a particle in the standoff");
            }
            amrex::Print() << "STANDOFF_GEOMETRY_PASS roof_m=" << roof
                           << " particles=" << pc.TotalNumberOfParticles() << "\n";
        }
        if (restart_test && pass == 0) {
            auto& pc = w.GetPartContainer().GetParticleContainer(0);
            restart_count = pc.TotalNumberOfParticles();
            // Deliberately put one marker per owned tile in the collection band.
            // The checkpoint must retain these; the next InitData must collect them.
            for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                if (pti.numParticles() == 0) { continue; }
                ++restart_removed;
                auto const a = pti.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int i) {
                    a.m_rdata[PIdx::x][i] = 0.185;
                    a.m_rdata[PIdx::y][i] = 0.0;
                });
            }
            pc.Redistribute();
            amrex::ParallelDescriptor::ReduceLongSum(restart_removed);
            w.GetMultiDiags().NewIteration();
            w.GetMultiDiags().FilterComputePackFlush(0, true);
        }
        WarpX::ResetInstance();
        if (restart_test && pass == 0) {
            amrex::ParmParse("amr").add("restart", std::string("standoff_chk000000"));
        }
        }
    }
    warpx::initialization::finalize_external_libraries();
}
