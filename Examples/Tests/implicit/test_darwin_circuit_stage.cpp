/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL.
 */
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Python/callbacks.H"
#include "WarpX.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <vector>

using namespace amrex::literals;

// A manufactured algebraic B-to-coil map, not a physical circuit model.
// Exercise installed callbacks, including rejected residual/Jv evaluations.
int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        amrex::Real feedback = 10._rt, radius = .18_rt, theta = .5_rt;
        amrex::ParmParse test("circuit_test");
        test.query("feedback", feedback);
        test.query("radius", radius);
        amrex::ParmParse("implicit_evolve").query("theta", theta);
        amrex::Real start_time = 0._rt, dt = 0._rt, linkage = 0._rt;
        std::vector<amrex::Real> accepted, candidate;
        int starts = 0, probes = 0, commits = 0;

        auto measure = [&] () {
            auto const& bz = *simulation.m_fields.get(
                warpx::fields::FieldType::Bfield_fp, ablastr::fields::Direction{2}, 0);
            auto const& geometry = simulation.Geom(0);
            auto const dx = geometry.CellSizeArray();
            auto const lo = geometry.ProbLoArray();
            int const plane = geometry.Domain().smallEnd(2) + geometry.Domain().length(2)/2;
            auto const owner = amrex::OwnerMask(bz, geometry.periodicity());
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> operation;
            amrex::ReduceData<amrex::Real, amrex::Real> data(operation);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(bz); mfi.isValid(); ++mfi) {
                auto const field = bz.const_array(mfi);
                auto const mask = owner->const_array(mfi);
                operation.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple {
                        amrex::Real const x = lo[0] + (i + .5_rt)*dx[0];
                        amrex::Real const y = lo[1] + (j + .5_rt)*dx[1];
                        if (k != plane || mask(i,j,k) == 0 || x*x + y*y >= radius*radius) {
                            return {0._rt, 0._rt};
                        }
                        return {field(i,j,k), 1._rt};
                    });
            }
            auto const values = data.value(operation);
            amrex::Real sums[2] = {amrex::get<0>(values), amrex::get<1>(values)};
            amrex::ParallelDescriptor::ReduceRealSum(sums, 2);
            AMREX_ALWAYS_ASSERT(sums[1] > 0._rt);
            return sums[0]/sums[1];
        };
        auto external = [&] () -> ExternalVectorPotential& {
            return *simulation.get_pointer_HybridPICModel()->m_external_vector_potential;
        };
        InstallPythonCallback("beforestep", [&] () {
            AMREX_ALWAYS_ASSERT(starts == commits);
            start_time = simulation.gett_new(0);
            dt = simulation.getdt(0);
            linkage = measure(); // Actual delivered endpoint B, after the preceding step.
            accepted.resize(external().nFields());
            candidate.resize(external().nFields());
            for (int coil = 0; coil < external().nFields(); ++coil) {
                accepted[coil] = external().TimeScale(coil, start_time);
                candidate[coil] = accepted[coil] + dt*5.e7_rt/(coil + 1);
                external().SetScale(external().FieldName(coil), accepted[coil], candidate[coil],
                                    start_time, start_time + dt);
            }
            ++starts;
        });
        InstallPythonCallback("externalcoiltheta", [&] () {
            AMREX_ALWAYS_ASSERT(starts == commits + 1);
            AMREX_ALWAYS_ASSERT(simulation.getdt(0) == dt);
            amrex::Real const trial = measure();
            for (int coil = 0; coil < external().nFields(); ++coil) {
                // With two coils only the second feeds back: a first-coil-only
                // convergence test cannot pass the residual replay regression.
                amrex::Real const coupling = coil + 1 == external().nFields() ? feedback : 0._rt;
                candidate[coil] = accepted[coil] + dt*5.e7_rt/(coil + 1)
                    - coupling*(trial - linkage)/theta;
                external().SetScale(external().FieldName(coil), accepted[coil], candidate[coil],
                                    start_time, start_time + dt);
            }
            ++probes;
        });
        InstallPythonCallback("externalcoilfinish", [&] () {
            // The theta callback already proposes the endpoint circuit state.
            // Commit it once; do not re-integrate using a stale endpoint B probe.
            AMREX_ALWAYS_ASSERT(starts == commits + 1 && probes > 0);
            for (int coil = 0; coil < external().nFields(); ++coil) {
                AMREX_ALWAYS_ASSERT(std::abs(external().TimeScale(coil, start_time + dt)
                    - candidate[coil]) < 1.e-12_rt);
            }
            ++commits;
        });
        simulation.InitData();
        simulation.Evolve();
        AMREX_ALWAYS_ASSERT(starts == 2 && commits == starts && probes > 10);
        amrex::Print() << "CIRCUIT_STAGE_TEST: starts=" << starts << ", commits=" << commits
                       << ", probes=" << probes << '\n';
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
