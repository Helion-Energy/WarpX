/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/ImplicitSolvers/DarwinABoundary.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "ablastr/coarsen/sample.H"

#include <AMReX_ParmParse.H>

#include <cmath>
#include <memory>

namespace {
using Vector = amrex::Array<amrex::MultiFab, 3>;

void
initialize_potential (Vector& a, amrex::Geometry const& geom) {
    auto const dx = geom.CellSizeArray();
    auto const lo = geom.ProbLoArray();
    auto const hi = geom.ProbHiArray();
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{}, pmc_lo{}, pmc_hi{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        periodic[d] = geom.isPeriodic(d);
        pmc_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
        pmc_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
    }
    for (int c = 0; c < 3; ++c) {
        auto const stag = a[c].ixType().toIntVect();
        amrex::MultiFab imposed(a[c].boxArray(), a[c].DistributionMap(), 1, 4);
        imposed.setVal(0.);
        for (amrex::MFIter mfi(a[c]); mfi.isValid(); ++mfi) {
            auto const value = a[c].array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                  int k) {
                int const p[3]{i, j, k};
                amrex::Real shape = .001 * (c + 1);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    amrex::Real const x =
                        (p[d] + .5 * (1 - stag[d])) * dx[d] / (hi[d] - lo[d]);
                    amrex::Real const frequency = periodic[d] ? 2. : 1.;
                    amrex::Real const angle =
                        frequency * 3.14159265358979323846 * x;
                    if (d == AMREX_SPACEDIM - 1) {
                        shape *= stag[d] ? std::cos(angle) : std::sin(angle);
                    } else {
#if defined(WARPX_DIM_RZ)
                        // Az is even at the axis; Ar and Atheta are odd.
                        shape *= c == 2 ? .5 * (1. + std::cos(angle))
                                        : std::sin(angle);
#else
                            shape *= std::sin(angle);
#endif
                    }
                }
                value(i, j, k) = shape;
            });
        }
        a[c].OverrideSync(geom.periodicity());
        a[c].FillBoundary(geom.periodicity());
        ApplyDarwinCellCenteredABoundary(a[c], imposed, geom,
#if defined(WARPX_DIM_RZ)
                                         true,
#else
                                         false,
#endif
                                         nullptr, pmc_lo, pmc_hi);
    }
}

void
assemble (Vector& a, Vector& b, Vector& current, amrex::MultiFab& nodal,
          amrex::Geometry const& geom, FiniteDifferenceSolver& solver) {
    initialize_potential(a, geom);
    ablastr::fields::VectorField av{&a[0], &a[1], &a[2]};
    ablastr::fields::VectorField bv{&b[0], &b[1], &b[2]};
    ablastr::fields::VectorField jv{&current[0], &current[1], &current[2]};
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> no_eb;
    solver.ComputeCurlA(
        bv, av, no_eb, 0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo, WarpX::field_boundary_hi));
    for (auto& component : b) {
        component.OverrideSync(geom.periodicity());
        component.FillBoundary(geom.periodicity());
    }
    solver.CalculateCurrentAmpere(jv, bv, no_eb, 0);
    // The same production Yee-to-node sampler used by electron inertia.
    // Its physical-cap stencil reads the mixed corners in curl(A).
    for (int c = 0; c < 3; ++c) {
        amrex::GpuArray<int, 3> stagger{1, 1, 1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            stagger[d] = current[c].ixType().nodeCentered(d);
        }
        for (amrex::MFIter mfi(nodal); mfi.isValid(); ++mfi) {
            auto const j = current[c].const_array(mfi);
            auto const n = nodal.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int jidx, int k) {
                    n(i, jidx, k, c) = ablastr::coarsen::sample::Interp(
                        j, stagger, {1, 1, 1}, {1, 1, 1}, i, jidx, k, 0);
                });
        }
    }
}
} // namespace

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        auto const& geom = simulation.Geom(0);
        auto& solver = *simulation.get_pointer_fdtd_solver_fp(0);
        auto const electric = simulation.m_fields.get_alldirs(
            warpx::fields::FieldType::Efield_fp, 0);
        auto const magnetic = simulation.m_fields.get_alldirs(
            warpx::fields::FieldType::Bfield_fp, 0);
        amrex::BoxArray single(geom.Domain());
        amrex::DistributionMapping single_dm(amrex::Vector<int>{0});
        int input_guard = 4, output_guard = 4;
        amrex::ParmParse("curl_guard_test").query("input_guard", input_guard);
        amrex::ParmParse("curl_guard_test").query("output_guard", output_guard);
        amrex::IntVect input_grow(4), output_grow(4);
        input_grow[0] = input_guard;
        output_grow[0] = output_guard;
        amrex::Print() << "CURL_ALLOC registry_E=" << electric[0]->nGrowVect()
                       << " registry_B=" << magnetic[0]->nGrowVect()
                       << " fixture_A=" << input_grow
                       << " fixture_B=" << output_grow << "\n";
        Vector a, b, current, reference_a, reference_b, reference_j;
        for (int d = 0; d < 3; ++d) {
            auto const electric_type = electric[d]->ixType();
            auto const magnetic_type = magnetic[d]->ixType();
            a[d].define(electric[d]->boxArray(), electric[d]->DistributionMap(),
                        1, input_grow);
            current[d].define(electric[d]->boxArray(),
                              electric[d]->DistributionMap(), 1, 4);
            b[d].define(magnetic[d]->boxArray(), magnetic[d]->DistributionMap(),
                        1, output_grow);
            reference_a[d].define(amrex::convert(single, electric_type),
                                  single_dm, 1, 4);
            reference_j[d].define(amrex::convert(single, electric_type),
                                  single_dm, 1, 4);
            reference_b[d].define(amrex::convert(single, magnetic_type),
                                  single_dm, 1, 4);
            current[d].setVal(0.);
            reference_j[d].setVal(0.);
        }
        amrex::MultiFab nodal(
            amrex::convert(a[0].boxArray(), amrex::IntVect(1)),
            a[0].DistributionMap(), 3, 0);
        amrex::MultiFab reference_nodal(
            amrex::convert(single, amrex::IntVect(1)), single_dm, 3, 0);
        assemble(reference_a, reference_b, reference_j, reference_nodal, geom,
                 solver);
        assemble(a, b, current, nodal, geom, solver);
        amrex::Real curl_error = 0., curl_scale = 0., parity_error = 0.;
        for (int d = 0; d < 3; ++d) {
            amrex::MultiFab expected(b[d].boxArray(), b[d].DistributionMap(), 1,
                                     1);
            expected.setVal(0.);
            expected.ParallelCopy(reference_b[d], 0, 0, 1, 1, 1,
                                  geom.periodicity());
            curl_scale = std::max(curl_scale, expected.norminf(0, 1));
            amrex::MultiFab::Subtract(expected, b[d], 0, 0, 1, 1);
            curl_error = std::max(curl_error, expected.norminf(0, 1));
            // Independent boundary oracle: curl of homogeneous PMC images of
            // A has odd tangential B and even normal B at each cap. Check the
            // mixed corners too, not just shared points inside the domain.
            constexpr int axial = AMREX_SPACEDIM - 1;
            auto const domain = amrex::convert(geom.Domain(), b[d].ixType());
            auto const lower = domain.smallEnd();
            auto const upper = domain.bigEnd();
            int const node = b[d].ixType().nodeCentered(axial);
            bool const pmc_lo =
                WarpX::field_boundary_lo[axial] == FieldBoundaryType::PMC;
            bool const pmc_hi =
                WarpX::field_boundary_hi[axial] == FieldBoundaryType::PMC;
            expected.setVal(0.);
            for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
                auto const field = b[d].const_array(mfi);
                auto const error = expected.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    int const point[3]{i, j, k};
                    int mirror[3]{i, j, k};
                    for (int t = 0; t < axial; ++t) {
                        if (point[t] < lower[t] || point[t] > upper[t]) {
                            return;
                        }
                    }
                    if (pmc_lo && point[axial] < lower[axial]) {
                        mirror[axial] =
                            2 * lower[axial] - (1 - node) - point[axial];
                    } else if (pmc_hi && point[axial] > upper[axial]) {
                        mirror[axial] =
                            2 * upper[axial] + (1 - node) - point[axial];
                    } else {
                        return;
                    }
                    error(i, j, k) = field(i, j, k) -
                                     (node ? 1. : -1.) *
                                         field(mirror[0], mirror[1], mirror[2]);
                });
            }
            parity_error = std::max(parity_error, expected.norminf(0, 1));
        }
        amrex::MultiFab expected(nodal.boxArray(), nodal.DistributionMap(), 3,
                                 0);
        expected.ParallelCopy(reference_nodal, geom.periodicity());
        amrex::Real current_error = 0., current_scale = 0.;
        for (int d = 0; d < 3; ++d) {
            current_scale = std::max(current_scale, expected.norminf(d));
        }
        amrex::MultiFab::Subtract(expected, nodal, 0, 0, 3, 0);
        for (int d = 0; d < 3; ++d) {
            current_error = std::max(current_error, expected.norminf(d));
        }
        amrex::Print().SetPrecision(17)
            << "CURL_GUARDS curl_error=" << curl_error
            << " curl_scale=" << curl_scale
            << " pmc_parity_error=" << parity_error
            << " nodal_current_error=" << current_error
            << " current_scale=" << current_scale
            << " boxes=" << a[0].boxArray().size()
            << " ranks=" << amrex::ParallelDescriptor::NProcs() << "\n";
        AMREX_ALWAYS_ASSERT(curl_scale > 0. && current_scale > 0.);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            curl_error < 5.e-13 * curl_scale,
            "curl(A) physical/inter-box corners depend on decomposition");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(parity_error < 5.e-13 * curl_scale,
                                         "curl(A) violates PMC magnetic parity "
                                         "at a physical/inter-box corner");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(current_error < 5.e-13 * current_scale,
                                         "native Ampere/nodal electron-current "
                                         "assembly depends on decomposition");
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
