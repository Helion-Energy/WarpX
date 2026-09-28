/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "Initialization/WarpXInit.H"
#include "NonlinearSolvers/HybridPICPC.H"
#include <AMReX_ParmParse.H>
#include <cmath>

namespace {
struct FrozenOps {
    HybridPICModel const* model;
    amrex::MultiFab const* density;
    amrex::Vector<amrex::Array<amrex::MultiFab*,3>> const* mass = nullptr;
    HybridPICModel const*
    GetHybridPICModel () const {
        return model;
    }
    amrex::Real
    GetTheta () const {
        return .5;
    }
    std::pair<amrex::MultiFab const*, int>
    GetOhmDensityForPC (int) const {
        return {density, 0};
    }
    auto
    GetBfieldThetaForPC (int) const {
        auto v = WarpX::GetInstance().m_fields.get_alldirs(
            warpx::fields::FieldType::Bfield_fp, 0);
        return amrex::Array<amrex::MultiFab const*, 3>{v[0], v[1], v[2]};
    }
    auto
    GetIonCurrentForPC (int) const {
        auto v = WarpX::GetInstance().m_fields.get_alldirs(
            warpx::fields::FieldType::current_fp, 0);
        return amrex::Array<amrex::MultiFab const*, 3>{v[0], v[1], v[2]};
    }
    auto const&
    GetFieldBoundaryLo () const {
        return WarpX::field_boundary_lo;
    }
    auto const&
    GetFieldBoundaryHi () const {
        return WarpX::field_boundary_hi;
    }
    amrex::Vector<amrex::Array<amrex::MultiFab*, 3>> const*
    GetMassMatricesCoeff () const {
        return mass;
    }
    bool
    HasPressureUnknownForPC () const {
        return false;
    }
    amrex::Real
    PressureScaleForPC () const {
        return 1.;
    }
    amrex::Real
    PressureConductivityForPC () const {
        return 0.;
    }
};
} // namespace
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        amrex::ParmParse pc_inputs("pc_hybrid_pic");
        pc_inputs.add("inner_max", 63);
        pc_inputs.add("inner_rtol", 1.e-10);
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        using warpx::fields::FieldType;
        auto const* rho = sim.m_fields.get(FieldType::rho_fp, 0);
        amrex::MultiFab density(rho->boxArray(), rho->DistributionMap(), 1, 3);
        density.setVal(1.);
        for (int c = 0; c < 3; ++c) {
            sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0)[c]->setVal(
                c == 2 ? 1. : 0.);
            sim.m_fields.get_alldirs(FieldType::current_fp, 0)[c]->setVal(0.);
            sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0)[c]
                ->setVal(0.);
        }
        bool ion_test = false, pmc_test = false;
        amrex::ParmParse("hall_test").query("ion",ion_test);
        amrex::ParmParse("hall_test").query("pmc",pmc_test);
        amrex::Array<std::unique_ptr<amrex::MultiFab>,3> mass;
        amrex::Vector<amrex::Array<amrex::MultiFab*,3>> mass_ptr(1);
        for (int c = 0; c < 3; ++c) {
            auto const& e = *sim.m_fields.get_alldirs(FieldType::Efield_fp,0)[c];
            mass[c] = std::make_unique<amrex::MultiFab>(e.boxArray(),e.DistributionMap(),1,3);
            mass[c]->setVal(.2*(c+1)); mass_ptr[0][c] = mass[c].get();
        }
        amrex::Parser eta_parser("0.05"), hyper_parser("0.0001");
        eta_parser.registerVariables({"rho","J","t"});
        hyper_parser.registerVariables({"rho","B"});
        auto* model = sim.get_pointer_HybridPICModel();
        if (ion_test) {
            AMREX_ALWAYS_ASSERT(WarpX::grid_type == GridType::Collocated);
            model->m_eta = eta_parser.compile<3>();
            model->m_eta_h = hyper_parser.compile<2>();
            model->m_include_hyper_resistivity_term = true;
        }
        FrozenOps ops{model, &density, ion_test ? &mass_ptr : nullptr};
        WarpXSolverVec exact, rhs, solution;
        exact.Define(&sim, "Efield_fp");
        rhs.Define(exact);
        solution.Define(exact);
        exact.zero();
        rhs.zero();
        solution.zero();
        HybridPICPC<WarpXSolverVec, FrozenOps> pc;
        pc.Define(exact, &ops);
        amrex::Real const cw = .05, dt = 2. * PhysConst::mu0 * cw;
        pc.CurTimeStep(dt);
        pc.Update(exact);
        amrex::Vector<int> mode{0, 0, 1};
        int gradient = 0;
        amrex::ParmParse("hall_test").queryarr("mode", mode);
        amrex::ParmParse("hall_test").query("gradient", gradient);
        AMREX_ALWAYS_ASSERT(mode.size() == 3);
        auto const dx = sim.Geom(0).CellSizeArray();
        bool const staggered = WarpX::grid_type == GridType::Staggered;
        amrex::GpuArray<amrex::Real, 3> wave{}, q{}, average{},
            cosine{1., 0., 0.}, sine{0., 1., 0.};
        if (pmc_test) { sine = {0.,0.,0.}; cosine = {1.,1.,0.}; }
        amrex::Real q2 = 0., lap_symbol = 0.;
        for (int d = 0; d < 3; ++d) {
            wave[d] = 2. * 3.14159265358979323846 * mode[d];
            q[d] = staggered ? 2. * std::sin(.5 * wave[d] * dx[d]) / dx[d]
                             : std::sin(wave[d] * dx[d]) / dx[d];
            average[d] = staggered ? std::cos(.5 * wave[d] * dx[d]) : 1.;
            q2 += q[d] * q[d];
            lap_symbol += 4.*std::pow(std::sin(.5*wave[d]*dx[d])/dx[d],2);
        }
        if (gradient) {
            cosine = q;
            sine = {0., 0., 0.};
        }
        auto action = [&] (amrex::GpuArray<amrex::Real, 3> v) {
            auto a = v;
            amrex::Real dot = 0.;
            for (int d = 0; d < 3; ++d) {
                dot += q[d] * v[d];
            }
            amrex::GpuArray<amrex::Real, 3> k{};
            for (int d = 0; d < 3; ++d) {
                k[d] = average[d] * (q2 * v[d] - q[d] * dot);
            }
            a[0] += cw * average[0] * k[1];
            a[1] -= cw * average[1] * k[0];
            if (ion_test) {
                // Independent Fourier symbol of I+S+(H+D+S D)K.
                amrex::Real const dissipative = cw*(.05+.0001*lap_symbol);
                for (int c = 0; c < 3; ++c) { a[c] += dissipative*k[c]; }
                a[0] += .4*(v[1]+dissipative*k[1]);
                a[1] -= .2*(v[0]+dissipative*k[0]);
            }
            return a;
        };
        auto const bc = action(cosine), bs = action(sine);
        for (int c = 0; c < 3; ++c) {
            auto& a = *exact.getArrayVec()[0][c];
            auto& b = *rhs.getArrayVec()[0][c];
            auto const ix = a.ixType().toIntVect();
            for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi) {
                auto u = a.array(mfi), v = b.array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        amrex::Real const phase =
                            (pmc_test ? 0. : .27) + wave[0] * (i + (ix[0] ? 0. : .5)) * dx[0] +
                            wave[1] * (j + (ix[1] ? 0. : .5)) * dx[1] +
                            wave[2] * (k + (ix[2] ? 0. : .5)) * dx[2];
                        u(i, j, k) = cosine[c] * std::cos(phase) +
                                     sine[c] * std::sin(phase);
                        v(i, j, k) =
                            bc[c] * std::cos(phase) + bs[c] * std::sin(phase);
                    });
            }
        }
        pc.Apply(solution, rhs);
        solution -= exact;
        amrex::Real const error = solution.norm2() / exact.norm2();
        amrex::Print() << "CARTESIAN_HALL_PC staggered=" << staggered
                       << " mode=" << mode[0] << "," << mode[1] << ","
                       << mode[2] << " gradient=" << gradient
                       << " error=" << error << '\n';
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(error) && error < 3.e-9,
            "preserved Cartesian Hall mode inverse failed");
        amrex::Gpu::streamSynchronize();
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
