/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronInertiaConvection.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronInertiaElliptic.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>

namespace
{
    using namespace amrex::literals;
    using Field = ablastr::fields::VectorField;
    using warpx::fields::FieldType;
    using amrex::MultiFab;

    struct Scratch
    {
        std::array<MultiFab, 3> values;
        Field field;
        explicit Scratch (Field const& like)
        {
            for (int c = 0; c < 3; ++c) {
                values[c].define(like[c]->boxArray(), like[c]->DistributionMap(), 1,
                                 like[c]->nGrowVect());
                values[c].setVal(0.0_rt);
                field[c] = &values[c];
            }
        }
    };

    amrex::Real norm (Field const& a)
    {
        amrex::Real result = 0.0_rt;
        for (int c = 0; c < 3; ++c) { result = std::max(result, a[c]->norminf()); }
        return result;
    }

    void copy (Field const& dst, Field const& src)
    {
        for (int c = 0; c < 3; ++c) { MultiFab::Copy(*dst[c], *src[c], 0, 0, 1, 0); }
    }

    void compare (Field const& measured, Field const& exact, char const* label,
                  amrex::Real tolerance = 2.e-9_rt)
    {
        Scratch delta(measured);
        copy(delta.field, measured);
        for (int c = 0; c < 3; ++c) {
            MultiFab::Subtract(delta.values[c], *exact[c], 0, 0, 1, 0);
        }
        amrex::Real const scale = std::max(1.e-20_rt, norm(exact));
        amrex::Real const relative = norm(delta.field) / scale;
        amrex::Print() << "CONVECTION " << label << " relative=" << relative
                       << " signal_V_per_m=" << scale << '\n';
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(relative) && relative < tolerance, label);
    }
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT(hp.m_include_electron_inertia_elliptic);
        AMREX_ALWAYS_ASSERT(hp.m_electron_inertia_convection);
        auto const& geom = w.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const lower = geom.ProbLoArray();
        auto& rho = *w.m_fields.get(FieldType::rho_fp, 0);
        auto* ped = const_cast<MultiFab*>(hp.DensityPedestal(0));
        AMREX_ALWAYS_ASSERT(ped != nullptr);
        auto const J = w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        auto const Ji = w.m_fields.get_alldirs(FieldType::current_fp, 0);
        auto const B = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto const floor = hp.m_n_floor * PhysConst::q_e;
        constexpr amrex::Real density = 1.e19_rt * PhysConst::q_e;
        constexpr amrex::Real A = 2.e5_rt, Omega = 3.e5_rt, C = -4.e5_rt, U = 2.e4_rt;
        amrex::Real const wave = 2.0_rt*MathConst::pi / (geom.ProbHi(1)-lower[1]);
        constexpr amrex::Real smooth_U = 5.e5_rt, potential_D = 1.e7_rt;
        Scratch expected(E), first(E), actual(E), recovered(E);
        ElectronInertiaConvection convection;
        amrex::Print() << std::setprecision(17);

        // Affine velocity has an independent exact material acceleration:
        // u=(A r, Omega r, C z+U), a=((A^2-Omega^2)r,2 A Omega r,C(C z+U)).
        // Nonzero Ji=3 rho u and J=2 rho u deliberately distinguish electron
        // flow from either ordinary deposited j or the Ampere current alone.
        auto manufacture = [&] (int kind, amrex::Real scale) {
            bool const variable = kind == 1;
            bool const common_flow = kind == 2;
            bool const smooth = kind == 3, potential = kind == 4;
            ped->setVal(variable ? 0.01_rt*density : 0.2_rt*density);
            for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
                auto const den = rho.array(mfi);
                int const components = rho.nComp();
                amrex::ParallelFor(mfi.fabbox(), components,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                        amrex::Real const r = lower[0] + i*dx[0];
                        den(i,j,k,n) = variable ? density*(0.01_rt + 2.0_rt*r*r) : density;
                    });
            }
            for (int c = 0; c < 3; ++c) {
                B[c]->setVal(0.0_rt);
                E[c]->setVal(0.0_rt);
                auto const it = J[c]->ixType().toIntVect();
                for (amrex::MFIter mfi(*J[c]); mfi.isValid(); ++mfi) {
                    auto const jp = J[c]->array(mfi), ji = Ji[c]->array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        amrex::Real const r = lower[0] + (i+(it[0]?0.0_rt:0.5_rt))*dx[0];
                        amrex::Real const z = lower[1] + (j+(it[1]?0.0_rt:0.5_rt))*dx[1];
                        amrex::Real const charge = variable
                            ? amrex::max(floor, density*(0.02_rt+2.0_rt*r*r)) : 1.2_rt*density;
                        amrex::Real const velocity = scale*(
                            smooth ? (c==0 ? 0.0_rt : c==1
                                ? Omega*r*std::sin(wave*(z-lower[1])) : smooth_U)
                            : potential ? (c==0 ? A*r+potential_D*r*z*z : c==1
                                ? 0.0_rt : C*z+potential_D*r*r*z)
                            : variable ? (c==1 ? Omega*r : 0.0_rt)
                            : c==0 ? A*r : c==1 ? Omega*r : C*z+U);
                        ji(i,j,k) = 3.0_rt*charge*velocity;
                        jp(i,j,k) = (common_flow ? 3.0_rt : 2.0_rt)*charge*velocity;
                    });
                }
                // Expected edge values are analytic, independent of the
                // implemented derivative/interpolation or velocity scratch.
                auto const et = E[c]->ixType().toIntVect();
                for (amrex::MFIter mfi(expected.values[c]); mfi.isValid(); ++mfi) {
                    auto const exact = expected.values[c].array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        amrex::Real const r = lower[0] + (i+(et[0]?0.0_rt:0.5_rt))*dx[0];
                        amrex::Real const z = lower[1] + (j+(et[1]?0.0_rt:0.5_rt))*dx[1];
                        amrex::Real const f = std::sin(wave*(z-lower[1]));
                        amrex::Real const accel = common_flow ? 0.0_rt
                            : smooth ? (c==0 ? -Omega*Omega*r*f*f : c==1
                                ? smooth_U*Omega*r*wave*std::cos(wave*(z-lower[1])) : 0.0_rt)
                            : variable ? (c==0 ? -Omega*Omega*r : 0.0_rt)
                            : c==0 ? (A*A-Omega*Omega)*r
                            : c==1 ? 2.0_rt*A*Omega*r : C*(C*z+U);
                        exact(i,j,k) = -PhysConst::m_e/PhysConst::q_e * scale*scale*accel;
                    });
                }
            }
        };
        manufacture(0, 1.0_rt);
        convection.AddToRHS(E, J, Ji, rho, ped, floor, geom);
        compare(E, expected.field, "affine_all_components_and_axis");
        copy(first.field, E);
        manufacture(0, 2.0_rt);
        convection.AddToRHS(E, J, Ji, rho, ped, floor, geom);
        compare(E, expected.field, "stage_refresh_quadratic_scaling");
        manufacture(1, 1.0_rt);
        convection.AddToRHS(E, J, Ji, rho, ped, floor, geom);
        compare(E, expected.field, "rotation_density_floor_and_pedestal");
        manufacture(2, 1.0_rt);
        convection.AddToRHS(E, J, Ji, rho, ped, floor, geom);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(norm(E) == 0.0_rt,
            "Equal ion and plasma currents must give zero electron convection");
        amrex::Print() << "CONVECTION equal_current_null PASS\n";

        // Independent nonlinear smooth solution. Its truncation error must
        // decrease quadratically with dz; CTest runs several resolutions.
        manufacture(3, 1.0_rt);
        convection.AddToRHS(E, J, Ji, rho, ped, floor, geom);
        compare(E, expected.field, "smooth_sinusoid_second_order", wave*wave*dx[1]*dx[1]);

        // Irrotational but genuinely two-dimensional flow: u=grad(phi),
        // phi=A*r^2/2+C*z^2/2+D*r^2*z^2/2. Its convective acceleration
        // is a gradient and must not create any magnetic field.
        manufacture(4, 1.0_rt);
        convection.AddToRHS(E, J, Ji, rho, ped, floor, geom);
        Scratch potential_curl(B);
        w.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(
            potential_curl.field, E, w.GetEBUpdateBFlag()[0], 0);
        amrex::Real const curl_relative = norm(potential_curl.field)
            * std::min(dx[0],dx[1]) / norm(E);
        amrex::Print() << "CONVECTION potential_flow_curl_null relative=" << curl_relative << '\n';
        AMREX_ALWAYS_ASSERT(curl_relative < 1.e-11_rt);

        // A frozen EB row remains exactly unchanged (even for nonzero RHS).
        manufacture(0, 1.0_rt);
        std::array<amrex::iMultiFab, 3> masks;
        std::array<amrex::iMultiFab const*, 3> mask_ptr{};
        for (int c = 0; c < 3; ++c) {
            masks[c].define(E[c]->boxArray(), E[c]->DistributionMap(), 1, 0);
            mask_ptr[c] = &masks[c];
            for (amrex::MFIter mfi(masks[c]); mfi.isValid(); ++mfi) {
                auto const mask = masks[c].array(mfi);
                auto const exact = expected.values[c].array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    mask(i,j,k) = (i%5 != 0) ? 1 : 0;
                    if (mask(i,j,k) == 0) { exact(i,j,k) = 0.0_rt; }
                });
            }
        }
        convection.AddToRHS(E, J, Ji, rho, ped, floor, geom, mask_ptr);
        compare(E, expected.field, "frozen_edge_mask");

        // Assemble an independently known analytic convection RHS, apply
        // the production boundaries, and recover it with the existing
        // elliptic operator. The wrapper must produce this same field.
        manufacture(0, 1.0_rt);
        copy(E, expected.field);
        w.ApplyEfieldBoundary(0, PatchType::fine, 0.0_rt);
        copy(first.field, E);
        MultiFab effective(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        MultiFab::Copy(effective,rho,0,0,1,rho.nGrowVect());
        MultiFab::Add(effective,*ped,0,0,1,rho.nGrowVect());
        ElectronInertiaElliptic elliptic;
        elliptic.m_rtol = hp.m_electron_inertia_rtol;
        elliptic.m_max_iter = hp.m_electron_inertia_max_iters;
        elliptic.Define(E, 0);
        elliptic.PrepareCoefficients(effective, floor, 0);
        elliptic.Solve(E, 0);
        w.ApplyEfieldBoundary(0, PatchType::fine, 0.0_rt);
        copy(recovered.field, E);
        for (int c = 0; c < 3; ++c) {
            MultiFab::Subtract(first.values[c], *E[c], 0, 0, 1, 0);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(norm(first.field) > 1.e-6_rt*norm(recovered.field),
            "Fixture must distinguish convection before vs after elliptic recovery");
        bool sabotage = false;
        amrex::ParmParse("convection_test").query("disable_in_wrapper", sabotage);
        hp.m_electron_inertia_convection = !sabotage;
        hp.HybridPICSolveE(E, Ji, B, rho, w.GetEBUpdateEFlag()[0], 0, true);
        compare(E, recovered.field, "full_ohm_faraday_before_recovery");
        hp.HybridPICSolveE(E, Ji, B, rho, w.GetEBUpdateEFlag()[0], 0, false);
        compare(E, recovered.field, "full_ohm_particle_gather");
        hp.m_electron_inertia_convection = false;
        hp.HybridPICSolveE(E, Ji, B, rho, w.GetEBUpdateEFlag()[0], 0, true);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(norm(E) < 1.e-20_rt,
            "Legacy off switch must remove the entire convection RHS");
        amrex::Print() << "CONVECTION wrapper_default_on_and_legacy_off PASS\n";

        // Exercise the actual subcycled RK field advance. At B=0, the
        // Ampere current is zero and our prescribed Ji makes ue=3*u.
        // The initial Faraday rate follows independently from 9*analytic
        // convection followed by recovery. A short RK step must approach
        // this derivative, with a nonzero magnetic response.
        manufacture(0, 1.0_rt);
        hp.m_electron_inertia_convection = true;
        for (int c = 0; c < 3; ++c) {
            J[c]->setVal(0.0_rt);
            expected.values[c].mult(9.0_rt);
        }
        copy(E, expected.field);
        w.ApplyEfieldBoundary(0, PatchType::fine, 0.0_rt);
        elliptic.Solve(E, 0);
        w.ApplyEfieldBoundary(0, PatchType::fine, 0.0_rt);
        Scratch rate(B);
        w.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(
            rate.field, E, w.GetEBUpdateBFlag()[0], 0);
        constexpr amrex::Real dt_probe = 1.e-13_rt;
        for (int c = 0; c < 3; ++c) { rate.values[c].mult(-dt_probe); }
        AMREX_ALWAYS_ASSERT(norm(rate.field) > 1.e-12_rt);
        hp.BfieldEvolve(
            w.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0),
            w.m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, 0),
            w.m_fields.get_mr_levels_alldirs(FieldType::current_fp, 0),
            w.m_fields.get_mr_levels(FieldType::rho_fp, 0),
            w.GetEBUpdateEFlag(), 0, dt_probe, 0, SubcyclingHalf::None,
            B[0]->nGrowVect(), std::nullopt);
        compare(B, rate.field, "rk_field_advance_initial_derivative", 2.e-5_rt);
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
