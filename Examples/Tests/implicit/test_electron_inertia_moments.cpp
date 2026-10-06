/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronInertiaMoments.H"
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
        amrex::Print() << "MOMENTS " << label << " relative=" << relative
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
        AMREX_ALWAYS_ASSERT(hp.m_electron_inertia_moment_terms);
        hp.m_electron_inertia_convection = false;
        auto const& geom = w.Geom(0);
        auto const dx = geom.CellSizeArray(), lower = geom.ProbLoArray();
        auto& rho = *w.m_fields.get(FieldType::rho_fp, 0);
        auto* ped = const_cast<MultiFab*>(hp.DensityPedestal(0));
        AMREX_ALWAYS_ASSERT(ped != nullptr);
        auto const J = w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        auto const Ji = w.m_fields.get_alldirs(FieldType::current_fp, 0);
        auto const B = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto& solver = *w.get_pointer_fdtd_solver_fp(0);
        auto const floor = hp.m_n_floor * PhysConst::q_e;
        constexpr amrex::Real density = 1.e19_rt * PhysConst::q_e;
        constexpr amrex::Real A = 2.e5_rt, Omega = 3.e5_rt, C = -1.e5_rt, U = 2.e4_rt;
        constexpr amrex::Real ar = 2.e10_rt, at = -3.e10_rt, az = 4.e10_rt, au = 2.e9_rt;
        constexpr amrex::Real interval = 1.e-7_rt;
        Scratch older(Ji), newer(Ji), expected(E), recovered(E), first(E);
        ElectronInertiaMoments moments;
        amrex::Print() << std::setprecision(17);

        // Exact affine ion current with a linear time slope. Plasma current
        // differs from Ji; all three components and RZ div(Jr)=2*dJr/dr matter.
        auto fill_current = [&] (Field const& field, amrex::Real charge,
                                 amrex::Real factor, amrex::Real t) {
            for (int c = 0; c < 3; ++c) {
                auto const it = field[c]->ixType().toIntVect();
                for (amrex::MFIter mfi(*field[c]); mfi.isValid(); ++mfi) {
                    auto const out = field[c]->array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        amrex::Real const r = lower[0]+(i+(it[0]?0.0_rt:0.5_rt))*dx[0];
                        amrex::Real const z = lower[1]+(j+(it[1]?0.0_rt:0.5_rt))*dx[1];
                        amrex::Real const v = c==0 ? A*r : c==1 ? Omega*r : C*z+U;
                        amrex::Real const rate = c==0 ? ar*r : c==1 ? at*r : az*z+au;
                        out(i,j,k) = charge*(factor*v+t*rate);
                    });
                }
            }
        };
        auto manufacture = [&] (amrex::Real raw, amrex::Real pedestal,
                                 amrex::Real t, amrex::Real plasma_factor,
                                 bool with_slope) {
            rho.setVal(raw); ped->setVal(pedestal);
            amrex::Real const charge = std::max(floor, raw+pedestal);
            fill_current(Ji, charge, 3.0_rt, t);
            fill_current(J, charge, plasma_factor, 0.0_rt);
            amrex::Real const divergence_over_charge = raw+pedestal > floor
                ? 3.0_rt*(2.0_rt*A+C)+t*(2.0_rt*ar+az) : 0.0_rt;
            for (int c = 0; c < 3; ++c) {
                E[c]->setVal(0.0_rt); B[c]->setVal(0.0_rt);
                auto const it = E[c]->ixType().toIntVect();
                for (amrex::MFIter mfi(expected.values[c]); mfi.isValid(); ++mfi) {
                    auto const out = expected.values[c].array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        amrex::Real const r = lower[0]+(i+(it[0]?0.0_rt:0.5_rt))*dx[0];
                        amrex::Real const z = lower[1]+(j+(it[1]?0.0_rt:0.5_rt))*dx[1];
                        amrex::Real const v = c==0 ? A*r : c==1 ? Omega*r : C*z+U;
                        amrex::Real const rate = c==0 ? ar*r : c==1 ? at*r : az*z+au;
                        amrex::Real const ue = (3.0_rt-plasma_factor)*v+t*rate;
                        out(i,j,k) = -PhysConst::m_e/PhysConst::q_e *
                            ((with_slope ? rate : 0.0_rt)+ue*divergence_over_charge);
                    });
                }
            }
        };
        auto prepare = [&] (amrex::Real charge) {
            fill_current(older.field, charge, 3.0_rt, -0.5_rt*interval);
            fill_current(newer.field, charge, 3.0_rt, 0.5_rt*interval);
            moments.SetCurrentSlope(older.field, newer.field, interval);
        };
        prepare(1.2_rt*density);
        // Overwriting the source deposits must not change the cached slope.
        for (int c=0;c<3;++c) { older.values[c].setVal(1.e30_rt); newer.values[c].setVal(0); }
        for (amrex::Real t : {0.0_rt, 0.5_rt*interval, interval}) {
            manufacture(density, 0.2_rt*density, t, 2.0_rt, true);
            moments.AddToRHS(E,J,Ji,rho,ped,floor,geom,solver);
            compare(E,expected.field,"affine_first_half_second_half_extrapolation");
        }
        manufacture(density, 0.2_rt*density, 0.0_rt, 2.5_rt, true);
        moments.AddToRHS(E,J,Ji,rho,ped,floor,geom,solver);
        compare(E,expected.field,"refresh_plasma_current_keep_ion_slope");
        manufacture(density, 0.2_rt*density, 0.0_rt, 2.0_rt, false);
        moments.SetCurrentSlope(Ji,Ji,interval);
        moments.AddToRHS(E,J,Ji,rho,ped,floor,geom,solver);
        compare(E,expected.field,"compression_without_current_slope");
        prepare(floor);
        manufacture(0.1_rt*floor, 0.2_rt*floor, 0.0_rt, 2.0_rt, true);
        moments.AddToRHS(E,J,Ji,rho,ped,floor,geom,solver);
        compare(E,expected.field,"clipped_density_zero_compression");
        prepare(1.2_rt*density);
        manufacture(0.1_rt*density, 1.1_rt*density, 0.0_rt, 2.0_rt, true);
        moments.AddToRHS(E,J,Ji,rho,ped,floor,geom,solver);
        compare(E,expected.field,"static_pedestal_in_denominator");

        // Constant electron velocity with translating linear density:
        // rho=q*(1+beta*z), Ji=rho*U ez, dot(Ji)=-q*beta*U^2 ez.
        // Continuity compression and current slope must cancel exactly.
        constexpr amrex::Real beta=2.0_rt;
        ped->setVal(0.0_rt);
        for (amrex::MFIter mfi(rho);mfi.isValid();++mfi) {
            auto const den=rho.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),rho.nComp(),
                [=] AMREX_GPU_DEVICE(int i,int j,int k,int n) {
                    den(i,j,k,n)=density*(1.0_rt+beta*(lower[1]+j*dx[1]));
                });
        }
        for (int c=0;c<3;++c) {
            J[c]->setVal(0); E[c]->setVal(0);
            auto const it=Ji[c]->ixType().toIntVect();
            for (amrex::MFIter mfi(*Ji[c]);mfi.isValid();++mfi) {
                auto const ji=Ji[c]->array(mfi), old=older.values[c].array(mfi);
                auto const next=newer.values[c].array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    amrex::Real const z=lower[1]+(j+(it[1]?0.0_rt:0.5_rt))*dx[1];
                    amrex::Real const value=c==2 ? density*(1.0_rt+beta*z)*U : 0.0_rt;
                    amrex::Real const rate=c==2 ? -density*beta*U*U : 0.0_rt;
                    ji(i,j,k)=value;
                    old(i,j,k)=value-0.5_rt*interval*rate;
                    next(i,j,k)=value+0.5_rt*interval*rate;
                });
            }
        }
        moments.SetCurrentSlope(older.field,newer.field,interval);
        moments.AddToRHS(E,J,Ji,rho,ped,floor,geom,solver);
        amrex::Real const cancellation=norm(E)/(PhysConst::m_e/PhysConst::q_e*beta*U*U);
        amrex::Print()<<"MOMENTS translating_density_cancellation relative="<<cancellation<<'\n';
        AMREX_ALWAYS_ASSERT(cancellation<2.e-10_rt);

        for (int c=0;c<3;++c) { E[c]->setVal(0); }
        moments.AddMomentumFluxToRHS(E,J,Ji,rho,ped,floor,geom,solver);
        auto const flux_cancellation=norm(E)/(PhysConst::m_e/PhysConst::q_e*beta*U*U);
        amrex::Print()<<"MOMENTS flux_translating_density_cancellation relative="
                      <<flux_cancellation<<'\n';
        AMREX_ALWAYS_ASSERT(flux_cancellation<2.e-10_rt);

        // Independent clipped-floor limit: rho_eff is constant, so an affine
        // axial flow has material acceleration uz*d(uz)/dz, with no density
        // compression term. This detects dropping or halving the floor source.
        rho.setVal(0.1_rt*floor); ped->setVal(0.2_rt*floor);
        for (int c=0;c<3;++c) {
            E[c]->setVal(0); J[c]->setVal(0); expected.values[c].setVal(0);
            auto const it=Ji[c]->ixType().toIntVect();
            for (amrex::MFIter mfi(*Ji[c]);mfi.isValid();++mfi) {
                auto const ji=Ji[c]->array(mfi), exact=expected.values[c].array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const z=lower[1]+(j+(it[1]?0.0_rt:0.5_rt))*dx[1];
                    ji(i,j,k)=c==2 ? floor*(C*z+U) : 0.0_rt;
                    exact(i,j,k)=c==2 ? -PhysConst::m_e/PhysConst::q_e*C*(C*z+U) : 0.0_rt;
                });
            }
        }
        moments.SetCurrentSlope(Ji,Ji,interval);
        moments.AddMomentumFluxToRHS(E,J,Ji,rho,ped,floor,geom,solver);
        compare(E,expected.field,"flux_clipped_floor_material_acceleration");

        prepare(1.2_rt*density);
        manufacture(density,0.2_rt*density,0.0_rt,2.0_rt,true);
        std::array<amrex::iMultiFab,3> masks;
        std::array<amrex::iMultiFab const*,3> mask_ptr{};
        for (int c=0;c<3;++c) {
            masks[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,0);
            mask_ptr[c]=&masks[c];
            for (amrex::MFIter mfi(masks[c]);mfi.isValid();++mfi) {
                auto const mask=masks[c].array(mfi);
                auto const exact=expected.values[c].array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    mask(i,j,k)=(i%5!=0)?1:0;
                    if (mask(i,j,k)==0) { exact(i,j,k)=0.0_rt; }
                });
            }
        }
        moments.AddToRHS(E,J,Ji,rho,ped,floor,geom,solver,mask_ptr);
        compare(E,expected.field,"frozen_edge_mask");

        // Independent analytic RHS, physical boundaries and elliptic recovery.
        // Both Faraday and the final particle gather must use the same ordering.
        manufacture(density,0.2_rt*density,0.0_rt,2.0_rt,true);
        fill_current(older.field,1.2_rt*density,3.0_rt,-0.5_rt*interval);
        fill_current(newer.field,1.2_rt*density,3.0_rt,0.5_rt*interval);
        hp.PrepareElectronInertiaCurrentSlope(0,older.field,newer.field,interval);
        MultiFab effective(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        ElectronInertiaElliptic elliptic;
        elliptic.m_rtol=hp.m_electron_inertia_rtol;
        elliptic.m_max_iter=hp.m_electron_inertia_max_iters;
        elliptic.Define(E,0);
        auto recover = [&] (MultiFab const& density_field) {
            MultiFab::Copy(effective,density_field,0,0,1,effective.nGrowVect());
            MultiFab::Add(effective,*ped,0,0,1,effective.nGrowVect());
            elliptic.PrepareCoefficients(effective,floor,0);
            w.ApplyEfieldBoundary(0,PatchType::fine,0.0_rt);
            elliptic.Solve(E,0);
            w.ApplyEfieldBoundary(0,PatchType::fine,0.0_rt);
            w.FillBoundaryE(E[0]->nGrowVect(),std::nullopt);
        };
        copy(E,expected.field); copy(first.field,E);
        recover(rho); copy(recovered.field,E);
        for (int c=0;c<3;++c) { MultiFab::Subtract(first.values[c],*E[c],0,0,1,0); }
        AMREX_ALWAYS_ASSERT(norm(first.field)>1.e-6_rt*norm(recovered.field));
        bool sabotage=false;
        amrex::ParmParse("moment_test").query("disable_in_wrapper",sabotage);
        hp.m_electron_inertia_moment_terms=!sabotage;
        hp.HybridPICSolveE(E,Ji,B,rho,w.GetEBUpdateEFlag()[0],0,true);
        compare(E,recovered.field,"full_ohm_faraday_before_recovery");
        hp.HybridPICSolveE(E,Ji,B,rho,w.GetEBUpdateEFlag()[0],0,false);
        compare(E,recovered.field,"full_ohm_particle_gather");
        hp.m_electron_inertia_moment_terms=false;
        hp.HybridPICSolveE(E,Ji,B,rho,w.GetEBUpdateEFlag()[0],0,true);
        AMREX_ALWAYS_ASSERT(norm(E)<1.e-20_rt);
        hp.m_electron_inertia_moment_terms=true;

        // Exercise the production deposition/averaging/subcycling/gather path.
        // Frozen ions deposit Ji_new=0. Prescribe Ji_old=-dt*q*at*r e_theta:
        // div(Ji)=0 in both halves and the extrapolated gather. With all other
        // Ohm terms off, E in each half is independent of the changing B.
        // Independently recovering the two known RHS values predicts the
        // entire magnetic update, for any number of RK substeps.
        hp.m_solve_electron_energy_equation=false;
        w.HybridPICDepositRhoAndJ();
        auto& oldrho=*w.m_fields.get(FieldType::hybrid_rho_fp_temp,0);
        auto const oldji=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_temp,0);
        oldrho.setVal(density); ped->setVal(0.2_rt*density);
        MultiFab half(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        MultiFab::LinComb(half,0.5_rt,oldrho,0,0.5_rt,rho,0,0,1,half.nGrowVect());
        auto rhs_for = [&] (MultiFab const& den) {
            for (int c=0;c<3;++c) { E[c]->setVal(0); }
            for (amrex::MFIter mfi(*E[1]);mfi.isValid();++mfi) {
                auto const out=E[1]->array(mfi);
                auto const charge=den.const_array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    amrex::Real const r=lower[0]+i*dx[0];
                    out(i,j,k)=-PhysConst::m_e/PhysConst::q_e*density*at*r /
                        amrex::max(floor,charge(i,j,k)+0.2_rt*density);
                });
            }
        };
        Scratch predicted_B(B), curl(B);
        amrex::Real const dt=w.getdt(0);
        for (MultiFab const* den : {&oldrho,&half}) {
            rhs_for(*den); recover(*den);
            solver.ComputeCurlA(curl.field,E,w.GetEBUpdateBFlag()[0],0);
            for (int c=0;c<3;++c) {
                MultiFab::Saxpy(predicted_B.values[c],-0.5_rt*dt,curl.values[c],0,0,1,0);
            }
        }
        rhs_for(rho); recover(rho); copy(recovered.field,E);
        for (int c=0;c<3;++c) {
            B[c]->setVal(0); oldji[c]->setVal(0);
            if (c==1) {
                for (amrex::MFIter mfi(*oldji[c]);mfi.isValid();++mfi) {
                    auto const old=oldji[c]->array(mfi);
                    amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        old(i,j,k)=-dt*density*at*(lower[0]+i*dx[0]);
                    });
                }
            }
        }
        w.HybridPICEvolveFields();
        compare(E,recovered.field,"production_deposit_slope_and_final_gather",2.e-8_rt);
        compare(B,predicted_B.field,"production_both_halves_and_all_rk_substeps",2.e-8_rt);

        // Nearly longitudinal fields must satisfy the ORIGINAL elliptic
        // equation at the requested relative tolerance, without chasing the
        // tiny transverse correction below floating-point resolution.
        effective.setVal(floor);
        elliptic.PrepareCoefficients(effective,floor,0);
        auto const rwall=geom.ProbHi(0);
        for (int phase=0;phase<8;++phase) {
            for (int c=0;c<3;++c) {
                auto const it=expected.values[c].ixType().toIntVect();
                for (amrex::MFIter mfi(expected.values[c]);mfi.isValid();++mfi) {
                    auto const out=expected.values[c].array(mfi);
                    amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        auto const r=lower[0]+(i+(it[0]?0.0_rt:0.5_rt))*dx[0];
                        out(i,j,k)=c==0 ? 0.01_rt*r/rwall : c==1
                            ? (phase+1)*1.e-12_rt*(i%2 ? 1.0_rt : -1.0_rt)*r/rwall : 0.0_rt;
                    });
                }
            }
            elliptic.ApplyOperator(expected.values,first.values,0);
            copy(E,first.field);
            elliptic.Solve(E,0);
            copy(recovered.field,E);
            elliptic.ApplyOperator(recovered.values,newer.values,0);
            for (int c=0;c<3;++c) {
                MultiFab::Subtract(newer.values[c],first.values[c],0,0,1,0);
            }
            auto const residual=norm(newer.field)/norm(first.field);
            amrex::Print()<<"MOMENTS original_equation_residual phase="<<phase
                          <<" relative="<<residual<<'\n';
            AMREX_ALWAYS_ASSERT(residual<5.0_rt*elliptic.m_rtol);
        }
        amrex::Print()<<"MOMENTS production_continuity_completion PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
