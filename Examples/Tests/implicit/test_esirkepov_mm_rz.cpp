/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/ImplicitSolvers/MassMatrixDensityProjection.H"
#include "FieldSolver/ImplicitSolvers/ThetaImplicitHybrid.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>

class EsirkepovProbe : public ThetaImplicitHybrid
{
public:
    void Configure (WarpX& sim)
    {
        m_WarpX = &sim;
        m_num_amr_levels = 1;
        m_dt = sim.getdt(0);
        m_nlsolver_type = NonlinearSolverType::newton;
        m_max_particle_iterations = 100;
        m_particle_tolerance = 1.e-12;
        m_use_mass_matrices = true;
        m_use_mass_matrices_jacobian = true;
        m_esirkepov_mass_matrices = true;
        m_mass_matrices_pc_width = 0;
    }
    void Evaluate (bool linear) { PreRHSOp(0.5*m_dt, 1, linear); }
    void Diagonal () { SyncMassMatricesPCAndApplyBCs(); }
    void CheckCacheLifecycle () {
        using warpx::fields::FieldType;
        auto matrix = [&] () { return m_WarpX->m_fields.get(
            FieldType::MassMatrices_X,ablastr::fields::Direction{0},0); };
        m_mass_matrices_deposit_interval = 3;
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) > 0.);
        matrix()->setVal(0.);
        PreLinearSolve(1);
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) == 0.);
        InvalidateMassMatrices();
        PreLinearSolve(2);
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) > 0.);
        matrix()->setVal(0.);
        PreLinearSolve(0); // every new nonlinear solve starts with a fresh tangent
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) > 0.);
        matrix()->setVal(0.);
        auto const dt = m_dt;
        m_dt *= 1.1;
        PreLinearSolve(1);
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) > 0.);
        m_dt = dt;
        m_mass_matrices_reuse_within_step = true;
        PreLinearSolve(-1);
        matrix()->setVal(0.);
        PreLinearSolve(0);
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) == 0.);
        PreLinearSolve(0);
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) == 0.);
        PreLinearSolve(0); // cadence counts across nested solve entries
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) > 0.);
        auto const time = m_WarpX->gett_new(0);
        m_WarpX->sett_new(0,time+dt);
        matrix()->setVal(0.);
        PreLinearSolve(0);
        AMREX_ALWAYS_ASSERT(matrix()->norminf(matrix()->nComp()/2) > 0.);
        m_WarpX->sett_new(0,time);
        m_mass_matrices_reuse_within_step = false;
        m_mass_matrices_deposit_interval = 1;
        PreLinearSolve(0);
        amrex::Print() << "MM_CACHE PASS reuse invalidation solve_entry dt_change step_change bounded_cross_solve\n";
    }
    void CheckPCSurvivesResidual () {
        m_use_mass_matrices_pc = true;
        m_mass_matrices_deposit_interval = 3;
        Evaluate(false);
        m_use_mass_matrices_pc = false;
        m_mass_matrices_deposit_interval = 1;
    }
    int Width () const { return m_ncomp_xx[0]; }
};

amrex::MultiFab Clone (amrex::MultiFab const& field, int component = 0)
{
    amrex::MultiFab copy(field.boxArray(),field.DistributionMap(),1,field.nGrowVect());
    amrex::MultiFab::Copy(copy,field,component,0,1,field.nGrowVect());
    return copy;
}

amrex::Real Error (amrex::MultiFab const& a, amrex::MultiFab const& b)
{
    auto diff = Clone(a);
    amrex::MultiFab::Subtract(diff,b,0,0,1,0);
    return diff.norminf();
}

void Density (WarpX& sim, amrex::MultiFab& rho)
{
    ablastr::fields::MultiLevelScalarField refs{&rho};
    sim.GetPartContainer().DepositCharge(refs,0.0);
    sim.SyncRho(refs,{},{});
    sim.ApplyRhofieldBoundary(0,&rho,PatchType::fine);
    rho.FillBoundary(sim.Geom(0).periodicity());
}

void Electric (WarpX& sim, int direction, amrex::Real amplitude)
{
    using warpx::fields::FieldType;
    auto const spacing = sim.Geom(0).CellSizeArray();
    auto const lower = sim.Geom(0).ProbLoArray();
    auto const pi = std::acos(-1.0);
    auto const kr = 0.5*pi/sim.Geom(0).ProbLength(0);
    auto const kz = 2*pi/sim.Geom(0).ProbLength(1);
    for (int d = 0; d < 3; ++d) {
        auto& field = *sim.m_fields.get(FieldType::Efield_fp,ablastr::fields::Direction{d},0);
        auto const nodal = field.ixType().toIntVect();
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            auto const out = field.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                auto const r = (i+0.5*(1-nodal[0]))*spacing[0];
                auto const z = lower[1]+(j+0.5*(1-nodal[1]))*spacing[1];
                out(i,j,k) = d == direction
                    ? amplitude*(d == 2 ? std::cos(kr*r) : std::sin(kr*r))
                        *(1.0+0.2*std::sin(kz*z)) : 0.0;
            });
        }
    }
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        using warpx::fields::FieldType;
        AMREX_ALWAYS_ASSERT(WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov);
        AMREX_ALWAYS_ASSERT(WarpX::field_gathering_algo == GatheringAlgo::MomentumConserving);
        auto& rho = *sim.m_fields.get(FieldType::rho_fp,0);
        auto current = sim.m_fields.get_alldirs(FieldType::current_fp,0);
        int const mid = rho.nComp()/2;
        bool oblique = true;
        amrex::ParmParse("qualification").query("oblique_b",oblique);
        std::array<amrex::Real,3> const magnetic{oblique ? 0.4 : 0.0,
                                               oblique ? 0.6 : 0.0,2.0};
        for (int d = 0; d < 3; ++d) {
            sim.m_fields.get(FieldType::Bfield_fp,ablastr::fields::Direction{d},0)->setVal(
                magnetic[d]*1.e-9/sim.getdt(0));
        }
        auto old = Clone(rho);
        Density(sim,old);
        sim.SaveParticlesAtImplicitStepStart();
        MassMatrixDensityProjection predictor;
        predictor.CaptureStepStart(old,0);
        EsirkepovProbe probe;
        probe.Configure(sim);
        amrex::Real perturbation = 1.e-5;
        amrex::ParmParse("qualification").query("perturbation",perturbation);
        auto const amplitude = perturbation/sim.getdt(0);
        amrex::Real worst_current = 0.0, worst_density = 0.0, worst_continuity = 0.0;
        for (int direction = 0; direction < 3; ++direction) {
            Electric(sim,direction,0.0);
            probe.Evaluate(false);
            predictor.DepositEndpointAverage(sim,0,rho,mid);
            predictor.Capture(rho,mid,current);
            auto base_rho = Clone(rho,mid);
            std::array<amrex::MultiFab,3> base_j{
                Clone(*current[0]),Clone(*current[1]),Clone(*current[2])};
            probe.PreLinearSolve();
            int repetitions = 0;
            amrex::ParmParse("qualification").query("deposit_repetitions", repetitions);
            if (direction == 0 && repetitions > 0) {
                // Benchmark the same live nonlinear particle state. Each call
                // independently clears and rebuilds all nine response blocks.
                amrex::Gpu::synchronize();
                auto const start = amrex::second();
                for (int repeat = 0; repeat < repetitions; ++repeat) {
                    sim.DepositMassMatrices();
                }
                amrex::Gpu::synchronize();
                amrex::Print() << "ESIRKEPOV_MM deposit_seconds="
                    << (amrex::second()-start)/repetitions
                    << " repetitions=" << repetitions << "\n";
                // Finish boundary synchronization before checking the response.
                probe.PreLinearSolve();
            }
            bool check_pc = false;
            amrex::ParmParse("qualification").query("mm_pc",check_pc);
            if (check_pc && direction == 0) {
                probe.CheckCacheLifecycle();
                probe.Diagonal();
                auto const diag = sim.m_fields.get_alldirs(FieldType::MassMatrices_PC,0);
                auto keep = Clone(*diag[0]);
                probe.CheckPCSurvivesResidual();
                AMREX_ALWAYS_ASSERT(Error(keep,*diag[0]) == 0.);
                auto const electric = sim.m_fields.get_alldirs(FieldType::Efield_fp,0);
                auto const cells = sim.Geom(0).Domain().length();
                amrex::Real worst = 0.;
                for (int c = 0; c < 3; ++c) {
                    // Axis, interior, box seam, each wall and corners.
                    for (auto point : {amrex::IntVect(0,0),amrex::IntVect(0,5),
                            amrex::IntVect(1,1),amrex::IntVect(7,15),
                            amrex::IntVect(cells[0]-2,1),
                            amrex::IntVect(cells[0]-1,cells[1]-1),
                            amrex::IntVect(cells[0]-(c == 0),cells[1]-(c == 2))}) {
                        if (c == 1 && point[0] == 0) { continue; }
                        for (int d = 0; d < 3; ++d) {
                            electric[d]->setVal(0.);
                            if (c != d) { continue; }
                            int const ii = point[0], jj = point[1];
                            bool const periodic_z = sim.Geom(0).isPeriodic(1);
                            int const nz = cells[1];
                            for (amrex::MFIter mfi(*electric[d]); mfi.isValid(); ++mfi) {
                                auto e = electric[d]->array(mfi);
                                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                                    e(i,j,k) = i == ii && (periodic_z ? j%nz == jj%nz : j == jj) ? 1. : 0.;
                                });
                            }
                        }
                        // Native field boundary and MC auxiliary interpolation; independent
                        // of the reduced-matrix index/parity implementation.
                        sim.FillBoundaryE(0,sim.get_ng_fieldgather());
                        sim.ApplyEfieldBoundary(0,PatchType::fine,0.);
                        sim.ApplyFieldBoundaryOnAxis(electric[0],electric[1],electric[2],0);
                        probe.Evaluate(true);
                        amrex::Real error = 0., scale = 0.;
                        for (amrex::MFIter mfi(*current[c]); mfi.isValid(); ++mfi) {
                            if (!mfi.validbox().contains(point)) { continue; }
                            amrex::FArrayBox h(mfi.validbox(),3,amrex::The_Pinned_Arena());
                            h.copy<amrex::RunOn::Device>((*current[c])[mfi],mfi.validbox(),0,
                                                       mfi.validbox(),0,1);
                            h.copy<amrex::RunOn::Device>(base_j[c][mfi],mfi.validbox(),0,
                                                       mfi.validbox(),1,1);
                            h.copy<amrex::RunOn::Device>((*diag[c])[mfi],mfi.validbox(),0,
                                                       mfi.validbox(),2,1);
                            amrex::Gpu::synchronize();
                            auto const expected = h(point,0)-h(point,1);
                            scale = std::max(scale,std::abs(expected));
                            error = std::max(error,std::abs(expected-h(point,2)));
                        }
                        amrex::ParallelDescriptor::ReduceRealMax(error);
                        amrex::ParallelDescriptor::ReduceRealMax(scale);
                        worst = std::max(worst,error/std::max(scale,1.e-20));
                        amrex::Print() << "MM_PC diagonal c=" << c << " p=" << point
                            << " error=" << error/std::max(scale,1.e-20) << "\n";
                    }
                }
                AMREX_ALWAYS_ASSERT(worst < 1.e-8);
                amrex::Print() << "MM_PC PASS native_effective_diagonal " << worst << "\n";
                Electric(sim,0,0.);
            }
            probe.Evaluate(true);
            predictor.Apply(sim,0,rho,mid,current,0.5*sim.getdt(0));
            AMREX_ALWAYS_ASSERT(Error(Clone(rho,mid),base_rho) < 1.e-13*old.norminf());
            for (int d = 0; d < 3; ++d) {
                AMREX_ALWAYS_ASSERT(Error(*current[d],base_j[d]) <
                    1.e-13*std::max(base_j[d].norminf(),1.0));
            }
            Electric(sim,direction,amplitude);
            probe.Evaluate(true);
            predictor.Apply(sim,0,rho,mid,current,0.5*sim.getdt(0));
            std::array<amrex::MultiFab,3> mm_derivative{
                Clone(*current[0]),Clone(*current[1]),Clone(*current[2])};
            auto mm_rho = Clone(rho,mid);
            Electric(sim,direction,-amplitude);
            probe.Evaluate(true);
            predictor.Apply(sim,0,rho,mid,current,0.5*sim.getdt(0));
            for (int d = 0; d < 3; ++d) {
                amrex::MultiFab::Subtract(mm_derivative[d],*current[d],0,0,1,0);
            }
            amrex::MultiFab::Subtract(mm_rho,rho,mid,0,1,0);
            // Independent complete implicit push/deposit at +/- the same
            // electric perturbation. This covers all nine current/E blocks.
            Electric(sim,direction,amplitude);
            probe.Evaluate(false);
            predictor.DepositEndpointAverage(sim,0,rho,mid);
            std::array<amrex::MultiFab,3> exact_derivative{
                Clone(*current[0]),Clone(*current[1]),Clone(*current[2])};
            auto exact_rho = Clone(rho,mid);
            auto check = Clone(old);
            MassMatrixDensityProjection::ComputeDivergence(sim,0,current,check);
            amrex::MultiFab::LinComb(check,1.0,old,0,-0.5*sim.getdt(0),check,0,0,1,0);
            worst_continuity = std::max(worst_continuity,Error(check,exact_rho)/old.norminf());
            Electric(sim,direction,-amplitude);
            probe.Evaluate(false);
            predictor.DepositEndpointAverage(sim,0,rho,mid);
            for (int d = 0; d < 3; ++d) {
                amrex::MultiFab::Subtract(exact_derivative[d],*current[d],0,0,1,0);
            }
            amrex::MultiFab::Subtract(exact_rho,rho,mid,0,1,0);
            amrex::Real signal = 0.0, error = 0.0;
            for (int d = 0; d < 3; ++d) {
                signal = std::max(signal,exact_derivative[d].norminf());
                error = std::max(error,Error(mm_derivative[d],exact_derivative[d]));
            }
            worst_current = std::max(worst_current,error/signal);
            auto const density_signal = exact_rho.norminf();
            auto const density_error = Error(mm_rho,exact_rho);
            worst_density = std::max(worst_density,density_error/
                std::max(density_signal,1.e-10*old.norminf()));
            amrex::Print() << "ESIRKEPOV_MM direction=" << direction
                << " width=" << probe.Width() << " current_relative_error=" << error/signal
                << " density_relative_error=" << density_error/
                    std::max(density_signal,1.e-10*old.norminf()) << "\n";
        }
        amrex::Print() << "ESIRKEPOV_MM worst_current=" << worst_current
            << " worst_density=" << worst_density
            << " midpoint_continuity=" << worst_continuity << "\n";
        AMREX_ALWAYS_ASSERT(worst_current < 2.e-4);
        AMREX_ALWAYS_ASSERT(worst_density < 2.e-3);
        AMREX_ALWAYS_ASSERT(worst_continuity < 2.e-12);
        amrex::Print() << "ESIRKEPOV_MM PASS native_9blocks density continuity reference_identity\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
