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
#include <cmath>

class DensityDepositProbe : public ThetaImplicitHybrid
{
public:
    void Run (WarpX& sim)
    {
        m_WarpX = &sim;
        m_dt = sim.getdt(0);
        m_nlsolver_type = NonlinearSolverType::newton;
        m_max_particle_iterations = 100;
        m_particle_tolerance = 1.e-12;
        PreRHSOp(0.5*m_dt, 0, false);
    }
};

void DepositDensity (WarpX& sim, amrex::MultiFab& rho)
{
    ablastr::fields::MultiLevelScalarField refs{&rho};
    sim.GetPartContainer().DepositCharge(refs, 0.0);
    sim.SyncRho(refs, {}, {});
    sim.ApplyRhofieldBoundary(0, &rho, PatchType::fine);
}

amrex::Real Difference (amrex::MultiFab const& a, int ca,
                        amrex::MultiFab const& b, int cb, bool ghosts = false)
{
    amrex::MultiFab diff(a.boxArray(), a.DistributionMap(), 1, a.nGrowVect());
    amrex::MultiFab::LinComb(diff, 1.0, a, ca, -1.0, b, cb, 0, 1, a.nGrowVect());
    return diff.norminf(0, ghosts ? diff.nGrow() : 0);
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        using warpx::fields::FieldType;
        for (auto type : {FieldType::Efield_fp, FieldType::Bfield_fp}) {
            for (auto* field : sim.m_fields.get_alldirs(type, 0)) { field->setVal(0.0); }
        }
        bool const esirkepov = WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov;
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto current = sim.m_fields.get_alldirs(FieldType::current_fp, 0);
        int const mid = rho.nComp()/2;
        amrex::MultiFab old(rho.boxArray(), rho.DistributionMap(), 1, rho.nGrowVect());
        amrex::MultiFab actual(rho.boxArray(), rho.DistributionMap(), 1, rho.nGrowVect());
        amrex::MultiFab saved(rho.boxArray(), rho.DistributionMap(), 1, rho.nGrowVect());
        DepositDensity(sim, old);
        sim.SaveParticlesAtImplicitStepStart();
        DensityDepositProbe probe;
        probe.Run(sim);
        MassMatrixDensityProjection predictor;
        predictor.Capture(rho, mid, current);
        auto& ez = *sim.m_fields.get(FieldType::Efield_fp, ablastr::fields::Direction{2}, 0);
        auto const dz = sim.Geom(0).CellSize(1);
        auto const length = sim.Geom(0).ProbLength(1);
        amrex::Real const wave = 2.0*std::acos(-1.0)/length;
        amrex::Real const offset = ez.ixType().nodeCentered(1) ? 0.0 : 0.5;
        for (amrex::MFIter mfi(ez); mfi.isValid(); ++mfi) {
            auto const e = ez.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                e(i,j,k) = 1.e7*std::sin(wave*(j+offset)*dz);
            });
        }
        probe.Run(sim);
        amrex::MultiFab::Copy(actual, rho, mid, 0, 1, rho.nGrowVect());
        auto const signal = Difference(actual, 0, old, 0);
        AMREX_ALWAYS_ASSERT(signal > 1.e-8*old.norminf());
        predictor.Apply(sim, 0, rho, mid, current, 0.5*sim.getdt(0));
        auto const error = Difference(rho, mid, actual, 0)/signal;
        amrex::Print() << "MM_DENSITY particle_response esirkepov=" << esirkepov
                       << " nz=" << sim.Geom(0).Domain().length(1)
                       << " relative_error=" << error << " signal=" << signal << "\n";
        // Direct deposition: independently pushed particles must agree with
        // the continuity approximation to second-order spatial accuracy.
        if (!esirkepov) { AMREX_ALWAYS_ASSERT(error < 2.0*wave*wave*dz*dz); }
        amrex::MultiFab::Copy(saved, rho, mid, 0, 1, rho.nGrowVect());
        predictor.Apply(sim, 0, rho, mid, current, 0.5*sim.getdt(0));
        AMREX_ALWAYS_ASSERT(Difference(rho, mid, saved, 0, true) == 0.0);

        // Esirkepov conserves endpoint charge; do not confuse a deposit at
        // midpoint particle positions with the average of endpoint deposits.
        amrex::MultiFab div(rho.boxArray(), rho.DistributionMap(), 1, 0);
        MassMatrixDensityProjection::ComputeDivergence(sim,0,current,div);
        sim.FinishImplicitParticleUpdate(sim.getdt(0));
        DepositDensity(sim, actual);
        amrex::MultiFab::LinComb(saved, 1.0, old, 0, -sim.getdt(0), div, 0, 0, 1, 0);
        auto const continuity = Difference(actual, 0, saved, 0)/old.norminf();
        amrex::Print() << "MM_DENSITY endpoint_continuity esirkepov=" << esirkepov
                       << " defect_over_density=" << continuity << "\n";
        if (esirkepov) { AMREX_ALWAYS_ASSERT(continuity < 2.e-12); }
        else { AMREX_ALWAYS_ASSERT(continuity > 1.e-10); }

        // Off-axis: div(a*r,0,b*z)=2a+b. At the axis the charge deposit
        // uses its configured lumped volume, giving3a/2+b with Verboncoeur.
        // Include physical corner ghosts and multiple boxes.
        rho.setVal(2.0);
        for (auto* field : current) { field->setVal(0.0); }
        predictor.Capture(rho, mid, current);
        auto const dr = sim.Geom(0).CellSize(0);
        for (int d : {0,2}) {
            auto& field = *current[d];
            amrex::Real const stagger = field.ixType().nodeCentered(d == 0 ? 0 : 1)
                ? 0.0 : 0.5;
            for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
                auto const a = field.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    a(i,j,k) = d == 0 ? 0.2*(i+stagger)*dr : 0.3*(j+stagger)*dz;
                });
            }
        }
        predictor.Apply(sim, 0, rho, mid, current, 1.0);
        saved.setVal(1.3);
        if (sim.UseVerboncoeurAxisCorrection()) {
            for (amrex::MFIter mfi(saved); mfi.isValid(); ++mfi) {
                auto const out = saved.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    if (i == 0) { out(i,j,k) = 1.4; }
                });
            }
        }
        AMREX_ALWAYS_ASSERT(Difference(rho, mid, saved, 0, true) < 1.e-13);
        // Repeat and zero response must start from the held base, never
        // from the last projected density or a second boundary summation.
        predictor.Apply(sim, 0, rho, mid, current, 1.0);
        AMREX_ALWAYS_ASSERT(Difference(rho, mid, saved, 0, true) < 1.e-13);
        for (auto* field : current) { field->setVal(0.0); }
        predictor.Apply(sim, 0, rho, mid, current, 1.0);
        saved.setVal(2.0);
        AMREX_ALWAYS_ASSERT(Difference(rho, mid, saved, 0, true) == 0.0);
        // A rebase must replace all held data, including zero/subfloor cells.
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
            auto const r = rho.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                r(i,j,k,mid) = i % 3 == 0 ? 0.0 : (i % 3 == 1 ? 0.05 : 2.0);
            });
        }
        amrex::MultiFab::Copy(saved, rho, mid, 0, 1, rho.nGrowVect());
        predictor.Capture(rho, mid, current);
        predictor.Apply(sim, 0, rho, mid, current, 1.0);
        AMREX_ALWAYS_ASSERT(Difference(rho, mid, saved, 0, true) == 0.0);
        // A below-constitutive-floor charge still has a two-sided continuity
        // response. Freezing its negative perturbations would make the MM
        // Jacobian nonlinear even for arbitrarily small probes.
        rho.setVal(0.05);
        for (auto* field : current) { field->setVal(0.0); }
        predictor.Capture(rho,mid,current);
        for (amrex::MFIter mfi(*current[2]); mfi.isValid(); ++mfi) {
            auto const out = current[2]->array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                out(i,j,k) = 0.01*(j+0.5)*dz;
            });
        }
        predictor.Apply(sim,0,rho,mid,current,1.0);
        saved.setVal(0.04);
        AMREX_ALWAYS_ASSERT(Difference(rho,mid,saved,0,true) < 1.e-14);
        amrex::Print() << "MM_DENSITY PASS analytic_axis_ghosts replay zero rebase vacuum subfloor\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
