/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Independent material-acceleration reference for a smooth compressing edge.
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronInertiaMoments.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>

using namespace amrex::literals;
using warpx::fields::FieldType;
using Field = ablastr::fields::VectorField;

struct Scratch
{
    std::array<amrex::MultiFab, 3> values;
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

struct Profile
{
    amrex::Real core, background, pedestal, radius, width, strain;

    AMREX_GPU_HOST_DEVICE
    void density (amrex::Real r, amrex::Real& rho, amrex::Real& derivative) const
    {
        auto const t = std::tanh((r-radius)/width);
        rho = background + (core-background)*0.5_rt*(1.0_rt-t);
        derivative = -(core-background)*0.5_rt*(1.0_rt-t*t)/width;
    }

    AMREX_GPU_HOST_DEVICE
    amrex::Real exact_e (amrex::Real r) const
    {
        amrex::Real rho, drho;
        density(r,rho,drho);
        amrex::Real const q = rho + pedestal, f = rho/q;
        // rho(r,t)=exp(2*a*t)*rho0(r*exp(a*t)), ui=-a*r, Jplasma=0.
        // ue=ui*rho/(rho+ped); direct material differentiation, with fixed ped:
        // D_e ue = a^2*r*(3*f^2-2*f) - a^2*r^2*ped^2*drho/(rho+ped)^3.
        amrex::Real const acceleration = strain*strain *
            (r*(3.0_rt*f*f-2.0_rt*f)-r*r*pedestal*pedestal*drho/(q*q*q));
        return -PhysConst::m_e/PhysConst::q_e * acceleration;
    }
};

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        auto const& geom = w.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const plo = geom.ProbLoArray();
        auto& rho = *w.m_fields.get(FieldType::rho_fp,0);
        auto* ped = const_cast<amrex::MultiFab*>(hp.DensityPedestal(0));
        AMREX_ALWAYS_ASSERT(ped != nullptr && geom.isPeriodic(1));
        auto const Ji = w.m_fields.get_alldirs(FieldType::current_fp,0);
        auto const J = w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp,0);
        auto& solver = *w.get_pointer_fdtd_solver_fp(0);
        Scratch older(Ji), newer(Ji);
        ElectronInertiaMoments moments;
        amrex::MultiFab measures(E[0]->boxArray(),E[0]->DistributionMap(),6,0);
        amrex::MultiFab continuity(rho.boxArray(),rho.DistributionMap(),1,0);
        amrex::Real fraction = 0.05_rt;
        bool density_average = false;
        amrex::ParmParse pp("edge_probe");
        pp.query("pedestal_fraction",fraction);
        pp.query("density_average",density_average);
        Profile base{1.e19_rt*PhysConst::q_e, 1.e15_rt*PhysConst::q_e,
                     fraction*1.e19_rt*PhysConst::q_e, 0.15_rt, 0.0_rt, 1.e5_rt};
        AMREX_ALWAYS_ASSERT(base.pedestal >= hp.m_n_floor*PhysConst::q_e);
        constexpr amrex::Real interval = 1.e-8_rt;
        amrex::Print() << std::setprecision(17);
        // No finite-time approximation here: the two synthetic deposits encode
        // the exact instantaneous slope. The profile is not time integrated.
        amrex::Real const cells=0.01875_rt/dx[0];
        {
            for (amrex::Real phase : {0.0_rt,0.25_rt,0.5_rt,0.75_rt}) {
                auto profile = base;
                profile.radius += phase*dx[0];
                profile.width = cells*dx[0];
                for (int c = 0; c < 3; ++c) { J[c]->setVal(0); E[c]->setVal(0); }
                ped->setVal(profile.pedestal);
                for (amrex::MFIter mfi(rho);mfi.isValid();++mfi) {
                    auto const den = rho.array(mfi);
                    amrex::ParallelFor(mfi.fabbox(),rho.nComp(),
                        [=] AMREX_GPU_DEVICE(int i,int j,int k,int n) {
                            amrex::Real val, drho;
                            auto const r = std::abs(plo[0]+i*dx[0]);
                            profile.density(r,val,drho);
                            if (density_average) {
                                auto const left = amrex::max(0.0_rt,r-0.5_rt*dx[0]);
                                auto const right = r+0.5_rt*dx[0];
                                val = 0.0_rt;
                                constexpr int quadrature = 1024;
                                for (int q = 0; q < quadrature; ++q) {
                                    auto const rq = left+(q+0.5_rt)*(right-left)/quadrature;
                                    amrex::Real nq, dq;
                                    profile.density(rq,nq,dq);
                                    val += nq*rq;
                                }
                                val *= 2.0_rt*(right-left)/
                                    (quadrature*(right*right-left*left));
                            }
                            den(i,j,k,n) = val;
                        });
                }
                for (int c = 0; c < 3; ++c) {
                    auto const it = Ji[c]->ixType().toIntVect();
                    for (amrex::MFIter mfi(*Ji[c]);mfi.isValid();++mfi) {
                        auto const ji = Ji[c]->array(mfi);
                        auto const old = older.values[c].array(mfi);
                        auto const next = newer.values[c].array(mfi);
                        amrex::ParallelFor(mfi.fabbox(),
                            [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                                auto const r = plo[0]+(i+(it[0]?0.0_rt:0.5_rt))*dx[0];
                                amrex::Real val, drho;
                                profile.density(r,val,drho);
                                auto const u = c == 0 ? -profile.strain*r : 0.0_rt;
                                auto const value = val*u;
                                auto const rate = u*profile.strain*(2.0_rt*val+r*drho);
                                ji(i,j,k) = value;
                                old(i,j,k) = value-0.5_rt*interval*rate;
                                next(i,j,k) = value+0.5_rt*interval*rate;
                            });
                    }
                }
                // Finite-volume continuity check, using exact radial surface fluxes.
                // This is the correct derivative of the volume-averaged density.
                solver.ComputeDivE(Ji,continuity);
                for (amrex::MFIter mfi(continuity);mfi.isValid();++mfi) {
                    auto const residual = continuity.array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                            auto const r = plo[0]+i*dx[0];
                            auto const left = amrex::max(0.0_rt,r-0.5_rt*dx[0]);
                            auto const right = r+0.5_rt*dx[0];
                            amrex::Real nl,dl,nr,dr;
                            profile.density(left,nl,dl);
                            profile.density(right,nr,dr);
                            auto const rate = 2.0_rt*profile.strain*
                                (right*right*nr-left*left*nl)/(right*right-left*left);
                            residual(i,j,k) = std::abs(residual(i,j,k)+rate)/
                                (profile.core*profile.strain);
                        });
                }
                auto const continuity_error = continuity.norminf();
                AMREX_ALWAYS_ASSERT(continuity_error < 1.e-10_rt);
                moments.SetCurrentSlope(older.field,newer.field,interval);
                moments.AddMomentumFluxToRHS(E,J,Ji,rho,ped,hp.m_n_floor*PhysConst::q_e,geom,solver);
                for (amrex::MFIter mfi(measures);mfi.isValid();++mfi) {
                    auto const out = measures.array(mfi);
                    auto const er = E[0]->const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                            // Compare to an independently integrated cell-average
                            // force, so a thin layer is not tested at one lucky point.
                            amrex::Real exact = 0.0_rt;
                            constexpr int quadrature = 256;
                            for (int q = 0; q < quadrature; ++q) {
                                exact += profile.exact_e(plo[0]+(i+(q+0.5_rt)/quadrature)*dx[0]);
                            }
                            exact /= quadrature;
                            out(i,j,k,0) = std::abs(er(i,j,k)-exact);
                            out(i,j,k,1) = std::abs(exact);
                            out(i,j,k,2) = er(i,j,k);
                            out(i,j,k,3) = exact;
                            out(i,j,k,4) = std::abs(er(i,j,k));
                            out(i,j,k,5) = er(i,j,k)-exact;
                        });
                }
                auto const l1 = measures.sum_unique(0,false,geom.periodicity());
                auto const scale = measures.sum_unique(1,false,geom.periodicity());
                auto const signed_e = measures.sum_unique(2,false,geom.periodicity());
                auto const signed_exact = measures.sum_unique(3,false,geom.periodicity());
                auto const norm = measures.norminf(1);
                auto const refinement=geom.Domain().length(0)/32.0_rt;
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(l1/scale < 0.16_rt/(refinement*refinement),
                    "Smooth momentum-flux force must converge at second order");
                amrex::Print() << "EDGE_PROBE {\"nr\":" << geom.Domain().length(0)
                    << ",\"width_cells\":" << cells << ",\"phase\":" << phase
                    << ",\"pedestal_fraction\":" << fraction
                    << ",\"density_average\":" << (density_average ? "true" : "false")
                    << ",\"continuity_error\":" << continuity_error
                    << ",\"l1_relative\":" << l1/scale
                    << ",\"linf_relative\":" << measures.norminf(0)/norm
                    << ",\"signed_integral_ratio\":" << signed_e/signed_exact
                    << ",\"force_max_V_m\":" << measures.norminf(4)
                    << ",\"force_exact_max_V_m\":" << norm << "}\n";
            }
        }
        amrex::Print() << "EDGE_PROBE_COMPLETE\n";
    }
    warpx::initialization::finalize_external_libraries();
}
