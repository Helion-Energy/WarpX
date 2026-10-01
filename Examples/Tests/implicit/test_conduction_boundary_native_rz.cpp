/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <iomanip>
#include <string>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& hp = *sim.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const plo = geom.ProbLoArray();
        auto const phi = geom.ProbHiArray();
        auto const dlo = geom.Domain().smallEnd();
        auto const dhi = geom.Domain().bigEnd();
        auto const own = amrex::OwnerMask(te, geom.periodicity());
        bool const periodic_z = geom.isPeriodic(1);
        amrex::Real const kelvin = PhysConst::q_e / PhysConst::kb;
        amrex::Real const density = 2.e18;
        rho.setVal(density * PhysConst::q_e);
        auto const* pedestal = hp.DensityPedestal(0);
        // Independent analytic annular-prism volumes, without the production helper.
        auto energy = [&] ()
        {
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<amrex::Real> data(op);
            using Tuple = typename decltype(data)::Type;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.const_array(mfi);
                auto const mask = own->const_array(mfi);
                auto const ped = pedestal ? pedestal->const_array(mfi)
                                          : amrex::Array4<amrex::Real const>{};
                amrex::Real const floor = hp.m_qdsmc_n_floor;
                op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple {
                    if (!mask(i,j,k)) { return {0.0}; }
                    amrex::Real const r = plo[0] + (i-dlo[0])*dx[0];
                    amrex::Real const rl = amrex::max(plo[0], r-dx[0]/2);
                    amrex::Real const rh = amrex::min(phi[0], r+dx[0]/2);
                    amrex::Real const z = plo[1] + (j-dlo[1])*dx[1];
                    amrex::Real const dz = periodic_z ? dx[1]
                        : amrex::min(phi[1], z+dx[1]/2)-amrex::max(plo[1], z-dx[1]/2);
                    amrex::Real const ne = amrex::max(density +
                        (ped ? ped(i,j,k)/PhysConst::q_e : 0.0), floor);
                    return {1.5*PhysConst::kb*ne*t(i,j,k)*MathConst::pi*(rh*rh-rl*rl)*dz};
                });
            }
            amrex::Real value = amrex::get<0>(data.value(op));
            amrex::ParallelDescriptor::ReduceRealSum(value);
            return value;
        };
        amrex::ParmParse pp("boundary_test");
        std::string mode = "flux";
        pp.query("mode", mode);
        amrex::Real dt = 1.e-7;
        pp.query("dt", dt);
        for (int d=0; d<AMREX_SPACEDIM; ++d) {
            for (int s=0; s<2; ++s) { hp.m_cond_bc[d][s] = 0; }
        }
        te.setVal(100.0*kelvin);
        if (mode == "flux") {
            hp.m_cond_bc[0][1] = 2;
            hp.m_cond_bc_q[0][1] = 1.e4;
            amrex::Real const area = 2*MathConst::pi*phi[0]*(phi[1]-plo[1]);
            amrex::Real const expected = 1.e4*area*dt;
            auto const before = energy();
            auto const tally0 = hp.GetQdsmcWallTally(0,1);
            hp.QdsmcConductionOnceFDAtState(0,dt,false,rho,0.0);
            auto const actual = energy()-before;
            auto const heat = (hp.GetQdsmcWallTally(0,1)-tally0)*dx[0]*dx[1];
            amrex::Print() << std::setprecision(17) << "NATIVE_BOUNDARY mode=flux dt=" << dt
                << " energy_ratio=" << actual/expected << " tally_ratio=" << heat/expected
                << " delta_energy_J=" << actual << " expected_J=" << expected << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(actual/expected-1) < 1.e-9);
            AMREX_ALWAYS_ASSERT(std::abs(heat/expected-1) < 1.e-9);
        } else if (mode == "robin") {
            AMREX_ALWAYS_ASSERT(!periodic_z);
            hp.m_cond_bc[1][1] = 3;
            hp.m_cond_leg_flux_limit = 0;
            hp.m_cond_leg_length = 6;
            hp.m_cond_leg_Te_wall = 0.5;
            amrex::Real const tb = 1.999, ti = 2.0;
            int const wall = dhi[1]+1;
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto const t = te.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) {
                    t(i,j,k) = kelvin*(j == wall ? tb : ti);
                });
            }
            hp.QdsmcConductionOnceFDAtState(0,dt,false,rho,0.0);
            amrex::MultiFab rate(te.boxArray(),te.DistributionMap(),1,0);
            for (amrex::MFIter mfi(rate); mfi.isValid(); ++mfi) {
                auto const out=rate.array(mfi); auto const t=te.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) {
                    out(i,j,k) = (j == wall) ? (t(i,j,k)/kelvin-tb)/dt : 0.0;
                });
            }
            // chi=100 in the input deck, uniform density, exactly one inward face.
            amrex::Real const interior=100*(ti-tb)/dx[1]*2/dx[1];
            amrex::Real const external=100*(tb-0.5)/6*2/dx[1];
            auto const expected=interior-external;
            auto const actual=-rate.norminf();
            amrex::Print() << std::setprecision(17) << "NATIVE_BOUNDARY mode=robin dt=" << dt
                << " measured_eV_s=" << actual << " one_face_eV_s=" << expected
                << " doubled_face_eV_s=" << 2*interior-external << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(actual-expected) < 1.e-4*(std::abs(interior)+external));
        } else {
            AMREX_ALWAYS_ASSERT(mode == "closed");
            auto const before = energy();
            hp.QdsmcConductionOnceFDAtState(0,dt,false,rho,0.0);
            auto const after = energy();
            amrex::Print() << "NATIVE_BOUNDARY mode=closed relative_energy=" << (after-before)/before << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(after-before) < 1.e-11*before);
        }
        AMREX_ALWAYS_ASSERT(te.is_finite());
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
