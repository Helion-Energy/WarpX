/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <iomanip>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    int exit_code = 0;
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& h = *w.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& te = *w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
        auto& entropy = *w.m_fields.get(FieldType::hybrid_entropy_fp,0);
        auto& weights = *w.m_fields.get(FieldType::hybrid_qdsmc_weights_fp,0);
        auto& rho = *w.m_fields.get(FieldType::rho_fp,0);
        auto& pedestal = *const_cast<amrex::MultiFab*>(h.DensityPedestal(0));
        bool const has_eb = EB::enabled();
        auto const* phi = has_eb ? w.m_fields.get(FieldType::distance_to_eb,0) : nullptr;
        auto const dx = w.Geom(0).CellSizeArray();
        amrex::Real const volume = dx[0]*dx[1]*dx[2];
        amrex::Real const old_te = 500*PhysConst::q_e/PhysConst::kb;
        amrex::Real const recovered_te = 100*PhysConst::q_e/PhysConst::kb;
        amrex::Real const density = 1.e18;
        amrex::Real const ped_density = 1.e16;
        amrex::Real const gamma = h.m_gamma;
        amrex::ParmParse pp("recovery_test");
        amrex::Real covered_entropy = 1.e30;
        bool zero_weight = false;
        pp.query("covered_entropy",covered_entropy);
        pp.query("zero_weight",zero_weight);
        amrex::MultiFab expected(te.boxArray(),te.DistributionMap(),1,te.nGrowVect());
        te.setVal(old_te);
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const ph = has_eb ? phi->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            auto const ph_box = has_eb ? phi->fabbox(mfi.index()) : mfi.fabbox();
            auto const ke = entropy.array(mfi), wt = weights.array(mfi);
            auto const den = rho.array(mfi), ped = pedestal.array(mfi);
            auto const exact = expected.array(mfi);
            AMREX_ALWAYS_ASSERT(entropy.fabbox(mfi.index()).contains(mfi.fabbox()));
            AMREX_ALWAYS_ASSERT(weights.fabbox(mfi.index()).contains(mfi.fabbox()));
            AMREX_ALWAYS_ASSERT(rho.fabbox(mfi.index()).contains(mfi.fabbox()));
            AMREX_ALWAYS_ASSERT(pedestal.fabbox(mfi.index()).contains(mfi.fabbox()));
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) {
                bool const covered = has_eb && ph_box.contains(amrex::IntVect(i,j,k)) &&
                    ph(i,j,k)<=0;
                amrex::Real const nped = covered ? 0 : ped_density;
                amrex::Real const n = covered ? -density : density;
                amrex::Real const weight = covered ? (zero_weight ? 0 : 1.e-6*density) :
                    0.5*density;
                den(i,j,k) = n*PhysConst::q_e;
                ped(i,j,k) = nped*PhysConst::q_e;
                wt(i,j,k) = weight;
                ke(i,j,k) = covered ? covered_entropy : recovered_te*
                    std::pow(density+ped_density,1-gamma)*weight*volume*
                    PhysConst::kb/PhysConst::q_e;
                exact(i,j,k) = covered ? old_te :
                    (ped_density*old_te+weight*recovered_te)/(ped_density+weight);
            });
        }
        h.QDSMCUpdateTe(0);
        amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax,
                         amrex::ReduceOpSum,amrex::ReduceOpSum> ops;
        amrex::ReduceData<amrex::Real,amrex::Real,amrex::Long,amrex::Long> data(ops);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto const t = te.const_array(mfi), exact = expected.const_array(mfi);
            auto const ph = has_eb ? phi->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            auto const ph_box = has_eb ? phi->fabbox(mfi.index()) : mfi.fabbox();
            ops.eval(mfi.fabbox(),data,[=] AMREX_GPU_DEVICE (int i,int j,int k)->Tuple {
                bool const covered = has_eb && ph_box.contains(amrex::IntVect(i,j,k)) &&
                    ph(i,j,k)<=0;
                auto const error = std::abs(t(i,j,k)-exact(i,j,k))/old_te;
                bool const bad = !std::isfinite(t(i,j,k));
                return {covered ? error : 0, covered ? 0 : error,
                        covered ? amrex::Long(1) : amrex::Long(0),bad ? amrex::Long(1) : amrex::Long(0)};
            });
        }
        auto const result = data.value();
        auto covered_error=amrex::get<0>(result),open_error=amrex::get<1>(result);
        auto covered_nodes=amrex::get<2>(result),nonfinite=amrex::get<3>(result);
        amrex::ParallelDescriptor::ReduceRealMax(covered_error);
        amrex::ParallelDescriptor::ReduceRealMax(open_error);
        amrex::ParallelDescriptor::ReduceLongSum(covered_nodes);
        amrex::ParallelDescriptor::ReduceLongSum(nonfinite);
        bool const passed = covered_error==0 && open_error<=1.e-12 && nonfinite==0 &&
            (!has_eb || covered_nodes>0);
        amrex::Print()<<std::setprecision(17)<<"COVERED_RECOVERY_RESULT covered_error="
            <<covered_error<<" open_error="<<open_error<<" covered_copies="<<covered_nodes
            <<" nonfinite="<<nonfinite<<'\n'
            <<(passed ? "COVERED_RECOVERY_PASS\n" : "COVERED_RECOVERY_FAIL\n");
        exit_code = passed ? 0 : 1;
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
    return exit_code;
}
