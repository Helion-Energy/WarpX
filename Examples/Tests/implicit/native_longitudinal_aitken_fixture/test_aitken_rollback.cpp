/* Native accepted Yee companion checkpoint rejection/restore/retry diagnostic. */
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/ThermalRandomCheckpoint.H"
#include "NativeStateBytes.H"
#include "FieldSolver/ImplicitSolvers/NativeVacuumConstraint.H"
#include <AMReX_ParmParse.H>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iomanip>
using warpx::fields::FieldType;
using ablastr::fields::Direction;
using Real=amrex::Real;
using MF=amrex::MultiFab; using View=ablastr::fields::VectorField;
using Result=warpx::darwin::NativeVacuumConstraintResult;
static int calls=0;static bool reject=false;
extern "C" Result RealCorrect(WarpX&,MF const&,int,Real,View const&,View const&,MF&,View const&) asm("__real__ZN5warpx6darwin31CorrectNativeVacuumLongitudinalER5WarpXRKN5amrex8MultiFabEidRKSt5arrayIPS4_Lm3EESB_RS4_SB_");
extern "C" Result WrappedCorrect(WarpX&,MF const&,int,Real,View const&,View const&,MF&,View const&) asm("__wrap__ZN5warpx6darwin31CorrectNativeVacuumLongitudinalER5WarpXRKN5amrex8MultiFabEidRKSt5arrayIPS4_Lm3EESB_RS4_SB_");
extern "C" Result WrappedCorrect(WarpX& w,MF const& rho,int component,Real interval,View const& raw,View const& held,MF& phi,View const& delta){
 auto result=RealCorrect(w,rho,component,interval,raw,held,phi,delta);
 if(reject && ++calls==5){AMREX_ALWAYS_ASSERT(result.compatible&&result.converged);result.converged=false;amrex::Print()<<"AITKEN_INJECTED_SOLVER_REJECTION after=5\n";}
 return result;
}
std::vector<amrex::MultiFab const*> scratch(WarpX& w) {
    std::vector<amrex::MultiFab const*> v;
    for(int lev=0;lev<=w.finestLevel();++lev)
        for(auto t:{FieldType::MassMatrices_X,FieldType::MassMatrices_Y,FieldType::MassMatrices_Z})
            for(int d=0;d<3;++d)if(w.m_fields.has(t,Direction{d},lev))v.push_back(w.m_fields.get(t,Direction{d},lev));
    return v;
}
std::vector<int> components(std::vector<amrex::MultiFab const*> const& v,std::string const& phase) {
    std::vector<int> n;int i=0;
    for(auto* f:v){n.push_back(f->nComp());amrex::Print()<<"NATIVE_MM phase="<<phase<<" index="<<i++<<" components="<<f->nComp()<<"\n";}
    return n;
}
std::string random_bytes(std::string const& dir) {
    if(amrex::ParallelDescriptor::IOProcessor())std::filesystem::create_directory(dir);
    amrex::ParallelDescriptor::Barrier();
    warpx::thermal::WriteThermalRandomCheckpoint(dir);
    std::ifstream in(warpx::thermal::ThermalRandomFile(dir),std::ios::binary);
    AMREX_ALWAYS_ASSERT(bool(in));
    return std::string(std::istreambuf_iterator<char>(in),{});
}
int main(int argc,char** argv){
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        std::string restart;amrex::ParmParse("amr").get("restart",restart);
        AMREX_ALWAYS_ASSERT(!restart.empty());
        auto& w=WarpX::GetInstance();w.InitData();
        // Match WarpX::Evolve entry: restart reconstructs deposited rho/J and U-derived Te/Pe here.
        w.HybridPICInitializeRhoJandB();
        amrex::OutStream().precision(17);
        bool fail_first=true;Real dt=7.8125e-13,fail_factor=1.;
        auto pp=amrex::ParmParse("rollback_test");pp.query("fail_first",fail_first);
        pp.query("dt",dt);pp.query("fail_factor",fail_factor);
        auto* solver=w.get_pointer_ImplicitSolver();AMREX_ALWAYS_ASSERT(solver);
        auto const time=w.gett_new(0);auto const step=w.getistep(0);
        auto const excluded=scratch(w);auto const original_components=components(excluded,"entry");
        auto const before=radial_fixture::NativeStateBytes::Capture(w,true,true,excluded);
        before.Write("entry");auto const random_before=random_bytes("rng-entry");
        if(fail_first){
            reject=true;calls=0;w.setdt(dt*fail_factor);
            int const status=solver->OneStep(time,dt*fail_factor,step);
            amrex::Print()<<"NATIVE_REJECT status="<<status<<" time="<<time<<" step="<<step<<"\n";
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status<0,"Fixture requires the injected fifth native outer correction rejection");
            auto const after_components=components(scratch(w),"after_reject");
            int grew=0;AMREX_ALWAYS_ASSERT(original_components.size()==after_components.size());
            for(std::size_t i=0;i<original_components.size();++i)grew+=after_components[i]>original_components[i];
            amrex::ParallelDescriptor::ReduceIntMax(grew);
            amrex::Print()<<"NATIVE_MM_RESIZE grew="<<grew<<"\n";
            // This smooth tiny orbit stays within its native shape-index support;
            // dynamic MM growth is qualified separately by the R35 crossing case.
            // Retain every observed component count; this gate concerns accepted
            // native Je/D/Ei and physical particle/field/RNG restoration.
            amrex::ignore_unused(original_components);
            bool inject=false;pp.query("inject_physical",inject);
            if(inject) {
                for(auto const& pc:w.GetPartContainer()) {
                    bool done=false;
                    for(auto& level:pc->GetParticles()) {
                        for(auto& [key,tile]:level) {
                            auto& values=tile.GetStructOfArrays().GetRealData(PIdx::w);
                            if(!values.empty()) {
                                amrex::ParticleReal old_weight;
                                amrex::Gpu::dtoh_memcpy(&old_weight,values.data(),sizeof(old_weight));
                                auto const altered=old_weight*2;
                                amrex::Gpu::htod_memcpy(values.data(),&altered,sizeof(altered));done=true;break;
                            }
                        }
                        if(done)break;
                    }
                    if(done)break;
                }
                amrex::Print()<<"NATIVE_INJECTED_PHYSICAL_WEIGHT_MUTATION\n";
            }
            before.RequireUnchanged(w,"failed_physical_step",true,true,excluded,true);
            auto const random_after=random_bytes("rng-rejected");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(random_before==random_after,"Rejected residual advanced host/device RNG");
            AMREX_ALWAYS_ASSERT(w.gett_new(0)==time && w.getistep(0)==step);
            amrex::Print()<<"NATIVE_ROLLBACK_PASS physical_bytes_rng_time_step\n";
        }
        reject=false;calls=0;w.setdt(dt);
        int const accepted=solver->OneStep(time,dt,step);
        amrex::Print()<<"NATIVE_RETRY status="<<accepted<<" dt="<<dt<<" failed_first="<<fail_first<<"\n";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(accepted>=0,"Baseline retry must converge under original gates");
        radial_fixture::NativeStateBytes::Capture(w,true,true,scratch(w)).Write("accepted");
        auto const rng=random_bytes("rng-accepted");amrex::ignore_unused(rng);
        amrex::Print()<<"NATIVE_RETRY_PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
