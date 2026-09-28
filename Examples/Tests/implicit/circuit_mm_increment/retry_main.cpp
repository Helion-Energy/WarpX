/* Diagnostic-only rejection after an actual companion Freeze. No production hook. */
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "Circuit/CircuitCoupling.H"
#include <AMReX_VisMF.H>
#include <iomanip>
#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/ThermalRandomCheckpoint.H"
#include "NativeStateBytes.H"
#include <AMReX_ParmParse.H>
#include <filesystem>
#include <fstream>
#include <iterator>
using warpx::fields::FieldType;
using ablastr::fields::Direction;
using Response=warpx::particles::NativeEndpointCurrentResponse;
namespace {
bool reject_phase=false;
Response* last_response=nullptr;
std::uint64_t last_frozen_epoch=0;
int freezes=0,increment_actions=0;
std::vector<amrex::MultiFab const*> Scratch(WarpX& w) {
    std::vector<amrex::MultiFab const*> out;
    for(auto t:{FieldType::MassMatrices_X,FieldType::MassMatrices_Y,FieldType::MassMatrices_Z})
        for(int d=0;d<3;++d)if(w.m_fields.has(t,Direction{d},0))
            out.push_back(w.m_fields.get(t,Direction{d},0));
    return out;
}
std::string Random(std::string const& dir) {
    if(amrex::ParallelDescriptor::IOProcessor())std::filesystem::create_directory(dir);
    amrex::ParallelDescriptor::Barrier();
    warpx::thermal::WriteThermalRandomCheckpoint(dir);
    std::ifstream in(warpx::thermal::ThermalRandomFile(dir),std::ios::binary);
    AMREX_ALWAYS_ASSERT(in.good());
    return {std::istreambuf_iterator<char>(in),{}};
}
}
void MMCompanionRetryHook(Response* response) {
    AMREX_ALWAYS_ASSERT(response && response->IsValid());
    last_response=response;last_frozen_epoch=response->Epoch();++freezes;
    // Freeze only records the tensor; failure is injected after a real
    // retained current increment has been formed and composed.

}
void MMCircuitAfterCurrentIncrementHook() {
    ++increment_actions;
    if(reject_phase) {
        AMREX_ALWAYS_ASSERT(last_response && last_response->IsValid());
        last_response->Invalidate();
        amrex::Print()<<"INJECTED_AFTER_RETAINED_CURRENT actions="<<increment_actions
            <<" frozen_epoch="<<last_frozen_epoch<<" invalid_epoch="<<last_response->Epoch()<<"\n";
    }
}
std::string CircuitState(WarpX&w,std::string const&tag) {
    auto dir=tag+"-rank"+std::to_string(amrex::ParallelDescriptor::MyProc());
    std::filesystem::create_directory(dir);auto& c=*w.get_pointer_CircuitCoupling()->Coupler();
    c.Plugin()->WriteCheckpoint(dir);c.WriteMemoryCheckpoint(dir);
    std::vector<std::filesystem::path> files;for(auto const& e:std::filesystem::directory_iterator(dir))if(e.is_regular_file())files.push_back(e.path());std::sort(files.begin(),files.end());std::string bytes;for(auto const& f:files){bytes+=f.filename().string();std::ifstream in(f,std::ios::binary);bytes.append(std::istreambuf_iterator<char>(in),{});}return bytes;
}
int main(int argc,char** argv) {
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w=WarpX::GetInstance();w.InitData();w.HybridPICInitializeRhoJandB();
        bool fail_first=true,inject_weight=false;
        amrex::ParmParse pp("retry_test");pp.query("fail_first",fail_first);
        pp.query("inject_weight",inject_weight);
        auto* solver=w.get_pointer_ImplicitSolver();AMREX_ALWAYS_ASSERT(solver);
        auto const time=w.gett_new(0);auto const step=w.getistep(0);auto const dt=w.getdt(0);
        auto const excluded=Scratch(w);
        auto const before=radial_fixture::NativeStateBytes::Capture(w,true,true,excluded);
        before.Write("entry");auto const random_before=Random("rng-entry");auto const circuit_before=CircuitState(w,"circuit-entry");
        if(fail_first) {
            reject_phase=true;
            int const status=solver->OneStep(time,dt,step);
            amrex::Print()<<"REJECTED_COMPANION_STEP status="<<status<<" freezes="<<freezes<<"\n";
            AMREX_ALWAYS_ASSERT(status<0 && freezes>0 && increment_actions>0 && last_response);
            AMREX_ALWAYS_ASSERT(!last_response->IsValid() && last_response->Epoch()>last_frozen_epoch);
            radial_fixture::NativeStateBytes::Capture(w,true,true,excluded).Write("rejected");
            if(inject_weight)for(auto const& pc:w.GetPartContainer()) {
                bool done=false;
                for(WarpXParIter pti(*pc,0);pti.isValid();++pti)if(pti.numParticles()>0) {
                    auto* weight=pti.GetAttribs(PIdx::w).dataPtr();amrex::ParticleReal value;
                    amrex::Gpu::dtoh_memcpy(&value,weight,sizeof(value));value*=2;
                    amrex::Gpu::htod_memcpy(weight,&value,sizeof(value));done=true;break;
                }
                if(done)break;
            }
            before.RequireUnchanged(w,"rejected_companion",true,true,excluded,true);
            AMREX_ALWAYS_ASSERT(random_before==Random("rng-rejected"));AMREX_ALWAYS_ASSERT(circuit_before==CircuitState(w,"circuit-rejected"));
            AMREX_ALWAYS_ASSERT(w.gett_new(0)==time && w.getistep(0)==step);
            amrex::Print()<<"COMPANION_ROLLBACK_PASS fields_particles_rng_time_step_invalid_cache\n";
        }
        reject_phase=false;int const old_freezes=freezes;
        int const status=solver->OneStep(time,dt,step);
        AMREX_ALWAYS_ASSERT(status>=0 && freezes>old_freezes && last_response);
        radial_fixture::NativeStateBytes::Capture(w,true,true,Scratch(w)).Write("accepted");
        auto const circuit_accepted=CircuitState(w,"circuit-accepted");amrex::ignore_unused(circuit_accepted);auto const rng=Random("rng-accepted");amrex::ignore_unused(rng);
        amrex::Print()<<"COMPANION_RETRY_PASS status="<<status<<" freezes="<<freezes
            <<" final_epoch="<<last_response->Epoch()<<"\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
