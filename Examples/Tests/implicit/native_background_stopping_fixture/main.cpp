// Reuse the qualified native loading, raw-state and zero-background event
// oracle without modifying its standalone entry point.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
// The included fixture main relies on C++ main's implicit return; this renamed,
// unused entry is excluded from our driver. Reuse only its fixture utilities.
#define main NativePECFixtureEntryUnused
#include "../native_pec_plasma_fixture/main.cpp"
#undef main
#pragma GCC diagnostic pop
#include "BackgroundEvent.H"
int main(int argc,char** argv){
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        InstallPythonCallback("beforeInitEsolve",[](){Load(WarpX::GetInstance());});
        auto& w=WarpX::GetInstance();w.InitData();w.HybridPICInitializeRhoJandB();
        DumpRegistry(w,"initialized");
        bool zero=false;amrex::ParmParse("fixture").query("background_zero",zero);
        if(!zero)Event(w);
        BackgroundEvent(w);
        InstallPythonCallback("afterstep",[&](){DumpRegistry(w,"accepted_"+std::to_string(w.getistep(0)));});
        w.Evolve();amrex::Print()<<"BACKGROUND_CONTINUATION_PASS steps="<<w.getistep(0)<<"\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
