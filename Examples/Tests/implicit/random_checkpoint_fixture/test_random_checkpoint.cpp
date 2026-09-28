#include "FieldSolver/ImplicitSolvers/ThermalRandomCheckpoint.H"
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <vector>

std::vector<amrex::Real> draw () {
    int constexpr n=4096;
    std::vector<amrex::Real> host(32*amrex::OpenMP::get_max_threads());
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        auto const t=amrex::OpenMP::get_thread_num();
        for(int i=0;i<32;++i) { host[32*t+i]=amrex::RandomNormal(0.,1.); }
    }
    amrex::Gpu::DeviceVector<amrex::Real> values(n);
    auto* data=values.data();
    amrex::ParallelForRNG(n,[=] AMREX_GPU_DEVICE (int i,amrex::RandomEngine const& e) {
        data[i]=amrex::RandomNormal(0.,1.,e)+amrex::Random(e);
    });
    auto const old=host.size();host.resize(old+n);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,values.begin(),values.end(),host.begin()+old);
    return host;
}
int main (int argc,char** argv) {
    if(argc!=3) { return 2; }
    std::string const directory=argv[1], test=argv[2];
    int ac=1;amrex::Initialize(ac,argv);
    {
        if(amrex::ParallelDescriptor::IOProcessor()) { std::filesystem::create_directories(directory); }
        amrex::ParallelDescriptor::Barrier();
        amrex::ResetRandomSeed(7301+amrex::ParallelDescriptor::MyProc(),8123+amrex::ParallelDescriptor::MyProc());
        auto warm=draw();amrex::ignore_unused(warm);
        warpx::thermal::WriteThermalRandomCheckpoint(directory);
        auto expected=draw();auto noise=draw();amrex::ignore_unused(noise);
        auto const file=warpx::thermal::ThermalRandomFile(directory);
        if(test!="replay") {
            std::ifstream in(file,std::ios::binary);
            std::string data((std::istreambuf_iterator<char>(in)),{});in.close();
            if(test=="metadata") { data[0]='X'; }
            else if(test=="truncated") { data.resize(data.size()/2); }
            else if(test=="checksum") { data.back()^=1; }
            else if(test=="missing") { std::filesystem::remove(file); }
            else { amrex::Abort("unknown fixture case"); }
            if(test!="missing") { std::ofstream out(file,std::ios::binary|std::ios::trunc);out.write(data.data(),data.size()); }
        }
        warpx::thermal::ReadThermalRandomCheckpoint(directory);
        auto replay=draw();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(expected.size()==replay.size() &&
            std::memcmp(expected.data(),replay.data(),expected.size()*sizeof(amrex::Real))==0,
            "Native host/ParallelForRNG replay differs");
        AMREX_ALWAYS_ASSERT(test=="replay");
        amrex::Print()<<"PASS native RNG replay; host plus ParallelForRNG, device bytes "
                      <<warpx::thermal::ThermalRandomDeviceBytes()<<"\n";
    }
    amrex::Finalize();
}
