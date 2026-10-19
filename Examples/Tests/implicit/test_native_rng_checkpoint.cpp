#include "Initialization/WarpXInit.H"
#include "Python/callbacks.H"
#include "Utils/NativeRngState.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

#include <fstream>
#include <iomanip>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        bool restored = false;
        InstallPythonCallback("afterInitatRestart", [&]() {
            std::string checkpoint;
            amrex::ParmParse("amr").get("restart", checkpoint);
            auto const name = amrex::Concatenate("rank-", amrex::ParallelDescriptor::MyProc(), 6);
            auto const saved = warpx_rng::Read(checkpoint + "/NativeRNG/" + name + ".bin");
            auto const actual = warpx_rng::Capture();
            AMREX_ALWAYS_ASSERT(saved.host == actual.host && saved.device == actual.device);
            restored = true;
        });
        sim.InitData();
        bool const restarting = sim.getistep(0) != 0;
        AMREX_ALWAYS_ASSERT(restored == restarting);
        auto const rank = amrex::ParallelDescriptor::MyProc();
        std::ofstream output(amrex::Concatenate("samples-rank-", rank, 6) + ".jsonl");
        output << std::setprecision(17);
        InstallPythonCallback("particleinjection", [&]() {
            constexpr int count = 4096;
            amrex::Gpu::DeviceVector<amrex::Real> device(count * 4);
            amrex::Gpu::HostVector<amrex::Real> host(count * 4);
            auto* values = device.data();
            amrex::ParallelForRNG(count, [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                values[4*i] = amrex::Random(engine);
                values[4*i+1] = amrex::RandomNormal(0.0, 1.0, engine);
                values[4*i+2] = amrex::Real(amrex::Random_int(37, engine));
                values[4*i+3] = amrex::Real(amrex::RandomPoisson(2.5, engine));
            });
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, device.begin(), device.end(), host.begin());
            auto const uniform = amrex::Random();
            auto const normal = amrex::RandomNormal(0.0, 1.0);
            output << "{\"step\":" << sim.getistep(0) << ",\"uniform\":" << uniform
                   << ",\"normal\":" << normal << ",\"gpu\":[";
            for (int i = 0; i < count * 4; ++i) {
                if (i != 0) { output << ','; }
                output << host[i];
            }
            output << "]}\n";
            output.flush();
            AMREX_ALWAYS_ASSERT(output.good());
        });
        sim.Evolve();
        AMREX_ALWAYS_ASSERT(sim.getistep(0) == 20);
        auto const final_state = warpx_rng::Capture();
        warpx_rng::Write(amrex::Concatenate("final-rng-rank-", rank, 6) + ".bin", final_state);
        amrex::Print() << "NATIVE_RNG_CHECKPOINT_EVOLUTION_COMPLETE restored=" << restored << '\n';
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
