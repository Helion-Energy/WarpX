#ifndef WARPX_NATIVE_RNG_STATE_H_
#define WARPX_NATIVE_RNG_STATE_H_
#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_Gpu.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <cstdint>
#include <filesystem>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Captures the host engines and CUDA ParallelForRNG engine pool. Does not claim
// to capture the separate cuRAND host generator used by AMReX FillRandom APIs.
// The target WarpX source has no FillRandom/FillRandomNormal consumers.
namespace warpx_rng {
struct State {
    int ranks=0, rank=0, threads=0, cuda_version=0;
    std::size_t engine_count=0, engine_bytes=0;
    std::string host;
    std::vector<unsigned char> device;
};
inline State Describe () {
    State s;s.ranks=amrex::ParallelDescriptor::NProcs();s.rank=amrex::ParallelDescriptor::MyProc();
    s.threads=amrex::OpenMP::get_max_threads();
#ifdef AMREX_USE_CUDA
    s.cuda_version=CUDART_VERSION;
    s.engine_count=std::size_t(amrex::Gpu::Device::maxBlocksPerLaunch())*AMREX_GPU_MAX_THREADS;
    s.engine_bytes=sizeof(amrex::randState_t);
#elif defined(AMREX_USE_GPU)
    throw std::runtime_error("Native RNG persistence is not qualified for this GPU backend");
#endif
    return s;
}
inline void Validate (State const& s) {
    auto const expected=Describe();
    if(s.ranks!=expected.ranks || s.rank!=expected.rank || s.threads!=expected.threads ||
       s.cuda_version!=expected.cuda_version || s.engine_count!=expected.engine_count ||
       s.engine_bytes!=expected.engine_bytes || s.device.size()!=expected.engine_count*expected.engine_bytes ||
       s.host.empty() || s.host.size()>64*1024*1024)
        throw std::runtime_error("Native RNG checkpoint layout/backend/size mismatch");
}
inline State Capture () {
    State s=Describe();amrex::Gpu::synchronize();std::ostringstream host;
    amrex::SaveRandomState(host);s.host=host.str();
#ifdef AMREX_USE_CUDA
    s.device.resize(s.engine_count*s.engine_bytes);
    if(!amrex::getRandState())throw std::runtime_error("Native GPU RNG pool is not initialized");
    amrex::Gpu::dtoh_memcpy(s.device.data(),amrex::getRandState(),s.device.size());
#endif
    Validate(s);return s;
}
inline void Restore (State const& s) {
    Validate(s);
    // Validate all host engines before mutating any live state.
    std::istringstream probe(s.host);
    for (int thread = 0; thread < s.threads; ++thread) {
        std::mt19937 temporary;
        if (!(probe >> temporary)) {
            throw std::runtime_error("Invalid host RNG engine payload");
        }
    }
    probe >> std::ws;
    if (!probe.eof()) {
        throw std::runtime_error("Trailing host RNG engine payload");
    }
    amrex::Gpu::synchronize();std::istringstream host(s.host);
    amrex::RestoreRandomState(host,s.threads,0);
    if(!host)throw std::runtime_error("Invalid host RNG engine payload");
#ifdef AMREX_USE_CUDA
    amrex::Gpu::htod_memcpy(amrex::getRandState(),s.device.data(),s.device.size());
#endif
    amrex::Gpu::synchronize();
}
inline std::uint64_t Digest (State const& s) {
    // Accidental-corruption check, not authentication. Archival SHA256 is separate.
    std::uint64_t h=14695981039346656037ULL;
    for(unsigned char c:s.host){h^=c;h*=1099511628211ULL;}
    for(unsigned char c:s.device){h^=c;h*=1099511628211ULL;}
    return h;
}
inline void Write (std::string const& name, State const& s) {
    Validate(s);
    if (std::filesystem::exists(name)) {
        throw std::runtime_error("Refusing to overwrite native RNG state");
    }
    std::ofstream f(name,std::ios::binary|std::ios::trunc);
    f<<"WarpXNativeRNG 1\n"<<s.ranks<<' '<<s.rank<<' '<<s.threads<<' '<<s.cuda_version<<' '
     <<s.engine_count<<' '<<s.engine_bytes<<' '<<s.host.size()<<' '<<s.device.size()<<' '<<Digest(s)<<'\n';
    f.write(s.host.data(),std::streamsize(s.host.size()));
    f.write(reinterpret_cast<char const*>(s.device.data()),std::streamsize(s.device.size()));
    f.close();if(!f)throw std::runtime_error("Writing native RNG checkpoint failed");
}
inline State Read (std::string const& name) {
    std::ifstream f(name,std::ios::binary);std::string magic;std::getline(f,magic);
    if(magic!="WarpXNativeRNG 1")throw std::runtime_error("Native RNG checkpoint format mismatch");
    State s;std::size_t host_bytes=0,device_bytes=0;std::uint64_t digest=0;
    f>>s.ranks>>s.rank>>s.threads>>s.cuda_version>>s.engine_count>>s.engine_bytes>>host_bytes>>device_bytes>>digest;
    if(!f || f.get()!='\n')throw std::runtime_error("Invalid native RNG header");
    auto const expected=Describe();
    if(host_bytes==0 || host_bytes>64*1024*1024 || device_bytes!=expected.engine_count*expected.engine_bytes)
        throw std::runtime_error("Invalid native RNG payload lengths");
    s.host.resize(host_bytes);s.device.resize(device_bytes);
    f.read(s.host.data(),std::streamsize(host_bytes));
    f.read(reinterpret_cast<char*>(s.device.data()),std::streamsize(device_bytes));
    if(!f || f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Truncated or trailing RNG payload");
    Validate(s);if(Digest(s)!=digest)throw std::runtime_error("Native RNG payload checksum mismatch");
    return s;
}
inline bool Enabled ()
{
    bool enabled = false;
    amrex::ParmParse("warpx").query("checkpoint_rng_state", enabled);
    return enabled;
}

inline std::uint64_t LayoutDigest (std::string const& layout)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (unsigned char c : layout) {
        value ^= c;
        value *= 1099511628211ULL;
    }
    return value;
}

inline void WriteCheckpoint (std::string const& checkpoint, std::string const& layout)
{
    if (!Enabled()) { return; }
    namespace fs = std::filesystem;
    fs::path const directory = fs::path(checkpoint) / "NativeRNG";
    try {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (fs::exists(directory) || !fs::create_directory(directory)) {
                throw std::runtime_error("Native RNG checkpoint directory already exists or cannot be created");
            }
        }
        amrex::ParallelDescriptor::Barrier();
        auto const state = Capture();
        auto const name = amrex::Concatenate("rank-", state.rank, 6) + ".bin";
        auto const pending = directory / (name + ".partial");
        Write(pending.string(), state);
        fs::rename(pending, directory / name);
        amrex::ParallelDescriptor::Barrier();
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream header(directory / "Header.partial");
            header << "WarpXNativeRNGCheckpoint 1\n" << state.ranks << ' '
                   << LayoutDigest(layout) << '\n';
            header.close();
            if (!header) { throw std::runtime_error("Writing native RNG manifest failed"); }
            fs::rename(directory / "Header.partial", directory / "Header");
        }
        amrex::ParallelDescriptor::Barrier();
    } catch (std::exception const& error) {
        amrex::Abort(std::string("Native RNG checkpoint write: ") + error.what());
    }
}

inline bool ReadCheckpoint (std::string const& checkpoint, std::string const& layout)
{
    if (!Enabled()) { return false; }
    namespace fs = std::filesystem;
    fs::path const directory = fs::path(checkpoint) / "NativeRNG";
    try {
        bool required = false;
        amrex::ParmParse("warpx").query("require_restart_rng_state", required);
        if (!fs::exists(directory)) {
            if (required) { throw std::runtime_error("Required native RNG state is absent"); }
            amrex::Print() << "NATIVE_RNG_RESTART legacy_checkpoint_without_state; current seeded engines retained\n";
            return false;
        }
        std::ifstream header(directory / "Header");
        std::string magic;
        std::getline(header, magic);
        int ranks = 0;
        std::uint64_t digest = 0;
        header >> ranks >> digest;
        if (!header || magic != "WarpXNativeRNGCheckpoint 1" ||
            ranks != amrex::ParallelDescriptor::NProcs() || digest != LayoutDigest(layout)) {
            throw std::runtime_error("Incomplete RNG checkpoint or changed MPI/box ownership layout");
        }
        auto const name = amrex::Concatenate("rank-", amrex::ParallelDescriptor::MyProc(), 6) + ".bin";
        auto const state = Read((directory / name).string());
        // Every rank validates its complete payload before any live engine is restored.
        amrex::ParallelDescriptor::Barrier();
        Restore(state);
        amrex::ParallelDescriptor::Barrier();
        amrex::Print() << "NATIVE_RNG_RESTART restored_host_and_cuda_pool ranks=" << ranks << '\n';
        return true;
    } catch (std::exception const& error) {
        amrex::Abort(std::string("Native RNG checkpoint restore: ") + error.what());
    }
    return false;
}

}
#endif
