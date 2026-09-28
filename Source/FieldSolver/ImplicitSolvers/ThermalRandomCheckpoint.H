/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_THERMAL_RANDOM_CHECKPOINT_H_
#define WARPX_THERMAL_RANDOM_CHECKPOINT_H_
#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Random.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_Utility.H>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace warpx::thermal {
// Step-boundary I/O only. Covers the host engines and CUDA/HIP ParallelForRNG
// states used by native WarpX particle physics. AMReX's separate FillRandom
// generator is not exposed/serialized; it must not be used by this path.
// Raw device states deliberately require the same backend/library, device
// architecture, precision, rank count and OMP thread count on restart.
inline std::size_t ThermalRandomDeviceBytes () {
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    return std::size_t(amrex::Gpu::Device::maxBlocksPerLaunch()) *
        AMREX_GPU_MAX_THREADS * sizeof(amrex::randState_t);
#elif defined(AMREX_USE_GPU)
    amrex::Abort("Eulerian random checkpoint does not support this GPU RNG backend");
    return 0;
#else
    return 0;
#endif
}
inline std::string ThermalRandomCompatibility () {
    std::ostringstream s;
    s << "WarpXThermalRandom 1 " << amrex::ParallelDescriptor::NProcs() << ' '
      << amrex::ParallelDescriptor::MyProc() << ' ' << amrex::OpenMP::get_max_threads()
      << ' ' << AMREX_SPACEDIM << ' ' << sizeof(amrex::Real) << ' '
      << (std::endian::native==std::endian::little ? "little" : "big")
      << " compiler=" << __VERSION__;
#ifdef __GLIBCXX__
    s << " libstdcxx=" << __GLIBCXX__;
#endif
#ifdef _LIBCPP_VERSION
    s << " libcxx=" << _LIBCPP_VERSION;
#endif
#if defined(AMREX_USE_CUDA)
    s << " cuda=" << CUDART_VERSION << " curand=" << CURAND_VERSION;
#elif defined(AMREX_USE_HIP)
    s << " hip=" << HIP_VERSION << " hiprand=" << HIPRAND_VERSION;
#else
    s << " host";
#endif
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    s << " state=" << sizeof(amrex::randState_t)
      << " sm=" << amrex::Gpu::Device::devicePropMajor() << '.'
      << amrex::Gpu::Device::devicePropMinor()
      << " blocks=" << amrex::Gpu::Device::maxBlocksPerLaunch()
      << " threads=" << AMREX_GPU_MAX_THREADS;
#endif
    return s.str();
}
inline std::uint64_t ThermalRandomChecksum (
    std::string const& host, std::vector<char> const& device) {
    std::uint64_t hash=14695981039346656037ULL;
    for (unsigned char c:host) { hash=(hash^c)*1099511628211ULL; }
    for (unsigned char c:device) { hash=(hash^c)*1099511628211ULL; }
    return hash;
}
inline std::string ThermalRandomFile (std::string const& directory) {
    return directory+"/electron_random_"+
        amrex::Concatenate("",amrex::ParallelDescriptor::MyProc(),7)+".dat";
}
inline void WriteThermalRandomCheckpoint (std::string const& directory) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!amrex::OpenMP::in_parallel(),
        "Thermal random checkpoint must be outside an OMP region");
    amrex::Gpu::synchronize();
    std::ostringstream host; amrex::SaveRandomState(host);
    std::string const h=host.str();
    std::vector<char> device(ThermalRandomDeviceBytes());
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    AMREX_ALWAYS_ASSERT(amrex::getRandState()!=nullptr);
    amrex::Gpu::dtoh_memcpy(device.data(),amrex::getRandState(),device.size());
#endif
    std::ofstream out(ThermalRandomFile(directory),std::ios::binary|std::ios::trunc);
    out << ThermalRandomCompatibility() << '\n' << h.size() << ' ' << device.size()
        << ' ' << ThermalRandomChecksum(h,device) << '\n';
    out.write(h.data(),static_cast<std::streamsize>(h.size()));
    out.write(device.data(),static_cast<std::streamsize>(device.size()));
    out.close();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(out),"Unable to write Eulerian per-rank random checkpoint");
}
inline void ReadThermalRandomCheckpoint (std::string const& directory) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!amrex::OpenMP::in_parallel(),
        "Thermal random restart must be outside an OMP region");
    std::ifstream in(ThermalRandomFile(directory),std::ios::binary);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in),
        "Eulerian restart is missing per-rank random state; refusing to reseed stochastic physics");
    std::string compatibility;std::getline(in,compatibility);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(compatibility==ThermalRandomCompatibility(),
        "Eulerian random restart requires matching ranks, threads, precision and RNG backend/library/device");
    std::size_t nh=0,ng=0;std::uint64_t checksum=0;
    in>>nh>>ng>>checksum;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in) && in.get()=='\n' && nh>0 &&
        nh<=std::size_t(amrex::OpenMP::get_max_threads())*16384 &&
        ng==ThermalRandomDeviceBytes(),"Invalid Eulerian random checkpoint sizes");
    std::string host(nh,'\0');std::vector<char> device(ng);
    in.read(host.data(),static_cast<std::streamsize>(nh));
    in.read(device.data(),static_cast<std::streamsize>(ng));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in) && in.peek()==std::char_traits<char>::eof() &&
        checksum==ThermalRandomChecksum(host,device),"Corrupt or truncated Eulerian random checkpoint");
    // Validate every host engine before changing any live state.
    std::istringstream validate(host);
    for(int i=0;i<amrex::OpenMP::get_max_threads();++i) {
        std::mt19937 engine;validate>>engine;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(validate),"Invalid Eulerian host random engine state");
    validate>>std::ws;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(validate.eof(),"Trailing Eulerian host random engine state");
    amrex::Gpu::synchronize();
    std::istringstream restore(host);
    amrex::RestoreRandomState(restore,amrex::OpenMP::get_max_threads(),0);
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    AMREX_ALWAYS_ASSERT(amrex::getRandState()!=nullptr);
    amrex::Gpu::htod_memcpy(amrex::getRandState(),device.data(),ng);
#endif
    amrex::Gpu::synchronize();
}
} // namespace warpx::thermal
#endif
