/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_ENDPOINT_PARTICLE_HANDOFF_H_
#define WARPX_NATIVE_ENDPOINT_PARTICLE_HANDOFF_H_
#include "Particles/WarpXParticleContainer.H"
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
namespace warpx::darwin::detail {
/** Hashing locates an ID; every ID and attribute byte is then compared.
 * Duplicate IDs fail. Physical particles are never reordered or copied to host.
 * Scratch lives until the caller completes the one collective invalid check.
 */
struct EndpointHandoffScratch {
    // CUDA atomicCAS requires this exact unsigned 64-bit storage type.
    using Slot = unsigned long long;
    amrex::Gpu::DeviceVector<Slot> table;
    amrex::Gpu::DeviceVector<amrex::Long> permutation;
    amrex::Gpu::DeviceVector<int> seen;
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::uint64_t EndpointIDHash(std::uint64_t x) noexcept {
    x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);
    x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}
inline bool CompareEndpointTilePermutation(
    WarpXParticleContainer::ParticleTileType const& actual,
    WarpXParticleContainer::ParticleTileType const& expected,
    int* invalid,std::vector<std::unique_ptr<EndpointHandoffScratch>>& lifetime)
{
    using L=amrex::Long;using Slot=EndpointHandoffScratch::Slot;
    static_assert(std::numeric_limits<Slot>::digits>=std::numeric_limits<L>::digits);
    constexpr Slot empty=std::numeric_limits<Slot>::max();
    L const count=actual.numParticles();
    if(count!=expected.numParticles()||actual.numNeighborParticles()!=0||expected.numNeighborParticles()!=0||
       actual.NumRealComps()!=expected.NumRealComps()||actual.NumIntComps()!=expected.NumIntComps())return false;
    if(count<0||count>std::numeric_limits<L>::max()/4)return false;
    L capacity=1;while(capacity<2*count)capacity*=2;
    auto const& a=actual.GetStructOfArrays();auto const& b=expected.GetStructOfArrays();
    if(a.GetIdCPUData().size()!=std::size_t(count)||b.GetIdCPUData().size()!=std::size_t(count))return false;
    for(int c=0;c<actual.NumRealComps();++c)
        if(a.GetRealData(c).size()!=std::size_t(count)||b.GetRealData(c).size()!=std::size_t(count))return false;
    for(int c=0;c<actual.NumIntComps();++c)
        if(a.GetIntData(c).size()!=std::size_t(count)||b.GetIntData(c).size()!=std::size_t(count))return false;
    if(count==0)return true;
    auto memory=std::make_unique<EndpointHandoffScratch>();
    memory->table.resize(capacity);memory->permutation.resize(count);memory->seen.resize(count);
    auto* table=memory->table.data();auto* permutation=memory->permutation.data();auto* seen=memory->seen.data();
    auto const* aid=a.GetIdCPUData().data();auto const* bid=b.GetIdCPUData().data();
    amrex::ParallelFor(capacity,[=] AMREX_GPU_DEVICE(L i){table[i]=empty;});
    amrex::ParallelFor(count,[=] AMREX_GPU_DEVICE(L i){seen[i]=0;permutation[i]=-1;});
    auto const mask=std::uint64_t(capacity-1);
    amrex::ParallelFor(count,[=] AMREX_GPU_DEVICE(L i){
        auto slot=EndpointIDHash(bid[i])&mask;
        for(L probe=0;probe<capacity;++probe){
            Slot const previous=amrex::Gpu::Atomic::CAS(table+slot,empty,static_cast<Slot>(i));
            if(previous==empty)return;
            if(bid[previous]==bid[i]){amrex::Gpu::Atomic::Max(invalid,1);return;}
            slot=(slot+1)&mask;
        }
        amrex::Gpu::Atomic::Max(invalid,1);
    });
    amrex::ParallelFor(count,[=] AMREX_GPU_DEVICE(L i){
        auto slot=EndpointIDHash(aid[i])&mask;
        for(L probe=0;probe<capacity;++probe){
            Slot const previous=table[slot];if(previous==empty)break;
            if(bid[previous]==aid[i]){
                if(amrex::Gpu::Atomic::CAS(seen+previous,0,1)!=0){amrex::Gpu::Atomic::Max(invalid,1);return;}
                permutation[i]=static_cast<L>(previous);return;
            }
            slot=(slot+1)&mask;
        }
        amrex::Gpu::Atomic::Max(invalid,1);
    });
    for(int c=0;c<actual.NumRealComps();++c){
        auto const* ap=reinterpret_cast<unsigned char const*>(a.GetRealData(c).data());
        auto const* bp=reinterpret_cast<unsigned char const*>(b.GetRealData(c).data());
        amrex::ParallelFor(count,[=] AMREX_GPU_DEVICE(L i){
            L const j=permutation[i];if(j<0||j>=count)return;
            for(std::size_t byte=0;byte<sizeof(amrex::ParticleReal);++byte)
                if(ap[i*sizeof(amrex::ParticleReal)+byte]!=bp[j*sizeof(amrex::ParticleReal)+byte])
                    amrex::Gpu::Atomic::Max(invalid,1);
        });
    }
    for(int c=0;c<actual.NumIntComps();++c){
        auto const* ap=reinterpret_cast<unsigned char const*>(a.GetIntData(c).data());
        auto const* bp=reinterpret_cast<unsigned char const*>(b.GetIntData(c).data());
        amrex::ParallelFor(count,[=] AMREX_GPU_DEVICE(L i){
            L const j=permutation[i];if(j<0||j>=count)return;
            for(std::size_t byte=0;byte<sizeof(int);++byte)
                if(ap[i*sizeof(int)+byte]!=bp[j*sizeof(int)+byte])amrex::Gpu::Atomic::Max(invalid,1);
        });
    }
    lifetime.push_back(std::move(memory));return true;
}
} // namespace warpx::darwin::detail
#endif
