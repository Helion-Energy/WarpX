/* Fixture-only full raw particle/selected field byte oracle. */
#ifndef NATIVE_UNIFORM_STOPPING_BYTES_H_
#define NATIVE_UNIFORM_STOPPING_BYTES_H_
#include "FieldSolver/ImplicitSolvers/ThermalRandomCheckpoint.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <AMReX_Random.H>
#include <map>
#include <sstream>
#include <vector>
using RawBytes=std::vector<unsigned char>;
using RawState=std::map<std::string,RawBytes>;
template<typename T> RawBytes ReadBytes(T const* a,std::size_t n){
    RawBytes b(n*sizeof(T));if(n){amrex::Gpu::dtoh_memcpy(b.data(),a,b.size());}return b;
}
template<typename T> RawBytes ReadScalar(T const& a){auto const* p=reinterpret_cast<unsigned char const*>(&a);return {p,p+sizeof(T)};}
inline RawState ParticleBytes(WarpX& sim){
    amrex::Gpu::synchronize();RawState out;
    for(auto const& name:sim.GetPartContainer().GetSpeciesNames()){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
        for(std::size_t lev=0;lev<pc.GetParticles().size();++lev){
            auto const prefix=name+"/"+std::to_string(lev);
            out[prefix+"/tiles"]=ReadScalar(pc.GetParticles(lev).size());
            for(auto const& [key,tile]:pc.GetParticles(lev)){
                auto const label=prefix+"/"+std::to_string(key.first)+"/"+std::to_string(key.second);
                out[label+"/n"]=ReadScalar(tile.numParticles());out[label+"/neighbors"]=ReadScalar(tile.numNeighborParticles());
                auto const& soa=tile.GetStructOfArrays();auto const& id=soa.GetIdCPUData();
                out[label+"/id"]=ReadBytes(id.data(),id.size());
                for(int c=0;c<tile.NumRealComps();++c){auto const& a=soa.GetRealData(c);out[label+"/r"+std::to_string(c)]=ReadBytes(a.data(),a.size());}
                for(int c=0;c<tile.NumIntComps();++c){auto const& a=soa.GetIntData(c);out[label+"/i"+std::to_string(c)]=ReadBytes(a.data(),a.size());}
            }
        }
    }
    out["NextID"]=ReadScalar(WarpXParticleContainer::ParticleType::the_next_id);
    std::ostringstream stream;amrex::SaveRandomState(stream);auto const s=stream.str();out["rng_host"]={s.begin(),s.end()};
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    out["rng_device"]=ReadBytes(reinterpret_cast<unsigned char const*>(amrex::getRandState()),warpx::thermal::ThermalRandomDeviceBytes());
#endif
    return out;
}
inline void RestoreParticleBytes(WarpX& sim,RawState const& state){
    for(auto const& name:sim.GetPartContainer().GetSpeciesNames()){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
        for(std::size_t lev=0;lev<pc.GetParticles().size();++lev){for(auto& [key,tile]:pc.GetParticles(lev)){
            auto const label=name+"/"+std::to_string(lev)+"/"+std::to_string(key.first)+"/"+std::to_string(key.second);
            auto& soa=tile.GetStructOfArrays();for(int c=0;c<tile.NumRealComps();++c){auto& a=soa.GetRealData(c);auto const& b=state.at(label+"/r"+std::to_string(c));amrex::Gpu::htod_memcpy(a.data(),b.data(),b.size());}
        }}
    }amrex::Gpu::synchronize();
}
inline RawBytes FieldBytes(std::vector<amrex::MultiFab const*> const& fields){
    amrex::Gpu::synchronize();RawBytes out;
    for(auto const* f:fields){for(amrex::MFIter mfi(*f);mfi.isValid();++mfi){auto const& a=(*f)[mfi];auto b=ReadBytes(a.dataPtr(),a.size());out.insert(out.end(),b.begin(),b.end());}}
    return out;
}
#endif
