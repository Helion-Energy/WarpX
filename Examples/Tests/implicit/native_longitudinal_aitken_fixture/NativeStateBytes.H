/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef NATIVE_RADIAL_STATE_BYTES_H
#define NATIVE_RADIAL_STATE_BYTES_H

#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <algorithm>
#include <cstring>
#include <map>
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>

namespace radial_fixture {

// Fixture-only host snapshots. Compare every byte, including NaN payloads;
// reductions report counts only and cannot hide a compensating state mutation.
// Pointer values/capacity of the allocator are not physical particle topology.
class NativeStateBytes {
    using Bytes = std::vector<unsigned char>;
    std::map<std::string, Bytes> m_records;

    void host(std::string key, void const* p, std::size_t bytes) {
        Bytes value(bytes);
        if (bytes) { std::memcpy(value.data(),p,bytes); }
        m_records.emplace(std::move(key),std::move(value));
    }
    template<class T> void scalar(std::string const& key,T const& value) {
        host(key,&value,sizeof(value));
    }
    template<class T> void device(std::string key,T const* p,std::size_t count) {
        Bytes value(count*sizeof(T));
        if (!value.empty()) { amrex::Gpu::dtoh_memcpy(value.data(),p,value.size()); }
        m_records.emplace(std::move(key),std::move(value));
    }
    void string(std::string const& key,std::string const& value) {
        host(key,value.data(),value.size());
    }
    void boxes(std::string const& key,amrex::BoxArray const& ba,
               amrex::DistributionMapping const& dm) {
        scalar(key+"/size",ba.size());
        for(int i=0;i<ba.size();++i) {
            for(int d=0;d<AMREX_SPACEDIM;++d) {
                auto const k=key+"/"+std::to_string(i)+"/"+std::to_string(d);
                scalar(k+"/lo",ba[i].smallEnd(d));scalar(k+"/hi",ba[i].bigEnd(d));
                scalar(k+"/type",ba[i].type(d));
            }
        }
        auto const& map=dm.ProcessorMap();host(key+"/owners",map.data(),map.size()*sizeof(int));
    }
public:
    static NativeStateBytes Capture(WarpX& w, bool fields=true, bool fixture_attributes=true, std::vector<amrex::MultiFab const*> const& excluded={}) {
        amrex::Gpu::synchronize();
        NativeStateBytes result;
        int species=0;
        for(auto const& owner:w.GetPartContainer()) {
            auto const& pc=*owner;
            auto const prefix="particle/"+std::to_string(species++);
            result.string(prefix+"/name",owner->getName());
            auto const names=pc.GetRealSoANames(),ints=pc.GetIntSoANames();
            result.scalar(prefix+"/real_components",names.size());
            result.scalar(prefix+"/int_components",ints.size());
            for(std::size_t c=0;c<names.size();++c)result.string(prefix+"/real_name/"+std::to_string(c),names[c]);
            for(std::size_t c=0;c<ints.size();++c)result.string(prefix+"/int_name/"+std::to_string(c),ints[c]);
            auto const& levels=pc.GetParticles();
            result.scalar(prefix+"/levels",levels.size());
            for(int lev=0;lev<levels.size();++lev) {
                auto const level=prefix+"/level/"+std::to_string(lev);
                result.boxes(level+"/boxes",pc.ParticleBoxArray(lev),pc.ParticleDistributionMap(lev));
                result.scalar(level+"/tiles",levels[lev].size());
                for(auto const& [index,tile]:levels[lev]) {
                    auto const key=level+"/tile/"+std::to_string(index.first)+"/"+std::to_string(index.second);
                    auto const& soa=tile.GetStructOfArrays();
                    result.scalar(key+"/real_count",soa.numRealParticles());
                    result.scalar(key+"/neighbor_count",soa.numNeighborParticles());
                    auto const& ids=soa.GetIdCPUData();result.device(key+"/idcpu",ids.data(),ids.size());
                    for(int c=0;c<soa.NumRealComps();++c) {
                        if(!fixture_attributes && names[c].starts_with("wall_"))continue;
                        auto const& data=soa.GetRealData(c);result.device(key+"/real/"+std::to_string(c),data.data(),data.size());
                    }
                    for(int c=0;c<soa.NumIntComps();++c) {
                        auto const& data=soa.GetIntData(c);result.device(key+"/int/"+std::to_string(c),data.data(),data.size());
                    }
                }
            }
        }
        result.scalar("particle/species",species);
        auto names=fields ? w.m_fields.list() : std::vector<std::string>{};
        std::erase_if(names,[&](std::string const& n){return std::find(excluded.begin(),excluded.end(),w.m_fields.internal_get(n))!=excluded.end();});
        for(auto const& name:names) {
            auto const* f=w.m_fields.internal_get(name);
            auto const key="field/"+name;
            result.boxes(key+"/boxes",f->boxArray(),f->DistributionMap());
            result.scalar(key+"/components",f->nComp());
            for(int d=0;d<AMREX_SPACEDIM;++d)result.scalar(key+"/ghost/"+std::to_string(d),f->nGrowVect()[d]);
            for(amrex::MFIter mfi(*f);mfi.isValid();++mfi) {
                auto const& fab=(*f)[mfi];
                result.device(key+"/fab/"+std::to_string(mfi.index()),fab.dataPtr(),fab.size());
            }
        }
        return result;
    }
    void Write(std::string const& name) const {
        std::ofstream out(name+".rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".bin",std::ios::binary);
        for(auto const& [key,value]:m_records) {
            std::uint64_t n=key.size(),m=value.size();
            out.write(reinterpret_cast<char const*>(&n),sizeof(n));
            out.write(reinterpret_cast<char const*>(&m),sizeof(m));
            out.write(key.data(),n);out.write(reinterpret_cast<char const*>(value.data()),m);
        }
        out.close();AMREX_ALWAYS_ASSERT(bool(out));
    }
    bool IsSolverParticleScratch(std::string const& key) const {
        if(!key.starts_with("particle/"))return false;
        auto const species_end=key.find('/',9);
        if(species_end==std::string::npos)return false;
        auto const real=key.rfind("/real/"),integer=key.rfind("/int/");
        if(real==std::string::npos && integer==std::string::npos)return false;
        bool const is_real=real!=std::string::npos;
        auto const index=key.substr((is_real?real:integer)+(is_real?6:5));
        auto const entry=m_records.find(key.substr(0,species_end)+(is_real?"/real_name/":"/int_name/")+index);
        if(entry==m_records.end())return false;
        std::string const name(entry->second.begin(),entry->second.end());
        if(!is_real)return name=="esirkepov_chord_valid" || name=="diagnostic_impulse_valid";
        return name=="x_n" || name=="y_n" || name=="z_n" || name=="ux_n" || name=="uy_n" || name=="uz_n" ||
            name=="diagnostic_du_x" || name=="diagnostic_du_y" || name=="diagnostic_du_z" ||
            name=="esirkepov_chord_x" || name=="esirkepov_chord_y" || name=="esirkepov_chord_z";
    }
    void RequireUnchanged(WarpX& w,std::string const& label,
                          bool fields=true,bool fixture_attributes=true,std::vector<amrex::MultiFab const*> const& excluded={},bool permit_particle_solver_scratch=false) const {
        auto const now=Capture(w,fields,fixture_attributes,excluded);
        amrex::Long changed=0,bytes=0,scratch_changed=0;
        for(auto const& [name,value]:now.m_records) if(!m_records.contains(name)) {
            ++changed;
            amrex::AllPrint()<<"NATIVE_STATE_NEW_RECORD phase="<<label<<" record="<<name<<"\n";
        }
        for(auto const& [name,value]:m_records) {
            bytes+=static_cast<amrex::Long>(value.size());
            auto const p=now.m_records.find(name);
            if(p==now.m_records.end() || p->second.size()!=value.size() ||
               (!value.empty() && std::memcmp(value.data(),p->second.data(),value.size())!=0)) {
                bool const scratch=permit_particle_solver_scratch && IsSolverParticleScratch(name);
                if(scratch) ++scratch_changed;else ++changed;
                amrex::AllPrint()<<"NATIVE_STATE_CHANGED phase="<<label<<" record="<<name<<" solver_scratch="<<scratch<<"\n";
            }
        }
        amrex::Long records=static_cast<amrex::Long>(m_records.size());
        amrex::ParallelDescriptor::ReduceLongSum(changed);
        amrex::ParallelDescriptor::ReduceLongSum(scratch_changed);
        amrex::ParallelDescriptor::ReduceLongSum(records);
        amrex::ParallelDescriptor::ReduceLongSum(bytes);
        amrex::Print()<<"NATIVE_STATE_BYTES phase="<<label<<" records="<<records
            <<" compared_bytes="<<bytes<<" changed_records="<<changed<<" changed_solver_scratch="<<scratch_changed<<std::endl;
        AMREX_ALWAYS_ASSERT(changed==0);
    }
};
} // namespace radial_fixture
#endif
