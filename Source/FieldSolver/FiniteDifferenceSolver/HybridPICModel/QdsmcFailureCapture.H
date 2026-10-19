/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_QDSMC_FAILURE_CAPTURE_H_
#define WARPX_QDSMC_FAILURE_CAPTURE_H_
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/** Bounded observational dumps. Configuration is separate from physics inputs.
 * Registry lifetime is per process/output directory, including repeated WarpX
 * instances. A new output directory starts a new registry; disabling does not
 * reset its limits. All ranks must use the same environment configuration. */
class QdsmcFailureCapture {
public:
    using Field = std::pair<std::string, amrex::MultiFab const*>;
    static QdsmcFailureCapture* FromEnvironment () {
        char const* root=std::getenv("WARPX_CONDUCTION_CAPTURE_DIR");
        if (!root || !*root) { return nullptr; }
        static std::map<std::string,QdsmcFailureCapture> writers;
        auto& writer=writers[root];
        writer.m_root=root;
        return &writer;
    }
    static std::unique_ptr<amrex::MultiFab> RealMask (amrex::iMultiFab const& mask) {
        auto out=std::make_unique<amrex::MultiFab>(mask.boxArray(),mask.DistributionMap(),1,0);
        for(amrex::MFIter mfi(*out);mfi.isValid();++mfi) {
            auto const a=out->array(mfi);auto const b=mask.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
                a(i,j,k)=amrex::Real(b(i,j,k));
            });
        }
        return out;
    }
    long BeginCall () { return ++m_calls; }
    bool Wants (std::string const& key, bool terminal=false) const {
        return terminal ? !m_terminal && m_count<8
                        : m_count<7 && m_seen.count(key)==0;
    }
    static std::string Quote (std::string const& text) {
        std::ostringstream out;out<<'"';
        for(unsigned char c:text) {
            if(c=='"'||c=='\\')out<<'\\'<<char(c);
            else if(c=='\n')out<<"\\n";
            else if(c=='\r')out<<"\\r";
            else if(c=='\t')out<<"\\t";
            else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
            else out<<char(c);
        }
        out<<'"';return out.str();
    }
    static std::string Number (amrex::Real value) {
        if(!std::isfinite(value))return "null";
        std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();
    }
    void Write (std::string const& key, bool terminal, long call,
                std::string const& metadata,
                std::vector<Field> const& fields) {
        if(!Wants(key,terminal))return;
        ++m_count;m_seen.insert(key);if(terminal)m_terminal=true;
        std::ostringstream name;name<<m_root<<"/event"<<std::setw(2)<<std::setfill('0')<<m_count
                                   <<"-call"<<std::setw(10)<<call;
        std::string const path=name.str();
        if(amrex::ParallelDescriptor::IOProcessor()) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!std::filesystem::exists(path),
                "Conduction capture output exists: "+path);
            AMREX_ALWAYS_ASSERT(amrex::UtilCreateDirectory(path,0750));
        }
        amrex::ParallelDescriptor::Barrier();
        std::ostringstream inventory;inventory<<'[';bool first=true;
        auto const header=amrex::VisMF::GetHeaderVersion();
        amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::NoFabHeader_v1);
        for(auto const& item:fields) {
            if(!item.second)continue;
            auto const& mf=*item.second;
            amrex::VisMF::Write(mf,path+"/"+item.first);
            if(!first)inventory<<',';first=false;
            inventory<<"{\"name\":"<<Quote(item.first)<<",\"components\":"<<mf.nComp()
                     <<",\"stagger\":[";
            for(int d=0;d<AMREX_SPACEDIM;++d){if(d)inventory<<',';inventory<<mf.ixType()[d];}
            inventory<<"],\"ghost_cells\":[";
            for(int d=0;d<AMREX_SPACEDIM;++d){if(d)inventory<<',';inventory<<mf.nGrowVect()[d];}
            inventory<<"]}";
        }
        amrex::VisMF::SetHeaderVersion(header);inventory<<']';
        if(amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream f(path+"/CAPTURE.json",std::ios::out);
            f<<"{\"format\":1,\"event\":"<<m_count<<",\"call\":"<<call
             <<",\"key\":"<<Quote(key)<<",\"terminal\":"<<(terminal?"true":"false")
             <<",\"mpi_ranks\":"<<amrex::ParallelDescriptor::NProcs()
             <<",\"restart_checkpoint\":false,\"maximum_events\":8,"
             <<metadata<<",\"fields\":"<<inventory.str()<<"}\n";
            f.close();AMREX_ALWAYS_ASSERT(f.good());
        }
        amrex::ParallelDescriptor::Barrier();
        amrex::Print()<<"CONDUCTION_FIELD_CAPTURE "<<path<<" "<<key<<"\n";
    }
private:
    std::string m_root;
    std::set<std::string> m_seen;
    int m_count=0;
    long m_calls=0;
    bool m_terminal=false;
};
#endif
