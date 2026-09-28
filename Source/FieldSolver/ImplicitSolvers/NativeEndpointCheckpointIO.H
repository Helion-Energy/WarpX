/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_ENDPOINT_CHECKPOINT_IO_H_
#define WARPX_NATIVE_ENDPOINT_CHECKPOINT_IO_H_
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_VisMF.H>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace warpx::darwin::endpoint_checkpoint {
// Accepted-boundary I/O only. No field is materialized and no residual path
// performs a host transfer. Checksums detect damaged numeric records; they
// never substitute for fresh operator/configuration/physical validation.
inline std::uint64_t Checksum(unsigned char const* bytes,std::size_t size) {
    std::uint64_t value=14695981039346656037ULL;
    for(std::size_t i=0;i<size;++i)value=(value^bytes[i])*1099511628211ULL;
    return value;
}
inline void WriteText(std::string const& path,std::string const& body) {
    bool valid=true;
    if(amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream out(path,std::ios::binary);
        out<<"WarpXEndpointNumeric 1\n"<<body.size()<<' '
           <<Checksum(reinterpret_cast<unsigned char const*>(body.data()),body.size())<<'\n';
        out.write(body.data(),static_cast<std::streamsize>(body.size()));out.close();valid=bool(out);
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,"Unable to write accepted endpoint checkpoint: "+path);
}
inline std::string ReadText(std::string const& path) {
    amrex::Vector<char> bytes;amrex::ParallelDescriptor::ReadAndBcastFile(path,bytes);
    std::istringstream in(std::string(bytes.data()));
    std::string magic;int version=0;std::size_t size=0;std::uint64_t checksum=0;
    in>>magic>>version>>size>>checksum;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in)&&magic=="WarpXEndpointNumeric"&&version==1&&
        size>0&&size<=1048576&&in.get()=='\n',"Invalid accepted endpoint numeric header: "+path);
    std::string body(size,'\0');in.read(body.data(),static_cast<std::streamsize>(size));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in)&&in.peek()==std::char_traits<char>::eof()&&
        checksum==Checksum(reinterpret_cast<unsigned char const*>(body.data()),body.size()),
        "Corrupt accepted endpoint numeric record: "+path);
    return body;
}
inline std::vector<int> FieldChecksum(amrex::MultiFab const& field) {
    // Four 16-bit lanes per global FAB make the reduction independent of
    // DistributionMapping without signed integer overflow or rank-local hashes.
    std::vector<int> values(4*field.boxArray().size(),0),owners(field.boxArray().size(),0);
    for(amrex::MFIter it(field);it.isValid();++it) {
        auto const& fab=field[it];std::vector<amrex::Real> host(fab.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,fab.dataPtr(),fab.dataPtr()+fab.size(),host.begin());
        auto const hash=Checksum(reinterpret_cast<unsigned char const*>(host.data()),host.size()*sizeof(amrex::Real));
        for(int k=0;k<4;++k)values[4*it.index()+k]=int((hash>>(16*k))&65535ULL);
        owners[it.index()]=1;
    }
    amrex::ParallelDescriptor::ReduceIntMax(values.data(),int(values.size()));
    amrex::ParallelDescriptor::ReduceIntSum(owners.data(),int(owners.size()));
    for(int n:owners)AMREX_ALWAYS_ASSERT(n==1);
    return values;
}
inline void Write(std::string const& path,amrex::MultiFab const& field) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field.is_finite(0,field.nComp(),field.nGrowVect()),
        "Nonfinite accepted endpoint checkpoint field: "+path);
    auto const checksum=FieldChecksum(field);
    amrex::VisMF::Write(field,path);
    std::ostringstream out;out<<checksum.size()<<'\n';for(int x:checksum)out<<x<<'\n';
    WriteText(path+".checksum",out.str());
}
inline void Allocate(std::string const& path,amrex::MultiFab& field,
                     amrex::MultiFab const& maximum_shape,int components=1) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::VisMF::Exist(path),"Missing endpoint numeric array: "+path);
    amrex::Vector<char> bytes;amrex::VisMF::ReadFAHeader(path,bytes);
    std::istringstream in(std::string(bytes.data()));amrex::VisMF::Header saved;in>>saved;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in)&&saved.m_ba==maximum_shape.boxArray()&&
        saved.m_ncomp==components&&saved.m_ngrow.allGE(amrex::IntVect(0))&&
        maximum_shape.nGrowVect().allGE(saved.m_ngrow),"Invalid endpoint numeric allocation: "+path);
    field.define(saved.m_ba,maximum_shape.DistributionMap(),components,saved.m_ngrow);
}
inline void Read(std::string const& path,amrex::MultiFab& field) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::VisMF::Exist(path),
        "Missing accepted endpoint checkpoint field: "+path);
    amrex::Vector<char> bytes;amrex::VisMF::ReadFAHeader(path,bytes);
    std::istringstream header(std::string(bytes.data()));amrex::VisMF::Header saved;header>>saved;
    auto const boxes=field.boxArray();auto const owners=field.DistributionMap();
    auto const ghosts=field.nGrowVect();int const components=field.nComp();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(header)&&saved.m_ba==boxes&&
        saved.m_ncomp==components&&saved.m_ngrow==ghosts,
        "Accepted endpoint checkpoint layout mismatch: "+path);
    std::istringstream in(ReadText(path+".checksum"));std::size_t n=0;in>>n;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in)&&n==4*boxes.size(),
        "Invalid accepted endpoint field checksum count: "+path);
    std::vector<int> expected(n);for(int& x:expected)in>>x;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in),"Truncated endpoint field checksums: "+path);
    in>>std::ws;AMREX_ALWAYS_ASSERT_WITH_MESSAGE(in.eof(),"Trailing endpoint field checksums: "+path);
    amrex::VisMF::Read(field,path);
    bool layout=field.boxArray()==boxes&&field.DistributionMap()==owners&&
        field.nGrowVect()==ghosts&&field.nComp()==components;
    amrex::ParallelDescriptor::ReduceBoolAnd(layout);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(layout,"Changed endpoint checkpoint layout: "+path);
    bool const finite=field.is_finite(0,components,ghosts);
    auto const actual=FieldChecksum(field);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(finite&&actual==expected,"Damaged accepted endpoint checkpoint field: "+path);
}
} // namespace warpx::darwin::endpoint_checkpoint
#endif
