#ifndef WARPX_NATIVE_PARTICLE_ID_STATE_H_
#define WARPX_NATIVE_PARTICLE_ID_STATE_H_

#include "Particles/WarpXParticleContainer.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Utils/NativeRngState.H"

#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// AMReX writes the MPI maximum NextID into each particle header. That maintains
// uniqueness after redistribution, but cannot reproduce the rank-local future
// IDs of a continuation on the same MPI/box layout. Save both distinct SoA
// counters after a synchronous checkpoint, and restore after all initialization.
namespace warpx_particle_ids {
using Physical = WarpXParticleContainer::ParticleType;
using Marker = QdsmcParticleContainer::ParticleType;

struct State {
    int ranks = 0, rank = 0;
    int physical_components = 0, marker_components = 0;
    amrex::Long physical_next = 0, marker_next = 0;
};

inline State Capture ()
{
    // Reading the public static state does not allocate an ID. Calling NextID()
    // to inspect it would mutate the state that this observer must preserve.
    return {amrex::ParallelDescriptor::NProcs(), amrex::ParallelDescriptor::MyProc(),
            Physical::NArrayReal, Marker::NArrayReal,
            Physical::the_next_id, Marker::the_next_id};
}

inline void Validate (State const& s)
{
    auto const expected = Capture();
    if (s.ranks != expected.ranks || s.rank != expected.rank ||
        s.physical_components != expected.physical_components ||
        s.marker_components != expected.marker_components ||
        s.physical_next <= 0 || s.marker_next <= 0 ||
        s.physical_next > amrex::LongParticleIds::LastParticleID ||
        s.marker_next > amrex::LongParticleIds::LastParticleID) {
        throw std::runtime_error("Native particle-ID rank/type/range mismatch");
    }
}

inline std::string Payload (State const& s)
{
    std::ostringstream text;
    text << s.ranks << ' ' << s.rank << ' ' << s.physical_components << ' '
         << s.marker_components << ' ' << s.physical_next << ' ' << s.marker_next << '\n';
    return text.str();
}

inline void Write (std::string const& file, State const& s)
{
    Validate(s);
    if (std::filesystem::exists(file)) {
        throw std::runtime_error("Refusing to overwrite native particle-ID state");
    }
    auto const payload = Payload(s);
    std::ofstream stream(file);
    stream << "WarpXNativeParticleIDs 1\n" << payload << warpx_rng::LayoutDigest(payload) << '\n';
    stream.close();
    if (!stream) { throw std::runtime_error("Writing native particle-ID state failed"); }
}

inline State Read (std::string const& file)
{
    std::ifstream stream(file);
    std::string magic;
    std::getline(stream, magic);
    State s;
    std::uint64_t digest = 0;
    stream >> s.ranks >> s.rank >> s.physical_components >> s.marker_components
           >> s.physical_next >> s.marker_next >> digest;
    if (!stream || magic != "WarpXNativeParticleIDs 1") {
        throw std::runtime_error("Invalid native particle-ID payload");
    }
    stream >> std::ws;
    if (!stream.eof() || digest != warpx_rng::LayoutDigest(Payload(s))) {
        throw std::runtime_error("Native particle-ID checksum or trailing-data mismatch");
    }
    Validate(s);
    return s;
}

inline void Restore (State const& s)
{
    Validate(s);
    Physical::NextID(s.physical_next);
    Marker::NextID(s.marker_next);
}

inline void WriteCheckpoint (std::string const& checkpoint, std::string const& layout)
{
    namespace fs = std::filesystem;
    fs::path const directory = fs::path(checkpoint) / "NativeParticleIDs";
    try {
        if (amrex::ParallelDescriptor::IOProcessor() &&
            (fs::exists(directory) || !fs::create_directory(directory))) {
            throw std::runtime_error("Native particle-ID directory exists or cannot be created");
        }
        amrex::ParallelDescriptor::Barrier();
        auto const state = Capture();
        auto const name = amrex::Concatenate("rank-", state.rank, 6) + ".txt";
        Write((directory / (name + ".partial")).string(), state);
        fs::rename(directory / (name + ".partial"), directory / name);
        amrex::ParallelDescriptor::Barrier();
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream header(directory / "Header.partial");
            header << "WarpXNativeParticleIDCheckpoint 1\n" << state.ranks << ' '
                   << warpx_rng::LayoutDigest(layout) << '\n';
            header.close();
            if (!header) { throw std::runtime_error("Writing particle-ID manifest failed"); }
            fs::rename(directory / "Header.partial", directory / "Header");
        }
        amrex::ParallelDescriptor::Barrier();
    } catch (std::exception const& error) {
        amrex::Abort(std::string("Native particle-ID checkpoint write: ") + error.what());
    }
}

inline bool ReadCheckpoint (std::string const& checkpoint, std::string const& layout)
{
    namespace fs = std::filesystem;
    fs::path const directory = fs::path(checkpoint) / "NativeParticleIDs";
    try {
        bool required = false;
        amrex::ParmParse("warpx").query("require_restart_rng_state", required);
        if (!fs::exists(directory)) {
            if (required) { throw std::runtime_error("Required native particle-ID state is absent"); }
            amrex::Print() << "NATIVE_PARTICLE_IDS legacy_checkpoint_without_state\n";
            return false;
        }
        std::ifstream header(directory / "Header");
        std::string magic;
        int ranks = 0;
        std::uint64_t digest = 0;
        std::getline(header, magic);
        header >> ranks >> digest;
        if (!header || magic != "WarpXNativeParticleIDCheckpoint 1" ||
            ranks != amrex::ParallelDescriptor::NProcs() ||
            digest != warpx_rng::LayoutDigest(layout)) {
            throw std::runtime_error("Incomplete particle-ID checkpoint or changed MPI/box layout");
        }
        header >> std::ws;
        if (!header.eof()) { throw std::runtime_error("Trailing particle-ID manifest data"); }
        auto const name = amrex::Concatenate("rank-", amrex::ParallelDescriptor::MyProc(), 6) + ".txt";
        auto const state = Read((directory / name).string());
        // All ranks validate before any rank mutates either live counter.
        amrex::ParallelDescriptor::Barrier();
        Restore(state);
        amrex::ParallelDescriptor::Barrier();
        amrex::Print() << "NATIVE_PARTICLE_IDS restored_physical_and_marker_counters ranks=" << ranks << '\n';
        return true;
    } catch (std::exception const& error) {
        amrex::Abort(std::string("Native particle-ID checkpoint restore: ") + error.what());
    }
    return false;
}
}
#endif
