#include "Initialization/WarpXInit.H"
#include "Utils/NativeParticleIdState.H"
#include <AMReX_ParmParse.H>
#include <filesystem>
#include <fstream>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        namespace ids = warpx_particle_ids;
        auto const original = ids::Capture();
        auto const rank = amrex::ParallelDescriptor::MyProc();
        auto const name = amrex::Concatenate("probe-", rank, 6);
        auto state = original;
        state.physical_next = 12345 + 131*rank;
        state.marker_next = 67890 + 379*rank;
        ids::Restore(state);
        AMREX_ALWAYS_ASSERT(ids::Payload(ids::Capture()) == ids::Payload(state));
        ids::Write(name + ".txt", state);
        AMREX_ALWAYS_ASSERT(ids::Payload(ids::Read(name + ".txt")) == ids::Payload(state));
        AMREX_ALWAYS_ASSERT(ids::Payload(ids::Capture()) == ids::Payload(state));

        int rejected = 0;
        for (int control = 0; control < 6; ++control) {
            auto bad = state;
            if (control == 0) { bad.physical_next = 0; }
            if (control == 1) { bad.marker_next = 0; }
            if (control == 2) { ++bad.rank; }
            if (control == 3) { ++bad.ranks; }
            if (control == 4) { ++bad.physical_components; }
            if (control == 5) { ++bad.marker_components; }
            try { ids::Restore(bad); }
            catch (std::runtime_error const&) { ++rejected; }
            AMREX_ALWAYS_ASSERT(ids::Payload(ids::Capture()) == ids::Payload(state));
        }
        for (int control = 0; control < 3; ++control) {
            auto const file = name + "-bad-" + std::to_string(control) + ".txt";
            std::ofstream stream(file);
            if (control == 0) { stream << "WarpXNativeParticleIDs 1\n"; }
            if (control == 1) { stream << "WarpXNativeParticleIDs 1\n" << ids::Payload(state) << "0\n"; }
            if (control == 2) {
                stream << "WarpXNativeParticleIDs 1\n" << ids::Payload(state)
                       << warpx_rng::LayoutDigest(ids::Payload(state)) << "\ntrailing\n";
            }
            stream.close();
            try { auto const bad = ids::Read(file); amrex::ignore_unused(bad); }
            catch (std::runtime_error const&) { ++rejected; }
        }
        try { ids::Write(name + ".txt", state); }
        catch (std::runtime_error const&) { ++rejected; }
        AMREX_ALWAYS_ASSERT(rejected == 10);

        if (amrex::ParallelDescriptor::IOProcessor()) { std::filesystem::create_directory("checkpoint"); }
        amrex::ParallelDescriptor::Barrier();
        ids::WriteCheckpoint("checkpoint", "fixed-layout");
        ids::Physical::NextID(999999);
        ids::Marker::NextID(888888);
        std::string mode = "valid";
        amrex::ParmParse("test").query("mode", mode);
        if (mode == "missing") {
            ids::ReadCheckpoint("missing-checkpoint", "fixed-layout");
            amrex::Abort("Required missing-ID control incorrectly accepted");
        }
        ids::ReadCheckpoint("checkpoint", mode == "layout" ? "wrong-layout" : "fixed-layout");
        AMREX_ALWAYS_ASSERT(ids::Payload(ids::Capture()) == ids::Payload(state));
        AMREX_ALWAYS_ASSERT(ids::Physical::NextID() == state.physical_next);
        AMREX_ALWAYS_ASSERT(ids::Marker::NextID() == state.marker_next);
        ids::Restore(original);
        amrex::Print() << "NATIVE_PARTICLE_IDS_PASS ranks=" << original.ranks
                       << " negative_controls=" << rejected << '\n';
    }
    warpx::initialization::finalize_external_libraries();
}
