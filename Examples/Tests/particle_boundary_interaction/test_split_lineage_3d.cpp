/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "WarpX.H"

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <limits>
#include <vector>

namespace {
struct Record {
    std::uint64_t id;
    std::array<amrex::ParticleReal, PIdx::nattribs> real;
};
std::vector<Record> capture (PhysicalParticleContainer& pc)
{
    std::vector<Record> result;
    for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
        auto const a = pti.GetParticleTile().getConstParticleTileData();
        auto const n = pti.numParticles();
        std::vector<std::uint64_t> ids(n);
        std::array<std::vector<amrex::ParticleReal>, PIdx::nattribs> values;
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, a.m_idcpu, a.m_idcpu+n, ids.begin());
        for (int c = 0; c < PIdx::nattribs; ++c) {
            values[c].resize(n);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, a.m_rdata[c], a.m_rdata[c]+n,
                             values[c].begin());
        }
        for (long i = 0; i < n; ++i) {
            if (!amrex::ConstParticleIDWrapper{ids[i]}.is_valid()) { continue; }
            Record row; row.id = ids[i];
            for (int c = 0; c < PIdx::nattribs; ++c) { row.real[c] = values[c][i]; }
            result.push_back(row);
        }
    }
    std::sort(result.begin(), result.end(), [](Record const& a, Record const& b) {
        return a.id < b.id;
    });
    return result;
}
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    int exit_code = 0;
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& pc = dynamic_cast<PhysicalParticleContainer&>(
            w.GetPartContainer().GetParticleContainer(0));
        amrex::MultiFab rho(amrex::convert(pc.ParticleBoxArray(0),
            amrex::IntVect::TheNodeVector()), pc.ParticleDistributionMap(0), 1, 1);
        rho.setVal(1.0);
        auto const plo = w.Geom(0).ProbLoArray();
        auto const dx = w.Geom(0).CellSizeArray();
        auto const domain = w.Geom(0).Domain();
        auto const dlo = domain.smallEnd();
        auto const lengths = domain.length();
        int const cpu = amrex::ParallelDescriptor::MyProc();
        amrex::Long constexpr first_id = 10000000;
        std::vector<Record> reference;
        amrex::Long reference_next = 0;
        for (int pass = 0; pass < 3; ++pass) {
            pc.GetParticles(0).clear();
            std::vector<std::pair<int,amrex::Box>> boxes;
            for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
                boxes.emplace_back(mfi.index(), pc.ParticleBoxArray(0)[mfi.index()]);
            }
            if (pass == 2) { std::reverse(boxes.begin(), boxes.end()); }
            amrex::Long expected_parents = 0, expected_children = 0;
            for (auto const& [grid,box] : boxes) {
                auto const lo = box.smallEnd();
                auto const size = box.length();
                for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
                    for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                        for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                            int const cell = (i-dlo[0])+lengths[0]*
                                ((j-dlo[1])+lengths[1]*(k-dlo[2]));
                            int const count = 1 << (cell%6);
                            expected_parents += count;
                            expected_children += count*(count < 30 ? 1+29/count : 1);
                        }
                    }
                }
                int const n = int(box.numPts())*32;
                auto& tile = pc.DefineAndReturnParticleTile(0, grid, 0);
                tile.resize(n);
                auto const a = tile.getParticleTileData();
                amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int position) {
                    int const logical = pass == 0 ? position :
                        (pass == 1 ? n-1-position : (17*position+23)%n);
                    int const cell = logical/32;
                    int const parent = logical%32;
                    int const i = lo[0]+cell%size[0];
                    int const j = lo[1]+(cell/size[0])%size[1];
                    int const k = lo[2]+cell/(size[0]*size[1]);
                    int const global_cell = (i-dlo[0])+lengths[0]*
                        ((j-dlo[1])+lengths[1]*(k-dlo[2]));
                    int const count = 1 << (global_cell%6);
                    amrex::Long const parent_id = 1+32*amrex::Long(global_cell)+parent;
                    int const factor = count < 30 ? 1+29/count : 1;
                    for (int c = 0; c < PIdx::nattribs; ++c) { a.m_rdata[c][position] = 0; }
                    a.m_rdata[PIdx::x][position] = plo[0]+(i-dlo[0]+(parent+0.5)/count)*dx[0];
                    a.m_rdata[PIdx::y][position] = plo[1]+(j-dlo[1]+0.5)*dx[1];
                    a.m_rdata[PIdx::z][position] = plo[2]+(k-dlo[2]+0.5)*dx[2];
                    a.m_rdata[PIdx::w][position] = factor;
                    // The unchanged velocity is an independent parent-lineage tag.
                    a.m_rdata[PIdx::ux][position] = parent_id*1024;
                    a.m_rdata[PIdx::uy][position] = cpu+1;
                    a.m_rdata[PIdx::uz][position] = -parent_id*512;
                    a.m_idcpu[position] = parent < count ?
                        amrex::SetParticleIDandCPU(parent_id,cpu) : amrex::ParticleIdCpus::Invalid;
                });
            }
            amrex::Gpu::streamSynchronize();
            PhysicalParticleContainer::ParticleType::NextID(first_id);
            double const begin = amrex::second();
            pc.SplitDepletedBand(rho, 0.5, 1.5, 0);
            double elapsed = amrex::second()-begin;
            pc.Redistribute();
            auto const rows = capture(pc);
            amrex::Long wrong = 0, duplicate = 0, physical_wrong = 0;
            std::map<amrex::Long,int> descendants;
            std::map<amrex::Long,std::array<double,3>> centroids;
            if (amrex::Long(rows.size()) != expected_children) { ++physical_wrong; }
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (i > 0 && rows[i].id == rows[i-1].id) { ++duplicate; }
                auto const& a = rows[i].real;
                auto const parent = amrex::Long(a[PIdx::ux]/1024);
                int const cell = int((parent-1)/32);
                int const count = 1 << (cell%6);
                int const ordinal = int((parent-1)%32);
                if (ordinal >= count || a[PIdx::w] != 1 ||
                    a[PIdx::uz] != -parent*512 || a[PIdx::uy] != cpu+1) { ++physical_wrong; }
                ++descendants[parent];
                centroids[parent][0] += a[PIdx::x];
                centroids[parent][1] += a[PIdx::y];
                centroids[parent][2] += a[PIdx::z];
            }
            if (amrex::Long(descendants.size()) != expected_parents) { ++physical_wrong; }
            for (auto const& [parent,count_observed] : descendants) {
                int const count = 1 << (int((parent-1)/32)%6);
                int const factor = count < 30 ? 1+29/count : 1;
                if (count_observed != factor) { ++physical_wrong; }
                int const cell = int((parent-1)/32);
                std::array<double,3> expected = {
                    plo[0]+(cell%lengths[0]+(int((parent-1)%32)+0.5)/count)*dx[0],
                    plo[1]+((cell/lengths[0])%lengths[1]+0.5)*dx[1],
                    plo[2]+(cell/(lengths[0]*lengths[1])+0.5)*dx[2]};
                for (int d = 0; d < 3; ++d) {
                    double const scale = std::max(std::abs(expected[d]),double(dx[d]));
                    if (std::abs(centroids[parent][d]/count_observed-expected[d]) >
                        64*std::numeric_limits<amrex::ParticleReal>::epsilon()*scale) {
                        ++physical_wrong;
                    }
                }
            }
            auto const next = PhysicalParticleContainer::ParticleType::NextID();
            if (pass == 0) { reference = rows; reference_next = next; }
            else {
                if (rows.size() != reference.size() || next != reference_next) { ++wrong; }
                else {
                    for (std::size_t i = 0; i < rows.size(); ++i) {
                        if (rows[i].id != reference[i].id || rows[i].real != reference[i].real) {
                            ++wrong;
                        }
                    }
                }
            }
            amrex::ParallelDescriptor::ReduceLongSum(wrong);
            amrex::ParallelDescriptor::ReduceLongSum(duplicate);
            amrex::ParallelDescriptor::ReduceLongSum(physical_wrong);
            amrex::ParallelDescriptor::ReduceRealMax(elapsed);
            amrex::Print() << std::setprecision(17) << "LINEAGE_RESULT pass=" << pass
                << " identity_mismatches=" << wrong << " duplicate_ids=" << duplicate
                << " physical_mismatches=" << physical_wrong << " split_seconds=" << elapsed << '\n';
            if (wrong || duplicate || physical_wrong) { exit_code = 1; }
        }
        amrex::Print() << (exit_code ? "LINEAGE_PERMUTATION_FAIL\n" : "LINEAGE_PERMUTATION_PASS\n");
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
    return exit_code;
}
