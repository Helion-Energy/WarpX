/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 * Test-only common-state replay of the existing native viscous source.
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace {
using amrex::MultiFab;
using warpx::fields::FieldType;
using Field = ablastr::fields::VectorField;

void mkdir_all (std::string const& path)
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::filesystem::create_directories(path);
    }
    amrex::ParallelDescriptor::Barrier();
}

template <typename T>
void binary (std::ofstream& f, T const* data, std::size_t count)
{
    f.write(reinterpret_cast<char const*>(data), std::streamsize(count * sizeof(T)));
}

template <typename Container>
void particle_dump (Container const& pc, std::string const& path)
{
    std::ofstream f(path + "_rank" + std::to_string(amrex::ParallelDescriptor::MyProc()) +
                    ".bin", std::ios::binary);
    std::uint64_t header[4] = {std::uint64_t(pc.NumRealComps()),
                              std::uint64_t(pc.NumIntComps()), sizeof(amrex::ParticleReal),
                              std::uint64_t(pc.GetParticles(0).size())};
    binary(f, header, 4);
    for (auto const& [key, tile] : pc.GetParticles(0)) {
        auto const& soa = tile.GetStructOfArrays();
        auto const np = tile.numParticles();
        std::int64_t shape[3] = {key.first, key.second, np};
        binary(f, shape, 3);
        auto const& ids = soa.GetIdCPUData();
        amrex::Gpu::PinnedVector<std::uint64_t> id(np);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, ids.begin(), ids.begin()+np, id.begin());
        binary(f, id.data(), np);
        for (int n=0; n<pc.NumRealComps(); ++n) {
            auto const& a = soa.GetRealData(n);
            amrex::Gpu::PinnedVector<amrex::ParticleReal> v(np);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, a.begin(), a.begin()+np, v.begin());
            binary(f, v.data(), np);
        }
        for (int n=0; n<pc.NumIntComps(); ++n) {
            auto const& a = soa.GetIntData(n);
            amrex::Gpu::PinnedVector<int> v(np);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, a.begin(), a.begin()+np, v.begin());
            binary(f, v.data(), np);
        }
    }
    AMREX_ALWAYS_ASSERT(f.good());
}

void snapshot (WarpX& w, std::string const& path, bool checkpoint)
{
    mkdir_all(path + "/fields");
    auto const names = w.m_fields.list();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream f(path + "/field_names.txt");
        for (auto const& name : names) { f << name << '\n'; }
        std::ofstream clock(path + "/clock.txt");
        clock << std::setprecision(17) << w.getistep(0) << " " << w.gett_new(0)
              << " " << w.getdt(0) << '\n';
    }
    for (auto const& name : names) {
        amrex::VisMF::Write(*w.m_fields.internal_get(name), path + "/fields/" + name);
    }
    auto& mpc = w.GetPartContainer();
    for (auto const& name : mpc.GetSpeciesNames()) {
        auto& pc = mpc.GetParticleContainerFromName(name);
        particle_dump(pc, path + "/particles_" + name);
        if (checkpoint) { pc.Checkpoint(path, name); }
    }
    auto& markers = *w.get_pointer_HybridPICModel()->m_qdsmc_pc;
    particle_dump(markers, path + "/markers");
    if (checkpoint) { markers.Checkpoint(path, "raw_markers"); }
}

void restore_supplement (WarpX& w, std::string const& path)
{
    std::ifstream f(path + "/field_names.txt");
    AMREX_ALWAYS_ASSERT(f.good());
    std::string name;
    while (std::getline(f, name)) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(w.m_fields.internal_has(name), name);
        auto& target = *w.m_fields.internal_get(name);
        // Read each saved FAB directly into its existing distribution. A
        // ParallelCopy with grown source boxes can mix neighboring ghost
        // values and is not an exact serialized-state restore.
        amrex::VisMF::Read(target, path + "/fields/" + name);
    }
    auto& mpc = w.GetPartContainer();
    for (auto const& species : mpc.GetSpeciesNames()) {
        auto& pc = mpc.GetParticleContainerFromName(species);
        pc.clearParticles();
        pc.Restart(path, species);
    }
    auto& markers = *w.get_pointer_HybridPICModel()->m_qdsmc_pc;
    markers.clearParticles();
    markers.Restart(path, "raw_markers");
}

amrex::Real integral (MultiFab const& mf, amrex::Geometry const& geom)
{
    MultiFab weighted(mf.boxArray(), mf.DistributionMap(), 1, 0);
    auto const vol = MakeQdsmcVolumeElement(geom, mf.ixType());
    for (amrex::MFIter mfi(weighted); mfi.isValid(); ++mfi) {
        auto const q = mf.const_array(mfi);
        auto const w = weighted.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            w(i,j,k) = q(i,j,k) * vol(i,j,k);
        });
    }
    return weighted.sum_unique(0, false, geom.periodicity());
}

amrex::Real thermal_energy (WarpX& w)
{
    auto const& hp = *w.get_pointer_HybridPICModel();
    auto const& te = *w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
    auto const& rho = *w.m_fields.get(FieldType::rho_fp,0);
    auto const* ped = hp.DensityPedestal(0);
    auto const floor = hp.m_n_floor;
    MultiFab u(te.boxArray(),te.DistributionMap(),1,0);
    for (amrex::MFIter mfi(u);mfi.isValid();++mfi) {
        auto const a=u.array(mfi);
        auto const t=te.const_array(mfi), n=rho.const_array(mfi);
        amrex::Array4<amrex::Real const> p;
        if(ped) { p=ped->const_array(mfi); }
        bool const has_ped=ped!=nullptr;
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            auto const ne=n(i,j,k)/PhysConst::q_e;
            auto const cap=ne+(has_ped?p(i,j,k)/PhysConst::q_e:0.0);
            a(i,j,k)=ne>floor?1.5*cap*PhysConst::kb*t(i,j,k):0.0;
        });
    }
    return integral(u,w.Geom(0));
}

amrex::Real staggered_work (WarpX& w, std::string const& path)
{
    auto& hp=*w.get_pointer_HybridPICModel();
    auto const E=w.m_fields.get_alldirs(FieldType::Efield_fp,0);
    auto J=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    auto const Ji=w.m_fields.get_alldirs(FieldType::current_fp,0);
    auto const B=w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
    auto const& rho=*w.m_fields.get(FieldType::rho_fp,0);
    auto const& pe=*w.m_fields.get(FieldType::hybrid_electron_pressure_fp,0);
    std::array<MultiFab,3> scratch,visc;
    for(int c=0;c<3;++c) {
        scratch[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,E[c]->nGrowVect());
        visc[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,1);
        scratch[c].setVal(0.0);visc[c].setVal(0.0);
    }
    Field tmp{&scratch[0],&scratch[1],&scratch[2]};
    Field ev{&visc[0],&visc[1],&visc[2]};
    w.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(tmp,J,Ji,B,rho,pe,
        w.GetEBUpdateEFlag()[0],0,&hp,true,true,nullptr,&ev);
    mkdir_all(path);
    amrex::Real result=0.0;
    for(int c=0;c<3;++c) {
        MultiFab product(J[c]->boxArray(),J[c]->DistributionMap(),1,0);
        MultiFab::Copy(product,*J[c],0,0,1,0);
        MultiFab::Multiply(product,visc[c],0,0,1,0);
        result+=integral(product,w.Geom(0));
        amrex::VisMF::Write(visc[c],path+"/Evisc"+std::to_string(c));
    }
    return result;
}
} // namespace

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        amrex::ParmParse pp("v1");
        int mode=0,steps=20,instrument=1;
        std::string restore;
        pp.query("mode",mode);pp.query("steps",steps);pp.query("instrument",instrument);
        pp.query("restore",restore);
        auto& w=WarpX::GetInstance();w.InitData();
        auto& hp=*w.get_pointer_HybridPICModel();
        // This is the frozen hot-PIC source-only discriminator, not an FRC.
        AMREX_ALWAYS_ASSERT(hp.m_include_electron_viscosity && hp.m_visc_in_ohms_law);
        AMREX_ALWAYS_ASSERT(hp.m_visc_heating_work && !hp.m_include_joule_heating);
        AMREX_ALWAYS_ASSERT(!hp.m_include_temperature_relaxation && !hp.m_hyper_res_heating);
        AMREX_ALWAYS_ASSERT(!hp.m_use_implicit_visc_temperature);
        AMREX_ALWAYS_ASSERT(hp.m_te_shunt_eV==0.0);
        if(mode==0) {
            w.Evolve(steps);
            snapshot(w,"common",true);
        } else {
            AMREX_ALWAYS_ASSERT(!restore.empty());
            restore_supplement(w,restore);
            hp.m_energy_budget=instrument!=0;
            snapshot(w,"before",false);
            if(mode==1) {
                amrex::Real const dt=0.5*w.getdt(0);
                amrex::Real const u0=thermal_energy(w);
                amrex::Real const clamp0=hp.m_visc_clamp_J;
                amrex::Real const bracket0=hp.m_ebud_visc_bulk+hp.m_ebud_visc_band;
                amrex::Real work=0.0;
                if(instrument) { work=staggered_work(w,"source_probe"); }
                hp.ApplyQdsmcEnergySources(0,dt,false);
                amrex::Real const u1=thermal_energy(w);
                amrex::Real const clamp=hp.m_visc_clamp_J-clamp0;
                amrex::Real const native_bracket=hp.m_ebud_visc_bulk+hp.m_ebud_visc_band-bracket0;
                auto const& qw=*w.m_fields.get("hybrid_qdsmc_visc_work_fp",0);
                auto const& qnu=*w.m_fields.get("hybrid_qdsmc_visc_heating_fp",0);
                amrex::Real const deposited_work=dt*integral(qw,w.Geom(0));
                amrex::Real const qnu_energy=dt*integral(qnu,w.Geom(0));
                amrex::Real const eps=std::numeric_limits<amrex::Real>::epsilon();
                amrex::Real const tolerance=128*eps*std::abs(u0)+2.e-11*std::abs(dt*work);
                if(instrument) {
                    AMREX_ALWAYS_ASSERT(std::abs((u1-u0)-(dt*work+clamp))<=tolerance);
                    AMREX_ALWAYS_ASSERT(std::abs(deposited_work-dt*work)<=tolerance);
                    AMREX_ALWAYS_ASSERT(std::abs(native_bracket-(u1-u0))<=tolerance);
                }
                amrex::Print()<<std::setprecision(17)<<"V1_BRACKET {\"instrument\":"<<instrument
                    <<",\"dt_s\":"<<dt<<",\"U0_J\":"<<u0<<",\"U1_J\":"<<u1
                    <<",\"dU_J\":"<<u1-u0<<",\"staggered_work_J\":"<<dt*work
                    <<",\"nodal_work_J\":"<<deposited_work<<",\"strain_J\":"<<qnu_energy
                    <<",\"clamp_J\":"<<clamp<<",\"native_bracket_J\":"<<native_bracket
                    <<",\"bound_J\":"<<tolerance<<"}\n";
            } else {
                AMREX_ALWAYS_ASSERT(mode==2);
                w.Evolve(steps);
            }
            snapshot(w,"after",false);
        }
        WarpX::ResetInstance();
    }
    warpx::initialization::finalize_external_libraries();
}
