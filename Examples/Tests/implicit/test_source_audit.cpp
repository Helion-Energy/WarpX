// Independent source-audit oracle and on/off particle-state witness.
// Physical eligibility remains disabled in both diagnostic modes.
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_Gpu.H>
#include <cstdint>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
using namespace amrex::literals;
using warpx::fields::FieldType;
std::array<amrex::Real,4> particle_stats(WarpXParticleContainer& pc) {
    amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> op;
    amrex::ReduceData<amrex::Real,amrex::Real,amrex::Real,amrex::Real> data(op);
    using Tuple=typename decltype(data)::Type;
    auto const mass=pc.getMass();
    for(WarpXParIter pti(pc,0);pti.isValid();++pti) {
        auto const* w=pti.GetAttribs(PIdx::w).dataPtr();
        auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();
        auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();
        auto const* uz=pti.GetAttribs(PIdx::uz).dataPtr();
        op.eval(pti.numParticles(),data,[=] AMREX_GPU_DEVICE(int i)->Tuple {
            auto const u2=ux[i]*ux[i]+uy[i]*uy[i]+uz[i]*uz[i];
            // Independent stable relativistic energy from raw momenta u=gamma*v.
            auto const rel=mass*u2/(1._rt+std::sqrt(1._rt+u2/(PhysConst::c*PhysConst::c)));
            return {0.5_rt*mass*w[i]*u2,w[i],w[i]*w[i],w[i]*rel};
        });
    }
    auto sums=data.value();std::array<amrex::Real,4> result{amrex::get<0>(sums),amrex::get<1>(sums),amrex::get<2>(sums),amrex::get<3>(sums)};
    for(auto& x:result)amrex::ParallelDescriptor::ReduceRealSum(x);
    return result;
}
void write_particles (WarpXParticleContainer& pc)
{
    std::ofstream f("particles-rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".bin",
                    std::ios::binary);
    for (WarpXParIter pti(pc,0); pti.isValid(); ++pti) {
        std::uint64_t const count=pti.numParticles();
        f.write(reinterpret_cast<char const*>(&count),sizeof(count));
        for (int component : {PIdx::w,PIdx::ux,PIdx::uy,PIdx::uz}) {
            auto const* first=pti.GetAttribs(component).dataPtr();
            std::vector<amrex::ParticleReal> host(count);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,first,first+count,host.begin());
            f.write(reinterpret_cast<char const*>(host.data()),
                    static_cast<std::streamsize>(count*sizeof(amrex::ParticleReal)));
        }
    }
    AMREX_ALWAYS_ASSERT(f.good());
}
int main(int argc,char** argv) {
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w=WarpX::GetInstance();w.InitData();
        auto& h=*w.get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT(!h.m_source_guard.enabled() && h.m_include_temperature_relaxation);
        amrex::ParmParse p("audit");amrex::Real dt=1.e-8_rt,n=1.e20_rt,nped=0,te0=100,nu=1.e7_rt,ti0=0;
        bool production_rate=false;
        p.query("dt",dt);p.query("n",n);p.query("nped",nped);p.query("Te_eV",te0);p.query("nu",nu);p.query("production_rate",production_rate);p.query("Ti_eV",ti0);
        auto& te=*w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
        auto& rho=*w.m_fields.get(FieldType::rho_fp,0);auto const qe=PhysConst::q_e,kb=PhysConst::kb,K=qe/kb;
        rho.setVal(n*qe);w.m_fields.get("rho_fp_ions",0)->setVal(n*qe);
        auto* ped=const_cast<amrex::MultiFab*>(h.DensityPedestal(0));AMREX_ALWAYS_ASSERT(ped);ped->setVal(nped*qe);
        te.setVal(te0*K);
        for(int c=0;c<3;++c) {
            w.m_fields.get_alldirs("current_fp_ions",0)[c]->setVal(0);
            w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0)[c]->setVal(0);
        }
        amrex::MultiFab ti(amrex::convert(te.boxArray(),amrex::IntVect::TheCellVector()),te.DistributionMap(),1,0);ti.setVal(ti0);
        std::map<std::string,amrex::MultiFab*> deposits{{"ions",&ti}};
        auto& ions=w.GetPartContainer().GetParticleContainerFromName("ions");
        auto const speed=std::sqrt(qe*ti0/ions.getMass());
        for(WarpXParIter pti(ions,0);pti.isValid();++pti) {
            auto* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto* uy=pti.GetAttribs(PIdx::uy).dataPtr();auto* uz=pti.GetAttribs(PIdx::uz).dataPtr();
            AMREX_ALWAYS_ASSERT(pti.numParticles()%8==0);
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(int i){
                ux[i]=(i&1)?speed:-speed;uy[i]=(i&2)?speed:-speed;uz[i]=(i&4)?speed:-speed;
            });
        }
        auto before=particle_stats(ions);
        AMREX_ALWAYS_ASSERT(std::abs(before[0]-1.5_rt*qe*ti0*before[1])<=1.e-12_rt*amrex::max(before[0],1._rt));
        auto const& geom=w.Geom(0);
#if defined(WARPX_DIM_RZ)
        AMREX_ALWAYS_ASSERT(ti0==0);
        auto const vol=MathConst::pi*(geom.ProbHi(0)*geom.ProbHi(0)-
                                      geom.ProbLo(0)*geom.ProbLo(0))*geom.ProbLength(1);
#else
        auto const vol=geom.ProbLength(0)*geom.ProbLength(1)*geom.ProbLength(2);
#endif
        AMREX_ALWAYS_ASSERT(std::abs(before[1]-n*vol)<=1.e-12_rt*n*vol);
        auto electron_energy=[&](){return te.min(0)*vol*(n+nped)*kb/(h.m_gamma-1._rt);};
        auto const e0=electron_energy();
        // Independent constant-state oracle; production parser has its approved units.
        auto rate=[&](amrex::Real ev){return production_rate ? 3.971488e4_rt*n*qe/std::pow(amrex::max(ev,.1_rt),1.5_rt) : nu;};
        auto const nue=rate(te0);auto const alpha=3._rt*(h.m_gamma-1._rt)*n/(n+nped);
        auto const expected_te=ti0+(te0-ti0)*std::exp(-alpha*nue*dt);
        auto const nui=rate(expected_te);auto const b=-std::expm1(-2._rt*nui*dt);
        auto const expected_ion=-b*before[0]+1.5_rt*qe*expected_te*b*before[1];
        auto const sigma=std::sqrt((3._rt*(1._rt-b)*qe*qe*ti0*expected_te*b+1.5_rt*std::pow(qe*expected_te*b,2))*before[2]);
        h.m_source_guard.BeginBatch(0,rho,h);
        h.QDSMCAddTemperatureRelaxation(0,dt,rho,deposits);
        auto const e1=electron_energy();
        auto const tmin=te.min(0)/K,tmax=te.max(0)/K;
        auto const expected_e1=e0*expected_te/te0;
        AMREX_ALWAYS_ASSERT(std::abs(tmin-expected_te)<=1.e-12_rt*amrex::max(te0,ti0) && std::abs(tmax-expected_te)<=1.e-12_rt*amrex::max(te0,ti0));
        AMREX_ALWAYS_ASSERT(std::abs(e1-expected_e1)<=1.e-12_rt*amrex::max(e0,before[0]));
        h.QDSMCApplyIonHeating(0,dt,nullptr,&deposits);
        auto const after=particle_stats(ions);auto const e2=electron_energy();
        AMREX_ALWAYS_ASSERT(e2==e1 && after[1]==before[1] && after[2]==before[2]);
        auto const ion_delta=after[0]-before[0];auto const noise=ion_delta-expected_ion;
        AMREX_ALWAYS_ASSERT(std::isfinite(ion_delta) && std::abs(noise)<=7._rt*sigma+1.e-12_rt*e0);
        auto const systematic=expected_e1-e0+expected_ion;auto const actual=e2-e0+ion_delta;
        auto const actual_rel=after[3]-before[3];
        bool const audit=h.m_source_guard.auditing();
        for (auto c : {HybridSourceEligibility::Ohmic,HybridSourceEligibility::Thermal,
                       HybridSourceEligibility::Stopping}) {
            AMREX_ALWAYS_ASSERT(!h.m_source_guard.guards(c));
        }
        if (audit) {
            auto const scale=amrex::max(e0,amrex::max(before[3],1._rt));
            AMREX_ALWAYS_ASSERT(std::abs(h.m_source_guard.electronEnergy(
                HybridSourceEligibility::Thermal)-(e1-e0))<=1.e-12_rt*scale);
            AMREX_ALWAYS_ASSERT(std::abs(h.m_source_guard.ionEnergy()-actual_rel)
                                <=1.e-12_rt*scale);
            AMREX_ALWAYS_ASSERT(h.m_source_guard.epoch()==1);
        }
        h.m_source_guard.EndBatch();
        write_particles(ions);
        if(amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream f("EXCHANGE.json");f<<std::setprecision(17)
            <<"{\"operator_oracle_pass\":true,\"exact_pair_conservation_claim\":false,\"full_qualification\":false"
            <<",\"ranks\":"<<amrex::ParallelDescriptor::NProcs()<<",\"dt_s\":"<<dt
            <<",\"density_m3\":"<<n<<",\"pedestal_m3\":"<<nped<<",\"Te0_eV\":"<<te0
            <<",\"Ti0_eV\":"<<ti0<<",\"Te1_eV\":"<<tmax<<",\"electron_rate_s1\":"<<nue<<",\"ion_rate_s1\":"<<nui
            <<",\"electron_energy_initial_J\":"<<e0<<",\"electron_change_J\":"<<e2-e0
            <<",\"ion_change_J\":"<<ion_delta<<",\"ion_expectation_J\":"<<expected_ion
            <<",\"ion_sampling_sigma_J\":"<<sigma<<",\"ion_sampling_error_J\":"<<noise
            <<",\"expected_pair_residual_J\":"<<systematic<<",\"realized_pair_residual_J\":"<<actual
            <<",\"expected_pair_residual_relative\":"<<systematic/e0
            <<",\"realized_pair_residual_relative\":"<<actual/e0
            <<",\"expected_pair_residual_relative_to_initial_thermal_energy\":"<<systematic/(e0+before[0])
            <<",\"realized_pair_residual_relative_to_initial_thermal_energy\":"<<actual/(e0+before[0])
            <<",\"audit_enabled\":"<<(audit?"true":"false")
            <<",\"actual_relativistic_ion_change_J\":"<<actual_rel
            <<",\"audited_electron_J\":"<<h.m_source_guard.electronEnergy(HybridSourceEligibility::Thermal)
            <<",\"audited_ion_J\":"<<h.m_source_guard.ionEnergy()
            <<",\"source_guard_enabled\":false,\"uniform_symmetric_one_species\":true}\n";
        }
        amrex::Print()<<"SOURCE_AUDIT_ORACLE_PASS; PAIR_RESIDUAL_MEASURED_NOT_FORCED_ZERO\n";
    }
    WarpX::ResetInstance();warpx::initialization::finalize_external_libraries();
}
