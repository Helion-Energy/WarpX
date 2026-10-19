/* Read-only state dumps around a controlled synthetic timestep refinement. */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "Python/callbacks.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <fstream>
#include <iomanip>
#include <cmath>
void snapshot(WarpX& sim, std::string const& name) {
 if(amrex::ParallelDescriptor::IOProcessor()) AMREX_ALWAYS_ASSERT(amrex::UtilCreateDirectory(name,0750));
 amrex::ParallelDescriptor::Barrier();
 using warpx::fields::FieldType;
 auto const version=amrex::VisMF::GetHeaderVersion();
 amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::NoFabHeader_v1);
 for(auto const& entry : {std::pair{FieldType::hybrid_electron_temperature_fp,"Te"},std::pair{FieldType::hybrid_rho_fp_temp,"rho_old"},std::pair{FieldType::rho_fp,"rho_new"}})
  amrex::VisMF::Write(*sim.m_fields.get(entry.first,0),name+"/"+entry.second);
 auto const* ped=sim.get_pointer_HybridPICModel()->DensityPedestal(0);
 AMREX_ALWAYS_ASSERT(ped != nullptr);amrex::VisMF::Write(*ped,name+"/pedestal");
 amrex::VisMF::SetHeaderVersion(version);
 if(amrex::ParallelDescriptor::IOProcessor()) {
  std::ofstream f(name+"/STATE.json");f<<std::setprecision(17)<<"{\"step\":"<<sim.getistep(0)<<",\"time\":"<<sim.gett_new(0)<<",\"dt\":"<<sim.getdt(0)<<",\"synchronized\":"<<(sim.getis_synchronized()?"true":"false")<<"}\n";
 }
}
int main(int argc,char**argv) {
 warpx::initialization::initialize_external_libraries(argc,argv);
 {
  auto& sim=WarpX::GetInstance();sim.InitData();int factor=0;amrex::ParmParse("refine").get("factor",factor);
  AMREX_ALWAYS_ASSERT(factor==1 || factor==2 || factor==4);
  AMREX_ALWAYS_ASSERT(sim.getistep(0)==500);
  auto const begin=sim.gett_new(0), saved_dt=sim.getdt(0);
  sim.SynchronizeVelocityWithPosition();sim.setdt(saved_dt/factor);
  auto const end_step=500+500*factor;sim.updateMaxStep(end_step);
  bool initial_observed=false;
  InstallPythonCallback("particleinjection",[&](){if(!initial_observed){snapshot(sim,"initial");initial_observed=true;}});
  sim.Evolve();AMREX_ALWAYS_ASSERT(initial_observed);snapshot(sim,"final");
  AMREX_ALWAYS_ASSERT(sim.getistep(0)==end_step);
  AMREX_ALWAYS_ASSERT(std::abs(sim.gett_new(0)-(begin+500*saved_dt))<1.e-20);
  amrex::Print()<<"COUPLED_REFINEMENT_COMPLETED factor="<<factor<<"\n";
 }
 WarpX::ResetInstance();warpx::initialization::finalize_external_libraries();
}
