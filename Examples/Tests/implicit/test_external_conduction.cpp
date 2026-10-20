#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "Python/callbacks.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuLaunch.H>
#include <fstream>
#include <iomanip>
#include <cmath>
using warpx::fields::FieldType;
using ablastr::fields::Direction;
void snapshot(WarpX& sim, std::string const& name) {
 if(amrex::ParallelDescriptor::IOProcessor()) AMREX_ALWAYS_ASSERT(amrex::UtilCreateDirectory(name,0750));
 amrex::ParallelDescriptor::Barrier();
 auto const version=amrex::VisMF::GetHeaderVersion();
 amrex::VisMF::SetHeaderVersion(amrex::VisMF::Header::NoFabHeader_v1);
 amrex::VisMF::Write(*sim.m_fields.get(FieldType::hybrid_electron_temperature_fp,0),name+"/Te");
 amrex::VisMF::Write(*sim.m_fields.get(FieldType::rho_fp,0),name+"/rho");
 for(int d=0;d<3;++d) {
  amrex::VisMF::Write(*sim.m_fields.get(FieldType::Bfield_fp,Direction{d},0),name+"/B"+std::to_string(d));
  amrex::VisMF::Write(*sim.m_fields.get(FieldType::Efield_fp,Direction{d},0),name+"/E"+std::to_string(d));
 }
 amrex::VisMF::SetHeaderVersion(version);
}
int main(int argc,char**argv) {
 warpx::initialization::initialize_external_libraries(argc,argv);
 {
  auto& sim=WarpX::GetInstance();sim.InitData();bool initialized=false;int direct=0,axis=0;amrex::ParmParse pp("probe");pp.query("direct",direct);pp.query("temperature_axis",axis);AMREX_ALWAYS_ASSERT(axis>=0 && axis<AMREX_SPACEDIM);
  auto initialize_temperature=[&](){
   AMREX_ALWAYS_ASSERT(!initialized);initialized=true;
   auto& hp=*sim.get_pointer_HybridPICModel();
   auto& te=*sim.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
   auto const dx=sim.Geom(0).CellSizeArray();auto const lo=sim.Geom(0).ProbLoArray();
   for(amrex::MFIter mfi(te);mfi.isValid();++mfi) {
    auto const arr=te.array(mfi);
    amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
     int const node[3]={i,j,k};amrex::Real const x=lo[axis]+node[axis]*dx[axis];arr(i,j,k)=10000.0*(1.0+0.05*std::cos(3.14159265358979323846*x));
    });
   }
   hp.QDSMCFillElectronPressureFromTe(0);snapshot(sim,"initial");
  };
  AMREX_ALWAYS_ASSERT(!sim.get_pointer_HybridPICModel()->m_explicit_B_is_response);
  if(direct) {
   sim.HybridPICInitializeRhoJandB();initialize_temperature();
   sim.get_pointer_HybridPICModel()->QdsmcConductionOnceFD(0,sim.getdt(0),true);
  } else {
   InstallPythonCallback("particleinjection",initialize_temperature);sim.Evolve();
   AMREX_ALWAYS_ASSERT(sim.getistep(0)==1);
  }
  AMREX_ALWAYS_ASSERT(initialized && !sim.get_pointer_HybridPICModel()->m_explicit_B_is_response);snapshot(sim,"final");
  bool check_total_E_zero = false;
  pp.query("check_total_E_zero", check_total_E_zero);
  if (check_total_E_zero) {
      for (int d = 0; d < 3; ++d) {
          auto const& e = *sim.m_fields.get(FieldType::Efield_fp, Direction{d}, 0);
          AMREX_ALWAYS_ASSERT_WITH_MESSAGE(e.norm0(0, e.nGrow()) <= 1.e-12,
              "Restored total electric field is nonzero on valid or ghost nodes");
      }
      amrex::Print() << "RESTORED_TOTAL_E_ZERO_WITH_GHOSTS\n";
  }
  auto const& hp=*sim.get_pointer_HybridPICModel();auto const& a=hp.m_cond_last_report;
  AMREX_ALWAYS_ASSERT(a.completed && a.completed_time==a.requested_time);
  if(amrex::ParallelDescriptor::IOProcessor()) {
   std::ofstream f("RESULT.json");f<<std::setprecision(17)<<"{\"completed\":true,\"step\":"<<sim.getistep(0)<<",\"time\":"<<sim.gett_new(0)<<",\"conduction_half_interval\":"<<a.completed_time<<",\"capacity_energy_before\":"<<a.energy_before[0]<<",\"capacity_energy_after\":"<<a.energy_after[0]<<",\"native_residual\":"<<a.residual<<"}\n";
  }
  amrex::Print()<<"EXTERNAL_CONDUCTION_PROBE_COMPLETED\n";
 }
 WarpX::ResetInstance();warpx::initialization::finalize_external_libraries();
}
