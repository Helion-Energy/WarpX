/* Native RZ source/force application regression. License: BSD-3-Clause-LBNL */
#include "WarpX.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"
#include "Python/callbacks.H"
#include "Fields.H"
#include <AMReX_ParmParse.H>
#include <AMReX_GpuLaunch.H>
#include "Initialization/WarpXInit.H"
#include <iomanip>
#include "NativeEnergyInventory.H"
#include "NativeInertiaInventory.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
void LoadField() {
    auto& w=WarpX::GetInstance();
    using warpx::fields::FieldType;using ablastr::fields::Direction;
    auto B=w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
    auto const& g=w.Geom(0);auto dx=g.CellSizeArray(),lo=g.ProbLoArray();
    amrex::Real amplitude=.01;amrex::ParmParse("source_gate").query("B0",amplitude);
    for(int c=0;c<3;++c){auto& f=*B[c];auto const type=f.ixType().toIntVect();
      for(amrex::MFIter it(f);it.isValid();++it){auto a=f.array(it);
        amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
          auto const x=(lo[0]+(i+.5*(1-type[0]))*dx[0])/.25;
          auto const y=1-x*x;
          a(i,j,k)=c==1?amplitude*x*y*y*y*y:0.;
        });
      }
    }
    w.FillBoundaryB(B[0]->nGrowVect(),true);
    amrex::Print()<<"SOURCE_GATE_INITIAL_FIELD Bt_norm="<<B[1]->norminf()
      <<" proton_q_over_m="<<PhysConst::q_e/PhysConst::m_p<<"\n";
}
int main(int argc,char** argv){
  warpx::initialization::initialize_external_libraries(argc,argv);
  { InstallPythonCallback("beforeInitEsolve",LoadField);
    auto& w=WarpX::GetInstance();w.InitData();ClearPythonCallback("beforeInitEsolve");
    amrex::OutStream().precision(17);
    bool energy_audit=false;amrex::ParmParse("source_gate").query("energy_audit",energy_audit);
    std::unique_ptr<NativeEnergyInventory> inventory;
    bool inertia_audit=false;amrex::ParmParse("source_gate").query("inertia_audit",inertia_audit);
    NativeInertiaInventory* inertia=nullptr;
    if(energy_audit) inventory=std::make_unique<NativeEnergyInventory>(w);
    if(inertia_audit) {
      auto observer=std::make_unique<NativeInertiaInventory>(w);inertia=observer.get();
      w.reduced_diags->m_rd_names.push_back(observer->m_rd_name);
      w.reduced_diags->m_multi_rd.push_back(std::move(observer));
    }
    if(energy_audit || inertia_audit) {
      InstallPythonCallback("beforestep",[&] {if(inventory) inventory->Begin();if(inertia) inertia->Begin();});
      InstallPythonCallback("afterstep",[&] {if(inventory) inventory->End();if(inertia) inertia->End();});
    }
    w.Evolve();
    if(energy_audit || inertia_audit) {ClearPythonCallback("beforestep");ClearPythonCallback("afterstep");inventory.reset();}
    WarpX::Finalize(); }
  warpx::initialization::finalize_external_libraries();
}
