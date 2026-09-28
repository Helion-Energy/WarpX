/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermalRuntime.H"
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <string>
using namespace warpx::thermal;
namespace {
void check (bool okay, const char* message) { AMREX_ALWAYS_ASSERT_WITH_MESSAGE(okay,message); }
ElectronThermalModelConfig resolve (const std::string& prefix, const std::string& partner="off",
 const std::string& audit="off", amrex::Real relative=1.e-10, amrex::Real absolute=1.e-30,
 bool reference=false, bool channel=true) {
 amrex::ParmParse pp(prefix);
 pp.add("expected_ou_thermal_partner",partner);pp.add("expected_ou_audit",audit);
 pp.add("expected_ou_relative_tolerance",relative);pp.add("expected_ou_absolute_tolerance_joule",absolute);
 pp.add("expected_ou_relativistic_reference",int(reference));
 return ElectronThermalModelConfig::Resolve(pp,channel,false);
}
void replace (std::string& text,const std::string& from,const std::string& to) {
 auto const pos=text.find(from);check(pos!=std::string::npos,"fixture replacement exists");text.replace(pos,from.size(),to);
}
}
int main (int argc,char* argv[]) {
 amrex::Initialize(argc,argv);
 {
  std::string which="same";amrex::ParmParse("test").query("case",which);
  amrex::InitRandom(941);auto const expected_random=amrex::Random();amrex::InitRandom(941);
  ElectronEnergyMetadata saved{ElectronEnergyMode::DecoupledJFNK,5./3.,2.e18,true,{}};
  saved.thermal_model=resolve("saved","population_bounded_nr","off",1.e-3);
  auto requested=saved;
  if(which=="same") {}
  else if(which=="switch")requested.mode=ElectronEnergyMode::CoupledJFNK;
  else if(which=="reverse_switch")saved.mode=ElectronEnergyMode::CoupledJFNK;
  else if(which=="effective_partner") {
   requested.thermal_model=resolve("request","population_bounded_nr","bounded_nr",1.e-3);
   check(saved.thermal_model.expected_options.mode==ExpectedIonEnergyMode::NonrelativisticBounded,
    "population partner resolves effective bounded NR");
  } else if(which=="inactive_tolerance") {
   saved.thermal_model=resolve("off_saved","off","off",2.,3.);
   requested.thermal_model=resolve("off_requested","off","off",4.,5.);
  } else if(which=="reference") {
   requested.thermal_model=resolve("request","population_bounded_nr","off",1.e-3,1.e-30,true);
   check(saved.Encode()==requested.Encode(),"optional reference excluded from physical contract");
  } else if(which=="defaults") {
   amrex::ParmParse empty("empty");auto const config=ElectronThermalModelConfig::Resolve(empty,false,false);
   ExpectedIonEnergyOptions const defaults;
   check(!config.ExpectedAudit() && !config.PopulationPartner() && !config.relativistic_reference &&
    config.expected_options.mode==ExpectedIonEnergyMode::Unspecified &&
    config.expected_options.absolute_tolerance_joule==defaults.absolute_tolerance_joule &&
    config.expected_options.relative_component_tolerance==defaults.relative_component_tolerance,
    "shared helper defaults preserved");
   saved.thermal_model=config;requested.thermal_model=config;
  } else if(which=="active_default" || which=="default_matches_explicit" ||
            which=="default_budget" || which=="capability_default") {
   amrex::ParmParse pp("default_request");
   if(which=="default_budget")pp.add("expected_ou_relative_tolerance",1.e-3);
   auto const config=ElectronThermalModelConfig::Resolve(pp,true,false);
   ExpectedIonEnergyOptions const defaults;
   check(config.PopulationPartner() && config.ExpectedAudit() &&
    config.expected_options.mode==ExpectedIonEnergyMode::NonrelativisticBounded,
    "active relaxation defaults to bounded population partner");
   check(config.expected_options.relative_component_tolerance==
     (which=="default_budget" ? 1.e-3 : defaults.relative_component_tolerance) &&
    config.expected_options.absolute_tolerance_joule==defaults.absolute_tolerance_joule,
    "default never loosens the NR accuracy budget");
   saved.thermal_model=config;requested.thermal_model=config;
   if(which=="default_matches_explicit") {
    requested.thermal_model=resolve("explicit","population_bounded_nr");
    requested.mode=ElectronEnergyMode::CoupledJFNK;
   }
   if(which=="capability_default") {
    auto const temperature=15000.*PhysConst::q_e/PhysConst::kb;
    auto const mass=3.3435837724e-27;
    auto const strict=ExpectedIonExchangeWork({1.e6,0.,0.},{0.,0.,0.},0.,1.e4,
        temperature,0.,mass,1.e-9,config.expected_options);
    check(!strict.valid,"unchanged default rejects unsupported NR accuracy");
    amrex::ParmParse budget("declared_budget");budget.add("expected_ou_relative_tolerance",1.e-3);
    auto const declared=ElectronThermalModelConfig::Resolve(budget,true,false);
    auto const allowed=ExpectedIonExchangeWork({1.e6,0.,0.},{0.,0.,0.},0.,1.e4,
        temperature,0.,mass,1.e-9,declared.expected_options);
    check(allowed.valid,"explicit declared physical budget supports bounded case");
    auto const fast=ExpectedIonExchangeWork({1.e8,0.,0.},{0.,0.,0.},0.,1.e4,
        temperature,0.,mass,1.e-9,declared.expected_options);
    check(!fast.valid,"declared budget retains live rejection for faster ions");
   }
  } else if(which=="explicit_off") {
   auto const config=resolve("diagnostic","off");
   check(!config.PopulationPartner() && !config.ExpectedAudit(),"explicit diagnostic off honored");
   saved.thermal_model=config;requested.thermal_model=config;
  } else if(which=="redirect_default") {
   amrex::ParmParse pp("redirect");auto const config=ElectronThermalModelConfig::Resolve(pp,false,true);
   check(!config.PopulationPartner() && !config.ExpectedAudit(),"redirect-only default has no relaxation partner");
   saved.thermal_model=config;requested.thermal_model=config;
  } else if(which=="default_restart_off") {
   saved.thermal_model=resolve("old_default","off");
   amrex::ParmParse pp("new_default");
   requested.thermal_model=ElectronThermalModelConfig::Resolve(pp,true,false);
  } else if(which=="default_relativistic") {
   amrex::ParmParse pp("default_invalid");pp.add("expected_ou_audit",std::string("relativistic"));
   requested.thermal_model=ElectronThermalModelConfig::Resolve(pp,true,false);
  } else if(which=="partner_no_relaxation") {
   requested.thermal_model=resolve("invalid","population_bounded_nr","off",1.e-3,1.e-30,false,false);
  } else if(which=="partner")requested.thermal_model=resolve("request","off","bounded_nr",1.e-3);
  else if(which=="convention") {
   saved.thermal_model=resolve("c_saved","off","bounded_nr",1.e-3);
   requested.thermal_model=resolve("request","off","relativistic",1.e-3);
  } else if(which=="relative")requested.thermal_model=resolve("request","population_bounded_nr","off",2.e-3);
  else if(which=="absolute")requested.thermal_model=resolve("request","population_bounded_nr","off",1.e-3,2.e-30);
  else if(which=="gamma")requested.gamma=1.4;
  else if(which=="floor")requested.number_floor=1.e18;
  else if(which=="axis")requested.corrected_axis=false;
  else if(which=="legacy")requested.mode=ElectronEnergyMode::Legacy;
  else if(which=="negative_tolerance")requested.thermal_model=resolve("request","off","bounded_nr",-1.);
  else if(which=="no_channel")requested.thermal_model=resolve("request","off","bounded_nr",1.e-3,1.e-30,false,false);
  else if(which=="bad_partner")requested.thermal_model=resolve("request","invented");
  else if(which=="partner_relativistic")requested.thermal_model=resolve("request","population_bounded_nr","relativistic");
  else if(which=="reference_without_partner")requested.thermal_model=resolve("request","off","off",1.e-3,1.e-30,true);
  auto text=saved.Encode();
  if(which=="missing")text.erase(text.find("thermal_model"));
  else if(which=="version")replace(text,"WarpXElectronEnergy 2","WarpXElectronEnergy 1");
  else if(which=="model_version")replace(text,"thermal_model 1","thermal_model 2");
  else if(which=="policy")replace(text,"endpoint_absorption_frozen_cell_temperature","endpoint_absorption_hold_U");
  else if(which=="history")replace(text,"unqualified_endpoint_absorption_split_v1","changed_history_v2");
  else if(which=="source")replace(text,"midpoint_raw_species_fixed_old_ti_endpoint_ou_v1","changed_source_v2");
  else if(which=="trailing")text+="unexpected trailing data\n";
  else if(which=="malformed")replace(text,"expected_ou_relative_tolerance 0.001","expected_ou_relative_tolerance nan");
  else if(which=="duplicate")replace(text,"thermal_model 1\n","thermal_model 1\nthermal_model 1\n");
  auto const restored=ElectronEnergyMetadata::Decode(text);
  restored.ValidateRestart(requested);
  check(restored.Encode()==text,"metadata exact roundtrip");
  check(amrex::Random()==expected_random,"resolve/encode/decode/validate consume no RNG");
  amrex::Print()<<"PASS thermal checkpoint model "<<which<<"\n";
 }
 amrex::Finalize();
}
