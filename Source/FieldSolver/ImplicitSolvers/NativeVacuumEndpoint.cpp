/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */

#include "NativeLongitudinalIncrement.H"

#include "NativeVacuumEndpoint.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "NativeVacuumMassMatrixCapabilities.H"
#include "NativePECPlasma.H"
#include "NativeCircuitFieldRate.H"
#include "Circuit/CircuitCoupling.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "DarwinABoundary.H"
#include "DarwinVacuumAffineResponse.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "ImplicitFieldRollback.H"
#include "NativeEndpointField.H"
#include "NativeInstantaneousForce.H"
#include "NativeInstantaneousIonCurrent.H"
#include "NativeJouleWork.H"
#include "NativeVacuumJouleError.H"
#include "NativeSplitAmpere.H"
#include "NativeVacuumConstraint.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>
#include <ablastr/coarsen/sample.H>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
namespace {
using MF = amrex::MultiFab;
using Field = amrex::Array<MF, 3>;
using View = ablastr::fields::VectorField;
using FT = warpx::fields::FieldType;
#include "NativeVacuumFieldUtils.H"
} // namespace
bool NativeVacuumEndpointEnabled() {
  bool enabled = false;
  amrex::ParmParse("endpoint_diagnostic").query("joined_vacuum", enabled);
  return enabled;
}
bool NativeZeroExternalDriveSupported(WarpX& w) {
  using ablastr::fields::Direction;
  auto const& model=*w.get_pointer_HybridPICModel();
  bool prescribed=false,circuit=false;
  amrex::ParmParse endpoint("endpoint_diagnostic");
  endpoint.query("prescribed_drive",prescribed);
  endpoint.query("native_circuit",circuit);
  auto const* external=model.m_external_vector_potential.get();
  bool ready=w.finestLevel()==0 && !model.m_has_external_current &&
      !prescribed && !circuit &&
      (model.m_add_external_fields ? external!=nullptr : external==nullptr);
  amrex::ParallelDescriptor::ReduceBoolAnd(ready);
  if(!ready)return false;

  // Agree optional presence/count before inspecting a per-field list. In
  // particular a last-rank configuration fault must not change collective order.
  int count=model.m_add_external_fields ? external->nFields() : 0;
  int count_min=count,count_max=count;
  amrex::ParallelDescriptor::ReduceIntMin(count_min);
  amrex::ParallelDescriptor::ReduceIntMax(count_max);
  ready=count_min==count_max && count>=0 &&
      (!model.m_add_external_fields || count>0);
  int enabled=model.m_add_external_fields ? 1 : 0;
  int enabled_min=enabled,enabled_max=enabled;
  amrex::ParallelDescriptor::ReduceIntMin(enabled_min);
  amrex::ParallelDescriptor::ReduceIntMax(enabled_max);
  ready=ready && enabled_min==enabled_max;
  amrex::ParallelDescriptor::ReduceBoolAnd(ready);
  if(!ready)return false;
  if(count==0)return true;

  std::set<std::string> unique;
  std::ostringstream schema;
  for(int f=0;f<count;++f) {
    auto const& name=external->FieldName(f);
    ready=!name.empty() && unique.insert(name).second && ready;
    schema<<name.size()<<':'<<name;
    // The exact owned parser family proves constant zero. A profile that only
    // vanishes at the current time, or stale ParmParse text, cannot certify it.
    ready=external->HasConstantZeroTimeProfile(f) && ready;
  }
  std::string const local_schema=schema.str();
  ready=local_schema.size()<=static_cast<std::size_t>(std::numeric_limits<int>::max()) && ready;
  amrex::ParallelDescriptor::ReduceBoolAnd(ready);
  if(!ready)return false;
  int size_min=static_cast<int>(local_schema.size()),size_max=size_min;
  amrex::ParallelDescriptor::ReduceIntMin(size_min);
  amrex::ParallelDescriptor::ReduceIntMax(size_max);
  if(size_min!=size_max)return false;
  std::string reference_schema=local_schema;
  amrex::ParallelDescriptor::Bcast(reference_schema.data(),size_min,
      amrex::ParallelDescriptor::IOProcessorNumber());
  ready=reference_schema==local_schema;
  amrex::ParallelDescriptor::ReduceBoolAnd(ready);
  if(!ready)return false;

  std::vector<MF const*> units,driven;
  auto const E=w.m_fields.get_alldirs(FT::Efield_fp,0);
  auto const B=w.m_fields.get_alldirs(FT::Bfield_fp,0);
  auto append=[&](std::string const& name,int c,MF const& native,
                   std::vector<MF const*>& list) {
    if(!w.m_fields.has(name,Direction{c},0)){ready=false;return;}
    auto const* field=w.m_fields.get(name,Direction{c},0);
    bool const layout=field->nComp()==1 && field->boxArray()==native.boxArray() &&
        field->DistributionMap()==native.DistributionMap() &&
        field->nGrowVect().allGE(native.nGrowVect());
    ready=layout && ready;list.push_back(field);
  };
  for(int f=0;f<count;++f)for(int c=0;c<3;++c) {
    append(external->FieldName(f)+"_Aext",c,*E[c],units);
    append(external->FieldName(f)+"_curlAext",c,*B[c],units);
  }
  for(int c=0;c<3;++c) {
    // Use the register's canonical field names rather than interpreting an
    // arbitrary field with 'external' in its name. B_static remains physical.
    if(!w.m_fields.has(FT::hybrid_E_fp_external,Direction{c},0) ||
       !w.m_fields.has(FT::hybrid_B_fp_external,Direction{c},0)){ready=false;continue;}
    auto const* electric=w.m_fields.get(FT::hybrid_E_fp_external,Direction{c},0);
    auto const* magnetic=w.m_fields.get(FT::hybrid_B_fp_external,Direction{c},0);
    for(auto const pair:{std::make_pair(electric,E[c]),std::make_pair(magnetic,B[c])}) {
      ready=pair.first->nComp()==1 && pair.first->boxArray()==pair.second->boxArray() &&
          pair.first->DistributionMap()==pair.second->DistributionMap() &&
          pair.first->nGrowVect().allGE(pair.second->nGrowVect()) && ready;
      driven.push_back(pair.first);
    }
  }
  // Complete every rank's presence/layout preflight before per-field reductions.
  amrex::ParallelDescriptor::ReduceBoolAnd(ready);
  if(!ready)return false;
  for(auto const* field:units)
    ready=field->is_finite(0,1,field->nGrowVect()) && ready;
  for(auto const* field:driven) {
    ready=field->is_finite(0,1,field->nGrowVect()) && ready;
    ready=(field->norminf(0,1,field->nGrowVect())==0.) && ready;
  }
  amrex::ParallelDescriptor::ReduceBoolAnd(ready);
  return ready;
}
bool NativeStaticVacuumEndpointSupported(WarpX& w) {
#if defined(WARPX_DIM_RZ)
  auto const& m=*w.get_pointer_HybridPICModel();
  bool smooth=false;
  amrex::Real theta=0.5;
  amrex::ParmParse implicit("implicit_evolve");
  implicit.query("theta",theta);
  amrex::ParmParse endpoint("endpoint_diagnostic");
  endpoint.query("smooth_force",smooth);
  std::string policy;endpoint.query("vacuum_edge_policy",policy);
  return NativeVacuumEndpointEnabled() && NativeEndpointEnabled() &&
      NativeFullOhmLongitudinalEnabled() && NativeIonQuadratureEnabled() &&
      NativeCorrelatedIncrementEnabled() && m.UseCompatibleYeeInertia() &&
      m.UsesEulerianElectronEnergy() && theta==0.5 &&
      w.Geom(0).isPeriodic(1) &&
      WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC &&
      !NativeCircuitDriveEnabled() && !NativePrescribedDriveEnabled() &&
      !warpx::darwin::NativePECPlasmaEnabled() && !smooth &&
      !m.HasResistivity() && !m.DensityPedestal(0) &&
      !m.m_include_temperature_relaxation && !m.m_include_joule_heating &&
      !m.m_include_electron_viscosity && !m.m_has_energy_sink &&
      !m.m_has_electron_stopping && policy=="native_edge_candidate" &&
      m.m_darwin_vacuum_recovery &&
      m.m_darwin_vacuum_recovery_cadence=="half" &&
      m.m_darwin_vacuum_recovery_frozen_mask &&
      m.m_darwin_vacuum_recovery_components=="all" &&
      m.m_darwin_vacuum_recovery_mask=="vacuum" &&
      m.m_darwin_vacrec_relax_time==0. &&
      m.m_darwin_vacuum_recovery_operator=="poisson" &&
      m.m_vacuum_recovery_live_probes;
#else
  amrex::ignore_unused(w);
  return false;
#endif
}
bool NativeRetainedStaticVacuumSupported(WarpX& w) {
  if(!NativeStaticVacuumEndpointSupported(w))return false;
  auto const& m=*w.get_pointer_HybridPICModel();
  if(!m.m_add_external_fields)return true;
  auto const* external=m.m_external_vector_potential.get();
  if(!external || external->nFields()<1)return false;
  amrex::ParmParse parameters("external_vector_potential");
  for(int f=0;f<external->nFields();++f) {
    // The existing recovery uses add_external_fields to select its physical
    // A wall pin even for an identically zero scale. Admit that exact static
    // pin without admitting segments, a driver or a sampled-time guess.
    if(external->UsesPythonScale(f) || external->DeviceDriven(f))return false;
    std::string expression="1.0";
    parameters.query(external->FieldName(f)+".A_time_external_function(t)",expression);
    amrex::Parser parser(expression);
    if(!parser.symbols().empty() || external->TimeScale(f,0.)!=0.)return false;
  }
  return true;
}
bool NativeRetainedCircuitVacuumSupported(WarpX& w) {
#if defined(WARPX_DIM_RZ)
  auto const& m=*w.get_pointer_HybridPICModel();
  bool smooth=false,mm=false,density_projection=false;
  amrex::Real theta=0.5;
  amrex::ParmParse implicit("implicit_evolve");
  implicit.query("theta",theta);
  implicit.query("use_mass_matrices_jacobian",mm);
  implicit.query("mass_matrices_density_projection",density_projection);
  amrex::ParmParse endpoint("endpoint_diagnostic");
  endpoint.query("smooth_force",smooth);
  std::string policy;endpoint.query("vacuum_edge_policy",policy);
  return NativeVacuumEndpointEnabled() && NativeCircuitDriveEnabled() &&
      NativeEndpointEnabled() && NativeFullOhmLongitudinalEnabled() &&
      NativeIonQuadratureEnabled() && NativeCorrelatedIncrementEnabled() &&
      m.UseCompatibleYeeInertia() && m.UsesEulerianElectronEnergy() && theta==0.5 &&
      WarpX::field_centering_nox==2 && WarpX::field_centering_noz==2 &&
      w.Geom(0).isPeriodic(1) &&
      WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC &&
      !mm && !density_projection && !NativePrescribedDriveEnabled() &&
      !warpx::darwin::NativePECPlasmaEnabled() && !smooth &&
      m.m_add_external_fields && !m.HasResistivity() && !m.DensityPedestal(0) &&
      !m.m_end_region.holmstrom &&
      !m.m_include_temperature_relaxation && !m.m_include_joule_heating &&
      !m.m_include_electron_viscosity && !m.m_has_energy_sink &&
      !m.m_has_electron_stopping && policy=="native_edge_candidate" &&
      m.m_darwin_vacuum_recovery &&
      m.m_darwin_vacuum_recovery_cadence=="half" &&
      m.m_darwin_vacuum_recovery_frozen_mask &&
      m.m_darwin_vacuum_recovery_components=="all" &&
      m.m_darwin_vacuum_recovery_mask=="vacuum" &&
      m.m_darwin_vacrec_relax_time==0. &&
      m.m_darwin_vacuum_recovery_operator=="poisson" &&
      m.m_vacuum_recovery_live_probes;
#else
  amrex::ignore_unused(w);
  return false;
#endif
}
bool NativeVacuumMassMatrixSupported(WarpX& w) {
  bool density_projection=false;
  amrex::ParmParse("implicit_evolve").query(
      "mass_matrices_density_projection",density_projection);
  return density_projection && NativeStaticVacuumEndpointSupported(w);
}
char const *NativeVacuumSupportRecordName() {
  return "diagnostic_vacuum_support_nodal";
}
char const *NativeVacuumTransverseRecordName() {
  return "diagnostic_E_transverse_endpoint_fp";
}
void AllocateNativeVacuumSupportStorage(WarpX &w) {
  if (!NativeVacuumEndpointEnabled())
    return;
  auto const *name = NativeVacuumSupportRecordName();
  if (!w.m_fields.has(name, 0)) {
    auto const &rho = *w.m_fields.get(FT::rho_fp, 0);
    // Independent accepted endpoint density and interval-frozen recovery mask.
    w.m_fields.alloc_init(name, 0, rho.boxArray(), rho.DistributionMap(), 2,
                          rho.nGrowVect(), 0., true, true, true);
  }
  if (NativeCircuitDriveEnabled()) {
    auto const electric = w.m_fields.get_alldirs(FT::Efield_fp, 0);
    for (int c = 0; c < 3; ++c) {
      auto const dir = ablastr::fields::Direction{c};
      if (!w.m_fields.has(NativeVacuumTransverseRecordName(), dir, 0))
        w.m_fields.alloc_init(NativeVacuumTransverseRecordName(), dir, 0,
            electric[c]->boxArray(), electric[c]->DistributionMap(), 1,
            electric[c]->nGrowVect(), 0., true, true, true);
    }
  }
}
namespace {
void ValidateTransversePublication(WarpX &w, View const &transverse, View const &reference) {
  auto const full = w.m_fields.get_alldirs(FT::Efield_fp, 0);
  auto const ell = w.m_fields.get_alldirs("hybrid_E_long_fp", 0);
  auto expected = Clone(full);
  for (int c = 0; c < 3; ++c) {
    bool finite = transverse[c]->is_finite(0, 1, transverse[c]->nGrowVect());
    finite = full[c]->is_finite(0, 1, full[c]->nGrowVect()) && finite;
    finite = ell[c]->is_finite(0, 1, ell[c]->nGrowVect()) && finite;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(finite,
        "Native vacuum accepted full, longitudinal or transverse field is nonfinite");
    MF::LinComb(expected[c], 1., *transverse[c], 0, 1., *ell[c], 0,
                0, 1, expected[c].nGrowVect());
  }
  VectorImages(w, V(expected), true, &reference);
  for (auto const &field : expected)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field.is_finite(0, 1, field.nGrowVect()),
        "Native vacuum reconstructed publication is nonfinite");
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Error(V(expected), full) == 0.,
      "Native vacuum transverse companion does not reproduce accepted full E");
}
}
void ValidateNativeVacuumRestart(WarpX &w) {
  if (!NativeVacuumEndpointEnabled())
    return;
  ValidateNativeVacuumEndpointCapabilities(w);
  AuditNativeVacuumPotential(w, "restart", true);
  auto const &record = *w.m_fields.get(NativeVacuumSupportRecordName(), 0);
  MF physical(record, amrex::make_alias, 0, 1),
      mask(record, amrex::make_alias, 1, 1);
  auto reference = Clone(w.m_fields.get_alldirs(FT::Efield_fp, 0));
  Zero(V(reference));
  bool const circuit = NativeCircuitDriveEnabled();
  double vacuum_current_relative_error = 0.;
  if (circuit) {
    auto *subsystem = w.get_pointer_CircuitCoupling();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(subsystem && subsystem->Coupler() &&
        subsystem->NativeRestartBindingValidated(),
        "Vacuum/circuit restart requires the restored native provider binding");
    auto transverse = w.m_fields.get_alldirs(NativeVacuumTransverseRecordName(), 0);
    // The accepted field is already completed. Use standard boundary H once;
    // the condensed vacuum feedback map would count that completion twice.
    warpx::darwin::NativeCircuitFieldRate rate(w, *subsystem->Coupler());
    std::string error;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rate.Prepare(w.gett_new(0), error), error);
    rate.Apply(transverse, true);
    auto geometry = w.Geom(0);
#if defined(WARPX_DIM_RZ)
    geometry = amrex::Geometry(geometry.Domain(), geometry.ProbDomain(), 1,
                               geometry.isPeriodic());
#endif
    warpx::darwin::NativeInertiaSupport support(geometry, w.boxArray(0),
        w.DistributionMap(0), warpx::darwin::NativeVacuumSupportOptions(w));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(support.FreezeEdges(physical, mask),
        "Native vacuum restart support cannot define the accepted current constraint");
    double current_scale = 1., vacuum_error = 0.;
    for (int c = 0; c < 3; ++c) {
      auto const &current = rate.CurrentRate(c);
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(current.is_finite(0, 1, 0),
          "Native vacuum restart current rate is nonfinite");
      current_scale = std::max(current_scale, current.norminf(0));
      MF masked(current.boxArray(), current.DistributionMap(), 1, 0);
      for (amrex::MFIter it(masked); it.isValid(); ++it) {
        auto out = masked.array(it);
        auto value = current.const_array(it);
        auto recovery = support.RecoveryMask(c).const_array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
          out(i,j,k) = recovery(i,j,k) ? value(i,j,k) : 0.;
        });
      }
      vacuum_error = std::max(vacuum_error, masked.norminf(0));
      MF::Copy(reference[c], rate.ElectricReference(c), 0, 0, 1, reference[c].nGrowVect());
    }
    vacuum_current_relative_error = vacuum_error / current_scale;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(vacuum_current_relative_error) &&
        vacuum_current_relative_error <= 1.e-10,
        "Native vacuum transverse checkpoint violates the accepted vacuum-current constraint");
  }
  auto ref = V(reference);
  View transverse{};
  if (circuit) {
    transverse = w.m_fields.get_alldirs(NativeVacuumTransverseRecordName(), 0);
    ValidateTransversePublication(w, transverse, ref);
  }
  auto const result = warpx::darwin::CheckNativeVacuumAcceptedConstraint(
      w, physical, mask, circuit ? &ref : nullptr, circuit ? &transverse : nullptr);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      result.compatible && result.converged,
      "Native vacuum restart support or held E-EL violates the accepted "
      "range/divergence contract");
  auto &target = *w.m_fields.get("hybrid_rho_vacmask_fp", 0);
  AMREX_ALWAYS_ASSERT(record.nGrowVect().allGE(target.nGrowVect()));
  MF::Copy(target, record, 1, 0, 1, target.nGrowVect());
  bool dumps = false;
  amrex::ParmParse("endpoint_diagnostic").query("write_fields", dumps);
  if (dumps)
    amrex::VisMF::Write(record, "RESTART_vacuum_support");
  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::ofstream log("NATIVE_VACUUM_RESTART.json");
    log << std::setprecision(17)
        << "{\"range_defect_V_per_m\":" << result.null_defect
        << ",\"divergence_error_V_per_m\":" << result.divergence_error
        << ",\"vacuum_current_relative_error\":" << vacuum_current_relative_error
        << ",\"accepted_fields_reprojected\":false,\"independent_D_reseeded\":"
           "false}\n";
  }
  amrex::Print() << "restart: native vacuum support restored; accepted "
                    "E/EL/Je/D unchanged\n";
}
void AuditNativeVacuumPotential(WarpX &w, char const *label, bool required) {
  if (!NativeVacuumEndpointEnabled())
    return;
  bool dumps = false;
  amrex::ParmParse("endpoint_diagnostic").query("write_fields", dumps);
  if (!required && !dumps)
    return;
  auto const &phi = *w.m_fields.get("hybrid_phi_darwin_fp", 0);
  auto const longitudinal = w.m_fields.get_alldirs("hybrid_E_long_fp", 0);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      phi.is_finite(0, phi.nComp(), phi.nGrowVect()),
      "Native vacuum scalar potential contains nonfinite values");
  for (auto const *field : longitudinal)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field->is_finite(0, 1, field->nGrowVect()),
        "Native vacuum longitudinal field contains nonfinite values");
  auto gradient = Clone(longitudinal);
  Gradient(w, phi, V(gradient));
  amrex::GpuArray<int, AMREX_SPACEDIM> lo{}, hi{};
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
    hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
  }
  // Exact ComputeDarwinELong completion: no E/PEC gather image or new solve.
  double absolute = 0., scale = 1.;
  for (int c = 0; c < 3; ++c) {
    gradient[c].setBndry(0.);
    gradient[c].FillBoundary(w.Geom(0).periodicity());
    ApplyDarwinPMCVectorBoundary(gradient[c], w.Geom(0), lo, hi);
    MF difference(gradient[c].boxArray(), gradient[c].DistributionMap(), 1,
                  gradient[c].nGrowVect());
    MF::LinComb(difference, 1., gradient[c], 0, -1., *longitudinal[c], 0,
                0, 1, difference.nGrowVect());
    absolute = std::max(absolute, difference.norminf(0, 1, difference.nGrowVect()));
    scale = std::max(scale, longitudinal[c]->norminf(0, 1, longitudinal[c]->nGrowVect()));
  }
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      std::isfinite(absolute) && absolute <= 1.e-8 * scale,
      "Native vacuum scalar potential does not represent accepted longitudinal field");
  if (dumps) {
    std::string const prefix = std::string("PHI_") + label + "_" +
                               std::to_string(w.getistep(0)) + "_";
    amrex::VisMF::Write(phi, prefix + "phi");
    for (int c = 0; c < 3; ++c) {
      amrex::VisMF::Write(*longitudinal[c], prefix + "EL_" + std::to_string(c));
      amrex::VisMF::Write(gradient[c], prefix + "gradient_" + std::to_string(c));
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ofstream log("NATIVE_VACUUM_PHI.jsonl", std::ios::app);
      log << std::setprecision(17) << "{\"step\":" << w.getistep(0)
          << ",\"label\":\"" << label << "\",\"absolute_V_per_m\":"
          << absolute << ",\"scale_V_per_m\":" << scale
          << ",\"normalized\":" << absolute / scale << "}\n";
    }
  }
}
void ValidateNativeVacuumEntryDensity(WarpX &w) {
  if (!NativeVacuumEndpointEnabled())
    return;
  warpx::thermal::ValidateNativeVacuumJouleErrorEntry(w);
  auto const &record = *w.m_fields.get(NativeVacuumSupportRecordName(), 0);
  auto const &density = *w.m_fields.get(FT::rho_fp, 0);
  AMREX_ALWAYS_ASSERT(record.boxArray() == density.boxArray() &&
                      record.DistributionMap() == density.DistributionMap());
  MF difference(density.boxArray(), density.DistributionMap(), 1, 0);
  MF::LinComb(difference, 1., density, 0, -1., record, 0, 0, 1, 0);
  double const error = difference.norminf();
  double const scale = std::max({1., record.norminf(0), density.norminf(0)});
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      std::isfinite(error) && error <= 1.e-12 * scale,
      "Native vacuum entry density differs from the accepted support "
      "checkpoint; no silent reconstruction");
  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::ofstream log("NATIVE_VACUUM_ENTRY.jsonl", std::ios::app);
    log << std::setprecision(17) << "{\"step\":" << w.getistep(0)
        << ",\"rho_error_C_per_m3\":" << error
        << ",\"scale_C_per_m3\":" << scale << "}\n";
  }
}
void ValidateNativeVacuumEndpointCapabilities(WarpX &w) {
  if (!NativeVacuumEndpointEnabled())
    return;
  auto const &m = *w.get_pointer_HybridPICModel();
  auto const* pmc_owner=NativePrivatePMCMassMatrixOwner(w);
  bool mm = false, smooth = false;
  amrex::ParmParse("implicit_evolve").query("use_mass_matrices_jacobian", mm);
  amrex::ParmParse("endpoint_diagnostic").query("smooth_force", smooth);
  std::string policy;
  amrex::ParmParse("endpoint_diagnostic").query("vacuum_edge_policy", policy);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      NativeEndpointEnabled() && NativeFullOhmLongitudinalEnabled() &&
          m.UseCompatibleYeeInertia() && NativeCorrelatedIncrementEnabled() &&
          (!mm || NativeVacuumMassMatrixSupported(w) || pmc_owner) && (!m.HasResistivity() || smooth) && !m.DensityPedestal(0) &&
          !NativePrescribedDriveEnabled() && policy == "native_edge_candidate",
      "Joined vacuum prototype requires explicit native_edge_candidate, "
      "static full-Ohm/direct-Yee/correlated state with an explicit smooth "
      "force adapter for resistivity; MM requires source-free periodic-z RZ "
      "current+density response with live frozen-mask Poisson recovery or the "
      "private owner-bound PMC counterpart; "
      "vacuum MM source/circuit feedback and pedestal are outside this slice");
  if (NativeCircuitDriveEnabled()) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!smooth && !m.HasResistivity() &&
        !m.m_include_joule_heating && !m.m_include_temperature_relaxation &&
        !m.m_include_electron_viscosity && !m.m_has_energy_sink,
        "Native vacuum/circuit endpoint initially requires source-free coefficients");
  }
  if (smooth) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx::thermal::NativeJouleWorkEnabled() &&
            !m.m_include_electron_viscosity && !m.m_include_hyper_resistivity_term,
        "Joined vacuum smooth force requires native applied Joule work and "
        "constant common eta/geometric end terms; viscosity/hyper are unsupported");
    // This retains exact physical receiver/orphan and separate signed end-work
    // contracts; it does not turn removed vacuum Ohm work into electron heat.
    warpx::thermal::ValidateNativeJouleWork(w);
  }
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      m.m_darwin_vacuum_recovery &&
          m.m_darwin_vacuum_recovery_cadence == "half" &&
          m.m_darwin_vacuum_recovery_frozen_mask &&
          m.m_darwin_vacuum_recovery_components == "all" &&
          m.m_darwin_vacuum_recovery_mask == "vacuum" &&
          m.m_darwin_vacrec_relax_time == 0.,
      "Joined vacuum prototype requires frozen all-component instantaneous "
      "half-cadence "
      "recovery");
}
void CompleteNativeVacuumLongitudinalSource(WarpX &w, amrex::Real interval) {
  AMREX_ALWAYS_ASSERT(interval > 0. && std::isfinite(interval));
  using ablastr::coarsen::sample::Interp;
  auto const &model = *w.get_pointer_HybridPICModel();
  auto const &rho = *w.m_fields.get("hybrid_rho_vacmask_fp", 0);
  amrex::GpuArray<int, 3> const node{1, 1, 1}, ratio{1, 1, 1};
  double const threshold = PhysConst::q_e * model.m_n_floor *
                           model.m_darwin_vacuum_recovery_density_fraction;
  auto E = w.m_fields.get_alldirs(FT::Efield_fp, 0);
  for (int c = 0; c < 3; ++c) {
    auto const &a =
        *w.m_fields.get("hybrid_A_fp", ablastr::fields::Direction{c}, 0);
    auto const &old =
        *w.m_fields.get("hybrid_A_old_fp", ablastr::fields::Direction{c}, 0);
    auto const &el =
        *w.m_fields.get("hybrid_E_long_fp", ablastr::fields::Direction{c}, 0);
    auto const* low=warpx::darwin::increment::Low(w,c);
    auto iv = E[c]->ixType().toIntVect();
    amrex::GpuArray<int, 3> stagger{iv[0], iv[1], 1};
#if defined(WARPX_DIM_3D)
    stagger[2] = iv[2];
#endif
    for (amrex::MFIter mfi(*E[c]); mfi.isValid(); ++mfi) {
      auto out = E[c]->array(mfi);
      auto r = rho.const_array(mfi), x = a.const_array(mfi),
           x0 = old.const_array(mfi), l = el.const_array(mfi);
      auto const remainder=low?low->const_array(mfi):amrex::Array4<amrex::Real const>{};
      bool const compensated=low!=nullptr;
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                              int k) {
        if (Interp(r, node, stagger, ratio, i, j, k, 0) < threshold) {
          auto const transverse=-(x(i,j,k)-x0(i,j,k))/interval;
          out(i,j,k)=compensated?warpx::ohm::compensated::Add(
              warpx::ohm::compensated::Sum(transverse,l(i,j,k)),
              {remainder(i,j,k),0.}).hi:transverse+l(i,j,k);
        }
      });
    }
  }
  // Identical physical E and native PMC images, without touching recovery
  // target/cache fields. Caller restores the entire registered E allocation.
  VectorImages(w, E, true);
}
namespace {
bool PrivateVacuumTransverseOutputSupported(WarpX& w, View const* output) {
  int lo=output ? 1 : 0,hi=lo;
  amrex::ParallelDescriptor::ReduceIntMin(lo);
  amrex::ParallelDescriptor::ReduceIntMax(hi);
  if(lo!=hi) return false;
  if(!output) return true;
  bool valid=true;
  std::vector<std::pair<std::uintptr_t,std::uintptr_t>> registered,ranges;
  for(auto const& name:w.m_fields.list()) {
    auto const& f=*w.m_fields.internal_get(name);
    for(amrex::MFIter it(f);it.isValid();++it) {
      auto const begin=reinterpret_cast<std::uintptr_t>(f[it].dataPtr());
      registered.emplace_back(begin,begin+f[it].size()*sizeof(amrex::Real));
    }
  }
  auto const e=w.m_fields.get_alldirs(FT::Efield_fp,0);
  for(int c=0;c<3;++c) {
    auto const* f=(*output)[c];
    valid=f && f->nComp()==1 && f->boxArray()==e[c]->boxArray() &&
        f->DistributionMap()==e[c]->DistributionMap() &&
        e[c]->nGrowVect().allGE(f->nGrowVect()) && valid;
    if(!f) continue;
    for(amrex::MFIter it(*f);it.isValid();++it) {
      auto const begin=reinterpret_cast<std::uintptr_t>((*f)[it].dataPtr());
      auto const end=begin+(*f)[it].size()*sizeof(amrex::Real);
      for(auto const& r:registered) valid=(end<=r.first || begin>=r.second) && valid;
      for(auto const& r:ranges) valid=(end<=r.first || begin>=r.second) && valid;
      ranges.emplace_back(begin,end);
    }
  }
  amrex::ParallelDescriptor::ReduceBoolAnd(valid);
  return valid;
}
}
bool TryConstrainNativeVacuumEndpointField(WarpX &w, amrex::Real time,
                                        bool initial, View const &baseline,
                                        View const* exact_transverse) {
  // Collective metadata-only preflight precedes all temporary physical writes.
  if(!PrivateVacuumTransverseOutputSupported(w,exact_transverse)) return false;
  ValidateNativeEndpointCapabilities(w);
  ValidateNativeVacuumEndpointCapabilities(w);
  AllocateNativeEndpointStorage(w);
  auto* endpoint_owner=warpx::darwin::NativeEndpointPairedFields(w);
  bool const initial_pair=initial&&endpoint_owner&&endpoint_owner->InitialEndpointSelected();
  bool const paired_endpoint=warpx::darwin::NativeEndpointPairSelected(w)&&(!initial||initial_pair);
  int pair_min=paired_endpoint,pair_max=pair_min;
  amrex::ParallelDescriptor::ReduceIntMin(pair_min);amrex::ParallelDescriptor::ReduceIntMax(pair_max);
  if(pair_min!=pair_max)return false;
  warpx::darwin::NativePairedDarwinFields::EndpointHistoryView endpoint_history;
  if(paired_endpoint&&!initial){
    bool ready=endpoint_owner&&!endpoint_owner->EndpointPending();
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!endpoint_owner->EndpointHistory(endpoint_history)||
       !endpoint_owner->MatchesEndpointOrigin({baseline[0],baseline[1],baseline[2]}))return false;
  }
  warpx::darwin::NativePairedDarwinFields::EndpointDraft endpoint_draft;
  auto &model = *w.get_pointer_HybridPICModel();
  auto E = w.m_fields.get_alldirs(FT::Efield_fp, 0),
       B = w.m_fields.get_alldirs(FT::Bfield_fp, 0),
       Ji = w.m_fields.get_alldirs(FT::current_fp, 0),
       C = w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma, 0),
       Je = w.m_fields.get_alldirs("diagnostic_Je_endpoint_fp", 0),
       Ei = w.m_fields.get_alldirs("hybrid_E_inertial_fp", 0),
       EL = w.m_fields.get_alldirs("hybrid_E_long_fp", 0);
  bool baseline_layout = true;
  for (int c = 0; c < 3; ++c)
    baseline_layout = baseline[c] && baseline[c]->nComp() == 1 &&
        baseline[c]->boxArray() == E[c]->boxArray() &&
        baseline[c]->DistributionMap() == E[c]->DistributionMap() && baseline_layout;
  amrex::ParallelDescriptor::ReduceBoolAnd(baseline_layout);
  if (!baseline_layout) return false;
  auto const &rho = *w.m_fields.get(FT::rho_fp, 0);
  MF density(rho, amrex::make_alias, 0, 1);
  auto const &mask = *w.m_fields.get("hybrid_rho_vacmask_fp", 0);
  // Numerical candidate validity is checked collectively before temporary
  // force/current publication. Zero density is the admitted vacuum state;
  // a negative or nonfinite candidate must not enter an elliptic operator.
  bool candidate_valid=std::isfinite(time);
  candidate_valid=rho.is_finite(0,1,rho.nGrowVect()) && candidate_valid;
  candidate_valid=(rho.min(0)>=0.) && candidate_valid;
  candidate_valid=mask.is_finite(0,1,mask.nGrowVect()) && candidate_valid;
  candidate_valid=(mask.min(0)>=0.) && candidate_valid;
  for(int c=0;c<3;++c)
    candidate_valid=baseline[c]->is_finite(0,1,baseline[c]->nGrowVect()) && candidate_valid;
  amrex::ParallelDescriptor::ReduceBoolAnd(candidate_valid);
  if(!candidate_valid) return false;
  auto point = Clone(Ji), electron = Clone(Je), force = Clone(E),
       nores = Clone(E), reference = Clone(E), transverse = Clone(E),
       inertia = Clone(E), accepted = Clone(E);
  Zero(V(reference));
  warpx::implicit::FieldRollback saved;
  saved.Capture(w.m_fields);
  if (!warpx::darwin::TryCalculateNativeSplitDarwinAmpere(w))
    model.CalculatePlasmaCurrent(B, w.GetEBUpdateEFlag()[0], 0);
  VectorImages(w, C, false);
  if (!warpx::particles::DepositNativeInstantaneousIonCurrent(
          w, warpx::particles::InstantaneousIonState::Current, V(point))) {
    AMREX_ALWAYS_ASSERT(saved.Restore(w.m_fields));saved.Discard();return false;
  }
  Copy(Ji, V(point));
  for (int c = 0; c < 3; ++c) {
    if (initial)
      MF::LinComb(electron[c], 1., *C[c], 0, -1., point[c], 0, 0, 1,
                  electron[c].nGrowVect());
    MF::LinComb(*C[c], 1., point[c], 0, 1., electron[c], 0, 0, 1,
                C[c]->nGrowVect());
  }
  Zero(Ei);
  w.m_fields.get("hybrid_E_inertial_nodal", 0)->setVal(0.);
  if (!warpx::darwin::CaptureNativeInstantaneousForce(
          w, density, V(force), V(nores), nullptr)) {
    AMREX_ALWAYS_ASSERT(saved.Restore(w.m_fields));saved.Discard();return false;
  }
  bool smooth = false;
  amrex::ParmParse("endpoint_diagnostic").query("smooth_force", smooth);
  bool const corrected_force = smooth && model.HasResistivity() &&
                                model.m_implicit_push_excludes_resistive_field;
  auto nores_view = V(nores);
  bool const circuit = NativeCircuitDriveEnabled();
  warpx::darwin::NativeVacuumConstraintResult result;
  if (circuit) {
    auto *subsystem = w.get_pointer_CircuitCoupling();
    bool ready = subsystem && subsystem->Coupler() &&
        subsystem->Coupler()->DeviceTrials() && subsystem->Coupler()->SupportsNativeEndpointRate();
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if (ready) {
      // The legacy initialization recovery may leave driven A-wall values in
      // the initial transverse guess. The algebraic E/gather PEC rows are
      // homogeneous; preserve its interior and null modes in private scratch.
      // Solver vectors have no guards. Allocate the full native E layout
      // before applying gather images; only valid baseline values are data.
      auto accepted_baseline = Clone(E);
      Zero(V(accepted_baseline));
      for (int c = 0; c < 3; ++c)
        MF::Copy(accepted_baseline[c], *baseline[c], 0, 0, 1, 0);
      VectorImages(w, V(accepted_baseline), true);
      result = warpx::darwin::SolveNativeVacuumCircuitConstraint(
          w, *subsystem->Coupler(), time, density, mask, V(accepted_baseline),
          V(force), EL, V(transverse), V(inertia), V(reference),
          corrected_force ? &nores_view : nullptr);
    }
  } else if(paired_endpoint&&endpoint_history.paired) {
    // The existing joined spatial n+1 solve owns both origin parts. No old A,
    // BDF field estimate or instantaneous dJe is substituted for that origin.
    // Core clones these inputs; this legacy View typedef is mutable although
    // the producer never writes its baseline arguments.
    View low{const_cast<MF*>(endpoint_history.low[0]),
             const_cast<MF*>(endpoint_history.low[1]),
             const_cast<MF*>(endpoint_history.low[2])};
    if(!corrected_force)result=warpx::darwin::SolveNativeVacuumPairedEndpoint(
        w,density,mask,baseline,low,V(reference),V(force),EL,V(transverse),V(inertia),endpoint_draft);
  } else {
    result = warpx::darwin::SolveNativeVacuumConstraint(
        w, density, mask, baseline, V(reference), V(force), EL, V(transverse),
        V(inertia), corrected_force ? &nores_view : nullptr,
        paired_endpoint?&endpoint_draft:nullptr);
  }
  AMREX_ALWAYS_ASSERT(saved.Restore(w.m_fields));
  saved.Discard();
  amrex::Print() << "Native joined vacuum endpoint compatible="
                 << result.compatible << " converged=" << result.converged
                 << " null=" << result.null_defect
                 << " plasma=" << result.plasma_error
                 << " div=" << result.divergence_error
                 << " residual=" << result.residual
                 << " iterations=" << result.iterations << "\n";
  if (!result.compatible || !result.converged) return false;
  auto publish_initial_current=[&](){
    Copy(Ji, V(point));
    Copy(Je, V(electron));
    for (auto name :
         {"hybrid_Je_n_nodal", "hybrid_Je_nm1_nodal", "hybrid_Je_theta_nodal"})
      CopyNativeEndpointCurrentMirror(w, Je, *w.m_fields.get(name, 0));
    for (auto *f : w.m_fields.get_alldirs("diagnostic_D_endpoint_fp", 0))
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          f->norminf() == 0., "Joined initial D baseline must already be zero");
  };
  if(initial_pair){
    warpx::implicit::FieldRollback initial_saved;initial_saved.Capture(w.m_fields);
    publish_initial_current();
    if(!endpoint_owner->AdoptInitialEndpoint(std::move(endpoint_draft),time)){
      AMREX_ALWAYS_ASSERT(initial_saved.Restore(w.m_fields));initial_saved.Discard();return false;
    }
    initial_saved.Discard();
  } else if(paired_endpoint){
    // The saved native transaction has already been restored. Adoption owns
    // only E/Ei and their registered pair/mirror arrays; no history rotation.
    if(!endpoint_owner->AdoptEndpoint(std::move(endpoint_draft),time,
        static_cast<std::uint64_t>(w.getistep(0))+(endpoint_owner->EndpointPreSourceReclosure()?0:1)))return false;
  } else {
  for (int c = 0; c < 3; ++c)
    MF::LinComb(accepted[c], 1., transverse[c], 0, 1., *EL[c], 0, 0, 1,
                accepted[c].nGrowVect());
  Copy(E, V(accepted));
  w.FillBoundaryE(E[0]->nGrowVect(), true);
  w.ApplyEfieldBoundary(0, ablastr::utils::enums::PatchType::fine, time);
  auto ref = V(reference);
  VectorImages(w, E, true, circuit ? &ref : nullptr);
  if (circuit) {
    Copy(w.m_fields.get_alldirs(NativeVacuumTransverseRecordName(), 0), V(transverse));
    ValidateTransversePublication(w, V(transverse), ref);
  }
  Copy(Ei, V(inertia));
  PublishNativeYeeInertiaMirror(w);
  }
  // Fresh initialization prescribes D0=0. Retained adoption has already
  // seeded these actual moments before binding its immutable current origin.
  if (initial&&!initial_pair) publish_initial_current();
  auto &support_record = *w.m_fields.get(NativeVacuumSupportRecordName(), 0);
  AMREX_ALWAYS_ASSERT(density.nGrowVect().allGE(support_record.nGrowVect()) &&
                      mask.nGrowVect().allGE(support_record.nGrowVect()));
  MF::Copy(support_record, density, 0, 0, 1, support_record.nGrowVect());
  MF::Copy(support_record, mask, 0, 1, 1, support_record.nGrowVect());
  bool verify = false;
  amrex::ParmParse("endpoint_diagnostic").query("vacuum_schur_verify", verify);
  if (initial && verify)
    AMREX_ALWAYS_ASSERT(warpx::darwin::VerifyNativeVacuumFiniteSchur(
        w, rho, 0, .5 * w.getdt(0)));
  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::ofstream f("NATIVE_VACUUM_ENDPOINT.jsonl", std::ios::app);
    f << std::setprecision(17)
      << "{\"initial\":" << (initial ? "true" : "false") << ",\"time\":" << time
      << ",\"iterations\":" << result.iterations
      << ",\"actions\":" << result.actions
      << ",\"plasma_error_V_per_m\":" << result.plasma_error
      << ",\"divergence_error_V_per_m\":" << result.divergence_error
      << ",\"range_defect_V_per_m\":" << result.null_defect << "}\n";
  }
  if(exact_transverse) {
    // The solve consumed the old baseline before this point. In particular the
    // private solver m_E may serve as both baseline and destination at startup.
    for(int c=0;c<3;++c)
      MF::Copy(*(*exact_transverse)[c],transverse[c],0,0,1,
               (*exact_transverse)[c]->nGrowVect());
  }
  return true;
}
void ConstrainNativeVacuumEndpointField(WarpX &w, amrex::Real time,
                                      bool initial, View const &baseline,
                                      View const* exact_transverse) {
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      TryConstrainNativeVacuumEndpointField(w, time, initial, baseline, exact_transverse),
      "Native joined vacuum endpoint rejected incompatible held "
      "EL/null baseline or unconverged rows");
}
