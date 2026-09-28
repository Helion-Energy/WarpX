/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitHybrid.H"
#include "EmbeddedBoundary/Enabled.H"
#include "DarwinThermalAdvance.H"
#include "ThermalCurrentRemainder.H"
#include "DarwinVacuumJointSolve.H"
#include "NativeEndpointField.H"
#include "ThermalSourceAcceptance.H"
#include "ImplicitParticleEndpointAudit.H"
#include "Particles/MultiParticleContainer.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include <ablastr/coarsen/sample.H>
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>

namespace warpx::thermal {
using Real = amrex::Real;
using Vec = WarpXSolverVec;
using warpx::fields::FieldType;
using ablastr::fields::Direction;

DarwinThermalAdvance::DarwinThermalAdvance (ThetaImplicitHybrid& solver)
    : m_solver(solver), m_simulation(*solver.m_WarpX), m_model(*solver.m_hybrid_pic_model),
      m_geometry(m_model.ElectronThermalGeometry())
{
    m_coupled = m_model.GetElectronEnergyMode() == HybridPICModel::ElectronEnergyMode::CoupledJFNK;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_solver.m_continuity_density && m_solver.m_theta == 0.5,
        "Eulerian Darwin thermal advance requires endpoint-averaged Esirkepov density and theta=0.5");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_solver.m_darwin_segregated_solve,
        "Eulerian Darwin thermal advance requires an independent longitudinal outer solve");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_solver.m_external_field_iteration || m_solver.m_circuit_native,
        "Eulerian Darwin circuit iterations require the native C++ circuit plugin");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_model.m_pec_conductor_wall_rows,
        "Eulerian thermal trial pressure requires conductor moment closure before mapping; this wall option is not qualified");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_solver.m_skip_particle_picard_init,
        "Eulerian Darwin residual acceptance requires converged particle pushes on iteration zero");
    // These guards close only as the corresponding pure physical adapters are
    // integrated. No legacy source/marker/time-integrator may run by accident.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_model.m_eta_per_species.empty(),
        "Eulerian Darwin species resistivity requires a self-consistent per-species force response; this option is not qualified");
    amrex::ParmParse pp("implicit_evolve.thermal");
    pp.query("current_remainder",m_current_remainder_requested);
    // All ranks must enter the optional arithmetic/scope collectives together.
    // Reject an inconsistent selector before allocating or publishing scratch.
    int requested_min=m_current_remainder_requested ? 1 : 0;
    int requested_max=requested_min;
    amrex::ParallelDescriptor::ReduceIntMin(requested_min);
    amrex::ParallelDescriptor::ReduceIntMax(requested_max);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(requested_min==requested_max,
        "Thermal current remainder selector must agree on all ranks");
    if(m_current_remainder_requested) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(remainder::All(remainder::ArithmeticSupported()),
            "Thermal current remainder requires precise GNU/SSE2 CPU binary64 arithmetic, round-to-nearest, no FTZ/DAZ or implicit FMA target; GPU/other arithmetic is unsupported");
        bool source_free=!m_solver.m_circuit_native && !m_simulation.get_pointer_CircuitCoupling();
        amrex::ParallelDescriptor::ReduceBoolAnd(source_free);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source_free,
            "Thermal current remainder does not support native circuit attachment");
        bool valid=!m_solver.m_use_mass_matrices && !m_model.m_include_joule_heating &&
            !m_model.m_include_temperature_relaxation && !m_model.m_include_thermal_conduction &&
            m_solver.m_num_amr_levels==1 && !EB::enabled();
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,
            "Thermal current remainder first scope requires MM0/source0/conduction0, single level without EB");
    }
    pp.query("joint_vacuum",m_joint_vacuum_requested); // OFF leaves used-input metadata unchanged.
    InitializeIonElectricWork();
    ValidateNativeJouleWork(m_simulation);
    if (NativeJouleWorkEnabled()) {
        m_joule_work=std::make_unique<NativeJouleWork>(m_simulation);
    }
    // A separately registered bulk-drag collision must not duplicate the OU
    // mean relaxation on the same species. Legacy modes never enter this class.
    std::vector<std::string> collision_names,drag_species;
    amrex::ParmParse("collisions").queryarr("collision_names",collision_names);
    for(auto const& name:collision_names){
        amrex::ParmParse collision(name);std::string type;
        collision.query("type",type);
        if(type=="hybrid_resistive_drag"){
            std::vector<std::string> names;collision.getarr("species",names);
            drag_species.insert(drag_species.end(),names.begin(),names.end());
        }
    }
    for(auto const& name:m_simulation.GetPartContainer().GetSpeciesNames()){
        bool const charged=m_simulation.GetPartContainer().GetParticleContainerFromName(name).getCharge()!=0;
        bool const overlap=IonExchangeDragOverlaps(m_model.m_include_temperature_relaxation && charged,
            m_model.IsRelaxationExcluded(name),name,drag_species);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!overlap,
            "Eulerian implicit ion OU relaxation overlaps hybrid_resistive_drag for species "+name+
            "; the additional bulk momentum/work transfer is not qualified");
    }
    auto const& thermal_model=m_model.ElectronThermalModel();
    m_population_thermal_partner=thermal_model.PopulationPartner();
    m_relativistic_reference_audit=thermal_model.relativistic_reference;
    m_expected_ou_audit=thermal_model.ExpectedAudit();
    m_expected_options=thermal_model.expected_options;
    if(m_expected_ou_audit) {
        if(m_population_thermal_partner) {
            amrex::Print()<<"Expected OU population thermal partner enabled: only electron relaxation is replaced; "
                <<"redirect budget and mechanical work stay separate. Source PC retains the legacy damping surrogate.\n";
        } else {
            amrex::Print()<<"Expected OU endpoint audit enabled; physical electron source unchanged, mechanical pairing is measured separately.\n";
        }
    }
    pp.queryAdd("electric_scale", m_electric_scale);
    pp.queryAdd("energy_scale", m_energy_scale);
    pp.queryAdd("outer_max_iterations", m_outer_iterations);
    pp.queryAdd("push_max_iterations", m_push_iterations);
    pp.queryAdd("push_relative_tolerance", m_push_tolerance);
    m_options.relative_tolerance = 1.e-6;
    m_options.absolute_tolerance = 1.e-12;
    m_options.linear_relative_tolerance = 1.e-6;
    m_options.probe_rhs_scale_floor = true;
    pp.queryAdd("relative_tolerance", m_options.relative_tolerance);
    pp.queryAdd("absolute_tolerance", m_options.absolute_tolerance);
    pp.queryAdd("linear_relative_tolerance", m_options.linear_relative_tolerance);
    pp.queryAdd("adaptive_forcing", m_options.adaptive_forcing);
    pp.queryAdd("forcing_alpha", m_options.forcing_alpha);
    pp.queryAdd("forcing_gamma", m_options.forcing_gamma);
    pp.queryAdd("forcing_max", m_options.forcing_max);
    pp.queryAdd("probe_relative_size", m_options.probe_relative_size);
    pp.queryAdd("max_newton_iterations", m_options.max_newton_iterations);
    pp.queryAdd("max_linear_iterations", m_options.max_linear_iterations);
    pp.queryAdd("restart_length", m_options.restart_length);
    pp.queryAdd("use_preconditioner", m_options.use_preconditioner);
    pp.queryAdd("linear_verbosity", m_options.linear_verbosity);
    pp.queryAdd("pc_source_diagonal", m_source_diagonal_enabled);
    pp.queryAdd("initial_guess_relative_margin", m_guess_options.relative_margin);
    pp.queryAdd("pc_pressure_coupling", m_pressure_coupling);
    ThermalPCOptions pc;
    pp.queryAdd("pc_cycles", pc.cycles);
    pp.queryAdd("pc_agglomeration", pc.agglomeration);
    pp.queryAdd("pc_consolidation", pc.consolidation);
    pp.queryAdd("pc_semicoarsening", pc.semicoarsening);
    pp.queryAdd("pc_max_coarsening_level", pc.max_coarsening_level);
    pp.queryAdd("pc_max_semicoarsening_level", pc.max_semicoarsening_level);
    pp.queryAdd("pc_semicoarsening_direction", pc.semicoarsening_direction);
    pp.queryAdd("pc_transport", pc.transport);
    pp.queryAdd("pc_transport_inner_max_iterations", pc.transport_inner_max_iterations);
    pp.queryAdd("pc_transport_inner_relative_tolerance", pc.transport_inner_relative_tolerance);
    pp.queryAdd("pc_transport_upwind_stabilization", pc.transport_upwind_stabilization);
    m_options.preconditioner = pc;
    AMREX_ALWAYS_ASSERT(m_push_iterations > 0 && std::isfinite(m_push_tolerance) &&
        m_push_tolerance > 0 && m_push_tolerance < 1 &&
        m_outer_iterations > 0 && std::isfinite(m_electric_scale) &&
        m_electric_scale > 0 && std::isfinite(m_energy_scale) && m_energy_scale > 0);
    m_energy.Define(&m_simulation, "none", "none", {{energy_name,m_energy_scale}});
    m_combined.Define(&m_simulation, "Efield_fp", "none", {{energy_name,m_energy_scale}},
                      m_electric_scale);
    m_field.Define(m_solver.m_E);
    m_field_rhs.Define(m_solver.m_E);
    m_field_work.Define(m_solver.m_E);
    m_pressure_work.Define(m_solver.m_E);
    m_saved_field.Define(m_solver.m_E);
    m_saved_field_old.Define(m_solver.m_Eold);
    m_saved_field_prev.Define(m_solver.m_Eprev);
    m_field_pc.define(m_field, &m_solver, m_solver.m_nlsolver->GetPreconditionerType());
    auto const& u = m_energy.getMultiFabBlock(energy_name,0);
    auto const& rho = *m_simulation.m_fields.get(FieldType::rho_fp,0);
    m_old_energy.define(u.boxArray(),u.DistributionMap(),1,0);
    m_endpoint.define(u.boxArray(),u.DistributionMap(),1,0);
    m_bound_density.define(u.boxArray(),u.DistributionMap(),1,0);
    m_old_charge.define(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
    m_endpoint_charge.define(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
    m_push_charge.define(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
    m_push_difference.define(rho.boxArray(),rho.DistributionMap(),1,0);
    for (int d=0;d<3;++d) {
        auto const& j=*m_simulation.m_fields.get(FieldType::current_fp,Direction{d},0);
        m_old_current[d]=std::make_unique<amrex::MultiFab>(j.boxArray(),j.DistributionMap(),1,j.nGrowVect());
        m_push_current[d]=std::make_unique<amrex::MultiFab>(j.boxArray(),j.DistributionMap(),1,j.nGrowVect());
    }
    m_magnetic.define(u.boxArray(),u.DistributionMap(),3,0);
    m_source.define(u.boxArray(),u.DistributionMap(),1,0);
    m_nodal_rates.define(rho.boxArray(),rho.DistributionMap(),SourceComponent::Count,1);
    int species_components=1;
    if (m_model.m_include_joule_heating || m_model.m_include_temperature_relaxation) {
        auto descriptors=DescribeNativeThermalSpecies(m_simulation.GetPartContainer());
        for (auto& species:descriptors) {
            species.relaxation_excluded=m_model.IsRelaxationExcluded(species.name);
            auto const overlay=m_model.m_eta_per_species.find(species.name);
            if (overlay!=m_model.m_eta_per_species.end()) {
                species.has_resistivity_overlay=true;
                species.resistivity_overlay=overlay->second;
            }
        }
        species_components=std::max(1,static_cast<int>(descriptors.size())*IonSourceComponent::Count);
        KineticSpeciesOptions species_options;
        species_options.relaxation=m_model.m_include_temperature_relaxation;
        species_options.single_precision_comms=WarpX::do_single_precision_comms;
        auto const ghosts=amrex::max(m_simulation.get_ng_depos_J(),m_simulation.get_ng_depos_rho());
        m_species=std::make_unique<KineticThermalSpecies>(m_geometry,u.boxArray(),
            u.DistributionMap(),ghosts,std::move(descriptors),species_options);
        // Aggregate MM responses cannot identify individual raw species
        // deposits. Use one consistent materialized particle response in BOTH
        // blocks until a per-species tangent exists. Existing MM paths remain
        // unchanged for configurations that do not request these sources.
        bool const constant_eta_mm=NativeConstantEtaMassMatrixSupported(m_simulation);
        if (m_solver.m_use_mass_matrices_jacobian && constant_eta_mm) {
            // The sole populated species has raw_charge/raw_sum == 1.
            // Global constant-eta Joule heat depends only on the live total
            // current and physical density/gates, not a species tangent.
            // Both native MM operators already consume the same corrected Eaux.
            amrex::Print()<<"Eulerian single-species constant-eta Joule: corrected-force "
                "current+density and endpoint MM; fresh full-push acceptance\n";
        }
        if (m_solver.m_use_mass_matrices_jacobian && !constant_eta_mm) {
            m_solver.m_use_mass_matrices_jacobian=false;
            m_solver.m_mass_matrices_density_projection=false;
            m_solver.m_use_mass_matrices=m_solver.m_use_mass_matrices_pc;
            m_solver.InvalidateMassMatrices();
            amrex::Print()<<"Eulerian species sources: full particle Jacobian response; aggregate species MM unavailable\n";
        }
    }
    m_ion_rates.define(rho.boxArray(),rho.DistributionMap(),species_components,1);
    m_additive_eta.define(rho.boxArray(),rho.DistributionMap(),1,1);
    InitializeAdditiveResistivity();
    RefreshDensityDependentContext();
    auto const options = m_model.EulerianMomentOptions();
    if (m_coupled && m_pressure_coupling) {
        FrozenPressureFieldOptions pressure;
        pressure.gamma=m_model.m_gamma;
        pressure.boundary=options.boundary;
        pressure.azimuthal_modes=options.azimuthal_modes;
        pressure.transformed_ohm_form=m_model.m_esolve_tensor || m_model.m_esolve_curlcurl;
        m_pressure_pc=std::make_unique<FrozenPressureFieldCoupling>(m_geometry,
            u.boxArray(),u.DistributionMap(),pressure);
        for (int c=0;c<3;++c) {
            auto const& e=*m_solver.m_E.getArrayVec()[0][c];
            m_pressure_weight[c]=std::make_unique<amrex::MultiFab>(
                e.boxArray(),e.DistributionMap(),1,0);
        }
    }
    for (int d=0;d<AMREX_SPACEDIM;++d) {
        m_number_flux[d]=std::make_unique<amrex::MultiFab>(
            amrex::convert(u.boxArray(),amrex::IntVect::TheDimensionVector(d)),
            u.DistributionMap(),1,0);
    }
    m_stage_options.retain_current_remainder=m_current_remainder_requested;
    m_stage_options.gamma=m_model.m_gamma;
    m_stage_options.theta=m_solver.m_theta;
    m_stage_options.conduction_theta=m_solver.m_theta;
    pp.queryAdd("conduction_theta",m_stage_options.conduction_theta);
    m_stage_options.temperature_floor_ev=m_model.m_cond_te_floor;
    m_guess_options.theta=m_solver.m_theta;
    m_guess_options.gamma=m_model.m_gamma;
    m_guess_options.temperature_floor_ev=m_model.m_cond_te_floor;


    m_stage_options.order=m_model.m_cond_fd_order;
    m_stage_options.cross_mode=1;
    m_stage_options.limiter_width=m_model.m_cond_fd_limiter_width;
    m_stage_options.free_streaming_fraction=m_model.m_cond_flux_limit_factor;
    m_stage_options.use_compiled_conductivity=m_model.m_include_thermal_conduction;
    m_stage_options.parallel_executor=m_model.m_kappa_par;
    m_stage_options.perpendicular_executor=m_model.m_kappa_perp;
    if(m_model.m_kappa_par_parser)m_stage_options.parallel_owner=std::make_shared<amrex::Parser>(*m_model.m_kappa_par_parser);
    if(m_model.m_kappa_perp_parser)m_stage_options.perpendicular_owner=std::make_shared<amrex::Parser>(*m_model.m_kappa_perp_parser);
    m_stage_options.conductivity.isotropic=m_model.m_cond_isotropic;
    m_stage_options.conductivity.isotropic_B=m_model.m_cond_iso_B;
    m_stage_options.conductivity.unmagnetized_parallel=true;
    for (int d=0;d<AMREX_SPACEDIM;++d) {
        for (int side=0;side<2;++side) {
            auto& bc=m_stage_options.boundary[d][side];
            switch (m_model.m_cond_bc[d][side]) {
            case 0: bc.kind=BoundaryKind::Adiabatic; break;
            case 1: bc.kind=BoundaryKind::Reservoir; bc.value=m_model.m_cond_bc_Te[d][side]; break;
            case 2: bc.kind=BoundaryKind::PrescribedFlux; bc.value=-m_model.m_cond_bc_q[d][side]; break;
            case 3: bc.kind=BoundaryKind::Leg; bc.value=m_model.m_cond_leg_Te_wall; break;
            default: amrex::Abort("Unknown Eulerian thermal boundary");
            }
            if (bc.kind==BoundaryKind::Leg || bc.kind==BoundaryKind::Reservoir) {
                bc.leg_length=m_model.m_cond_leg_length;
                bc.flux_limit=bc.kind==BoundaryKind::Leg ? m_model.m_cond_leg_flux_limit
                                                         : m_model.m_cond_wall_flux_limit;
                bc.cap_form=m_model.m_cond_wall_flux_cap_form==1 ? WallCapForm::Sonic
                                                                : WallCapForm::FreeStreaming;
                if (bc.flux_limit>0 && bc.cap_form==WallCapForm::Sonic) {
                    bc.ion_mass=m_model.WallCapIonMass();
                }
            }
        }
    }
}

void DarwinThermalAdvance::RefreshDensityDependentContext ()
{
    AMREX_ALWAYS_ASSERT(!m_open);
    auto const& u=*m_simulation.m_fields.get(FieldType::hybrid_electron_energy_fp,0);
    if (m_model.m_include_electron_viscosity || m_model.m_include_hyper_resistivity_term) {
        TrialDissipationOptions dissipation;
        dissipation.viscosity=m_model.m_include_electron_viscosity;
        dissipation.viscous=m_model.ViscosityPointParams(0);
        dissipation.eta_h=m_model.m_eta_h;
        dissipation.eta_h_depends_on_B=m_model.m_hyper_resistivity_has_B_dependence;
        dissipation.transformed_electric_solve=m_model.m_esolve_tensor || m_model.m_esolve_curlcurl;
        if (m_model.m_include_hyper_resistivity_term) {
            dissipation.hyper=m_model.m_hyper_resistivity_curlcurl ? HyperMode::AmpereCurlCurl :
                (m_model.m_hyper_res_curl_curl ? HyperMode::InteriorCurlCurl : HyperMode::Laplacian);
        }
        for (int d=0;d<AMREX_SPACEDIM;++d) {
            for (int side=0;side<2;++side) {
                auto const boundary=side ? WarpX::field_boundary_hi[d] : WarpX::field_boundary_lo[d];
                auto& selected=dissipation.boundary[d][side];
                if (m_geometry.isPeriodic(d)) { selected=DissipationBoundary::Periodic; }
                else if (d==0 && side==0) { selected=DissipationBoundary::Axis; }
                else if (boundary==FieldBoundaryType::PEC) { selected=DissipationBoundary::PEC; }
                else if (boundary==FieldBoundaryType::PMC) { selected=DissipationBoundary::PMC; }
                else { amrex::Abort("Eulerian trial dissipation needs axis/PEC/PMC/periodic boundaries"); }
            }
        }
        m_dissipation=std::make_unique<EulerianDissipation>(m_geometry,u.boxArray(),u.DistributionMap(),dissipation);
    }
    auto options = m_model.EulerianMomentOptions();
    options.retain_current_remainder=m_current_remainder_requested;
    auto make_moments = [&] () {
        return std::make_unique<KineticThermalMoments>(m_geometry,u.boxArray(),
                                                      u.DistributionMap(),options);
    };
    m_old_moments=make_moments(); m_stage_moments=make_moments(); m_endpoint_moments=make_moments();
    ConductionActivityOptions activity;
    activity.closure=ConductionCellClosure::AllContributors;
    activity.legacy_density_floor=m_model.m_qdsmc_n_floor;
    activity.common_density_floor=m_model.m_n_floor;
    activity.halo_unfreeze=m_model.m_qdsmc_halo_unfreeze;
    m_conduction_activity=std::make_unique<NativeConductionActivity>(
        m_geometry,u.boxArray(),u.DistributionMap(),activity);
    m_density_epoch=m_model.DensityControlEpoch();
}

void DarwinThermalAdvance::InitializeAdditiveResistivity ()
{
    auto const end_region=m_model.EndRegion(0);
    amrex::GpuArray<int,3> const nodal{1,1,1};
    for (amrex::MFIter mfi(m_additive_eta);mfi.isValid();++mfi) {
        auto const eta=m_additive_eta.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            eta(i,j,k)=end_region.resistivity*end_region.Weight(i,j,k,nodal);
        });
    }
}

DarwinThermalAdvance::~DarwinThermalAdvance () = default;

} // namespace warpx::thermal
