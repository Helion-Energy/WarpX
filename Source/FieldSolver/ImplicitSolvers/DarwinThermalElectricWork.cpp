/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "DarwinThermalAdvance.H"
#include "ThetaImplicitHybrid.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <AMReX_ParmParse.H>

namespace warpx::thermal {
using warpx::fields::FieldType;
using ablastr::fields::Direction;

void DarwinThermalAdvance::InitializeIonElectricWork ()
{
    amrex::ParmParse("implicit_evolve.thermal").queryAdd(
        "audit_ion_electric_work",m_ion_electric_work_enabled);
    if (!m_ion_electric_work_enabled) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_simulation.finestLevel()==0 && !EB::enabled() &&
        !m_model.m_esolve_tensor && !m_model.m_esolve_curlcurl &&
        !WarpX::use_filter && !m_solver.m_adjoint_gather_ghosts &&
        m_simulation.GetPartContainer().m_E_ext_particle_s=="none" &&
        (!m_model.HasResistivity() || m_solver.m_use_resistive_push_correction),
        "Ion electric-work capture requires unfiltered single-level native Ohm fields, "
        "no EB/particle-external electric field, and the native resistive push correction");
    auto const& lo=m_simulation.GetFieldBoundaryLo();
    auto const& hi=m_simulation.GetFieldBoundaryHi();
    for (int d=0;d<AMREX_SPACEDIM;++d) {
        auto supported=[](FieldBoundaryType b) {
            return b==FieldBoundaryType::PEC || b==FieldBoundaryType::PMC ||
                b==FieldBoundaryType::Periodic;
        };
        bool low=supported(lo[d]);
#ifdef WARPX_DIM_RZ
        low=low || (d==0 && m_simulation.Geom(0).ProbLo(0)==0 && lo[d]==FieldBoundaryType::None);
#endif
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(low && supported(hi[d]),
            "Ion electric-work capture requires qualified PEC/PMC/periodic field boundaries");
    }
    for (int c=0;c<3;++c) {
        auto const& native=*m_simulation.m_fields.get(FieldType::Efield_fp,Direction{c},0);
        auto const& aux=*m_simulation.m_fields.get(FieldType::Efield_aux,Direction{c},0);
        AMREX_ALWAYS_ASSERT(aux.ixType().nodeCentered() &&
            aux.nGrowVect().allGE(m_simulation.get_ng_fieldgather()));
        m_ion_work_raw[c]=std::make_unique<amrex::MultiFab>(
            native.boxArray(),native.DistributionMap(),native.nComp(),native.nGrowVect());
        m_ion_work_aux[c]=std::make_unique<amrex::MultiFab>(
            aux.boxArray(),aux.DistributionMap(),aux.nComp(),aux.nGrowVect());
        m_ion_work_total[c]=std::make_unique<amrex::MultiFab>(
            aux.boxArray(),aux.DistributionMap(),aux.nComp(),aux.nGrowVect());
        m_ion_work_base_fp[c]=std::make_unique<amrex::MultiFab>(
            native.boxArray(),native.DistributionMap(),native.nComp(),native.nGrowVect());
        m_ion_work_remainder[c]=std::make_unique<amrex::MultiFab>(
            aux.boxArray(),aux.DistributionMap(),aux.nComp(),aux.nGrowVect());
        m_ion_work_raw[c]->setVal(0.);
        m_ion_work_raw_view[c]=m_ion_work_raw[c].get();
    }
}

ablastr::fields::VectorField const* DarwinThermalAdvance::IonElectricWorkOutput ()
{
    if (!m_ion_electric_work_enabled) { return nullptr; }
    m_ion_electric_work_ready=false;
    m_ion_work_base_ready=false;
    return &m_ion_work_raw_view;
}

void DarwinThermalAdvance::CaptureIonElectricWorkBase (
    ablastr::fields::VectorField const& nores)
{
    if (!m_ion_electric_work_enabled) { return; }
    m_ion_electric_work_ready=false;
    for (int c=0;c<3;++c) {
        auto const& field=*nores[c];
        amrex::MultiFab::Copy(*m_ion_work_base_fp[c],field,0,0,field.nComp(),field.nGrowVect());
    }
    m_ion_work_base_ready=true;
}

void DarwinThermalAdvance::PrepareIonElectricWorkField ()
{
    if (!m_ion_electric_work_enabled) { return; }
    m_ion_electric_work_ready=false;
    auto const& lo=m_simulation.GetFieldBoundaryLo();
    auto const& hi=m_simulation.GetFieldBoundaryHi();
    auto const ng=m_simulation.get_ng_fieldgather();
    auto const& geometry=m_simulation.Geom(0);
    auto const& periodic=geometry.periodicity();
    auto sync=[&](ablastr::fields::VectorField const& view) {
        if (WarpX::do_single_precision_comms) {
            for (auto* mf:view) {
                ablastr::utils::communication::FillBoundary(
                    *mf,mf->nGrowVect(),true,periodic,true);
            }
        } else {
            amrex::Vector<amrex::MultiFab*> fields(view.begin(),view.end());
            amrex::FillBoundaryAndSync_nowait(fields,periodic);
            amrex::FillBoundaryAndSync_finish(fields);
        }
    };
    // A freshly zeroed component has no tangential box/periodic ghosts.
    // Boundary parity reads those ghosts at physical-boundary/box corners;
    // populate them before reflection. The final native sync below remains
    // necessary after boundary rows have been projected.
    sync(m_ion_work_raw_view);
    // Apply the HOMOGENEOUS part of the same native Ohm boundary stack.
    // The resistive shell's prescribed B-dependent electric row is affine:
    // its global-eta component is zero, already imposed by the PEC projector.
    // Reapplying that affine drive to this component would double count work.
    PEC::ApplyPECtoEfield(m_ion_work_raw_view,lo,hi,FieldBoundaryType::PEC,
        ng,geometry,0,PatchType::fine,{});
    PEC::ApplyPECtoBfield(m_ion_work_raw_view,lo,hi,FieldBoundaryType::PMC,
        ng,geometry,0,PatchType::fine,{});
#ifdef WARPX_DIM_RZ
    m_simulation.ApplyFieldBoundaryOnAxis(m_ion_work_raw_view[0],
        m_ion_work_raw_view[1],m_ion_work_raw_view[2],0);
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_ion_work_base_ready,
        "Ion electric-work receipt has no current nores/base field capture");
    sync(m_ion_work_raw_view);
    ablastr::fields::VectorField aux,base,base_aux;
    for (int c=0;c<3;++c) {
        aux[c]=m_ion_work_aux[c].get();
        base[c]=m_ion_work_base_fp[c].get();
        base_aux[c]=m_ion_work_remainder[c].get();
    }
    m_simulation.InterpolateLevelZeroFieldToAux(aux,m_ion_work_raw_view);
    // The nores field already contains the native affine boundary rows. Keep
    // them and apply only the native sync/centering map to this private copy.
    sync(base);
    m_simulation.InterpolateLevelZeroFieldToAux(base_aux,base);
    // At finest_level=0 native FillBoundaryAux(ng) exchanges no levels:
    // interpolation already fills these guards from the synchronized source.
    // Another exchange would add an unmatched float rounding when single-
    // precision communication is selected.
}

void DarwinThermalAdvance::CaptureIonElectricWorkTotal ()
{
    if (!m_ion_electric_work_enabled) { return; }
    for (int c=0;c<3;++c) {
        auto const& applied=*m_simulation.m_fields.get(FieldType::Efield_aux,Direction{c},0);
        amrex::MultiFab::Copy(*m_ion_work_total[c],applied,0,0,applied.nComp(),applied.nGrowVect());
        // Actual P - native-centered N includes the difference between the
        // recovered/constrained iterate and raw Ohm field, stage differences,
        // and communication/addition rounding. It is not solely the nonlinear
        // residual. A new projector would discard affine/drive contributions.
        amrex::MultiFab::LinComb(*m_ion_work_remainder[c],1.,applied,0,
            -1.,*m_ion_work_remainder[c],0,0,applied.nComp(),applied.nGrowVect());
    }
    m_ion_electric_work_ready=true;
}

bool DarwinThermalAdvance::PrepareIonElectricWorkReceipt ()
{
    if (!m_ion_electric_work_enabled) { return true; }
    if (!m_ion_electric_work_ready) { return false; }
    IonElectricWorkFields fields;
    fields.filled_ghosts=m_simulation.get_ng_fieldgather();
    for (int c=0;c<3;++c) {
        fields.component[c]=m_ion_work_aux[c].get();
        fields.total[c]=m_ion_work_total[c].get();
        fields.remainder[c]=m_ion_work_remainder[c].get();
    }
    // Before endpoint extrapolation, absorption and OU: every charged
    // trajectory contributes, including excluded alphas and departing ions.
    m_ion_work_receipt=MeasureNativeImplicitIonElectricWork(
        m_simulation,0,m_dt,fields,
        NativeIonWorkContract::FullBorisGatherWithoutAdditionalMomentumChanges);
    return m_ion_work_receipt.valid && m_ion_work_receipt.has_total_field &&
        m_ion_work_receipt.has_remainder_field &&
        m_ion_work_receipt.gather_contract==IonElectricGatherContract::RecordedFinalGather;
}

void DarwinThermalAdvance::PrintIonElectricWorkReceipt () const
{
    if (!m_ion_electric_work_enabled) { return; }
    AMREX_ALWAYS_ASSERT(m_ion_work_receipt.valid);
    using C=IonElectricWorkComponent;
    std::size_t s=0;
    auto& particles=m_simulation.GetPartContainer();
    for (auto const& name:particles.GetSpeciesNames()) {
        if (particles.GetParticleContainerFromName(name).getCharge()==0) { continue; }
        auto const& diagnostic=m_ion_work_receipt.diagnostics.at(s);
        auto const& w=m_ion_work_receipt.species.at(s++);
        using D=IonElectricWorkDiagnostic;
        amrex::Print()<<"Eulerian accepted ion electric work [J]: species="<<name
            <<" global_eta="<<w[C::ComponentWork]<<" global_eta_abs="<<w[C::ComponentWorkAbs]
            <<" total_field="<<w[C::TotalFieldWork]<<" kinetic_change="<<w[C::KineticChange]
            <<" work_defect="<<w[C::TotalWorkDefect]<<" work_defect_abs="<<w[C::TotalWorkDefectAbs]
            <<" nr_global_eta="<<w[C::NonrelativisticWork]<<" physical_weight="<<w[C::Weight]
            <<" applied_remainder="<<diagnostic[D::RemainderWork]
            <<" applied_remainder_abs="<<diagnostic[D::RemainderWorkAbs]
            <<" nr_applied_remainder="<<diagnostic[D::NonrelativisticRemainderWork]
            <<" remainder_cauchy_bound="<<diagnostic[D::RemainderCauchyWorkBound]
            <<" global_eta_cauchy_bound="<<diagnostic[D::ComponentCauchyWorkBound]
            <<" kinetic_old="<<diagnostic[D::OldKineticEnergy]
            <<" kinetic_endpoint="<<diagnostic[D::EndpointKineticEnergy]
            <<" momentum_roundoff_reference="<<diagnostic[D::MomentumRoundoffReference]
            <<" resistive_push_correction="<<m_solver.m_use_resistive_push_correction<<"\n";
    }
}
} // namespace warpx::thermal
