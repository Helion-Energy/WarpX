/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeJouleWork.H"
#include "NativeEndpointField.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridOhmDampingFields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include <AMReX_Reduce.H>
#include <AMReX_ParallelDescriptor.H>
#include "BoundaryConditions/WarpX_PEC.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>

namespace warpx::thermal {
using warpx::fields::FieldType;
using ablastr::fields::Direction;
using ablastr::fields::VectorField;
using amrex::MultiFab;

bool NativeJouleWorkEnabled ()
{
    bool enabled=false;
    amrex::ParmParse("implicit_evolve.thermal").query("joule_applied_edge_work",enabled);
    return enabled;
}

void ValidateNativeJouleWork (WarpX& sim)
{
    if (!NativeJouleWorkEnabled()) { return; }
    auto const& model=*sim.get_pointer_HybridPICModel();
#ifndef WARPX_DIM_RZ
    amrex::Abort("Applied Joule edge work is initially qualified only in RZ");
#endif
    bool smooth=false;
    amrex::ParmParse("endpoint_diagnostic").query("smooth_force",smooth);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NativeEndpointEnabled() && smooth &&
        model.UsesEulerianElectronEnergy() && model.m_include_joule_heating &&
        sim.maxLevel()==0 && !EB::enabled() && WarpX::ncomps==1 &&
        !WarpX::use_filter && !WarpX::do_single_precision_comms,
        "Applied Joule edge work requires native smooth endpoint Eulerian Joule, one RZ level/mode, no EB/filter and double-precision communication");
    auto const symbols=model.m_resistivity_parser->symbols();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!model.m_has_heating_resistivity &&
        !model.m_has_per_species_eta && !model.m_joule_redirect_to_ions &&
        !model.m_joule_heating_taper && model.m_qdsmc_source_taper_n<=0 &&
        model.m_joule_heating_n_min<=model.m_n_floor &&
        !model.m_holmstrom_vacuum_region && !model.m_has_external_current &&
        !model.m_pec_conductor_wall_rows && !model.m_esolve_tensor && !model.m_esolve_curlcurl &&
        !symbols.contains("rho") && !symbols.contains("J") &&
        !symbols.contains("Te") && !symbols.contains("t"),
        "Applied Joule edge work initially requires constant common physical eta without heating/species overlays, density Holmstrom, redirect, extra density gates/tapers or transformed/conductor/external-current closures");
    // Validation only, outside residual assembly. The source itself never
    // recomputes this parser: it consumes the same-kernel recorded component.
    auto constant=amrex::ParmParse().makeParser(model.m_eta_expression,{});
    amrex::Real const eta=constant.compileHost<0>()();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(eta) && eta>=0,
        "Applied Joule work requires finite nonnegative constant eta");
    for (int d=0;d<AMREX_SPACEDIM;++d) {
        auto supported=[](FieldBoundaryType b) {
            return b==FieldBoundaryType::PEC || b==FieldBoundaryType::PMC ||
                b==FieldBoundaryType::Periodic;
        };
        bool low=supported(WarpX::field_boundary_lo[d]);
#ifdef WARPX_DIM_RZ
        low=low || (d==0 && sim.Geom(0).ProbLo(0)==0 &&
                    WarpX::field_boundary_lo[d]==FieldBoundaryType::None);
#endif
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(low && supported(WarpX::field_boundary_hi[d]),
            "Applied Joule work requires axis/PEC/PMC/periodic homogeneous electric boundaries");
    }
}

NativeJouleWork::NativeJouleWork (WarpX& sim)
    : m_geometry(sim.get_pointer_HybridPICModel()->ElectronThermalGeometry())
{
    ValidateNativeJouleWork(sim);
    auto const end=sim.get_pointer_HybridPICModel()->EndRegion(0);
    m_end_active=end.holmstrom || end.resistivity>0.;
    for (int c=0;c<3;++c) {
        auto const& native=*sim.m_fields.get(FieldType::Efield_fp,Direction{c},0);
        for (auto* f : {&m_saved_electric[c],&m_electric[c]}) {
            f->define(native.boxArray(),native.DistributionMap(),native.nComp(),native.nGrowVect());
            f->setVal(0.);
        }
        if (m_end_active) {
            for (auto* f : {&m_end_resistivity[c],&m_end_holmstrom[c]}) {
                f->define(native.boxArray(),native.DistributionMap(),native.nComp(),native.nGrowVect());
                f->setVal(0.);
            }
            m_end_power[c].define(native.boxArray(),native.DistributionMap(),4,0);
            m_end_power[c].setVal(0.);
            m_owner[c]=native.OwnerMask(m_geometry.periodicity());
        }
    }
}

bool NativeJouleWork::Evaluate (WarpX& sim, const MultiFab& charge,
                               EulerianDissipation* dissipation)
{
    m_valid=false; DiscardEndReceipt();
    auto const native=sim.m_fields.get_alldirs(FieldType::Efield_fp,0);
    VectorField component{&m_electric[0],&m_electric[1],&m_electric[2]};
    for (int c=0;c<3;++c) {
        AMREX_ALWAYS_ASSERT(native[c]->boxArray()==m_electric[c].boxArray() &&
            native[c]->DistributionMap()==m_electric[c].DistributionMap() &&
            native[c]->nComp()==m_electric[c].nComp() &&
            native[c]->nGrowVect()==m_electric[c].nGrowVect());
        MultiFab::Copy(m_saved_electric[c],*native[c],0,0,native[c]->nComp(),native[c]->nGrowVect());
        m_electric[c].setVal(0.);
    }
    VectorField end_eta{&m_end_resistivity[0],&m_end_resistivity[1],&m_end_resistivity[2]};
    VectorField end_gate{&m_end_holmstrom[0],&m_end_holmstrom[1],&m_end_holmstrom[2]};
    HybridOhmDampingFields damping{end_eta,end_gate};
    sim.get_pointer_HybridPICModel()->HybridPICSolveE(native,
        sim.m_fields.get_alldirs(FieldType::current_fp,0),
        sim.m_fields.get_alldirs(FieldType::Bfield_fp,0),charge,
        sim.GetEBUpdateEFlag()[0],0,false,true,dissipation,&component,m_end_active ? &damping : nullptr);
    for (int c=0;c<3;++c) {
        MultiFab::Copy(*native[c],m_saved_electric[c],0,0,native[c]->nComp(),native[c]->nGrowVect());
    }
    auto const& geometry=sim.Geom(0);
    auto project=[&] (VectorField const& fields) {
        auto sync=[&]() {
            amrex::Vector<MultiFab*> all(fields.begin(),fields.end());
            amrex::FillBoundaryAndSync_nowait(all,geometry.periodicity());
            amrex::FillBoundaryAndSync_finish(all);
        };
        // Tangential box/periodic images must precede normal corner parity.
        sync();
        auto const& lo=sim.GetFieldBoundaryLo();
        auto const& hi=sim.GetFieldBoundaryHi();
        auto const ng=sim.get_ng_fieldgather();
        PEC::ApplyPECtoEfield(fields,lo,hi,FieldBoundaryType::PEC,
            ng,geometry,0,PatchType::fine,{});
        PEC::ApplyPECtoBfield(fields,lo,hi,FieldBoundaryType::PMC,
            ng,geometry,0,PatchType::fine,{});
#ifdef WARPX_DIM_RZ
        sim.ApplyFieldBoundaryOnAxis(fields[0],fields[1],fields[2],0);
#endif
        sync();
    };
    project(component);
    bool valid=true;
    for (auto const& f:m_electric) { valid=f.is_finite(0,f.nComp(),0) && valid; }
    if (m_end_active) {
        project(end_eta); project(end_gate);
        auto const jp=sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        auto const ji=sim.m_fields.get_alldirs(FieldType::current_fp,0);
        for (int c=0;c<3;++c) {
            valid=m_end_resistivity[c].is_finite(0,1,0) &&
                m_end_holmstrom[c].is_finite(0,1,0) && valid;
            for (amrex::MFIter mfi(m_end_power[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
                auto const q=m_end_power[c].array(mfi);
                auto const p=jp[c]->const_array(mfi), ion=ji[c]->const_array(mfi);
                auto const e=m_end_resistivity[c].const_array(mfi);
                auto const h=m_end_holmstrom[c].const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    // Actual force/interval electron current, not endpoint or
                    // inertia Je. These are signed model work densities only.
                    amrex::Real const eta=p(i,j,k)*e(i,j,k);
                    amrex::Real const holm=(p(i,j,k)-ion(i,j,k))*h(i,j,k);
                    q(i,j,k,0)=eta; q(i,j,k,1)=std::abs(eta);
                    q(i,j,k,2)=holm; q(i,j,k,3)=std::abs(holm);
                });
            }
        }
    }
    m_valid=valid;
    return valid;
}

std::array<const MultiFab*,3> NativeJouleWork::Electric () const
{
    return {&m_electric[0],&m_electric[1],&m_electric[2]};
}

std::array<const MultiFab*,3> NativeJouleWork::EndResistivity () const
{
    AMREX_ALWAYS_ASSERT(m_end_active);
    return {&m_end_resistivity[0],&m_end_resistivity[1],&m_end_resistivity[2]};
}
std::array<const MultiFab*,3> NativeJouleWork::EndHolmstrom () const
{
    AMREX_ALWAYS_ASSERT(m_end_active);
    return {&m_end_holmstrom[0],&m_end_holmstrom[1],&m_end_holmstrom[2]};
}

bool NativeJouleWork::PrepareEndReceipt (amrex::Real dt)
{
    DiscardEndReceipt(); m_end_energy.fill(0.);
    if (!m_end_active) { return true; }
    if (!m_valid || !std::isfinite(dt) || dt<=0.) { return false; }
    for (int c=0;c<3;++c) {
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,
                         amrex::ReduceOpSum,amrex::ReduceOpSum> ops;
        amrex::ReduceData<amrex::Real,amrex::Real,amrex::Real,amrex::Real> data(ops);
        using Tuple=decltype(data)::Type;
        auto const volume=MakeQdsmcVolumeElement(m_geometry,m_end_power[c].ixType());
        for (amrex::MFIter mfi(m_end_power[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
            auto const q=m_end_power[c].const_array(mfi);
            auto const owner=m_owner[c]->const_array(mfi);
            ops.eval(mfi.tilebox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
                if (!owner(i,j,k)) { return {0.,0.,0.,0.}; }
                amrex::Real const v=volume(i,j,k);
                return {v*q(i,j,k,0),v*q(i,j,k,1),v*q(i,j,k,2),v*q(i,j,k,3)};
            });
        }
        auto const value=data.value();
        m_end_energy[0]+=amrex::get<0>(value);m_end_energy[1]+=amrex::get<1>(value);
        m_end_energy[2]+=amrex::get<2>(value);m_end_energy[3]+=amrex::get<3>(value);
    }
    amrex::ParallelDescriptor::ReduceRealSum(m_end_energy.data(),4);
    for (auto& e:m_end_energy) { e*=dt; if (!std::isfinite(e)) { return false; } }
    if (m_end_energy[1]<0. || m_end_energy[3]<0.) { return false; }
    m_end_prepared=true;
    return true;
}

void NativeJouleWork::CommitEndReceipt ()
{
    if (!m_end_active) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_end_prepared,
        "Numerical end work receipt must be prepared and consumed once");
    constexpr std::array<const char*,4> names{"end_resistivity","end_resistivity_abs",
                                             "end_holmstrom","end_holmstrom_abs"};
    for (int c=0;c<4;++c) {
        amrex::Print()<<"Eulerian accepted numerical sink [J]: channel="<<names[c]
            <<" energy="<<m_end_energy[c]<<"\n";
    }
    DiscardEndReceipt();
}
}
