/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#include "NativeCircuitFieldImpulse.H"
#include "NativeBoundaryFieldRate.H"
#include "DarwinABoundary.H"
#include "Circuit/CircuitCoupling.H"
#include "Circuit/Coils/FluxProbes.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
using MF=amrex::MultiFab;
using View=ablastr::fields::VectorField;
using Field=amrex::Array<MF,3>;
using FT=warpx::fields::FieldType;
namespace {
View V(Field& x){return {&x[0],&x[1],&x[2]};}
void Allocate(Field& x,View const& layout){for(int c=0;c<3;++c){x[c].define(layout[c]->boxArray(),
    layout[c]->DistributionMap(),1,layout[c]->nGrowVect());x[c].setVal(0.);}}
bool AllRanks(bool valid,std::string& error){int ok=valid?1:0;amrex::ParallelDescriptor::ReduceIntMin(ok);
    if(!ok && valid)error="Circuit field impulse preparation failed on another rank";
    return ok!=0;}
}
namespace warpx::darwin {
NativeCircuitFieldImpulse::NativeCircuitFieldImpulse(WarpX& w,CircuitCoupler& c):m_warp(w),m_coupler(c)
{
    auto const e=w.m_fields.get_alldirs(FT::Efield_fp,0);
    Allocate(m_zero,e);Allocate(m_reference,e);Allocate(m_impulse,e);
    Allocate(m_a,w.m_fields.get_alldirs("hybrid_A_fp",0));
    Allocate(m_b,w.m_fields.get_alldirs(FT::Bfield_fp,0));Allocate(m_coil_b,V(m_b));
    Allocate(m_j,w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0));Allocate(m_coil_j,V(m_j));
    for(auto const* field:{&m_zero,&m_reference,&m_a,&m_b,&m_j,&m_impulse,&m_coil_b,&m_coil_j})
        for(auto const& component:*field)for(amrex::MFIter it(component);it.isValid();++it){
            auto const& fab=component[it];auto const begin=reinterpret_cast<std::uintptr_t>(fab.dataPtr());
            m_owned_storage.emplace_back(begin,begin+fab.size()*sizeof(amrex::Real));
        }
}
bool NativeCircuitFieldImpulse::Prepare(amrex::Real time,CircuitCoupler::SourceImpulsePhase phase,std::string& error)
{
    m_ready=false;m_applied=false;m_application=0;m_initial_ready=false;m_coupler.InvalidateNativeEndpointRate();error.clear();
    auto& model=*m_warp.get_pointer_HybridPICModel();auto* subsystem=m_warp.get_pointer_CircuitCoupling();
    bool valid=subsystem && subsystem->Coupler()==&m_coupler && model.m_add_external_fields &&
        m_warp.maxLevel()==0 && !EB::enabled() && WarpX::ncomps==1 && std::isfinite(time) &&
        m_coupler.SupportsSourceImpulse() && m_coupler.RetainedSourcePhaseReady(phase) &&
        model.m_external_vector_potential && !model.m_has_external_current;
    if(!valid)error="Unsupported retained circuit field-impulse context";
    if(!AllRanks(valid,error))return false;
    auto const magnetic=m_warp.m_fields.get_alldirs(FT::Bfield_fp,0);
    for(int c=0;c<3;++c)valid=valid && magnetic[c]->nComp()==1 &&
        magnetic[c]->nGrowVect().allGE(amrex::IntVect(2));
    if(!valid)error="Native source coil inventory requires the consumed grow2 magnetic band";
    if(!AllRanks(valid,error))return false;
    auto const& ext=*model.m_external_vector_potential;auto const& coils=subsystem->GetCoilSet();
    valid=ext.nFields()==coils.size();
    for(int f=0;f<ext.nFields();++f){bool coupled=false;
        for(auto const& coil:coils.coils())coupled=coupled || coil.field_name==ext.FieldName(f);
        valid=valid && coupled && ext.UsesPythonScale(f) && !ext.DeviceDriven(f);
        for(int c=0;c<3;++c)for(auto const* suffix:{"_Aext","_curlAext"}){
            auto const& unit=*m_warp.m_fields.get(ext.FieldName(f)+suffix,ablastr::fields::Direction{c},0);
            auto const& layout=std::string(suffix)=="_Aext"?m_reference[c]:m_coil_b[c];
            valid=valid && unit.boxArray()==layout.boxArray() && unit.DistributionMap()==layout.DistributionMap() &&
                unit.nGrowVect().allGE(layout.nGrowVect());
        }
    }
    if(!valid)error="Circuit field-impulse unit fields or owned scale layout changed";
    if(!AllRanks(valid,error))return false;
    int cached_lo=m_coupler.NativeEndpointBoundaryResponse().empty()?0:1,cached_hi=cached_lo;
    amrex::ParallelDescriptor::ReduceIntMin(cached_lo);
    amrex::ParallelDescriptor::ReduceIntMax(cached_hi);
    if(cached_lo!=cached_hi){error="Circuit impulse H-cache readiness differs across ranks";return false;}
    if(!cached_lo){
        m_coupler.ConfigureDarwinMagneticResponse();
        std::vector<double> h(static_cast<std::size_t>(coils.size())*ext.nFields(),0.);
        for(int f=0;f<ext.nFields();++f){
            for(int c=0;c<3;++c)MF::Copy(m_reference[c],*m_warp.m_fields.get(
                ext.FieldName(f)+"_Aext",ablastr::fields::Direction{c},0),0,0,1,m_reference[c].nGrowVect());
            NativeBoundaryFieldRate(m_warp,V(m_zero),V(m_reference),V(m_a),V(m_b),V(m_j));
            for(int port=0;port<coils.size();++port)h[static_cast<std::size_t>(port)*ext.nFields()+f]=
                warpx::circuit::DiskFluxLinkage(coils.coil(port),m_b[2]);
        }
        bool const configured=m_coupler.SetNativeEndpointBoundaryResponse(h,error);
        if(!AllRanks(configured,error))return false;
    }
    if(!m_initial_coil)m_initial_coil=std::make_unique<NativeCoilField>(m_warp);
    if(!m_coupler.PrepareNativeSourceImpulse(time,phase,error))return false;
    m_initial_coil->SampleAtTime(time);
    m_initial_port_current=m_coupler.SourceEntryCurrents();
    // Register the new immutable inventory allocations once. Their layout is
    // fixed for this object; aliases must not become impulse inputs either.
    if(m_initial_storage.empty())for(int c=0;c<3;++c)
        for(auto const* field:{&m_initial_coil->MagneticField(c),&m_initial_coil->Current(c)})
            for(amrex::MFIter it(*field);it.isValid();++it){auto const& fab=(*field)[it];
                auto const begin=reinterpret_cast<std::uintptr_t>(fab.dataPtr());
                m_initial_storage.emplace_back(begin,begin+fab.size()*sizeof(amrex::Real));}
    m_time=time;m_phase=phase;m_initial_ready=true;
    m_generation=m_coupler.NativeSourceImpulseGeneration();m_ready=true;return true;
}
bool NativeCircuitFieldImpulse::Ready() const noexcept
{
    return m_ready && m_coupler.NativeSourceImpulseReady() &&
        m_generation==m_coupler.NativeSourceImpulseGeneration();
}
bool NativeCircuitFieldImpulse::Applied() const noexcept
{
    return Ready() && m_applied && m_application!=0 &&
        m_application==m_coupler.NativeSourceImpulseApplication();
}
bool NativeCircuitFieldImpulse::MatchesSourceContext(WarpX const& w,amrex::Real time,
    CircuitCoupler::SourceImpulsePhase phase) const noexcept
{
    return &w==&m_warp && m_initial_ready && Ready() && time==m_time && phase==m_phase;
}
void NativeCircuitFieldImpulse::Apply(View const& input)
{
    std::string error;
    bool const valid=TryApply(input,error);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,error);
}
bool NativeCircuitFieldImpulse::TryApply(View const& input,std::string& error)
{
    bool valid=Ready();
    for(int c=0;c<3;++c){auto const* f=input[c];
        valid=valid && f && f->nComp()==1 && f->boxArray()==m_a[c].boxArray() &&
            f->DistributionMap()==m_a[c].DistributionMap() && f->nGrowVect().allGE(m_a[c].nGrowVect());
        if(!f)continue;
        for(amrex::MFIter it(*f);it.isValid();++it){auto const& fab=(*f)[it];
            auto const begin=reinterpret_cast<std::uintptr_t>(fab.dataPtr());auto const end=begin+fab.size()*sizeof(amrex::Real);
            for(auto const& [lo,hi]:m_owned_storage)valid=valid && !(begin<hi && lo<end);
            for(auto const& [lo,hi]:m_initial_storage)valid=valid && !(begin<hi && lo<end);
        }
    }
    if(!valid)error="Circuit field impulse input is stale, has wrong layout, or overlaps owned storage";
    if(!AllRanks(valid,error))return false;
    NativeBoundaryFieldRate(m_warp,input,V(m_zero),V(m_a),V(m_b),V(m_j));
    m_coupler.ApplyNativeSourceImpulse(m_b[2]);
    auto const& ext=*m_warp.get_pointer_HybridPICModel()->m_external_vector_potential;
    auto const* increments=m_coupler.SourceFieldIncrements().data();
    for(int c=0;c<3;++c){m_reference[c].setVal(0.);m_coil_b[c].setVal(0.);m_coil_j[c].setVal(0.);
        for(int f=0;f<ext.nFields();++f){
            auto const& aunit=*m_warp.m_fields.get(ext.FieldName(f)+"_Aext",ablastr::fields::Direction{c},0);
            auto const& bunit=*m_warp.m_fields.get(ext.FieldName(f)+"_curlAext",ablastr::fields::Direction{c},0);
            for(amrex::MFIter it(m_reference[c]);it.isValid();++it){auto const a=m_reference[c].array(it);auto const u=aunit.const_array(it);
                amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)+=increments[f]*u(i,j,k);});}
            for(amrex::MFIter it(m_coil_b[c]);it.isValid();++it){auto const b=m_coil_b[c].array(it);auto const u=bunit.const_array(it);
                amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){b(i,j,k)+=increments[f]*u(i,j,k);});}
        }
    }
    NativeBoundaryFieldRate(m_warp,input,V(m_reference),V(m_a),V(m_b),V(m_j));
    for(int c=0;c<3;++c)for(amrex::MFIter it(m_impulse[c]);it.isValid();++it){
        auto const psi=m_impulse[c].array(it);auto const a=m_a[c].const_array(it);
        amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){psi(i,j,k)=-a(i,j,k);});
    }
    auto coil_current=V(m_coil_j);
    m_warp.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(coil_current,V(m_coil_b),m_warp.GetEBUpdateEFlag()[0],0);
    amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
    for(int d=0;d<AMREX_SPACEDIM;++d){lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
        hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;}
    for(auto& j:m_coil_j){j.OverrideSync(m_warp.Geom(0).periodicity());j.FillBoundary(m_warp.Geom(0).periodicity());
        ApplyDarwinPMCVectorBoundary(j,m_warp.Geom(0),lo,hi);}
    m_application=m_coupler.NativeSourceImpulseApplication();m_applied=true;error.clear();return true;
}
}
