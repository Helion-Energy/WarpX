/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#include "NativeCircuitFieldRate.H"
#include "NativeBoundaryFieldRate.H"
#include "Circuit/CircuitCoupler.H"
#include "Circuit/CircuitCoupling.H"
#include "Circuit/Coils/FluxProbes.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <cmath>
#include <set>
using MF=amrex::MultiFab;
using View=ablastr::fields::VectorField;
using Field=amrex::Array<MF,3>;
using FT=warpx::fields::FieldType;
namespace {
View V(Field& f){return {&f[0],&f[1],&f[2]};}
void Allocate(Field& out,View const& layout){for(int c=0;c<3;++c){out[c].define(layout[c]->boxArray(),layout[c]->DistributionMap(),1,layout[c]->nGrowVect());out[c].setVal(0.);}}
void BuildReferenceOnDevice(WarpX& simulation,Field& reference,Field& electric_reference,
    amrex::Gpu::DeviceVector<double> const& rates)
{
    auto const& ext=*simulation.get_pointer_HybridPICModel()->m_external_vector_potential;
    AMREX_ALWAYS_ASSERT(rates.size()==static_cast<std::size_t>(ext.nFields()));
    auto const* factors=rates.data();
    for(int c=0;c<3;++c) {
        reference[c].setVal(0.);
        for(int f=0;f<ext.nFields();++f) {
            auto const& unit=*simulation.m_fields.get(ext.FieldName(f)+"_Aext",ablastr::fields::Direction{c},0);
            for(amrex::MFIter it(reference[c]);it.isValid();++it) {
                auto const a=reference[c].array(it);auto const u=unit.const_array(it);
                amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)+=factors[f]*u(i,j,k);});
            }
        }
        for(amrex::MFIter it(electric_reference[c]);it.isValid();++it) {
            auto const e=electric_reference[c].array(it);auto const a=reference[c].const_array(it);
            amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){e(i,j,k)=-a(i,j,k);});
        }
    }
}
}
namespace warpx::darwin {
NativeCircuitFieldRate::NativeCircuitFieldRate(WarpX& w,CircuitCoupler& c):m_warp(w),m_coupler(c)
{
    auto const e=w.m_fields.get_alldirs(FT::Efield_fp,0);
    Allocate(m_zero,e);Allocate(m_reference,e);Allocate(m_e_reference,e);
    Allocate(m_a,w.m_fields.get_alldirs("hybrid_A_fp",0));
    Allocate(m_b,w.m_fields.get_alldirs(FT::Bfield_fp,0));
    Allocate(m_j,w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0));
}
bool NativeCircuitFieldRate::Prepare(amrex::Real time,std::string& error)
{
    return Prepare(time,error,nullptr);
}
bool NativeCircuitFieldRate::Prepare(amrex::Real time,std::string& error,
                                    std::vector<double> const* boundary_response_override)
{
    m_ready=false;m_coupler.InvalidateNativeEndpointRate();
    auto fail=[&](char const* text){error=text;return false;};
    auto& model=*m_warp.get_pointer_HybridPICModel();
    auto* subsystem=m_warp.get_pointer_CircuitCoupling();
    if(!subsystem || subsystem->Coupler()!=&m_coupler || !model.m_add_external_fields ||
       m_warp.maxLevel()!=0 || EB::enabled() || WarpX::ncomps!=1 ||
       !std::isfinite(time) || !m_coupler.SupportsNativeEndpointRate())
        return fail("Unsupported native circuit field-rate context");
    auto const& ext=*model.m_external_vector_potential;
    auto const& coils=subsystem->GetCoilSet();
    std::set<std::string> coupled;
    for(auto const& coil:coils.coils())coupled.insert(coil.field_name);
    std::vector<double> prescribed(ext.nFields(),0.);
    amrex::ParmParse pp("external_vector_potential");
    for(int f=0;f<ext.nFields();++f) {
        for(int c=0;c<3;++c) {
            auto const& unit=*m_warp.m_fields.get(ext.FieldName(f)+"_Aext",ablastr::fields::Direction{c},0);
            if(unit.boxArray()!=m_reference[c].boxArray() ||
               unit.DistributionMap()!=m_reference[c].DistributionMap() ||
               !unit.nGrowVect().allGE(m_reference[c].nGrowVect()))
                return fail("Native circuit field-rate unit field layout changed");
        }
        if(coupled.contains(ext.FieldName(f)))continue;
        if(ext.DeviceDriven(f) || ext.UsesPythonScale(f))
            return fail("Non-circuit endpoint fields require an analytic prescribed rate");
        if(ext.HasAnalyticTimeProfile(f)) {
            auto const profile=ext.TimeProfile(f);
            prescribed[f]=profile.derivative(time);
        } else {
            std::string expression="1.0";pp.query(ext.FieldName(f)+".A_time_external_function(t)",expression);
            if(amrex::Parser(expression).symbols().contains("t"))
                return fail("Time-dependent endpoint field lacks an analytic derivative");
        }
        if(!std::isfinite(prescribed[f]))return fail("Nonfinite prescribed endpoint field rate");
    }
    if(m_coupler.NativeEndpointBoundaryResponse().empty()) {
        m_coupler.ConfigureDarwinMagneticResponse();
        std::vector<double> h(static_cast<std::size_t>(coils.size())*ext.nFields(),0.);
        for(int f=0;f<ext.nFields();++f) {
            for(int c=0;c<3;++c)MF::Copy(m_reference[c],*m_warp.m_fields.get(
                ext.FieldName(f)+"_Aext",ablastr::fields::Direction{c},0),0,0,1,m_reference[c].nGrowVect());
            NativeBoundaryFieldRate(m_warp,V(m_zero),V(m_reference),V(m_a),V(m_b),V(m_j));
            for(int port=0;port<coils.size();++port)
                h[static_cast<std::size_t>(port)*ext.nFields()+f]=
                    warpx::circuit::DiskFluxLinkage(coils.coil(port),m_b[2]);
        }
        if(!m_coupler.SetNativeEndpointBoundaryResponse(h,error))return false;
    }
    if(!m_coupler.PrepareNativeEndpointRate(time,prescribed,error,boundary_response_override))return false;
    m_generation=m_coupler.NativeEndpointRateGeneration();
    m_ready=true;Apply(V(m_zero),true);error.clear();return true;
}
void NativeCircuitFieldRate::BuildReference()
{
    BuildReferenceOnDevice(m_warp,m_reference,m_e_reference,m_coupler.EndpointFieldRates());
}
bool NativeCircuitFieldRate::Ready() const noexcept
{
    return m_ready && m_coupler.NativeEndpointRateReady() &&
        m_generation==m_coupler.NativeEndpointRateGeneration();
}
void NativeCircuitFieldRate::Apply(View const& transverse,bool affine)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Ready(),
        "Native circuit field-rate context is stale or unprepared");
    NativeBoundaryFieldRate(m_warp,transverse,V(m_zero),V(m_a),V(m_b),V(m_j));
    m_coupler.ApplyNativeEndpointRate(m_b[2],affine);
    BuildReference();
    NativeBoundaryFieldRate(m_warp,transverse,V(m_reference),V(m_a),V(m_b),V(m_j));
}
}
