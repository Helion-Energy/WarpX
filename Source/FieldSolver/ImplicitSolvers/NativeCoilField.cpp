/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#include "NativeCoilField.H"
#include "DarwinABoundary.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_MFIter.H>
#include <cmath>
namespace warpx::darwin {
using R=amrex::Real;using MF=amrex::MultiFab;
using V=ablastr::fields::VectorField;using D=ablastr::fields::Direction;
using warpx::fields::FieldType;
namespace {
V View(std::array<MF,3>& f){return {&f[0],&f[1],&f[2]};}
// NVCC extended lambdas need a public/free enclosing function and cannot
// capture a private class enum. Keep the sampling arithmetic device-local.
void AccumulateCoilUnit(MF& output,MF const& unit,double const* explicit_rates,
    bool device,bool value_sample,warpx::circuit::DeviceScaleView segment,
    int field,R time,R factor)
{
    for(amrex::MFIter mfi(output);mfi.isValid();++mfi) {
        auto const out=output.array(mfi);auto const in=unit.const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            R const scale=explicit_rates?explicit_rates[field]:
                device?(value_sample?segment.Value(field,time):segment.Slope(field)):factor;
            out(i,j,k)+=scale*in(i,j,k);
        });
    }
}
}
bool NativeCoilCurrentEnabled()
{
    bool enabled=false;
    amrex::ParmParse("endpoint_diagnostic").query("circuit_plasma_current",enabled);
    return enabled;
}
NativeCoilField::NativeCoilField(WarpX& w):m_warp(w)
{
    auto const& model=*w.get_pointer_HybridPICModel();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(w.maxLevel()==0 && !EB::enabled() &&
        WarpX::ncomps==1 && model.m_add_external_fields &&
        !model.m_has_external_current && model.m_external_vector_potential,
        "Native coil current needs one level, no EB, full imposed fields and no independent J_ext");
    auto const b=w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
    auto const j=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    for(int c=0;c<3;++c) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(b[c]->nGrowVect().allGE(amrex::IntVect(2)),
            "Native coil current requires the consumed grow2 magnetic band");
        // Native grow1 Ampere consumes no more than the grow2 magnetic band.
        m_b[c].define(b[c]->boxArray(),b[c]->DistributionMap(),1,2);
        m_j[c].define(j[c]->boxArray(),j[c]->DistributionMap(),1,1);
        m_b[c].setVal(0.);m_j[c].setVal(0.);
    }
}
void NativeCoilField::SampleAtTime(R time){Apply(Sample::Value,time,0.,nullptr);}
void NativeCoilField::SampleSecant(R time,R duration){Apply(Sample::Secant,time,duration,nullptr);}
void NativeCoilField::SampleRates(amrex::Gpu::DeviceVector<double> const& rates)
{Apply(Sample::Rate,0.,0.,&rates);}
void NativeCoilField::Apply(Sample kind,R time,R duration,
                          amrex::Gpu::DeviceVector<double> const* rates)
{
    auto const& ext=*m_warp.get_pointer_HybridPICModel()->m_external_vector_potential;
    AMREX_ALWAYS_ASSERT(std::isfinite(time));
    AMREX_ALWAYS_ASSERT(kind!=Sample::Secant || (std::isfinite(duration)&&duration>0.));
    AMREX_ALWAYS_ASSERT(kind!=Sample::Rate || (rates && rates->size()==static_cast<std::size_t>(ext.nFields())));
    m_sampled=false;
    for(auto& b:m_b)b.setVal(0.);
    for(int field=0;field<ext.nFields();++field) {
        bool const device=ext.DeviceDriven(field);
        auto const segment=ext.DeviceScales();
        double const* explicit_rates=rates?rates->data():nullptr;
        R factor=0.;
        if(kind!=Sample::Rate) {
            if(device) {
                AMREX_ALWAYS_ASSERT(segment.start && segment.end && segment.dt>0. &&
                    std::isfinite(segment.t0) && std::isfinite(segment.dt));
                if(kind==Sample::Secant)
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(segment.t0==time && segment.dt==duration,
                        "Coil-current secant differs from its native trial segment");
                else AMREX_ALWAYS_ASSERT_WITH_MESSAGE(time>=segment.t0 && time<=segment.t0+segment.dt,
                        "Coil-current clock lies outside its native trial segment");
            } else if(kind==Sample::Value)factor=ext.TimeScale(field,time);
            else if(ext.HasAnalyticTimeProfile(field)) {
                auto const profile=ext.TimeProfile(field);
                AMREX_ALWAYS_ASSERT(profile.ConsistentIncrement(time,duration));
                factor=profile.increment(time,duration)/duration;
            } else {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!ext.UsesPythonScale(field),
                    "A native circuit secant needs an active device segment");
                // A static non-circuit profile has no rate. Native endpoint
                // scope separately rejects an unaccompanied time dependence.
                factor=0.;
            }
            AMREX_ALWAYS_ASSERT(std::isfinite(factor));
        }
        for(int c=0;c<3;++c) {
            auto const& unit=*m_warp.m_fields.get(ext.FieldName(field)+"_curlAext",D{c},0);
            AMREX_ALWAYS_ASSERT(unit.boxArray()==m_b[c].boxArray() &&
                unit.DistributionMap()==m_b[c].DistributionMap() && unit.nComp()==1 &&
                unit.nGrowVect().allGE(m_b[c].nGrowVect()));
            AccumulateCoilUnit(m_b[c],unit,explicit_rates,device,
                kind==Sample::Value,segment,field,time,factor);
        }
    }
    for(auto& b:m_b){b.OverrideSync(m_warp.Geom(0).periodicity());b.FillBoundary(m_warp.Geom(0).periodicity());}
    for(auto& j:m_j)j.setVal(0.);
    auto current=View(m_j);
    m_warp.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(
        current,View(m_b),m_warp.GetEBUpdateEFlag()[0],0);
    amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
    for(int d=0;d<AMREX_SPACEDIM;++d){lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;}
    for(auto& j:m_j){j.OverrideSync(m_warp.Geom(0).periodicity());j.FillBoundary(m_warp.Geom(0).periodicity());ApplyDarwinPMCVectorBoundary(j,m_warp.Geom(0),lo,hi);}
    m_sampled=true;
}
void NativeCoilField::SubtractCurrent(V const& total)const
{
    AMREX_ALWAYS_ASSERT(m_sampled);
    for(int c=0;c<3;++c) {
        AMREX_ALWAYS_ASSERT(total[c] && total[c]->boxArray()==m_j[c].boxArray() &&
            total[c]->DistributionMap()==m_j[c].DistributionMap() && total[c]->nComp()==1 &&
            total[c]->nGrowVect().allGE(amrex::IntVect(1)));
        for(int d=0;d<3;++d)AMREX_ALWAYS_ASSERT(total[c]!=&m_j[d] && total[c]!=&m_b[d]);
        MF::Subtract(*total[c],m_j[c],0,0,1,1);
    }
}
void NativeCoilField::CalculatePlasmaCurrent(V const& total_B,R time)
{
    auto& model=*m_warp.get_pointer_HybridPICModel();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!model.m_has_external_current,
        "Native coil and independent external-current subtraction cannot be combined");
    SampleAtTime(time);
    model.CalculatePlasmaCurrent(total_B,m_warp.GetEBUpdateEFlag()[0],0);
    SubtractCurrent(m_warp.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0));
}
MF const& NativeCoilField::MagneticField(int c)const{AMREX_ALWAYS_ASSERT(m_sampled);return m_b[c];}
MF const& NativeCoilField::Current(int c)const{AMREX_ALWAYS_ASSERT(m_sampled);return m_j[c];}
}
