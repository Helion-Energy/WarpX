/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_DARWIN_DISPLACEMENT_CURRENT_H_
#define WARPX_DARWIN_DISPLACEMENT_CURRENT_H_
#include "ThermalCurrentRemainder.H"
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuLaunch.H>
namespace warpx::darwin {
/** Ampere current minus epsilon0*(EL-ELold)/interval, in SI A/m^2.
 * Caller supplies the SAME existing epsilon0/interval coefficient. Form the
 * field difference first so a constant longitudinal field gives exactly zero
 * displacement current, independent of its large common offset. This changes
 * only represented arithmetic, not time labels, constraints, BCs or histories.
 * CalculateCurrentAmpere (including split Ampere) refreshes valid cells and
 * exactly one guard cell. Restrict this subtraction to that same region:
 * wider guards are storage, not fresh Ampere data. Subtracting there would
 * accumulate a history-dependent displacement current on every residual.
 * Consumers needing wider current images must fill them from valid data.
 */
inline void SubtractDisplacementCurrent(amrex::MultiFab& current,
    amrex::MultiFab const& longitudinal,amrex::MultiFab const& old,
    amrex::Real factor,amrex::MultiFab const* low=nullptr,
    amrex::MultiFab* current_low=nullptr)
{
    AMREX_ALWAYS_ASSERT(current.nGrowVect().allGE(amrex::IntVect(1)) &&
        current.boxArray()==longitudinal.boxArray() &&
        current.boxArray()==old.boxArray() &&
        current.DistributionMap()==longitudinal.DistributionMap() &&
        current.DistributionMap()==old.DistributionMap() &&
        longitudinal.nGrowVect().allGE(current.nGrowVect()) &&
        old.nGrowVect().allGE(current.nGrowVect()) &&
        longitudinal.nComp()>=current.nComp() && old.nComp()>=current.nComp());
    if(low) AMREX_ALWAYS_ASSERT(low->boxArray()==current.boxArray() &&
        low->DistributionMap()==current.DistributionMap() &&
        low->nGrowVect().allGE(current.nGrowVect()) &&
        low->nComp()>=current.nComp());
    if(current_low) {
        using namespace warpx::thermal::remainder;
        bool valid=ArithmeticSupported() && Layout(*current_low,current,1) && !Overlap(*current_low,current) &&
            !Overlap(*current_low,longitudinal) && !Overlap(*current_low,old) &&
            (!low || !Overlap(*current_low,*low)) && std::isfinite(factor);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(All(valid),"Invalid displacement-current remainder output");
    }
    for(amrex::MFIter mfi(current);mfi.isValid();++mfi){
        auto const j=current.array(mfi);
        auto const e=longitudinal.const_array(mfi),e0=old.const_array(mfi);
        auto const remainder=low?low->const_array(mfi):amrex::Array4<amrex::Real const>{};
        bool const compensated=low!=nullptr;
        bool const keep=current_low!=nullptr;
        auto const small=keep?current_low->array(mfi):amrex::Array4<amrex::Real>{};
        amrex::ParallelFor(mfi.growntilebox(1),current.nComp(),
            [=] AMREX_GPU_DEVICE(int i,int k,int l,int c){
                auto const ampere=j(i,k,l,c);
                auto difference=e(i,k,l,c)-e0(i,k,l,c);
                if(compensated) difference+=remainder(i,k,l,c);
                j(i,k,l,c)-=factor*difference;
                if(keep) {
                    namespace dd=warpx::ohm::compensated;
                    auto delta=dd::Sum(e(i,k,l,c),-e0(i,k,l,c));
                    if(compensated) delta=dd::Add(delta,{remainder(i,k,l,c),0.});
                    auto value=dd::Add({ampere,0.},dd::Negate(dd::Multiply(delta,factor)));
                    small(i,k,l,c)=dd::WithHigh(value,j(i,k,l,c)).lo;
                }
            });
    }
}
}
#endif
