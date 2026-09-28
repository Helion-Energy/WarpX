/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "DarwinThermalAdvance.H"
#include "ThetaImplicitHybrid.H"
#include "ThermalCurrentRemainder.H"
#include "WarpX.H"
#include <limits>
namespace warpx::thermal {
using warpx::fields::FieldType;
using ablastr::fields::Direction;
DarwinThermalAdvance::CurrentRemainderMap::CurrentRemainderMap(DarwinThermalAdvance& owner):a(owner) {
    active=a.m_current_remainder_requested && a.m_open;
    int local=active,minimum=local,maximum=local;
    amrex::ParallelDescriptor::ReduceIntMin(minimum);amrex::ParallelDescriptor::ReduceIntMax(maximum);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(minimum==maximum,"Rank-inconsistent thermal remainder map");
    if(!active)return;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(remainder::All(remainder::ArithmeticSupported() && !a.m_current_remainder_collecting &&
        a.m_current_remainder_generation<std::numeric_limits<std::uint64_t>::max()),"Nested or exhausted thermal remainder generation");
    a.InvalidateCurrentRemainder();++a.m_current_remainder_generation;a.m_current_remainder_collecting=true;
}
DarwinThermalAdvance::CurrentRemainderMap::~CurrentRemainderMap() {
    if(active && !finished)a.InvalidateCurrentRemainder();
}
bool DarwinThermalAdvance::CurrentRemainderMap::Finish() {
    if(!active){finished=true;return true;}
    bool const valid=remainder::All(a.m_current_remainder_collecting &&
        a.m_current_remainder_ready==a.m_current_remainder_generation && a.m_current_remainder_ready!=0);
    if(!valid)return false;
    a.m_current_remainder_collecting=false;finished=true;return true;
}
void DarwinThermalAdvance::InvalidateCurrentRemainder() noexcept {
    m_current_remainder_collecting=false;m_current_remainder_ready=0;
    if(m_stage)m_stage->InvalidateResidualRemainder();
}
ablastr::fields::VectorField DarwinThermalAdvance::CurrentRemainderOutput(amrex::Real interval) {
    ablastr::fields::VectorField output{};
    if(!m_current_remainder_requested || !m_current_remainder_collecting)return output;
    bool valid=m_open && m_current_remainder_generation!=0 && interval==m_solver.m_theta*m_dt &&
        std::isfinite(interval) && interval>0 && m_solver.m_num_amr_levels==1;
    for(int c=0;c<3;++c) {
        auto const& current=*m_simulation.m_fields.get(FieldType::hybrid_current_fp_plasma,Direction{c},0);
        valid=valid && current.nComp()==1 && current.nGrowVect().allGE(amrex::IntVect(1));
        if(m_current_remainder[c])valid=valid && remainder::Layout(*m_current_remainder[c],current,1);
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(remainder::All(valid),"Thermal remainder producer context/layout mismatch");
    for(int c=0;c<3;++c) {
        auto const& current=*m_simulation.m_fields.get(FieldType::hybrid_current_fp_plasma,Direction{c},0);
        if(!m_current_remainder[c])m_current_remainder[c]=std::make_unique<amrex::MultiFab>(
            current.boxArray(),current.DistributionMap(),1,current.nGrowVect());
        output[c]=m_current_remainder[c].get();
    }
    m_current_remainder_ready=0;return output;
}
void DarwinThermalAdvance::PublishCurrentRemainder() {
    if(!m_current_remainder_requested || !m_current_remainder_collecting)return;
    bool valid=m_open;
    for(int c=0;c<3;++c)valid=valid && m_current_remainder[c]!=nullptr;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(remainder::All(valid),"Thermal remainder publication without producer");
    for(int c=0;c<3;++c)valid=m_current_remainder[c]->is_finite(0,1,0) && valid;
    if(valid)m_current_remainder_ready=m_current_remainder_generation;
}
bool DarwinThermalAdvance::CopyResidualRemainder(WarpXSolverVec& output) const {
    if(!remainder::All(m_current_remainder_requested && m_open && m_stage &&
        m_current_remainder_ready==m_current_remainder_generation && m_current_remainder_ready!=0))return false;
    output.zero();return m_stage->CopyResidualRemainder(output.getMultiFabBlock(energy_name,0));
}
bool DarwinThermalAdvance::DifferenceResidualRemainder(WarpXSolverVec& out,WarpXSolverVec const& p,
    WarpXSolverVec const& pl,WarpXSolverVec const& n,WarpXSolverVec const& nl,amrex::Real factor) const {
    return remainder::Difference(out.getMultiFabBlock(energy_name,0),p.getMultiFabBlock(energy_name,0),
        pl.getMultiFabBlock(energy_name,0),n.getMultiFabBlock(energy_name,0),nl.getMultiFabBlock(energy_name,0),factor);
}
}
