/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativeStoppingMaterialReaction.H"

#include "DarwinInitialRateSchur.H"
#include "StoppingVelocityImages.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Particles/Gather/StoppingGatherGeometry.H"
#include "Particles/ShapeFactors.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

namespace warpx::thermal {
namespace {
amrex::IntVect YeeType (int c)
{
    amrex::IntVect type(1);
#if defined(WARPX_DIM_RZ)
    if (c!=1) { type[c/2]=0; }
#else
    type[c]=0;
#endif
    return type;
}
void Scatter (amrex::Geometry const& geometry, amrex::Box const& cells,
    AcceptedStoppingOptions const& options, amrex::MFIter const& mfi,
    std::array<amrex::MultiFab,3>& extensive, StoppingIonImpulse const* tuples,
    amrex::Long count, amrex::Real* totals, int* bad)
{
    auto const gather_box=warpx::particles::StoppingGatherBox(cells);
    auto const lower=gather_box.smallEnd();
    auto const dx=geometry.CellSizeArray(), inverse=geometry.InvCellSizeArray();
    auto const problo=geometry.ProbLoArray();
    amrex::GpuArray<amrex::Array4<amrex::Real>,3> a{};
    amrex::GpuArray<StoppingVelocityImages,3> images{};
    amrex::GpuArray<amrex::IntVect,3> types{};
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> tile_lo{},tile_hi{},origin{};
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        tile_lo[d]=problo[d]+cells.smallEnd(d)*dx[d];
        tile_hi[d]=problo[d]+(cells.bigEnd(d)+1)*dx[d];
        origin[d]=problo[d]+lower[d]*dx[d];
    }
    for (int c=0; c<3; ++c) {
        a[c]=extensive[c].array(mfi); types[c]=extensive[c].ixType().toIntVect();
        images[c]=MakeStoppingVelocityImages(geometry,options,extensive[c].ixType(),c);
    }
    // Scatter iterations overlap: For, not the CPU-SIMD ParallelFor.
    amrex::For(count,[=] AMREX_GPU_DEVICE(amrex::Long p) {
        auto const t=tuples[p];
        bool valid=std::isfinite(t.weight) && t.weight>=0.;
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> x{};
        for (int c=0; c<3; ++c) {
            valid=valid && std::isfinite(t.position[c]) && std::isfinite(t.impulse[c]) &&
                std::isfinite(t.weight*t.impulse[c]);
        }
#if defined(WARPX_DIM_RZ)
        amrex::Real const radius=std::sqrt(t.position[0]*t.position[0]+t.position[1]*t.position[1]);
        x={radius,t.position[2]};
#else
        x={t.position[0],t.position[1],t.position[2]};
#endif
        for (int d=0; d<AMREX_SPACEDIM; ++d) {
            valid=valid && std::isfinite(x[d]) && x[d]>=tile_lo[d] && x[d]<=tile_hi[d];
        }
        if (!valid) { amrex::HostDevice::Atomic::Add(bad,1); return; }
        auto impulse=t.impulse;
#if defined(WARPX_DIM_RZ)
        auto const co=radius>0.?t.position[0]/radius:1.;
        auto const si=radius>0.?t.position[1]/radius:0.;
        impulse={co*t.impulse[0]+si*t.impulse[1],co*t.impulse[1]-si*t.impulse[0],t.impulse[2]};
#endif
        amrex::Real absolute=0.;
        for (int c=0; c<3; ++c) {
            absolute+=std::abs(t.weight*t.impulse[c]);
            valid=valid && std::isfinite(t.weight*impulse[c]);
        }
        if (!valid || !std::isfinite(absolute)) { amrex::HostDevice::Atomic::Add(bad,1); return; }
        for (int c=0; c<3; ++c) {
            amrex::HostDevice::Atomic::Add(totals+c,t.weight*t.impulse[c]);
        }
        amrex::HostDevice::Atomic::Add(totals+3,absolute);
        Compute_shape_factor<3> const shape;
        for (int c=0; c<3; ++c) {
            amrex::Real s[AMREX_SPACEDIM][4];
            amrex::IntVect first;
            for (int d=0; d<AMREX_SPACEDIM; ++d) {
                auto const q=(x[d]-origin[d])*inverse[d]-.5*(1-types[c][d]);
                first[d]=lower[d]+shape(s[d],q);
            }
#if defined(WARPX_DIM_RZ)
            for (int j=0; j<4; ++j) { for (int i=0; i<4; ++i) {
                amrex::IntVect index(first[0]+i,first[1]+j);
                auto const parity=images[c](index);
                if (!a[c].contains(index)) { amrex::HostDevice::Atomic::Add(bad,1); continue; }
                amrex::HostDevice::Atomic::Add(&a[c](index),
                    (s[0][i]*s[1][j])*(t.weight*impulse[c])*parity);
            }}
#else
            for (int k=0; k<4; ++k) { for (int j=0; j<4; ++j) { for (int i=0; i<4; ++i) {
                amrex::IntVect index(first[0]+i,first[1]+j,first[2]+k);
                auto const parity=images[c](index);
                if (!a[c].contains(index)) { amrex::HostDevice::Atomic::Add(bad,1); continue; }
                amrex::HostDevice::Atomic::Add(&a[c](index),
                    (s[0][i]*s[1][j]*s[2][k])*(t.weight*impulse[c])*parity);
            }}}
#endif
        }
    });
}
}

NativeStoppingMaterialReaction::NativeStoppingMaterialReaction(
    NativeStoppingMaterialSupport& material, StoppingMaterialLease lease,
    amrex::Geometry const& geometry, amrex::BoxArray const& cells,
    amrex::DistributionMapping const& dm, AcceptedStoppingOptions const& options)
    : m_material(material), m_lease(lease), m_geometry(geometry), m_cells(cells),
      m_distribution(dm), m_options(options), m_totals(4), m_bad(1)
{
    for(int c=0;c<3;++c){
        auto const boxes=amrex::convert(cells,YeeType(c));
        for(auto* f:{&m_extensive[c],&m_momentum[c]}){
            f->define(boxes,dm,1,options.ghosts);f->setVal(0.);
        }
        m_owner[c]=m_momentum[c].OwnerMask(geometry.periodicity());
    }
}
bool NativeStoppingMaterialReaction::Begin()
{
    bool const matches=m_material.Matches(m_lease);
    bool good=matches&&m_material.FootprintReady();
    amrex::ParallelDescriptor::ReduceBoolAnd(good);
    m_open=false;m_valid=false;m_local_bad=false;m_balance={};
    if(!good)return false;
    for(int c=0;c<3;++c){m_extensive[c].setVal(0.);m_momentum[c].setVal(0.);}
    amrex::Gpu::fillAsync(m_totals.begin(),m_totals.end(),
        [] AMREX_GPU_DEVICE(amrex::Real& v,amrex::Long){v=0.;});
    amrex::Gpu::fillAsync(m_bad.begin(),m_bad.end(),
        [] AMREX_GPU_DEVICE(int& v,amrex::Long){v=0;});
    m_open=true;return true;
}
void NativeStoppingMaterialReaction::DepositTile(amrex::MFIter const& it,
    StoppingIonImpulse const* tuples,amrex::Long count)
{
    if(!m_open||count<0||(count>0&&!tuples)||!it.isValid()||
       it.DistributionMap()!=m_distribution||it.index()<0||it.index()>=m_cells.size()||
       amrex::enclosedCells(it.validbox())!=m_cells[it.index()]){
        m_local_bad=true;return;
    }
    if(count)Scatter(m_geometry,m_cells[it.index()],m_options,it,m_extensive,
        tuples,count,m_totals.data(),m_bad.data());
}
bool NativeStoppingMaterialReaction::Finish()
{
    // Complete all local deposits, then one scalar validation, never per tile.
    bool const matches=m_material.Matches(m_lease);
    bool good=matches&&m_open&&!m_local_bad;
    amrex::ParallelDescriptor::ReduceBoolAnd(good);
    m_open=false;m_valid=false;if(!good)return false;
    amrex::Gpu::synchronize();int invalid=0;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_bad.begin(),m_bad.end(),&invalid);
    good=invalid==0;amrex::ParallelDescriptor::ReduceBoolAnd(good);
    if(!good)return false;
    for(int c=0;c<3;++c){
        auto& x=m_extensive[c];x.SumBoundary(m_geometry.periodicity());
        x.OverrideSync(m_geometry.periodicity());
        auto const volume=MakeQdsmcVolumeElement(m_geometry,x.ixType());
        for(amrex::MFIter it(x);it.isValid();++it){
            auto const a=x.const_array(it);auto b=m_momentum[c].array(it);
            amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                b(i,j,k)=a(i,j,k)/volume(i,j,k);
            });
        }
        bool const finite=m_momentum[c].is_finite(0,1,0);good=finite&&good;
    }
    amrex::Real totals[4];
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_totals.begin(),m_totals.end(),totals);
    amrex::ParallelDescriptor::ReduceRealSum(totals,4);
    for(int c=0;c<3;++c){
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Real> data(op);using T=typename decltype(data)::Type;
        for(amrex::MFIter it(m_extensive[c]);it.isValid();++it){
            auto a=m_extensive[c].const_array(it);auto own=m_owner[c]->const_array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T{
                return{own(i,j,k)?a(i,j,k):0.};
            });
        }
        auto sum=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(sum);
        m_balance.conjugate_integral[c]=sum;m_balance.particle_cartesian[c]=totals[c];
#if defined(WARPX_DIM_RZ)
        m_balance.electron_cartesian[c]=c==2?-sum:0.;
#else
        m_balance.electron_cartesian[c]=-sum;
#endif
        m_balance.unrepresented_cartesian[c]=totals[c]+m_balance.electron_cartesian[c];
        good=good&&std::isfinite(sum)&&std::isfinite(totals[c]);
    }
    m_balance.absolute_particle_impulse=totals[3];good=good&&std::isfinite(totals[3]);
    amrex::ParallelDescriptor::ReduceBoolAnd(good);m_valid=good;return good;
}
ablastr::fields::ConstVectorField NativeStoppingMaterialReaction::Momentum() const
{
    AMREX_ALWAYS_ASSERT(m_valid);return{&m_momentum[0],&m_momentum[1],&m_momentum[2]};
}
StoppingReactionBalance const& NativeStoppingMaterialReaction::Balance() const
{
    AMREX_ALWAYS_ASSERT(m_valid);return m_balance;
}
} // namespace warpx::thermal
