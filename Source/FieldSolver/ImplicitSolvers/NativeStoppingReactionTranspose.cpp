/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativeStoppingReactionTranspose.H"

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
NativeStoppingReactionTranspose::NativeStoppingReactionTranspose (
    NativeAcceptedStoppingContext const& context) : m_context(context), m_totals(4), m_bad(1)
{
    for (int c=0; c<3; ++c) {
        auto const boxes=amrex::convert(context.Cells(),YeeType(c));
        for (auto* f:{&m_extensive[c],&m_pi[c],&m_current[c],&m_inertia[c]}) {
            f->define(boxes,context.Distribution(),1,context.Options().ghosts);
            f->setVal(0.);
        }
        m_owner[c]=m_pi[c].OwnerMask(context.Geometry().periodicity());
    }
}
bool NativeStoppingReactionTranspose::ContextUnchanged () const
{
    return m_context.Valid() && m_generation==m_context.Generation();
}
bool NativeStoppingReactionTranspose::Reject (char const* reason)
{
    m_open=false; m_valid=false; m_failure=reason; m_balance={};
    for (int c=0; c<3; ++c) {
        for (auto* f:{&m_extensive[c],&m_pi[c],&m_current[c],&m_inertia[c]}) { f->setVal(0.); }
    }
    return false;
}
void NativeStoppingReactionTranspose::Invalidate () { Reject("invalidated"); }
bool NativeStoppingReactionTranspose::Begin ()
{
    Reject("preparing");
    int bad=!m_context.Valid();
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        bad=bad || m_context.Geometry().Domain().length(d)<4;
    }
    amrex::ParallelDescriptor::ReduceIntMax(bad);
    if (bad) { return Reject("invalid context or domain narrower than cubic support"); }
    amrex::Gpu::fillAsync(m_totals.begin(),m_totals.end(),[] AMREX_GPU_DEVICE (amrex::Real& v, amrex::Long) { v=0.; });
    amrex::Gpu::fillAsync(m_bad.begin(),m_bad.end(),[] AMREX_GPU_DEVICE (int& v, amrex::Long) { v=0; });
    m_generation=m_context.Generation(); m_open=true; m_host_bad=false; m_failure="";
    return true;
}
void NativeStoppingReactionTranspose::DepositTile (
    amrex::MFIter const& mfi, StoppingIonImpulse const* tuples, amrex::Long count)
{
    if (!m_open || !ContextUnchanged() || count<0 || (count>0 && !tuples) ||
        !mfi.isValid() || mfi.DistributionMap()!=m_context.Distribution() || mfi.index()<0 || mfi.index()>=m_context.Cells().size() ||
        amrex::enclosedCells(mfi.validbox())!=m_context.Cells()[mfi.index()]) {
        m_host_bad=true; return;
    }
    if (count==0) { return; }
    Scatter(m_context.Geometry(),m_context.Cells()[mfi.index()],m_context.Options(),mfi,
        m_extensive,tuples,count,m_totals.data(),m_bad.data());
}
bool NativeStoppingReactionTranspose::Finish ()
{
    amrex::Gpu::synchronize();
    int bad=0;
    if (m_open) { amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_bad.begin(),m_bad.end(),&bad); }
    bad=bad || m_host_bad || !m_open || !ContextUnchanged();
    amrex::ParallelDescriptor::ReduceIntMax(bad);
    if (bad) { return Reject("invalid impulse tuple, layout or context generation"); }
    auto const& geometry=m_context.Geometry();
    auto const factor=PhysConst::q_e/m_context.Options().model_electron_mass;
    for (int c=0; c<3; ++c) {
        auto& extensive=m_extensive[c];
        extensive.SumBoundary(geometry.periodicity());
        extensive.OverrideSync(geometry.periodicity());
        auto const volume=MakeQdsmcVolumeElement(geometry,extensive.ixType());
        for (amrex::MFIter mfi(extensive); mfi.isValid(); ++mfi) {
            auto const source=extensive.const_array(mfi);
            auto const pi=m_pi[c].array(mfi),current=m_current[c].array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                pi(i,j,k)=source(i,j,k)/volume(i,j,k);
                current(i,j,k)=factor*pi(i,j,k);
            });
        }
        if (!m_pi[c].is_finite(0,1,0) || !m_current[c].is_finite(0,1,0)) {
            return Reject("nonfinite reaction density");
        }
    }
    ablastr::fields::ConstVectorField in{&m_current[0],&m_current[1],&m_current[2]};
    ablastr::fields::VectorField out{&m_inertia[0],&m_inertia[1],&m_inertia[2]};
    auto const& o=m_context.Options();
    warpx::darwin::ApplyYeeInertiaMass(geometry,m_context.Kappa(),in,out,
        o.model_electron_mass/(PhysConst::q_e*PhysConst::q_e*o.reference_number_density));
    for (auto const& f:m_inertia) {
        if (!f.is_finite(0,1,0)) { return Reject("nonfinite inertia impulse"); }
    }
    amrex::Real totals[4];
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_totals.begin(),m_totals.end(),totals);
    amrex::ParallelDescriptor::ReduceRealSum(totals,4);
    for (int c=0; c<3; ++c) {
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Real> data(op);
        using Tuple=typename decltype(data)::Type;
        for (amrex::MFIter mfi(m_extensive[c]); mfi.isValid(); ++mfi) {
            auto const value=m_extensive[c].const_array(mfi);
            auto const owner=m_owner[c]->const_array(mfi);
            op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
                return {owner(i,j,k)?value(i,j,k):0.};
            });
        }
        auto sum=amrex::get<0>(data.value()); amrex::ParallelDescriptor::ReduceRealSum(sum);
        m_balance.conjugate_integral[c]=sum;
        m_balance.particle_cartesian[c]=totals[c];
#if defined(WARPX_DIM_RZ)
        m_balance.electron_cartesian[c]=c==2?-sum:0.;
#else
        m_balance.electron_cartesian[c]=-sum;
#endif
        m_balance.unrepresented_cartesian[c]=totals[c]+m_balance.electron_cartesian[c];
    }
    m_balance.absolute_particle_impulse=totals[3];
    for (int c=0; c<3; ++c) {
        if (!std::isfinite(m_balance.unrepresented_cartesian[c]) ||
            !std::isfinite(m_balance.conjugate_integral[c])) { return Reject("nonfinite impulse sum"); }
    }
    if (!std::isfinite(totals[3])) { return Reject("nonfinite absolute impulse sum"); }
    m_open=false; m_valid=true; m_failure=""; return true;
}
ablastr::fields::ConstVectorField NativeStoppingReactionTranspose::ConjugateMomentumDensity () const
{
    AMREX_ALWAYS_ASSERT(Valid()); return {&m_pi[0],&m_pi[1],&m_pi[2]};
}
ablastr::fields::ConstVectorField NativeStoppingReactionTranspose::ElectronCurrentImpulse () const
{
    AMREX_ALWAYS_ASSERT(Valid()); return {&m_current[0],&m_current[1],&m_current[2]};
}
ablastr::fields::ConstVectorField NativeStoppingReactionTranspose::InertiaImpulse () const
{
    AMREX_ALWAYS_ASSERT(Valid()); return {&m_inertia[0],&m_inertia[1],&m_inertia[2]};
}
StoppingReactionBalance const& NativeStoppingReactionTranspose::Balance () const
{
    AMREX_ALWAYS_ASSERT(Valid()); return m_balance;
}
} // namespace warpx::thermal
