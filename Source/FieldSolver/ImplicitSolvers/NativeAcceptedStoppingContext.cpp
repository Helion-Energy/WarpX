/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativeAcceptedStoppingContext.H"

#include "DarwinInitialRateSchur.H"
#include "Fields.H"
#include "StoppingVelocityImages.H"
#include "Particles/Gather/StoppingGatherGeometry.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>

namespace warpx::thermal {
namespace {
void FillVelocityImages (amrex::Geometry const&, AcceptedStoppingOptions const&,
                         std::array<amrex::MultiFab,3>&);
amrex::IntVect YeeType (int component)
{
    amrex::IntVect result(1);
#if defined(WARPX_DIM_RZ)
    if (component != 1) { result[component/2] = 0; }
#elif defined(WARPX_DIM_3D)
    result[component] = 0;
#endif
    return result;
}
}
NativeAcceptedStoppingContext::NativeAcceptedStoppingContext (
    amrex::Geometry const& geometry, amrex::BoxArray const& cells,
    amrex::DistributionMapping const& distribution, AcceptedStoppingOptions const& options)
    : m_geometry(geometry), m_cells(cells), m_nodes(amrex::convert(cells, amrex::IntVect(1))),
      m_distribution(distribution), m_options(options), m_kappa(m_nodes, distribution, 1, 1)
{
    AMREX_ALWAYS_ASSERT(cells.ixType().cellCentered() && cells.isDisjoint() &&
        cells.minimalBox() == geometry.Domain() && cells.numPts() == geometry.Domain().numPts());
    AMREX_ALWAYS_ASSERT(options.ghosts.allGE(amrex::IntVect(2)));
    for (int c=0; c<3; ++c) {
        m_velocity[c].define(amrex::convert(cells,YeeType(c)),distribution,1,options.ghosts);
        m_velocity[c].setVal(0.);
    }
    m_kappa.setVal(0.);
}
AcceptedStoppingBinding NativeAcceptedStoppingContext::Bind (
    ablastr::fields::MultiFabRegister const& fields, bool pedestal_active,
    amrex::Real time, std::uint64_t epoch)
{
    AcceptedStoppingBinding b;
    b.endpoint_time=time; b.endpoint_epoch=epoch;
    using ablastr::fields::Direction;
    for (int c=0; c<3; ++c) {
        if (fields.has(AcceptedCurrentName,Direction{c},0)) {
            b.electron_current[c]=fields.get(AcceptedCurrentName,Direction{c},0);
        }
    }
    if (fields.has(warpx::fields::FieldType::rho_fp,0)) {
        b.raw_charge=fields.get(warpx::fields::FieldType::rho_fp,0);
    }
    if (pedestal_active && fields.has("hybrid_rho_pedestal_fp",0)) {
        b.pedestal=fields.get("hybrid_rho_pedestal_fp",0);
    }
    // A missing requested pedestal remains an invalid binding, not zero fill.
    if (pedestal_active && !b.pedestal) { b.raw_charge=nullptr; }
    return b;
}
bool NativeAcceptedStoppingContext::Supported () const
{
#if defined(WARPX_DIM_RZ)
    if (!m_geometry.IsRZ() || m_geometry.ProbLo(0)!=0. || m_geometry.isPeriodic(0)) { return false; }
#elif defined(WARPX_DIM_3D)
    if (m_geometry.Coord()!=0) { return false; }
#else
    return false;
#endif
    if (m_geometry.Domain().smallEnd()!=amrex::IntVect(0)) { return false; }
    auto const& o=m_options;
    if (o.levels!=1 || o.azimuthal_modes!=1 || o.shape_order!=3 || o.embedded_boundary ||
        o.moving_window || o.galilean || o.filter || o.current_centering || o.single_precision_communications ||
        o.galerkin_interpolation || !std::isfinite(o.number_density_floor) || o.number_density_floor<0. ||
        !std::isfinite(o.reference_number_density) || o.reference_number_density<=0. ||
        !std::isfinite(o.model_electron_mass) || o.model_electron_mass<=0.) { return false; }
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        if (m_geometry.isPeriodic(d)) { continue; }
        for (int side=0; side<2; ++side) {
#if defined(WARPX_DIM_RZ)
            if (d==0 && side==0) { continue; }
#endif
            auto const f=side?o.field_hi[d]:o.field_lo[d];
            auto const p=side?o.particle_hi[d]:o.particle_lo[d];
            if ((f!=FieldBoundaryType::PEC && f!=FieldBoundaryType::PMC) ||
                (p!=ParticleBoundaryType::Reflecting && p!=ParticleBoundaryType::Absorbing &&
                 p!=ParticleBoundaryType::None)) { return false; }
        }
    }
    return true;
}
bool NativeAcceptedStoppingContext::Reject (char const* reason)
{
    ++m_generation;
    m_valid=false; m_failure=reason; m_binding={}; m_kappa.setVal(0.);
    for (auto& v:m_velocity) { v.setVal(0.); }
    return false;
}
bool NativeAcceptedStoppingContext::Prepare (
    ablastr::fields::MultiFabRegister const& fields, AcceptedStoppingBinding const& b)
{
    Reject("preparing");
    if (!Supported()) { return Reject("unsupported capability"); }
    if (!std::isfinite(b.endpoint_time) || !b.raw_charge || b.charge_component!=0 ||
        !fields.has(warpx::fields::FieldType::rho_fp,0) ||
        b.raw_charge!=fields.get(warpx::fields::FieldType::rho_fp,0)) {
        return Reject("not an accepted endpoint density binding");
    }
    using ablastr::fields::Direction;
    for (int c=0; c<3; ++c) {
        if (!fields.has(AcceptedCurrentName,Direction{c},0) || !b.electron_current[c] ||
            b.electron_current[c]!=fields.get(AcceptedCurrentName,Direction{c},0)) {
            return Reject("not the accepted electron-current binding");
        }
        auto const& j=*b.electron_current[c];
        if (j.boxArray()!=m_velocity[c].boxArray() || j.DistributionMap()!=m_distribution ||
            j.nComp()!=1 || !j.is_finite(0,1,0)) { return Reject("invalid accepted current"); }
    }
    auto const& rho=*b.raw_charge;
    if (rho.boxArray()!=m_nodes || rho.DistributionMap()!=m_distribution || rho.nComp()<1 ||
        !rho.is_finite(0,1,0)) { return Reject("invalid accepted charge"); }
    amrex::Real const floor=PhysConst::q_e*m_options.number_density_floor;
    if (rho.min(0,0)<=floor) { return Reject("raw density at or below native inertia floor"); }
    if (b.pedestal && (!fields.has("hybrid_rho_pedestal_fp",0) ||
        b.pedestal!=fields.get("hybrid_rho_pedestal_fp",0) ||
        b.pedestal->boxArray()!=m_nodes || b.pedestal->DistributionMap()!=m_distribution ||
        b.pedestal->nComp()!=1 || !b.pedestal->is_finite(0,1,0) || b.pedestal->min(0,0)<0.)) {
        return Reject("invalid accepted pedestal");
    }
    amrex::Real const reference=PhysConst::q_e*m_options.reference_number_density;
    bool const pedestal=b.pedestal!=nullptr;
    for (amrex::MFIter mfi(m_kappa); mfi.isValid(); ++mfi) {
        auto const out=m_kappa.array(mfi);
        auto const charge=rho.const_array(mfi);
        amrex::Array4<amrex::Real const> ped;
        if (pedestal) { ped=b.pedestal->const_array(mfi); }
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            out(i,j,k)=reference/amrex::max(floor,charge(i,j,k)+(pedestal?ped(i,j,k):0.));
        });
    }
    m_kappa.OverrideSync(m_geometry.periodicity());
    m_kappa.FillBoundary(m_geometry.periodicity());
    if (!m_kappa.is_finite(0,1,0) || m_kappa.min(0,0)<=0.) { return Reject("invalid native kappa"); }
    std::array<amrex::MultiFab*,3> out{&m_velocity[0],&m_velocity[1],&m_velocity[2]};
    amrex::Real const mass_scale=m_options.model_electron_mass/
        (PhysConst::q_e*PhysConst::q_e*m_options.reference_number_density);
    warpx::darwin::ApplyYeeInertiaMass(m_geometry,m_kappa,b.electron_current,out,mass_scale);
    for (auto& v:m_velocity) {
        v.mult(-PhysConst::q_e/m_options.model_electron_mass,0,1,0);
        v.OverrideSync(m_geometry.periodicity());
        v.FillBoundary(m_geometry.periodicity());
    }
    FillVelocityImages(m_geometry,m_options,m_velocity);
    for (auto const& v:m_velocity) {
        if (!v.is_finite(0,1,v.nGrowVect())) { return Reject("nonfinite prepared velocity"); }
    }
    m_binding=b; m_failure=""; m_valid=true; return true;
}
namespace {
void FillVelocityImages (amrex::Geometry const& m_geometry,
    AcceptedStoppingOptions const& m_options, std::array<amrex::MultiFab,3>& m_velocity)
{
    // Exact HPM CalculateElectronFluidVelocity parity, without deposit folding.
    for (int c=0; c<3; ++c) {
        auto& velocity=m_velocity[c];
        auto const image=MakeStoppingVelocityImages(m_geometry,m_options,velocity.ixType(),c);
        for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
            auto const v=velocity.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                amrex::IntVect source(AMREX_D_DECL(i,j,k));
                auto const original=source;
                auto const parity=image(source);
                if (source!=original) { v(i,j,k)=parity*v(source); }
            });
        }
    }
}
} // namespace
ablastr::fields::ConstVectorField NativeAcceptedStoppingContext::Velocity () const
{
    AMREX_ALWAYS_ASSERT(m_valid);
    return {&m_velocity[0],&m_velocity[1],&m_velocity[2]};
}
AcceptedStoppingGather NativeAcceptedStoppingContext::GatherView (
    amrex::MFIter const& mfi, ablastr::fields::ConstVectorField const& magnetic) const
{
    AMREX_ALWAYS_ASSERT(m_valid);
    AcceptedStoppingGather v;
    auto const box=m_cells[mfi.index()];
    auto const gather_box=warpx::particles::StoppingGatherBox(box);
    auto const lo=amrex::lbound(gather_box);
    auto const dx=m_geometry.CellSizeArray(),inv=m_geometry.InvCellSizeArray(),p=m_geometry.ProbLoArray();
    auto const domlo=m_geometry.Domain().smallEnd();
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        v.tile_lo[d]=p[d]+(box.smallEnd(d)-domlo[d])*dx[d];
        v.tile_hi[d]=p[d]+(box.bigEnd(d)+1-domlo[d])*dx[d];
    }
#if defined(WARPX_DIM_RZ)
    v.inverse={inv[0],1.,inv[1]};
    v.origin={p[0]+gather_box.smallEnd(0)*dx[0],0.,p[1]+gather_box.smallEnd(1)*dx[1]};
#else
    v.inverse={inv[0],inv[1],inv[2]};
    v.origin={p[0]+gather_box.smallEnd(0)*dx[0],p[1]+gather_box.smallEnd(1)*dx[1],p[2]+gather_box.smallEnd(2)*dx[2]};
#endif
    v.lower=lo;
    for (int c=0; c<3; ++c) {
        AMREX_ALWAYS_ASSERT(magnetic[c] && magnetic[c]->DistributionMap()==m_distribution &&
            amrex::convert(magnetic[c]->boxArray(),amrex::IntVect(0))==m_cells &&
            magnetic[c]->nComp()==1 && magnetic[c]->nGrowVect().allGE(amrex::IntVect(2)));
        v.velocity[c]=m_velocity[c].const_array(mfi);v.velocity_type[c]=m_velocity[c].ixType();
        v.magnetic[c]=magnetic[c]->const_array(mfi);v.magnetic_type[c]=magnetic[c]->ixType();
    }
    v.ready=true;return v;
}
} // namespace warpx::thermal
