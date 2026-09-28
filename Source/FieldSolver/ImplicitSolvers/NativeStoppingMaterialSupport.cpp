/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativeStoppingMaterialSupport.H"
#include "NativeStoppingCarryCertificate.H"
#include "NativeInertiaSupport.H"
#include "NativeVacuumParticleSupport.H"
#include "StoppingVelocityImages.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Particles/Deposition/EsirkepovCurrentRate.H"
#include "Particles/Gather/StoppingGatherGeometry.H"
#include "Particles/ShapeFactors.H"
#include "WarpX.H"
#include <ablastr/particles/NodalFieldGather.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace warpx::thermal {
namespace {
using R = amrex::Real;
using CV = NativeStoppingMaterialSupport::ConstVector;
using Fields = std::array<amrex::MultiFab,3>;
using FV = ablastr::fields::VectorField;
bool All(bool value) { amrex::ParallelDescriptor::ReduceBoolAnd(value); return value; }
CV Const(Fields const& f) { return {&f[0],&f[1],&f[2]}; }
FV Mutable(Fields& f) { return {&f[0],&f[1],&f[2]}; }
amrex::IntVect Yee(int c) {
    amrex::IntVect t(1);
#if defined(WARPX_DIM_RZ)
    if(c!=1)t[c/2]=0;
#else
    t[c]=0;
#endif
    return t;
}
void Define(Fields& f,amrex::BoxArray const& cells,
            amrex::DistributionMapping const& dm,amrex::IntVect ghosts) {
    for(int c=0;c<3;++c){f[c].define(amrex::convert(cells,Yee(c)),dm,1,ghosts);f[c].setVal(0.);}
}
bool Layout(amrex::MultiFab const* f,amrex::BoxArray const& boxes,
            amrex::DistributionMapping const& dm,int components=1) {
    return f&&f->boxArray()==boxes&&f->DistributionMap()==dm&&f->nComp()==components;
}
bool SameValid(amrex::MultiFab const& a,amrex::MultiFab const& b) {
    amrex::Gpu::DeviceScalar<int> bad(0);auto* fail=bad.dataPtr();
    for(amrex::MFIter it(a);it.isValid();++it){auto x=a.const_array(it),y=b.const_array(it);
        amrex::For(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            auto const* p=reinterpret_cast<unsigned char const*>(&x(i,j,k));
            auto const* q=reinterpret_cast<unsigned char const*>(&y(i,j,k));
            for(std::size_t n=0;n<sizeof(R);++n)if(p[n]!=q[n]){
                amrex::HostDevice::Atomic::Add(fail,1);break;
            }
        });
    }
    return All(bad.dataValue()==0);
}
bool Finite(CV const& f, bool ghosts=false) {
    bool good=true;
    // Never short-circuit a MultiFab collective on a local predicate.
    for(auto const* a:f){bool const finite=a->is_finite(0,1,ghosts?a->nGrowVect():amrex::IntVect(0));good=good&&finite;}
    return good;
}
void Sync(Fields& f,amrex::Geometry const& g) {
    for(auto& a:f){a.setBndry(0.);a.OverrideSync(g.periodicity());a.FillBoundary(g.periodicity());}
}
void VelocityImages(Fields& f,amrex::Geometry const& g,AcceptedStoppingOptions const& o) {
    for(int c=0;c<3;++c){auto const image=MakeStoppingVelocityImages(g,o,f[c].ixType(),c);
        for(amrex::MFIter it(f[c]);it.isValid();++it){auto v=f[c].array(it);
            amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                amrex::IntVect source(AMREX_D_DECL(i,j,k));auto const original=source;
                R const parity=image(source);if(source!=original)v(i,j,k)=parity*v(source);
            });
        }
    }
}
bool Agreement(AcceptedStoppingOptions const& o,AcceptedStoppingBinding const& b) {
    // Small setup metadata only; no host bulk field transfer or rank-local
    // conditional collectives. Integer-valued options are exactly represented.
    std::vector<R> local{o.number_density_floor,o.reference_number_density,o.model_electron_mass,
        b.endpoint_time,R(o.levels),R(o.azimuthal_modes),R(o.shape_order),R(o.embedded_boundary),
        R(o.moving_window),R(o.galilean),R(o.filter),R(o.current_centering),
        R(o.single_precision_communications),R(o.galerkin_interpolation),R(b.charge_component)};
    for(int d=0;d<AMREX_SPACEDIM;++d){local.insert(local.end(),{R(o.ghosts[d]),R(o.field_lo[d]),
        R(o.field_hi[d]),R(o.particle_lo[d]),R(o.particle_hi[d])});}
    auto root=local;int const io=amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(root.data(),static_cast<int>(root.size()),io);
    auto epoch=b.endpoint_epoch;amrex::ParallelDescriptor::Bcast(&epoch,1,io);
    return All(root==local&&epoch==b.endpoint_epoch);
}
bool LocalScope(WarpX& w,AcceptedStoppingOptions const& o) {
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(w,o);return false;
#else
    // WarpX RZ field kernels may carry Coord()==0 metadata. Use the same
    // physical radial geometry as native thermal contexts, never mutate Geom.
    auto const* model=w.get_pointer_HybridPICModel();
    if(!model||(w.Geom(0).Coord()!=0&&w.Geom(0).Coord()!=1))return false;
    auto const g=model->ElectronThermalGeometry();auto const& cells=w.boxArray(0);
    bool good=sizeof(R)==sizeof(double)&&sizeof(amrex::ParticleReal)==sizeof(double)&&
        w.finestLevel()==0&&w.maxLevel()==0&&!EB::enabled()&&!w.getdo_moving_window()&&
        WarpX::grid_type==GridType::Staggered&&WarpX::nox==3&&WarpX::ncomps==1&&
        WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving&&
        WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov&&
        !WarpX::do_shared_mem_current_deposition&&!WarpX::use_filter&&
        !WarpX::do_single_precision_comms&&!w.do_current_centering&&!WarpX::galerkin_interpolation&&
        WarpX::field_centering_nox==2&&WarpX::field_centering_noy==2&&WarpX::field_centering_noz==2&&
        g.IsRZ()&&g.ProbLo(0)==0.&&g.Domain().smallEnd()==amrex::IntVect(0)&&
        !g.isPeriodic(0)&&g.isPeriodic(1)&&cells.ixType().cellCentered()&&cells.isDisjoint()&&
        cells.minimalBox()==g.Domain()&&cells.numPts()==g.Domain().numPts()&&
        o.levels==1&&o.azimuthal_modes==1&&o.shape_order==3&&!o.embedded_boundary&&
        !o.moving_window&&!o.galilean&&!o.filter&&!o.current_centering&&
        !o.single_precision_communications&&!o.galerkin_interpolation&&
        o.ghosts.allGE(amrex::IntVect(2))&&std::isfinite(o.number_density_floor)&&o.number_density_floor>0.&&
        std::isfinite(o.reference_number_density)&&o.reference_number_density>0.&&
        std::isfinite(o.model_electron_mass)&&o.model_electron_mass>0.;
    for(auto v:w.m_v_galilean)good=good&&v==0.;
    // ShiftGalileanBoundary may leave inactive RZ y metadata nonfinite.
    // Only physical x/z shifts enter this RZ geometry; velocity still checks
    // every component above. The non-RZ scope rejection is unchanged.
    good=good&&w.m_galilean_shift[0]==0.&&w.m_galilean_shift[2]==0.;
    if(auto const* model=w.get_pointer_HybridPICModel())good=good&&!model->m_density_pedestal;
    good=good&&o.field_lo[0]==FieldBoundaryType::None&&o.field_hi[0]==FieldBoundaryType::PEC&&
        o.particle_lo[0]==ParticleBoundaryType::None&&
        (o.particle_hi[0]==ParticleBoundaryType::Reflecting||o.particle_hi[0]==ParticleBoundaryType::Absorbing||
         o.particle_hi[0]==ParticleBoundaryType::None)&&
        o.field_lo[1]==FieldBoundaryType::Periodic&&o.field_hi[1]==FieldBoundaryType::Periodic&&
        o.particle_lo[1]==ParticleBoundaryType::Periodic&&o.particle_hi[1]==ParticleBoundaryType::Periodic;
    for(int d=0;d<AMREX_SPACEDIM;++d){
        int const width=std::max(o.ghosts[d],w.get_ng_depos_J()[d]);
        good=good&&o.field_lo[d]==WarpX::field_boundary_lo[d]&&o.field_hi[d]==WarpX::field_boundary_hi[d]&&
            o.particle_lo[d]==WarpX::particle_boundary_lo[d]&&o.particle_hi[d]==WarpX::particle_boundary_hi[d]&&
            width<=g.Domain().length(d);
        // Sufficient allocation coverage for one physical reflection. No
        // producer-undefined mirror cell is inferred from an arbitrary halo.
        for(int n=0;n<cells.size();++n)good=good&&cells[n].length(d)>=width;
    }
    R const floor=PhysConst::q_e*o.number_density_floor;
    R const reference=PhysConst::q_e*o.reference_number_density;
    R const scale=o.model_electron_mass/(PhysConst::q_e*PhysConst::q_e*o.reference_number_density);
    return good&&std::isfinite(floor)&&floor>0.&&std::isfinite(reference)&&reference>0.&&
        std::isfinite(scale)&&scale>0.&&std::isfinite(PhysConst::q_e/o.model_electron_mass);
#endif
}
bool Named(WarpX& w,AcceptedStoppingBinding const& b) {
    using ablastr::fields::Direction;
    auto const& cells=w.boxArray(0);auto const& dm=w.DistributionMap(0);
    bool good=!b.pedestal&&b.charge_component==0&&std::isfinite(b.endpoint_time)&&
        w.m_fields.has(warpx::fields::FieldType::rho_fp,0)&&
        b.raw_charge==w.m_fields.get(warpx::fields::FieldType::rho_fp,0)&&
        b.raw_charge&&b.raw_charge->nComp()>=1&&
        Layout(b.raw_charge,amrex::convert(cells,amrex::IntVect(1)),dm,b.raw_charge->nComp());
    for(int c=0;c<3;++c){
        bool const exists=w.m_fields.has(NativeAcceptedStoppingContext::AcceptedCurrentName,Direction{c},0);
        good=good&&exists&&b.electron_current[c]&&
            b.electron_current[c]==(exists?w.m_fields.get(NativeAcceptedStoppingContext::AcceptedCurrentName,Direction{c},0):nullptr)&&
            Layout(b.electron_current[c],amrex::convert(cells,Yee(c)),dm);
    }
    return good;
}
} // namespace

struct NativeStoppingMaterialSupport::Impl {
    struct Output { Fields velocity,inertia,endpoint,residual; StoppingMaterialWork work; };
    amrex::Geometry geometry;
    amrex::BoxArray cells;
    amrex::DistributionMapping distribution;
    AcceptedStoppingBinding binding;
    amrex::Real clock=0.;int step=0,native_coord=0;
    bool legacy=false;
    amrex::MultiFab rho, bound_charge;
    Fields current,mass,unit,scaled_pi,original_current;
    bool projected=false;
    std::array<amrex::iMultiFab,3> forbidden;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> owner;
    std::unique_ptr<warpx::darwin::NativeInertiaSupport> support;
    std::unique_ptr<NativeAcceptedStoppingContext> positive;
    std::unique_ptr<Output> published,trial;
    amrex::Gpu::DeviceVector<amrex::Long> counts;
    Impl(WarpX& w,AcceptedStoppingOptions const& o,AcceptedStoppingBinding const& b,CV const* projected_current)
        :geometry(w.get_pointer_HybridPICModel()->ElectronThermalGeometry()),
         cells(w.boxArray(0)),distribution(w.DistributionMap(0)),binding(b),
         clock(w.gett_new(0)),step(w.getistep(0)),native_coord(w.Geom(0).Coord()),
         rho(amrex::convert(cells,amrex::IntVect(1)),distribution,1,1),
         bound_charge(amrex::convert(cells,amrex::IntVect(1)),distribution,1,0),counts(8) {
        // Preserve every original valid component-0 replica for exact lifetime
        // checks. Canonical coefficient rho below has a different image contract.
        amrex::MultiFab::Copy(bound_charge,*b.raw_charge,0,0,1,0);
        rho.setVal(0.);amrex::MultiFab::Copy(rho,*b.raw_charge,0,0,1,0);
        rho.OverrideSync(geometry.periodicity());rho.FillBoundary(geometry.periodicity());
        Define(current,cells,distribution,amrex::IntVect(0));
        projected=projected_current!=nullptr;
        if(projected)Define(original_current,cells,distribution,amrex::IntVect(0));
        Define(mass,cells,distribution,amrex::IntVect(0));
        Define(unit,cells,distribution,amrex::IntVect(0));
        Define(scaled_pi,cells,distribution,amrex::IntVect(0));
        amrex::IntVect grow=o.ghosts;
        for(int d=0;d<AMREX_SPACEDIM;++d)grow[d]=std::max(grow[d],w.get_ng_depos_J()[d]);
        for(int c=0;c<3;++c){
            amrex::MultiFab::Copy(current[c],*(projected?(*projected_current)[c]:b.electron_current[c]),0,0,1,0);
            if(projected)amrex::MultiFab::Copy(original_current[c],*b.electron_current[c],0,0,1,0);
            current[c].OverrideSync(geometry.periodicity());unit[c].setVal(1.);
            forbidden[c].define(current[c].boxArray(),distribution,1,grow);forbidden[c].setVal(1);
            owner[c]=current[c].OwnerMask(geometry.periodicity());
        }
        warpx::darwin::NativeInertiaSupportOptions options;
        options.coefficient=warpx::darwin::InertiaEdgePolicy::NativeEdgeCandidate;
        options.recovery=warpx::darwin::InertiaRecoveryMask::None;
        options.components=warpx::darwin::InertiaRecoveryComponents::All;
        options.charge_floor=PhysConst::q_e*o.number_density_floor;
        options.reference_charge_density=PhysConst::q_e*o.reference_number_density;
#if defined(WARPX_DIM_RZ)
        options.lower={warpx::darwin::InitialRateBoundary::Axis,warpx::darwin::InitialRateBoundary::Periodic};
        options.upper={warpx::darwin::InitialRateBoundary::PEC,warpx::darwin::InitialRateBoundary::Periodic};
#endif
        support=std::make_unique<warpx::darwin::NativeInertiaSupport>(geometry,cells,distribution,options);
        for(auto* slot:{&published,&trial}){
            *slot=std::make_unique<Output>();
            for(auto* f:{&(*slot)->velocity,&(*slot)->inertia,&(*slot)->endpoint,&(*slot)->residual})
                Define(*f,cells,distribution,o.ghosts);
        }
    }
    // Keep extended device lambdas in the public implementation type.
    // CUDA rejects a lambda enclosed by the private PrepareImpl member.
    bool PreparePhysicalSupport() {
        amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
        for(int c=0;c<3;++c){auto const& support=this->support->PhysicalSupport(c);
            for(amrex::MFIter it(support);it.isValid();++it){auto p=support.const_array(it);
                auto j=this->current[c].const_array(it),mass=this->mass[c].const_array(it);auto v=this->forbidden[c].array(it);
                amrex::For(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int k,int z){
                    bool const physical=p(i,k,z)!=0;v(i,k,z)=!physical;
                    if(!std::isfinite(mass(i,k,z))||(physical?mass(i,k,z)<=0.:(mass(i,k,z)!=0.||j(i,k,z)!=0.)))
                        amrex::HostDevice::Atomic::Add(invalid,1);
                });
            }
        }
        return All(bad.dataValue()==0);
    }
    void ApplyMass(CV const& in,FV const& out,AcceptedStoppingOptions const& o) const {
        R const scale=o.model_electron_mass/(PhysConst::q_e*PhysConst::q_e*o.reference_number_density);
        if(legacy)warpx::darwin::ApplyYeeInertiaMass(geometry,positive->Kappa(),in,out,scale);
        else support->ApplyMass(in,out,scale);
    }
    bool SourceUnchanged(WarpX& w) const {
        bool good=Named(w,binding)&&w.boxArray(0)==cells&&w.DistributionMap(0)==distribution&&
            w.gett_new(0)==clock&&w.getistep(0)==step&&w.Geom(0).Domain()==geometry.Domain()&&
            w.Geom(0).Coord()==native_coord;
        for(int d=0;d<AMREX_SPACEDIM;++d)good=good&&w.Geom(0).isPeriodic(d)==geometry.isPeriodic(d)&&
            w.Geom(0).ProbLo(d)==geometry.ProbLo(d)&&w.Geom(0).ProbHi(d)==geometry.ProbHi(d)&&
            w.Geom(0).CellSize(d)==geometry.CellSize(d);
        if(!All(good))return false;
        good=SameValid(bound_charge,*binding.raw_charge);
        for(int c=0;c<3;++c){bool const same=SameValid(projected?original_current[c]:current[c],*binding.electron_current[c]);good=good&&same;}
        return good;
    }
};

NativeStoppingMaterialSupport::NativeStoppingMaterialSupport(WarpX& w,AcceptedStoppingOptions const& o)
    :m_sim(w),m_options(o){}
NativeStoppingMaterialSupport::~NativeStoppingMaterialSupport()=default;
bool NativeStoppingMaterialSupport::Fail(StoppingMaterialFailure reason,bool invalidate) noexcept {
    m_failure=reason;m_action=false;
    if(invalidate){m_prepared=false;m_open=false;m_footprint=false;}
    return false;
}
void NativeStoppingMaterialSupport::Invalidate() noexcept {
    ++m_generation;Fail(StoppingMaterialFailure::StaleMaterial,true);
}
bool NativeStoppingMaterialSupport::Prepare(AcceptedStoppingBinding const& binding) {
    return PrepareImpl(binding,nullptr);
}
bool NativeStoppingMaterialSupport::PrepareCarried(AcceptedStoppingBinding const& binding,
    NativeStoppingCarryCertificate const& carry) {
    if(!carry.ValidateSource(m_sim,binding.endpoint_time,binding.endpoint_epoch)){
        ++m_generation;return Fail(StoppingMaterialFailure::StaleMaterial,true);
    }
    auto const projected=carry.MaterialCurrent();
    bool valid=true;for(int c=0;c<3;++c)valid=valid&&projected[c]&&
        binding.electron_current[c]&&Layout(projected[c],binding.electron_current[c]->boxArray(),binding.electron_current[c]->DistributionMap());
    if(!All(valid)){++m_generation;return Fail(StoppingMaterialFailure::InvalidBinding,true);}
    // Even with a requested carry producer, an entirely positive material
    // uses the original path and operation sequence.
    auto const& report=carry.Report();
    if(report.free_rows+report.fixed_rows+report.axis_rows==0)return Prepare(binding);
    return PrepareImpl(binding,&projected);
}
bool NativeStoppingMaterialSupport::PrepareImpl(AcceptedStoppingBinding const& binding,CV const* projected_current) {
    ++m_generation;m_prepared=false;m_action=false;m_open=false;m_footprint=false;
    if(!All(LocalScope(m_sim,m_options)))return Fail(StoppingMaterialFailure::Unsupported);
    if(!Agreement(m_options,binding)||!All(Named(m_sim,binding)))return Fail(StoppingMaterialFailure::InvalidBinding);
    bool const finite_rho=binding.raw_charge->is_finite(0,1,0);
    R const minimum=binding.raw_charge->min(0,0);
    bool const finite_j=Finite(binding.electron_current);
    if(!All(finite_rho&&minimum>=0.&&finite_j))return Fail(StoppingMaterialFailure::InvalidDensity);
    auto next=std::make_unique<Impl>(m_sim,m_options,binding,projected_current);
    if(!next->SourceUnchanged(m_sim))return Fail(StoppingMaterialFailure::InvalidBinding);
    if(!next->support->FreezeEdges(next->rho,next->rho))return Fail(StoppingMaterialFailure::InvalidDensity);
    next->legacy=minimum>PhysConst::q_e*m_options.number_density_floor;
    if(next->legacy){
        if(projected_current)return Fail(StoppingMaterialFailure::InvalidBinding);
        next->positive=std::make_unique<NativeAcceptedStoppingContext>(next->geometry,next->cells,next->distribution,m_options);
        bool const prepared=next->positive->Prepare(m_sim.m_fields,binding);
        if(!All(prepared))return Fail(StoppingMaterialFailure::InvalidDensity);
    }
    next->ApplyMass(Const(next->unit),Mutable(next->mass),m_options);
    if(!next->PreparePhysicalSupport())return Fail(StoppingMaterialFailure::EmptyCurrent);
    for(auto& mask:next->forbidden){if(!warpx::darwin::NativeVacuumParticleSupport::FillPullback(m_sim,mask))
        return Fail(StoppingMaterialFailure::Unsupported);}
    m_impl=std::move(next);m_prepared=true;m_failure=StoppingMaterialFailure::None;m_report={};return true;
}
StoppingMaterialLease NativeStoppingMaterialSupport::Lease() const {
    StoppingMaterialLease lease;if(m_prepared){lease.m_owner=this;lease.m_generation=m_generation;
        lease.m_time=m_impl->binding.endpoint_time;lease.m_epoch=m_impl->binding.endpoint_epoch;}return lease;
}
bool NativeStoppingMaterialSupport::LocalMatches(StoppingMaterialLease const& lease) const {
    return m_prepared&&m_impl&&lease.m_owner==this&&lease.m_generation==m_generation&&
        lease.m_time==m_impl->binding.endpoint_time&&lease.m_epoch==m_impl->binding.endpoint_epoch;
}
bool NativeStoppingMaterialSupport::Matches(StoppingMaterialLease const& lease) {
    if(!All(LocalMatches(lease)&&LocalScope(m_sim,m_options)))return Fail(StoppingMaterialFailure::StaleMaterial,true);
    if(!m_impl->SourceUnchanged(m_sim))return Fail(StoppingMaterialFailure::StaleMaterial,true);
    return true;
}
bool NativeStoppingMaterialSupport::BeginFootprint(StoppingMaterialLease const& lease) {
    m_action=false;
    bool const matches=Matches(lease);bool const idle=All(!m_open);
    if(!matches||!idle)return Fail(StoppingMaterialFailure::Footprint);
    m_footprint=false;m_local_tile_bad=false;m_report={};
    auto* count=m_impl->counts.data();amrex::ParallelFor(8,[=] AMREX_GPU_DEVICE(int i){count[i]=0;});
    m_open=true;m_failure=StoppingMaterialFailure::None;return true;
}

void NativeStoppingMaterialSupport::AddFootprintTile(amrex::MFIter const& it,
    amrex::Box const& tile,StoppingMaterialPoint const* points,amrex::Long count) {
    if(!m_open||!m_prepared||!m_impl||count<0||(count>0&&!points)||!it.isValid()||
       it.DistributionMap()!=m_impl->distribution||it.index()<0||it.index()>=m_impl->cells.size()||
       amrex::enclosedCells(it.validbox())!=m_impl->cells[it.index()]||!tile.ixType().cellCentered()||
       !m_impl->cells[it.index()].contains(tile)){
        m_local_tile_bad=true;return;
    }
    if(count==0)return;
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(points);m_local_tile_bad=true;
#else
    auto& s=*m_impl;auto const& g=s.geometry;
    auto current_box=tile;current_box.grow(m_sim.get_ng_depos_J());
    auto const co=WarpX::LowerCorner(current_box,0,0.),ci=WarpX::InvCellSize(0);
    auto const cl=amrex::lbound(current_box);
    auto const gather=warpx::particles::StoppingGatherBox(s.cells[it.index()]);
    auto const gl=gather.smallEnd();
    auto const dx=g.CellSizeArray(),inv=g.InvCellSizeArray(),plo=g.ProbLoArray(),phi=g.ProbHiArray();
    amrex::GpuArray<R,2> low{},high{},origin{};
    for(int d=0;d<2;++d){low[d]=plo[d]+tile.smallEnd(d)*dx[d];
        high[d]=plo[d]+(tile.bigEnd(d)+1)*dx[d];origin[d]=plo[d]+gl[d]*dx[d];}
    amrex::GpuArray<amrex::Array4<int const>,3> masks{};
    amrex::GpuArray<StoppingVelocityImages,3> images{};
    amrex::GpuArray<amrex::IntVect,3> types{};
    for(int c=0;c<3;++c){masks[c]=s.forbidden[c].const_array(it);types[c]=Yee(c);
        images[c]=MakeStoppingVelocityImages(g,m_options,s.forbidden[c].ixType(),c);}
    auto const rho=s.rho.const_array(it);R const floor=PhysConst::q_e*m_options.number_density_floor;
    auto* totals=s.counts.data();
    amrex::For(count,[=] AMREX_GPU_DEVICE(amrex::Long n){
        auto const p=points[n];amrex::HostDevice::Atomic::Add(totals,amrex::Long(1));
        bool good=std::isfinite(p.weight)&&p.weight>=0.&&std::isfinite(p.charge)&&p.charge>0.;
        for(auto q:p.position)good=good&&std::isfinite(q);
        R const r=std::sqrt(p.position[0]*p.position[0]+p.position[1]*p.position[1]);
        R const z=p.position[2];
        good=good&&std::isfinite(r)&&r>0.&&r>=low[0]&&r<=high[0]&&z>=low[1]&&z<=high[1]&&
            r>=plo[0]&&r<phi[0]&&z>=plo[1]&&z<phi[1];
        if(!good){amrex::HostDevice::Atomic::Add(totals+1,amrex::Long(1));return;}
        amrex::GpuArray<double,3> const x{(r-co.x)*ci.x,0.,(z-co.z)*ci.z};
        amrex::GpuArray<double,3> value{},unused{},zero{},unit{1.,0.,1.};
        if(!warpx::particles::EsirkepovCurrentRate<3,2>(x,unit,zero,{0,0,0},1.,0.,value,unused)){
            amrex::HostDevice::Atomic::Add(totals+6,amrex::Long(1));return;
        }
        // Guard the scalar gather's global floor/int conversion and all four
        // loaded nodes, including nodes whose represented weight is zero.
        R const qr=(r-plo[0])*inv[0],qz=(z-plo[1])*inv[1];
        if(!(qr>=0.&&qz>=0.&&qr<R(std::numeric_limits<int>::max()-8)&&qz<R(std::numeric_limits<int>::max()-8))){
            amrex::HostDevice::Atomic::Add(totals+6,amrex::Long(1));return;
        }
        int const ir=int(std::floor(qr)),iz=int(std::floor(qz));
        if(!rho.contains(ir,iz,0)||!rho.contains(ir+1,iz+1,0)){
            amrex::HostDevice::Atomic::Add(totals+5,amrex::Long(1));return;
        }
        R const density=ablastr::particles::doGatherScalarFieldNodal(p.position[0],p.position[1],p.position[2],rho,inv,plo);
        if(!std::isfinite(density)||density<=floor){amrex::HostDevice::Atomic::Add(totals+2,amrex::Long(1));return;}
        if(p.weight==0.)return;
        int const cr=int(x[0])-1,cz=int(x[2])-1;
        bool current_bad=false,reaction_bad=false,uncovered=false;
        for(int j=0;j<4;++j)for(int i=0;i<4;++i){
            bool const branch=warpx::particles::EsirkepovCurrentRate<3,2>(x,unit,zero,{cr+i,0,cz+j},1.,0.,value,unused);
            if(!branch){amrex::HostDevice::Atomic::Add(totals+6,amrex::Long(1));return;}
            int const ii=cr+i+cl.x,jj=cz+j+cl.y;
            for(int c=0;c<3;++c){
                bool const produced=(c!=0||i<3)&&(c!=2||j<3)&&value[c]!=0.;
                if(produced){if(!masks[c].contains(ii,jj,0))uncovered=true;
                    else if(masks[c](ii,jj,0))current_bad=true;}
            }
        }
        Compute_shape_factor<3> const shape;
        for(int c=0;c<3;++c){R sr[4],sz[4];
            R const a=(r-origin[0])*inv[0]-.5*(1-types[c][0]);
            R const b=(z-origin[1])*inv[1]-.5*(1-types[c][1]);
            if(!std::isfinite(a)||!std::isfinite(b)||a<0.||b<0.||
               a>R(std::numeric_limits<int>::max()-8)||b>R(std::numeric_limits<int>::max()-8)){
                amrex::HostDevice::Atomic::Add(totals+6,amrex::Long(1));return;
            }
            int const fr=gl[0]+shape(sr,a),fz=gl[1]+shape(sz,b);
            for(int j=0;j<4;++j)for(int i=0;i<4;++i){
                if(sr[i]*sz[j]==0.)continue;
                amrex::IntVect index(fr+i,fz+j);R const parity=images[c](index);
                if(parity==0.)continue;
                if(!masks[c].contains(index))uncovered=true;
                else if(masks[c](index))reaction_bad=true;
            }
        }
        if(current_bad)amrex::HostDevice::Atomic::Add(totals+3,amrex::Long(1));
        if(reaction_bad)amrex::HostDevice::Atomic::Add(totals+4,amrex::Long(1));
        if(uncovered)amrex::HostDevice::Atomic::Add(totals+5,amrex::Long(1));
    });
#endif
}
bool NativeStoppingMaterialSupport::FinishFootprint(StoppingMaterialLease const& lease) {
    bool const matches=Matches(lease);bool const open=All(m_open);m_open=false;m_action=false;
    if(!matches||!open)return Fail(StoppingMaterialFailure::Footprint);
    amrex::Gpu::streamSynchronize();
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_impl->counts.begin(),m_impl->counts.end(),m_report.counts.begin());
    if(m_local_tile_bad)++m_report.counts[7];
    amrex::ParallelDescriptor::ReduceLongSum(m_report.counts.data(),static_cast<int>(m_report.counts.size()));
    bool valid=true;for(int n=1;n<8;++n)valid=valid&&m_report.counts[n]==0;
    m_footprint=valid;m_report.valid=valid;
    if(!valid)return Fail(StoppingMaterialFailure::Footprint);
    m_failure=StoppingMaterialFailure::None;return true;
}

namespace {
bool Overlap(amrex::MultiFab const& a,amrex::MultiFab const& b) {
    // Read allocation metadata only. In particular, reject stale const views
    // aliasing the writable back buffer before even its halo is cleared.
    for(amrex::MFIter it(a);it.isValid();++it){
        auto const x=reinterpret_cast<std::uintptr_t>(a[it].dataPtr());
        auto const y=reinterpret_cast<std::uintptr_t>(b[it].dataPtr());
        auto const xn=a[it].nBytes(),yn=b[it].nBytes();
        if(x<=y?(y-x<xn):(x-y<yn))return true;
    }
    return false;
}
StoppingMaterialWork Measure(NativeStoppingMaterialSupport::Impl const& s,
    CV const& psi,CV const& pi,CV const& mean) {
    amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
        amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
        amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
        amrex::ReduceOpSum,amrex::ReduceOpSum> op;
    amrex::ReduceData<R,R,R,R,R,R,R,R,R,R,amrex::Long,amrex::Long> data(op);
    using Tuple=typename decltype(data)::Type;
    for(int c=0;c<3;++c){auto const volume=MakeQdsmcVolumeElement(s.geometry,s.mass[c].ixType());
        for(amrex::MFIter it(s.mass[c]);it.isValid();++it){
            auto m=s.mass[c].const_array(it),j0=s.current[c].const_array(it),j1=s.trial->endpoint[c].const_array(it);
            auto b=s.trial->inertia[c].const_array(it),ve=s.trial->velocity[c].const_array(it);
            auto x=psi[c]->const_array(it),p=pi[c]->const_array(it),j=mean[c]->const_array(it);
            auto support=s.support->PhysicalSupport(c).const_array(it),owner=s.owner[c]->const_array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int k,int z)->Tuple {
                if(!owner(i,k,z))return {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0,0};
                if(!support(i,k,z))return {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0,1};
                R const v=volume(i,k,z),old=j0(i,k,z),next=j1(i,k,z),mass=m(i,k,z);
                R const mid=.5*(old+next),delta=next-old;
                R const k0=.5*(old*(mass*old))*v,k1=.5*(next*(mass*next))*v;
                R const dk=(mid*(mass*delta))*v,source=(mid*(x(i,k,z)+b(i,k,z)))*v;
                R const vp=(ve(i,k,z)*p(i,k,z))*v,jb=(j(i,k,z)*b(i,k,z))*v;
                R const abs_source=(std::abs(mid*x(i,k,z))+std::abs(mid*b(i,k,z)))*v;
                return {k0,k1,dk,source,vp,jb,std::abs(k0)+std::abs(k1),std::abs(dk),
                    abs_source,std::abs(vp)+std::abs(jb),1,0};
            });
        }
    }
    auto const t=data.value();R a[10]{amrex::get<0>(t),amrex::get<1>(t),amrex::get<2>(t),
        amrex::get<3>(t),amrex::get<4>(t),amrex::get<5>(t),amrex::get<6>(t),
        amrex::get<7>(t),amrex::get<8>(t),amrex::get<9>(t)};
    amrex::Long rows[2]{amrex::get<10>(t),amrex::get<11>(t)};
    amrex::ParallelDescriptor::ReduceRealSum(a,10);amrex::ParallelDescriptor::ReduceLongSum(rows,2);
    StoppingMaterialWork w;w.kinetic_before=a[0];w.kinetic_after=a[1];w.kinetic_increment=a[2];
    w.constitutive_work=a[3];w.endpoint_rounding=a[2]-a[3];w.velocity_reaction=a[4];w.mean_reaction=a[5];
    w.reaction_rounding=a[4]+a[5];w.absolute_kinetic=a[6];w.absolute_increment=a[7];
    w.absolute_constitutive=a[8];w.absolute_reaction=a[9];w.physical_rows=rows[0];w.empty_rows=rows[1];return w;
}
bool Finite(StoppingMaterialWork const& w) {
    for(R x:{w.kinetic_before,w.kinetic_after,w.kinetic_increment,w.constitutive_work,
        w.endpoint_rounding,w.velocity_reaction,w.mean_reaction,w.reaction_rounding,
        w.absolute_kinetic,w.absolute_increment,w.absolute_constitutive,w.absolute_reaction})
        if(!std::isfinite(x))return false;
    return true;
}
}
bool NativeStoppingMaterialSupport::Apply(StoppingMaterialLease const& lease,CV const& psi,
    CV const& pi,CV const& mean) {
    m_action=false;
    bool const matches=Matches(lease);bool const coverage=All(m_footprint&&!m_open);
    if(!matches||!coverage)return Fail(matches?StoppingMaterialFailure::Footprint:StoppingMaterialFailure::StaleMaterial);
    auto& s=*m_impl;bool valid=true;
    for(int c=0;c<3;++c)for(auto const* f:{psi[c],pi[c],mean[c]})
        valid=valid&&Layout(f,s.current[c].boxArray(),s.distribution);
    if(!All(valid))return Fail(StoppingMaterialFailure::InvalidAction);
    // Every allocated writable field here has the same local FAB index map.
    for(auto const& input:{psi,pi,mean})for(auto const* f:input){
        for(auto const* group:{&s.trial->velocity,&s.trial->inertia,&s.trial->endpoint,&s.trial->residual,&s.scaled_pi})
            for(auto const& writable:*group)valid=valid&&!Overlap(*f,writable);
    }
    if(!All(valid))return Fail(StoppingMaterialFailure::InvalidAction);
    bool const finite_psi=Finite(psi),finite_pi=Finite(pi),finite_mean=Finite(mean);
    if(!All(finite_psi&&finite_pi&&finite_mean))return Fail(StoppingMaterialFailure::InvalidAction);
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(int c=0;c<3;++c)for(amrex::MFIter it(s.mass[c]);it.isValid();++it){
        auto p=pi[c]->const_array(it);auto support=s.support->PhysicalSupport(c).const_array(it);
        amrex::For(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            if(!support(i,j,k)&&p(i,j,k)!=0.)amrex::HostDevice::Atomic::Add(invalid,1);
        });
    }
    if(!All(bad.dataValue()==0))return Fail(StoppingMaterialFailure::EmptyCurrent);
    R const alpha=PhysConst::q_e/m_options.model_electron_mass;
    for(int c=0;c<3;++c){
        auto& a=s.scaled_pi[c];amrex::MultiFab::Copy(a,*pi[c],0,0,1,0);a.mult(alpha,0,1,0);
    }
    s.ApplyMass(Const(s.scaled_pi),Mutable(s.trial->inertia),m_options);
    s.ApplyMass(mean,Mutable(s.trial->velocity),m_options);
    for(auto& v:s.trial->velocity){v.mult(-alpha,0,1,0);v.OverrideSync(s.geometry.periodicity());v.FillBoundary(s.geometry.periodicity());}
    VelocityImages(s.trial->velocity,s.geometry,m_options);
    for(int c=0;c<3;++c){
        for(amrex::MFIter it(s.mass[c]);it.isValid();++it){
            auto m=s.mass[c].const_array(it),old=s.current[c].const_array(it),b=s.trial->inertia[c].const_array(it);
            auto x=psi[c]->const_array(it),j=mean[c]->const_array(it);auto support=s.support->PhysicalSupport(c).const_array(it);
            auto next=s.trial->endpoint[c].array(it),res=s.trial->residual[c].array(it);
            amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int k,int z){
                if(support(i,k,z)){
                    R const kick=(x(i,k,z)+b(i,k,z))/m(i,k,z);
                    next(i,k,z)=old(i,k,z)+kick;
                    res(i,k,z)=j(i,k,z)-.5*(old(i,k,z)+next(i,k,z));
                }else{
                    next(i,k,z)=0.;res(i,k,z)=j(i,k,z);
                }
            });
        }
    }
    Sync(s.trial->endpoint,s.geometry);Sync(s.trial->residual,s.geometry);
    bool const fv=Finite(Const(s.trial->velocity),true),fb=Finite(Const(s.trial->inertia),true),
        fj=Finite(Const(s.trial->endpoint),true),fr=Finite(Const(s.trial->residual),true);
    if(!All(fv&&fb&&fj&&fr))return Fail(StoppingMaterialFailure::InvalidAction);
    s.trial->work=Measure(s,psi,pi,mean);
    if(!All(Finite(s.trial->work)))return Fail(StoppingMaterialFailure::InvalidAction);
    s.published.swap(s.trial);m_action=true;m_failure=StoppingMaterialFailure::None;return true;
}

bool NativeStoppingMaterialSupport::LegacyPositiveBranch() const {
    AMREX_ALWAYS_ASSERT(m_prepared);return m_impl->legacy;
}
CV NativeStoppingMaterialSupport::CapturedCurrent() const {
    AMREX_ALWAYS_ASSERT(m_prepared);return Const(m_impl->current);
}
amrex::MultiFab const& NativeStoppingMaterialSupport::CapturedCharge() const {
    AMREX_ALWAYS_ASSERT(m_prepared);return m_impl->rho;
}
amrex::iMultiFab const& NativeStoppingMaterialSupport::PhysicalSupport(int c) const {
    AMREX_ALWAYS_ASSERT(m_prepared&&c>=0&&c<3);return m_impl->support->PhysicalSupport(c);
}
amrex::iMultiFab const& NativeStoppingMaterialSupport::CurrentPullback(int c) const {
    AMREX_ALWAYS_ASSERT(m_prepared&&c>=0&&c<3);return m_impl->forbidden[c];
}
CV NativeStoppingMaterialSupport::Mass() const { AMREX_ALWAYS_ASSERT(m_prepared);return Const(m_impl->mass); }
CV NativeStoppingMaterialSupport::Velocity() const { AMREX_ALWAYS_ASSERT(m_action);return Const(m_impl->published->velocity); }
CV NativeStoppingMaterialSupport::InertiaImpulse() const { AMREX_ALWAYS_ASSERT(m_action);return Const(m_impl->published->inertia); }
CV NativeStoppingMaterialSupport::EndpointCurrent() const { AMREX_ALWAYS_ASSERT(m_action);return Const(m_impl->published->endpoint); }
CV NativeStoppingMaterialSupport::MeanResidual() const { AMREX_ALWAYS_ASSERT(m_action);return Const(m_impl->published->residual); }
StoppingMaterialWork const& NativeStoppingMaterialSupport::Work() const {
    AMREX_ALWAYS_ASSERT(m_action);return m_impl->published->work;
}
AcceptedStoppingGather NativeStoppingMaterialSupport::GatherView(amrex::MFIter const& it,CV const& magnetic) const {
    AcceptedStoppingGather v;if(!m_action||!m_footprint||!m_prepared||!it.isValid()||
        it.DistributionMap()!=m_impl->distribution||it.index()<0||it.index()>=m_impl->cells.size()||
        amrex::enclosedCells(it.validbox())!=m_impl->cells[it.index()])return v;
    auto const& s=*m_impl;auto const& g=s.geometry;auto const box=s.cells[it.index()];
    for(int c=0;c<3;++c)if(!magnetic[c]||magnetic[c]->DistributionMap()!=s.distribution||
        amrex::convert(magnetic[c]->boxArray(),amrex::IntVect(0))!=s.cells||magnetic[c]->nComp()!=1||
        !magnetic[c]->nGrowVect().allGE(amrex::IntVect(2)))return v;
    auto const gather=warpx::particles::StoppingGatherBox(box);auto const dx=g.CellSizeArray(),inv=g.InvCellSizeArray(),p=g.ProbLoArray();
    for(int d=0;d<AMREX_SPACEDIM;++d){v.tile_lo[d]=p[d]+box.smallEnd(d)*dx[d];v.tile_hi[d]=p[d]+(box.bigEnd(d)+1)*dx[d];}
#if defined(WARPX_DIM_RZ)
    v.inverse={inv[0],1.,inv[1]};v.origin={p[0]+gather.smallEnd(0)*dx[0],0.,p[1]+gather.smallEnd(1)*dx[1]};
#elif defined(WARPX_DIM_3D)
    v.inverse={inv[0],inv[1],inv[2]};v.origin={p[0]+gather.smallEnd(0)*dx[0],p[1]+gather.smallEnd(1)*dx[1],p[2]+gather.smallEnd(2)*dx[2]};
#endif
    v.lower=amrex::lbound(gather);
    for(int c=0;c<3;++c){v.velocity[c]=s.published->velocity[c].const_array(it);v.velocity_type[c]=s.published->velocity[c].ixType();
        v.magnetic[c]=magnetic[c]->const_array(it);v.magnetic_type[c]=magnetic[c]->ixType();}
    v.ready=true;return v;
}
} // namespace warpx::thermal
