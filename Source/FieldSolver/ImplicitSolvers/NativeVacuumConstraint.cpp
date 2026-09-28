#include "NativeEndpointCheckpointIO.H"
/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeVacuumConstraint.H"
#include "NativeVacuumCircuitFieldRate.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "DarwinABoundary.H"
#include "DarwinLongitudinalSchur.H"
#include "DarwinVacuumAffineResponse.H"
#include "CompensatedNativeCurl.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "ImplicitFieldRollback.H"
#include "NativeInstantaneousIonRate.H"
#include "NativeInstantaneousForce.H"
#include "NativeVacuumEndpoint.H"
#include "NativeEndpointParticleHandoff.H"
#include "NonlinearSolvers/FlexibleGMRES.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>
#include <AMReX_GpuAtomic.H>
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Gather/GetExternalFields.H"
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <cfenv>
namespace warpx::darwin {
namespace {
using MF = amrex::MultiFab;
using Field = amrex::Array<MF, 3>;
using View = ablastr::fields::VectorField;
using FT = warpx::fields::FieldType;
using Core = DarwinVacuumAffineResponse;
#include "NativeVacuumFieldUtils.H"
#include "NativeVacuumLinearOps.H"
#include "NativeVacuumPhysicalStopping.H"
amrex::Geometry Geometry(WarpX &w) {
  auto g = w.Geom(0);
#if defined(WARPX_DIM_RZ)
  return amrex::Geometry(g.Domain(), g.ProbDomain(), 1, g.isPeriodic());
#else
  return g;
#endif
}
void DefineTrace(WarpX &w, NativeInertiaSupport &support,
                 Core::Options const &co,
                 amrex::Array<amrex::iMultiFab, 3> &trace,
                 amrex::Array<amrex::iMultiFab, 3> &global,
                 amrex::Array<amrex::iMultiFab, 3> &empty) {
  auto E = w.m_fields.get_alldirs(FT::Efield_fp, 0);
  for (int c = 0; c < 3; ++c) {
    for (auto *p : {&trace, &global, &empty}) {
      (*p)[c].define(E[c]->boxArray(), E[c]->DistributionMap(), 1, 0);
      (*p)[c].setVal(0);
    }
  }
  for (int c = 0; c < 3; ++c) {
    auto const domain = amrex::convert(w.Geom(0).Domain(), E[c]->ixType());
    auto const lo = domain.smallEnd(), hi = domain.bigEnd();
    auto const node = E[c]->ixType().toIntVect();
    auto const low = co.pmc_lo, high = co.pmc_hi;
    amrex::GpuArray<int, AMREX_SPACEDIM> per{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
      per[d] = w.Geom(0).isPeriodic(d);
    for (amrex::MFIter mfi(trace[c]); mfi.isValid(); ++mfi) {
      auto t = trace[c].array(mfi), g = global[c].array(mfi);
      auto m = support.RecoveryMask(c).const_array(mfi);
      amrex::ParallelFor(
          mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            int p[]{i, j, k};
            bool fixed = false;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
              bool lowwall = p[d] == lo[d];
#if defined(WARPX_DIM_RZ)
              if (d == 0)
                lowwall = false;
#endif
              fixed = fixed ||
                      (!per[d] && node[d] &&
                       ((lowwall && !low[d]) || (p[d] == hi[d] && !high[d])));
            }
#if defined(WARPX_DIM_RZ)
            fixed = fixed || (c == 1 && i == 0);
#endif
            g(i, j, k) = !fixed;
            t(i, j, k) = !fixed && !m(i, j, k);
          });
    }
  }
}
#include "NativeVacuumRetainedLinearOps.H"
#include "NativeVacuumFiniteSchur.H"
} // namespace
namespace {
using EndpointOwner=NativePairedDarwinFields;
using EndpointPart=EndpointOwner::EndpointPart;
using EndpointCV=EndpointOwner::CV;
using EndpointPC=WarpXParticleContainer;
bool EndpointAll(bool value){amrex::ParallelDescriptor::ReduceBoolAnd(value);return value;}
bool EndpointLayout(MF const* a,MF const& b,bool guards=true){
    return a&&a->boxArray()==b.boxArray()&&a->DistributionMap()==b.DistributionMap()&&
        a->nComp()==b.nComp()&&(!guards||a->nGrowVect()==b.nGrowVect());
}
void EndpointBytes(void const* left,void const* right,std::size_t count,int* bad){
    auto a=static_cast<unsigned char const*>(left),b=static_cast<unsigned char const*>(right);
    // Boolean mismatch flag cannot overflow; For keeps host iterations serial.
    // These owner methods do not launch an OpenMP parallel host region.
    amrex::For(count,[=] AMREX_GPU_DEVICE(std::size_t k){
        if(a[k]!=b[k])amrex::Gpu::Atomic::Exch(bad,1);
    });
}
void EndpointCompare(MF const& a,MF const& b,int* bad,bool region=false){
    for(amrex::MFIter it(a);it.isValid();++it){
        if(!region)EndpointBytes(a[it].dataPtr(),b[it].dataPtr(),a[it].size()*sizeof(amrex::Real),bad);
        else {
            auto x=a.const_array(it),y=b.const_array(it);int const n=a.nComp();
            amrex::For(it.fabbox(),n,[=] AMREX_GPU_DEVICE(int i,int j,int k,int c){
                auto const* p=reinterpret_cast<unsigned char const*>(&x(i,j,k,c));
                auto const* q=reinterpret_cast<unsigned char const*>(&y(i,j,k,c));
                bool mismatch=false;for(std::size_t z=0;z<sizeof(amrex::Real);++z)mismatch=mismatch||(p[z]!=q[z]);
                if(mismatch)amrex::Gpu::Atomic::Exch(bad,1);
            });
        }
    }
}
// Separate iterator lifetimes; actual FAB byte ranges include make_alias views.
using EndpointRange=std::pair<std::uintptr_t,std::uintptr_t>;
bool EndpointRanges(MF const& f,std::vector<EndpointRange>& ranges){
    bool good=true;
    for(amrex::MFIter it(f);it.isValid();++it){
        auto const begin=reinterpret_cast<std::uintptr_t>(f[it].dataPtr());
        auto const count=static_cast<std::uint64_t>(f[it].size());
        if(count>std::numeric_limits<std::uintptr_t>::max()/sizeof(amrex::Real)){good=false;continue;}
        auto const bytes=std::uintptr_t(count*sizeof(amrex::Real));
        if(begin>std::numeric_limits<std::uintptr_t>::max()-bytes||(!begin&&bytes)){good=false;continue;}
        ranges.emplace_back(begin,begin+bytes);
    }
    return good;
}
bool EndpointOverlap(std::vector<EndpointRange> const& a,std::vector<EndpointRange> const& b){
    for(auto const& x:a)for(auto const& y:b)if(x.first<y.second&&y.first<x.second)return true;
    return false;
}
// Bind the actual mutable rules consumed by image completion, centering,
// native rates and support coefficients. Capture once before Core construction;
// compare scalars/layouts first, then device coefficient bytes with the same
// single reduction as the protected population/field images. No trial host copy.
struct EndpointOperatorConfiguration {
    struct Species {
        std::string name;EndpointPC* pointer=nullptr;
        std::vector<int> flags;
        std::vector<amrex::ParticleReal> external;
    };
    struct Layout {
        std::string name;int component=0;MF const* pointer=nullptr;
        amrex::BoxArray boxes;amrex::DistributionMapping distribution;
        int components=0;amrex::IntVect grow;
    };
    std::vector<int> values;
    std::vector<amrex::Real> coefficients;
    std::vector<Layout> layouts;
    std::vector<Species> species;
    std::vector<std::string> names;
    std::string external_e,external_b;
    NativeInertiaSupportOptions support;
    std::array<amrex::Gpu::DeviceVector<amrex::Real>,3> centering;
    FiniteDifferenceSolver const* solver=nullptr;
    bool ready=false;
    static std::vector<int> Values(WarpX& w) {
        std::vector<int> v{w.maxLevel(),w.finestLevel(),WarpX::nox,WarpX::noy,WarpX::noz,
            WarpX::ncomps,int(WarpX::grid_type),int(WarpX::electromagnetic_solver_id),
            int(WarpX::field_gathering_algo),int(WarpX::current_deposition_algo),
            WarpX::field_centering_nox,WarpX::field_centering_noy,WarpX::field_centering_noz,
            int(WarpX::do_single_precision_comms),int(w.do_current_centering),
            int(WarpX::use_filter),int(EB::enabled())};
#if defined(WARPX_DIM_RZ)
        v.push_back(WarpX::n_rz_azimuthal_modes);
#endif
        for(int d=0;d<AMREX_SPACEDIM;++d){
            v.push_back(int(WarpX::field_boundary_lo[d]));v.push_back(int(WarpX::field_boundary_hi[d]));
            v.push_back(int(WarpX::particle_boundary_lo[d]));v.push_back(int(WarpX::particle_boundary_hi[d]));
            v.push_back(w.get_ng_fieldgather()[d]);
        }
        return v;
    }
    static std::vector<amrex::Real> Coefficients(WarpX& w) {
        auto const& model=*w.get_pointer_HybridPICModel();
        auto const cell=WarpX::CellSize(0);auto const inverse=WarpX::InvCellSize(0);
        return {model.m_electron_inertia_mass,model.m_n0_ref,model.m_n_floor,
            cell[0],cell[1],cell[2],inverse.x,inverse.y,inverse.z};
    }
    bool LayoutsMatch(WarpX& w)const {
        bool good=true;
        for(auto const& a:layouts){
            bool const exists=w.m_fields.has(a.name,ablastr::fields::Direction{a.component},0);
            if(!exists){good=false;continue;}
            auto const* f=w.m_fields.get(a.name,ablastr::fields::Direction{a.component},0);
            good=f==a.pointer&&f->boxArray()==a.boxes&&f->DistributionMap()==a.distribution&&
                f->nComp()==a.components&&f->nGrowVect()==a.grow&&good;
        }
        return good;
    }
    static std::vector<int> Flags(EndpointPC& pc) {
        auto const* physical=dynamic_cast<PhysicalParticleContainer const*>(&pc);
        bool const charged_scope=physical&&physical->ImplicitIonElectricWorkCaptureSupported();
        std::vector<int> v{int(pc.DoFieldIonization()),int(charged_scope),
            int(pc.do_not_deposit),int(pc.HasiAttrib("ionizationLevel"))};
#ifdef WARPX_QED
        for(bool flag:{pc.has_quantum_sync(),pc.has_breit_wheeler(),pc.has_virtual_photons(),
                      pc.has_virtual_photons_beam_size_effect()})v.push_back(int(flag));
#endif
        return v;
    }
    static std::vector<amrex::ParticleReal> External(EndpointPC& pc) {
        std::vector<amrex::ParticleReal> v;
        for(auto x:pc.m_E_external_particle)v.push_back(x);
        for(auto x:pc.m_B_external_particle)v.push_back(x);
        return v;
    }
    static std::array<amrex::Gpu::DeviceVector<amrex::Real> const*,3> Stencils(WarpX& w) {
        return {&w.device_field_centering_stencil_coeffs_x,
            &w.device_field_centering_stencil_coeffs_y,&w.device_field_centering_stencil_coeffs_z};
    }
    static bool SameSupport(NativeInertiaSupportOptions const& a,NativeInertiaSupportOptions const& b) {
        return a.lower==b.lower&&a.upper==b.upper&&a.coefficient==b.coefficient&&
            a.recovery==b.recovery&&a.components==b.components&&a.charge_floor==b.charge_floor&&
            a.reference_charge_density==b.reference_charge_density&&
            a.recovery_density_fraction==b.recovery_density_fraction;
    }
    static bool NoExternalAction(WarpX& w) {
        bool good=true;
        for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
            auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
            for(WarpXParIter it(pc,0);it.isValid();++it)good=GetExternalEBField(it).isNoOp()&&good;
        }
        return good;
    }
    bool Capture(WarpX& w) {
        if(!EndpointAll(!ready&&compensated_curl::Supported(w)))return false;
        values=Values(w);coefficients=Coefficients(w);support=NativeVacuumSupportOptions(w);
        bool layout=true;
        for(auto const* name:{"Efield_fp","Bfield_fp","Efield_aux","Bfield_aux","current_fp"})
            for(int c=0;c<3;++c)layout=w.m_fields.has(name,ablastr::fields::Direction{c},0)&&layout;
        if(!EndpointAll(layout))return false;
        for(auto const* name:{"Efield_fp","Bfield_fp","Efield_aux","Bfield_aux","current_fp"})
            for(int c=0;c<3;++c){auto const* f=w.m_fields.get(name,ablastr::fields::Direction{c},0);
                layouts.push_back({name,c,f,f->boxArray(),f->DistributionMap(),f->nComp(),f->nGrowVect()});}
        solver=w.get_pointer_fdtd_solver_fp(0);
        auto& particles=w.GetPartContainer();names=particles.GetSpeciesNames();
        external_e=particles.m_E_ext_particle_s;external_b=particles.m_B_ext_particle_s;
        bool charged_scope=true;
        for(auto const& name:names){auto& pc=particles.GetParticleContainerFromName(name);
            auto const flags=Flags(pc);
            charged_scope=(pc.getCharge()==0.||flags[1])&&charged_scope;
            species.push_back({name,&pc,flags,External(pc)});}
        if(!EndpointAll(charged_scope))return false;
        auto const input=Stencils(w);
        for(int c=0;c<3;++c)centering[c]=*input[c];
        ready=true;return EndpointAll(LocalMatches(w)&&NoExternalAction(w));
    }
    bool LocalMatches(WarpX& w)const {
        if(!ready||Values(w)!=values||Coefficients(w)!=coefficients||!LayoutsMatch(w)||!SameSupport(NativeVacuumSupportOptions(w),support)||
            solver!=w.get_pointer_fdtd_solver_fp(0))return false;
        auto& particles=w.GetPartContainer();
        if(particles.GetSpeciesNames()!=names||particles.m_E_ext_particle_s!=external_e||
            particles.m_B_ext_particle_s!=external_b)return false;
        auto const input=Stencils(w);
        for(int c=0;c<3;++c)if(input[c]->size()!=centering[c].size())return false;
        for(auto const& s:species){auto& pc=particles.GetParticleContainerFromName(s.name);
            if(&pc!=s.pointer||Flags(pc)!=s.flags||External(pc)!=s.external)return false;}
        return true;
    }
    void CompareStencils(WarpX& w,int* bad)const {
        auto const input=Stencils(w);
        for(int c=0;c<3;++c)EndpointBytes(input[c]->data(),centering[c].data(),
            centering[c].size()*sizeof(amrex::Real),bad);
    }
};
struct EndpointOrigin {
    struct Protected {std::string name;int component=-1;MF const* original=nullptr;MF copy;};
    struct Species {std::string name;EndpointPC* pc=nullptr;amrex::Real mass=0.,charge=0.;
        std::vector<std::string> reals,ints;EndpointPC::ParticleLevel old;};
    EndpointOperatorConfiguration configuration;
    WarpX* simulation=nullptr;
    amrex::Real clock=0.,time=0.,mass=0.,n0=0.,floor=0.;int step=0;
    bool accepted_boundary=false;
    std::uint64_t epoch=0;amrex::Long next_id=0;
    amrex::Geometry geometry;amrex::BoxArray cells;amrex::DistributionMapping distribution;
    std::vector<std::string> names;std::vector<Species> species;
    std::vector<std::unique_ptr<Protected>> fields;
    bool ready=false;
    void Protect(WarpX& w,char const* name,int c=-1){
        auto value=std::make_unique<Protected>();value->name=name;value->component=c;
        value->original=c<0?w.m_fields.get(name,0):w.m_fields.get(name,ablastr::fields::Direction{c},0);
        auto const& f=*value->original;value->copy.define(f.boxArray(),f.DistributionMap(),f.nComp(),f.nGrowVect());
        MF::Copy(value->copy,f,0,0,f.nComp(),f.nGrowVect());fields.push_back(std::move(value));
    }
    bool Capture(WarpX& w,amrex::Real t,std::uint64_t e,bool initial=false,bool restored=false,bool source_origin=false){
        bool const fresh_initial=!initial||(w.getistep(0)==0&&t==w.gett_new(0)&&
            !w.get_pointer_HybridPICModel()->m_darwin_checkpoint_restored);
        bool const restart_boundary=!restored||(!initial&&t==w.gett_new(0)&&
            w.get_pointer_HybridPICModel()->m_darwin_checkpoint_restored);
        bool const source_boundary=!source_origin||(!initial&&!restored&&t==w.gett_new(0)&&
            !w.get_pointer_HybridPICModel()->m_darwin_checkpoint_restored);
        if(!EndpointAll(!ready&&configuration.LocalMatches(w)&&std::isfinite(t)&&fresh_initial&&restart_boundary&&source_boundary&&
            e==std::uint64_t(w.getistep(0))+((initial||restored||source_origin)?0:1)))return false;
        simulation=&w;clock=w.gett_new(0);step=w.getistep(0);time=t;epoch=e;
        accepted_boundary=initial||restored||source_origin;
        geometry=w.Geom(0);cells=w.boxArray(0);distribution=w.DistributionMap(0);
        auto const& model=*w.get_pointer_HybridPICModel();
        mass=model.m_electron_inertia_mass;n0=model.m_n0_ref;floor=model.m_n_floor;
        next_id=EndpointPC::ParticleType::the_next_id;names=w.GetPartContainer().GetSpeciesNames();
        bool good=true;
        for(auto const& name:names){
            auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
            good=good&&pc.finestLevel()==0;
            species.emplace_back();auto& s=species.back();s.name=name;s.pc=&pc;
            s.mass=pc.getMass();s.charge=pc.getCharge();s.reals=pc.GetRealSoANames();s.ints=pc.GetIntSoANames();
            for(auto const& [key,tile]:pc.GetParticles(0)){
                good=good&&tile.numNeighborParticles()==0;
                auto& copy=s.old[key];copy.define(tile.NumRealComps()-PIdx::nattribs,
                    tile.NumIntComps()-IntIdx::nattribs,nullptr,nullptr,pc.arena());
                copy.GetStructOfArrays()=tile.GetStructOfArrays();
            }
        }
        for(auto const* name:{"rho_fp","hybrid_rho_vacmask_fp","hybrid_electron_energy_fp",
            "hybrid_electron_temperature_fp","hybrid_phi_darwin_fp"})good=w.m_fields.has(name,0)&&good;
        for(auto const* name:{"hybrid_E_long_fp","hybrid_A_fp","Bfield_fp","diagnostic_Je_endpoint_fp",
            "diagnostic_D_endpoint_fp","current_fp","hybrid_current_fp_plasma"})
            for(int c=0;c<3;++c)good=w.m_fields.has(name,ablastr::fields::Direction{c},0)&&good;
        if(!EndpointAll(good))return false;
        for(auto const* name:{"rho_fp","hybrid_rho_vacmask_fp","hybrid_electron_energy_fp",
            "hybrid_electron_temperature_fp","hybrid_phi_darwin_fp"})Protect(w,name);
        for(auto const* name:{"hybrid_E_long_fp","hybrid_A_fp","Bfield_fp","diagnostic_Je_endpoint_fp",
            "diagnostic_D_endpoint_fp","current_fp","hybrid_current_fp_plasma"})
            for(int c=0;c<3;++c)Protect(w,name,c);
        ready=true;return Matches(w);
    }
    bool Matches(WarpX& w,bool boundary_handoff=false,bool exact_particles=true)const {
        // A boundary handoff checks every captured physical/population/config
        // byte below and exactly one prescribed clock/step transition. It does
        // not waive redistribution or promote a stale pending receipt.
        bool const clocks=boundary_handoff?
            (!accepted_boundary&&w.gett_new(0)==time&&std::uint64_t(w.getistep(0))==epoch):
            (w.gett_new(0)==clock&&w.getistep(0)==step);
        bool good=ready&&simulation==&w&&configuration.LocalMatches(w)&&clocks&&
            epoch==std::uint64_t(step)+(accepted_boundary?0:1)&&w.finestLevel()==0&&w.boxArray(0)==cells&&w.DistributionMap(0)==distribution&&
            w.Geom(0).Domain()==geometry.Domain()&&w.Geom(0).Coord()==geometry.Coord()&&
            w.GetPartContainer().GetSpeciesNames()==names&&EndpointPC::ParticleType::the_next_id==next_id;
        auto const& model=*w.get_pointer_HybridPICModel();
        good=good&&model.m_electron_inertia_mass==mass&&model.m_n0_ref==n0&&model.m_n_floor==floor;
        for(int d=0;d<AMREX_SPACEDIM;++d)good=good&&w.Geom(0).ProbLo(d)==geometry.ProbLo(d)&&
            w.Geom(0).ProbHi(d)==geometry.ProbHi(d)&&w.Geom(0).isPeriodic(d)==geometry.isPeriodic(d);
        for(auto const& f:fields){
            bool const exists=f->component<0?w.m_fields.has(f->name,0):w.m_fields.has(f->name,ablastr::fields::Direction{f->component},0);
            auto const* now=exists?(f->component<0?w.m_fields.get(f->name,0):w.m_fields.get(f->name,ablastr::fields::Direction{f->component},0)):nullptr;
            good=now==f->original&&EndpointLayout(now,f->copy)&&good;
        }
        if(!EndpointAll(good))return false;
        for(auto const& s:species){
            auto& pc=w.GetPartContainer().GetParticleContainerFromName(s.name);
            good=good&& &pc==s.pc&&pc.finestLevel()==0&&pc.getMass()==s.mass&&pc.getCharge()==s.charge&&
                pc.GetRealSoANames()==s.reals&&pc.GetIntSoANames()==s.ints;
            if(!exact_particles)continue;
            good=pc.GetParticles(0).size()==s.old.size()&&good;
            for(auto const& [key,old]:s.old){
                auto it=pc.GetParticles(0).find(key);if(it==pc.GetParticles(0).end()){good=false;continue;}
                auto const& now=it->second;auto const& a=now.GetStructOfArrays();auto const& b=old.GetStructOfArrays();
                good=good&&now.numParticles()==old.numParticles()&&now.numNeighborParticles()==0&&
                    now.NumRealComps()==old.NumRealComps()&&now.NumIntComps()==old.NumIntComps()&&
                    a.GetIdCPUData().size()==b.GetIdCPUData().size();
                if(now.NumRealComps()==old.NumRealComps())for(int c=0;c<now.NumRealComps();++c)
                    good=good&&a.GetRealData(c).size()==b.GetRealData(c).size();
                if(now.NumIntComps()==old.NumIntComps())for(int c=0;c<now.NumIntComps();++c)
                    good=good&&a.GetIntData(c).size()==b.GetIntData(c).size();
            }
        }
        if(!EndpointAll(good))return false;
        if(!EndpointAll(configuration.NoExternalAction(w)))return false;
        amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
        configuration.CompareStencils(w,bad);
        for(auto const& f:fields)EndpointCompare(*f->original,f->copy,bad);
        if(exact_particles)for(auto const& s:species)for(auto const& [key,old]:s.old){
            auto const& now=s.pc->GetParticles(0).at(key);auto const& a=now.GetStructOfArrays();auto const& b=old.GetStructOfArrays();
            EndpointBytes(a.GetIdCPUData().data(),b.GetIdCPUData().data(),a.GetIdCPUData().size()*sizeof(std::uint64_t),bad);
            for(int c=0;c<now.NumRealComps();++c)EndpointBytes(a.GetRealData(c).data(),b.GetRealData(c).data(),a.GetRealData(c).size()*sizeof(amrex::ParticleReal),bad);
            for(int c=0;c<now.NumIntComps();++c)EndpointBytes(a.GetIntData(c).data(),b.GetIntData(c).data(),a.GetIntData(c).size()*sizeof(int),bad);
        }
        return EndpointAll(invalid.dataValue()==0);
    }
    bool RedistributeAccepted(WarpX& w,bool local,amrex::IntVect const& max_cells,
        bool sort,amrex::IntVect const& bins,bool deposition,amrex::IntVect const& index_type)
    {
        // All control values must agree before a clone or the live population
        // enters native communication. Zero-cell local movement is valid.
        std::vector<int> values{int(local),int(sort),int(deposition)};
        for(int d=0;d<AMREX_SPACEDIM;++d){values.push_back(max_cells[d]);
            values.push_back(bins[d]);values.push_back(index_type[d]);}
        bool good=accepted_boundary&&Matches(w);
        for(int value:values){int lo=value,hi=value;
            amrex::ParallelDescriptor::ReduceIntMin(lo);amrex::ParallelDescriptor::ReduceIntMax(hi);
            good=lo==hi&&good;}
        for(int d=0;d<AMREX_SPACEDIM;++d)good=max_cells[d]>=0&&
            (!local||max_cells[d]<geometry.Domain().length(d))&&
            (!sort||deposition||bins[d]>0)&&(!sort||!deposition||index_type[d]==0||index_type[d]==1)&&good;
        if(!EndpointAll(good))return false;
        amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
        for(auto const& s:species)for(auto const& [key,tile]:s.pc->GetParticles(0)){
            amrex::ignore_unused(key);auto const* ids=tile.GetStructOfArrays().GetIdCPUData().data();
            amrex::ParallelFor(tile.numParticles(),[=] AMREX_GPU_DEVICE(amrex::Long i){
                if(!amrex::ConstParticleIDWrapper(ids[i]).is_valid())amrex::Gpu::Atomic::Max(bad,1);
            });
        }
        if(!EndpointAll(invalid.dataValue()==0))return false;
        // Clone on the owning device. AMReX computes its own periodic image,
        // destination tile and MPI migration. This is an expected representation
        // only; no physical particle, field or source is reconstructed from it.
        struct AllAttributeClone final : EndpointPC::Base {
            explicit AllAttributeClone(EndpointPC const& pc):EndpointPC::Base(pc.make_alike<>()){
                std::fill(h_redistribute_real_comp.begin(),h_redistribute_real_comp.end(),1);
                std::fill(h_redistribute_int_comp.begin(),h_redistribute_int_comp.end(),1);
                SetParticleSize();
            }
        };
        std::vector<std::unique_ptr<AllAttributeClone>> predicted;
        for(auto const& s:species){
            auto clone=std::make_unique<AllAttributeClone>(*s.pc);
            clone->SetArena(s.pc->arena());clone->copyParticles(*s.pc,true);
            if(local)clone->Redistribute(0,0,amrex::IntVect(0),true,max_cells);
            else clone->Redistribute();
            // Redistribute is a representation handoff, never an absorbing
            // boundary action. Reject any native deletion before moving the
            // live population, even if the same deletion would be repeatable.
            amrex::Long before=0,after=0;
            for(auto const& item:s.pc->GetParticles(0))before+=item.second.numParticles();
            for(auto const& item:clone->GetParticles(0))after+=item.second.numParticles();
            amrex::ParallelDescriptor::ReduceLongSum(before);
            amrex::ParallelDescriptor::ReduceLongSum(after);
            good=before==after&&good;
            predicted.push_back(std::move(clone));
        }
        // Cloning may communicate, but must not alter the captured live origin.
        if(!EndpointAll(good)||!Matches(w))return false;
        auto& particles=w.GetPartContainer();
        // The retained origin includes implicit history attributes normally
        // omitted by native communication. Preserve them through this one
        // owned relocation; each container restores its original flags.
        for(auto const& s:species)s.pc->RedistributeAllAttributes(local,max_cells);
        if(sort)particles.SortParticlesByBin(bins,deposition,index_type);
        if(!Matches(w,false,false))return false;
        std::vector<std::unique_ptr<detail::EndpointHandoffScratch>> scratch;
        for(std::size_t n=0;n<species.size();++n){
            auto const& actual=species[n].pc->GetParticles(0);
            auto const& expected=predicted[n]->GetParticles(0);
            for(auto const& [key,tile]:actual){
                auto found=expected.find(key);
                if(found==expected.end()){good=tile.numParticles()==0&&tile.numNeighborParticles()==0&&good;continue;}
                good=detail::CompareEndpointTilePermutation(tile,found->second,bad,scratch)&&good;
            }
            for(auto const& [key,tile]:expected)if(actual.find(key)==actual.end())
                good=tile.numParticles()==0&&tile.numNeighborParticles()==0&&good;
        }
        bool const bytes=invalid.dataValue()==0; // Synchronizes every device comparison before scratch destruction.
        if(!EndpointAll(good&&bytes))return false;
        for(auto& s:species){
            s.old.clear();
            for(auto const& [key,tile]:s.pc->GetParticles(0)){
                auto& copy=s.old[key];copy.define(tile.NumRealComps()-PIdx::nattribs,
                    tile.NumIntComps()-IntIdx::nattribs,nullptr,nullptr,s.pc->arena());
                copy.GetStructOfArrays()=tile.GetStructOfArrays();
            }
        }
        return Matches(w);
    }
};
struct EndpointSolveContext {
    WarpX& w;
    std::unique_ptr<NativeInertiaSupport> support;
    std::unique_ptr<Core> core,all;
    amrex::Array<amrex::iMultiFab,3> trace,global,empty;
    MF density,mask;
    Field baseline,reference,force,ell,magnetic;
    std::unique_ptr<Field> baseline_low,auxiliary_high,auxiliary_low;
    RetainedEndpointSave saved;
    EndpointOrigin origin;
    EndpointSolveContext(WarpX& sim,MF const& rho,MF const& masks,View const& base,
        View const& ref,View const& f,View const& l,View const* low=nullptr):w(sim),baseline(Clone(base)),
        reference(Clone(ref)),force(Clone(f)),ell(Clone(l)),magnetic(Clone(sim.m_fields.get_alldirs(FT::Bfield_fp,0))) {
        if(low)baseline_low=std::make_unique<Field>(Clone(*low));
        for(auto const& x:{std::pair<MF*,MF const*>{&density,&rho},{&mask,&masks}}){
            x.first->define(x.second->boxArray(),x.second->DistributionMap(),x.second->nComp(),x.second->nGrowVect());
            MF::Copy(*x.first,*x.second,0,0,x.second->nComp(),x.second->nGrowVect());
        }
    }
    bool Prepare(){
        if(!origin.configuration.Capture(w))return false;
        support=std::make_unique<NativeInertiaSupport>(Geometry(w),w.boxArray(0),w.DistributionMap(0),NativeVacuumSupportOptions(w));
        if(!support->FreezeEdges(density,mask))return false;
        Core::Options co;co.relative_tolerance=1.e-14;co.absolute_tolerance=1.e-14;
        for(int d=0;d<AMREX_SPACEDIM;++d){co.pmc_lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
            co.pmc_hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;}
        DefineTrace(w,*support,co,trace,global,empty);
        auto e=w.m_fields.get_alldirs(FT::Efield_fp,0),b=w.m_fields.get_alldirs(FT::Bfield_fp,0);
        core=std::make_unique<Core>(w,e,b,co);all=std::make_unique<Core>(w,e,b,co);
        auto zero=Clone(e);Zero(V(zero));
        Core::Mask const recovery{&support->RecoveryMask(0),&support->RecoveryMask(1),&support->RecoveryMask(2)};
        Core::Mask const fixed_trace{&trace[0],&trace[1],&trace[2]};
        bool const a=baseline_low?
            core->FreezeRetained(recovery,fixed_trace,V(baseline),V(*baseline_low),V(reference)):
            core->Freeze(recovery,fixed_trace,V(baseline),V(reference));
        bool const c=all->Freeze({&global[0],&global[1],&global[2]},
            {&empty[0],&empty[1],&empty[2]},V(zero),V(reference));
        if(!EndpointAll(a&&c))return false;
        return RetainedFreeUnionMatches(*support,trace,global);
    }
    EndpointOwner::EndpointView ViewOfSaved(){
        EndpointOwner::EndpointView v;
        for(int k=0;k<EndpointOwner::endpoint_parts;++k)
            v.fields[k]={&saved.fields[k][0],&saved.fields[k][1],&saved.fields[k][2]};
        v.nodal_inertia_high=&saved.nodal_high;v.nodal_inertia_low=&saved.nodal_low;return v;
    }
    bool Equal(EndpointOwner::EndpointView const& v,bool published)const {
        bool good=saved.ready&&core->MatchesRetained(saved.recovered);
        for(int k=0;k<EndpointOwner::endpoint_parts;++k)for(int c=0;c<3;++c)
            good=EndpointLayout(v.fields[k][c],saved.fields[k][c])&&good;
        good=EndpointLayout(v.nodal_inertia_high,saved.nodal_high)&&EndpointLayout(v.nodal_inertia_low,saved.nodal_low)&&good;
        auto e=w.m_fields.get_alldirs(FT::Efield_fp,0),ei=w.m_fields.get_alldirs("hybrid_E_inertial_fp",0);
        auto const* node=w.m_fields.get("hybrid_E_inertial_nodal",0);
        if(published){
            for(int c=0;c<3;++c)good=EndpointLayout(e[c],saved.fields[int(EndpointPart::ElectricHigh)][c])&&
                EndpointLayout(ei[c],saved.fields[int(EndpointPart::InertiaHigh)][c],false)&&
                saved.fields[int(EndpointPart::InertiaHigh)][c].nGrowVect().allGE(ei[c]->nGrowVect())&&good;
            good=EndpointLayout(node,saved.nodal_high)&&good;
        }
        if(!EndpointAll(good))return false;
        amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
        for(int k=0;k<EndpointOwner::endpoint_parts;++k)for(int c=0;c<3;++c)
            EndpointCompare(*v.fields[k][c],saved.fields[k][c],bad);
        EndpointCompare(*v.nodal_inertia_high,saved.nodal_high,bad);EndpointCompare(*v.nodal_inertia_low,saved.nodal_low,bad);
        if(published){
            for(int c=0;c<3;++c){EndpointCompare(*e[c],saved.fields[int(EndpointPart::ElectricHigh)][c],bad);
                EndpointCompare(*ei[c],saved.fields[int(EndpointPart::InertiaHigh)][c],bad,true);}
            EndpointCompare(*node,saved.nodal_high,bad);
        }
        return EndpointAll(invalid.dataValue()==0);
    }
    NativeVacuumConstraintResult Check(EndpointOwner::EndpointView const& v){
        NativeVacuumConstraintResult result;
        if(!Equal(v,false)||!origin.Matches(w))return result;
        // This immutable Core owns its freeze and retained recovery throughout
        // the endpoint lifetime. The actual registered operands were compared
        // exactly before its action; no re-solve or rounded reconstruction.
        JoinedRetainedMatrixFree op(w,*support,*core,*all,trace,V(reference));
        op.observation.clear();op.capture_phase.clear();op.recovered=std::move(saved.recovered);
        struct ReturnRecovery {Core::RetainedRecovery& out;Core::RetainedRecovery& in;
            ~ReturnRecovery(){out=std::move(in);}} restore{saved.recovered,op.recovered};
        for(int c=0;c<3;++c){
            op.field[c].setVal(0.);op.field_low[c].setVal(0.);
            MF::Copy(op.field[c],*v.fields[int(EndpointPart::TransverseHigh)][c],0,0,1,0);
            MF::Copy(op.field_low[c],*v.fields[int(EndpointPart::TransverseLow)][c],0,0,1,0);
        }
        if(!core->ApplyRetainedMaskedCurl(V(op.curl),V(op.curl_low),op.recovered))return result;
        auto const& receipt=op.recovered.Receipt();
        if(!EndpointAll(receipt.ready&&receipt.fixed_trace_exact&&bool(receipt.solve)&&
            std::isfinite(receipt.solve.target)&&receipt.solve.target>0.&&
            std::isfinite(receipt.norm_roundoff_factor)&&receipt.norm_roundoff_factor>=1.))return result;
        double const hi=core->MetricDot(V(op.curl),V(op.curl)),lo=core->MetricDot(V(op.curl_low),V(op.curl_low));
        double const residual=std::nextafter((std::sqrt(hi)+std::sqrt(lo))*receipt.norm_roundoff_factor,
            std::numeric_limits<double>::infinity());
        if(!(hi>=0.)||!(lo>=0.)||!std::isfinite(residual)||residual>receipt.solve.target)return result;
        if(!op.Rates(true,&VrefEll())||!op.RateMass())return result;
        op.GradientMass(saved.node);auto row=op.makeVecRHS();op.Assemble(V(row.edge),false,&VrefForce(),&VrefEll());
        result.plasma_error=op.RowError(V(row.edge));result.divergence_error=op.DivergenceError();
        result.null_defect=op.length*(op.graph.Compatibility(op.divergence,support->NodeWeights())+
            op.graph.Compatibility(op.divergence_low,support->NodeWeights()));
        result.residual=residual;result.iterations=receipt.solve.iterations;
        result.compatible=!op.failed&&std::isfinite(result.null_defect)&&result.null_defect<=1.e-9;
        result.converged=result.compatible&&std::isfinite(result.plasma_error)&&result.plasma_error<1.e-8&&
            std::isfinite(result.divergence_error)&&result.divergence_error<1.e-8;
        return result;
    }
    // Stable value views avoid references to ephemeral aggregate expressions.
    View ell_view{},force_view{};
    View const& VrefEll(){ell_view=V(ell);return ell_view;}
    View const& VrefForce(){force_view=V(force);return force_view;}
};
} // namespace
struct NativePairedDarwinFields::EndpointDraft::Impl {
    std::unique_ptr<EndpointSolveContext> context;
};
NativePairedDarwinFields::EndpointDraft::EndpointDraft()=default;
NativePairedDarwinFields::EndpointDraft::~EndpointDraft()=default;
NativePairedDarwinFields::EndpointDraft::EndpointDraft(EndpointDraft&&)noexcept=default;
NativePairedDarwinFields::EndpointDraft& NativePairedDarwinFields::EndpointDraft::operator=(EndpointDraft&&)noexcept=default;
bool NativePairedDarwinFields::EndpointDraft::Ready()const noexcept {
    return m_impl&&m_impl->context&&m_impl->context->saved.ready&&
        m_impl->context->core->MatchesRetained(m_impl->context->saved.recovered);
}
bool NativePairedDarwinFields::EndpointDraft::Bind(WarpX& w,R time,std::uint64_t epoch,bool initial,bool restored,bool source_origin){
    if(!EndpointAll(Ready()&& &m_impl->context->w==&w&&!m_impl->context->origin.ready))return false;
    auto& c=*m_impl->context;
    // Verify the restored transaction still supplies the actual solve inputs.
    auto ell=w.m_fields.get_alldirs("hybrid_E_long_fp",0),b=w.m_fields.get_alldirs(FT::Bfield_fp,0);
    auto const& rho=*w.m_fields.get(FT::rho_fp,0);MF density(rho,amrex::make_alias,0,1);
    auto const& mask=*w.m_fields.get("hybrid_rho_vacmask_fp",0);
    bool layout=EndpointLayout(&density,c.density)&&EndpointLayout(&mask,c.mask);
    for(int k=0;k<3;++k)layout=EndpointLayout(ell[k],c.ell[k])&&EndpointLayout(b[k],c.magnetic[k])&&layout;
    if(!EndpointAll(layout))return false;
    amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
    EndpointCompare(density,c.density,bad);EndpointCompare(mask,c.mask,bad);
    for(int k=0;k<3;++k){EndpointCompare(*ell[k],c.ell[k],bad);EndpointCompare(*b[k],c.magnetic[k],bad);}
    if(!EndpointAll(invalid.dataValue()==0))return false;
    return c.origin.Capture(w,time,epoch,initial,restored,source_origin);
}
void NativePairedDarwinFields::EndpointDraft::WriteCheckpoint(std::string const& directory) const {
    namespace io=endpoint_checkpoint;
    AMREX_ALWAYS_ASSERT(Ready());auto& c=*m_impl->context;
    AMREX_ALWAYS_ASSERT(c.origin.accepted_boundary&&c.origin.Matches(c.w)&&c.Equal(c.ViewOfSaved(),true));
    io::WriteText(directory+"/context",std::to_string(bool(c.baseline_low))+"\n");
    for(auto const& entry:{std::pair<char const*,MF const*>{"density",&c.density},{"mask",&c.mask},
        {"node",&c.saved.node},{"nodal_high",&c.saved.nodal_high},{"nodal_low",&c.saved.nodal_low}})
        io::Write(directory+"/"+entry.first,*entry.second);
    for(auto const& entry:{std::pair<char const*,Field const*>{"baseline",&c.baseline},
        {"reference",&c.reference},{"force",&c.force},{"ell",&c.ell},{"magnetic",&c.magnetic},
        {"original_trace",&c.saved.original_trace}})
        for(int d=0;d<3;++d)io::Write(directory+"/"+entry.first+std::to_string(d),(*entry.second)[d]);
    if(c.baseline_low)for(int d=0;d<3;++d)
        io::Write(directory+"/baseline_low"+std::to_string(d),(*c.baseline_low)[d]);
    for(int k=0;k<endpoint_parts;++k)for(int d=0;d<3;++d)
        io::Write(directory+"/part"+std::to_string(k)+"_"+std::to_string(d),c.saved.fields[k][d]);
}
NativePairedDarwinFields::EndpointDraft NativePairedDarwinFields::EndpointDraft::ReadCheckpoint(
    WarpX& w,std::string const& directory,R time,std::uint64_t epoch)
{
    namespace io=endpoint_checkpoint;
    AMREX_ALWAYS_ASSERT(w.get_pointer_HybridPICModel()->m_darwin_checkpoint_restored);
    auto const e=w.m_fields.get_alldirs(FT::Efield_fp,0);
    auto const& rho=*w.m_fields.get(FT::rho_fp,0);MF density(rho,amrex::make_alias,0,1);
    auto const& mask=*w.m_fields.get("hybrid_rho_vacmask_fp",0);
    auto data=std::make_unique<Impl>();
    data->context=std::make_unique<EndpointSolveContext>(w,density,mask,e,e,e,e);
    auto& c=*data->context;
    std::istringstream metadata(io::ReadText(directory+"/context"));int low=-1;metadata>>low;
    AMREX_ALWAYS_ASSERT(bool(metadata)&&(low==0||low==1));metadata>>std::ws;AMREX_ALWAYS_ASSERT(metadata.eof());
    for(auto const& entry:{std::pair<char const*,MF*>{"density",&c.density},{"mask",&c.mask}})
        io::Read(directory+"/"+entry.first,*entry.second);
    // Private affine input guard widths are numeric layout, not permissions.
    // They must fit the actual electric allocation on the restarted mesh.
    for(auto const& entry:{std::pair<char const*,Field*>{"baseline",&c.baseline},
        {"reference",&c.reference},{"force",&c.force},{"ell",&c.ell},
        {"original_trace",&c.saved.original_trace}})
        for(int d=0;d<3;++d){auto const path=directory+"/"+entry.first+std::to_string(d);
            (*entry.second)[d].clear();io::Allocate(path,(*entry.second)[d],*e[d]);
            io::Read(path,(*entry.second)[d]);}
    for(int d=0;d<3;++d)io::Read(directory+"/magnetic"+std::to_string(d),c.magnetic[d]);
    if(low){c.baseline_low=std::make_unique<Field>();for(int d=0;d<3;++d){
        auto const path=directory+"/baseline_low"+std::to_string(d);
        io::Allocate(path,(*c.baseline_low)[d],*e[d]);io::Read(path,(*c.baseline_low)[d]);}}
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(c.Prepare(),"Cannot reconstruct accepted endpoint operator/support");
    for(int k=0;k<endpoint_parts;++k){c.saved.fields[k]=Clone(e);for(int d=0;d<3;++d)
        io::Read(directory+"/part"+std::to_string(k)+"_"+std::to_string(d),c.saved.fields[k][d]);}
    auto const& weights=c.support->NodeWeights();
    c.saved.node.define(weights.boxArray(),weights.DistributionMap(),1,1);
    io::Read(directory+"/node",c.saved.node);
    auto const& nodal=*w.m_fields.get("hybrid_E_inertial_nodal",0);
    for(auto const& entry:{std::pair<char const*,MF*>{"nodal_high",&c.saved.nodal_high},{"nodal_low",&c.saved.nodal_low}}){
        entry.second->define(nodal.boxArray(),nodal.DistributionMap(),nodal.nComp(),nodal.nGrowVect());
        io::Read(directory+"/"+entry.first,*entry.second);}
    View trace{&c.saved.original_trace[0],&c.saved.original_trace[1],&c.saved.original_trace[2]};
    auto const receipt=c.core->ReconstructAffineRetained(c.saved.recovered,
        c.saved.PartView(EndpointPart::TransverseHigh),c.saved.PartView(EndpointPart::TransverseLow),&trace);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(receipt.ready&&receipt.fixed_trace_exact&&bool(receipt.solve),
        "Restarted retained field does not satisfy its reconstructed affine operator");
    c.saved.ready=true; // Fresh numerical reconstruction above, never a serialized grant.
    EndpointDraft draft;draft.m_impl=std::move(data);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(draft.Bind(w,time,epoch,false,true),
        "Restarted endpoint inputs do not match the accepted physical state");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(c.Equal(c.ViewOfSaved(),true),
        "Restarted physical E/Ei differ from the saved retained high fields");
    auto const checked=c.Check(c.ViewOfSaved());
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(checked.compatible&&checked.converged,
        "Restarted endpoint failed fresh coupled physical constraints");
    c.saved.result=checked;c.saved.pair_null=checked.null_defect;
    return draft;
}
bool NativePairedDarwinFields::EndpointDraft::Matches(WarpX& w,EndpointView const& v)const {
    if(!EndpointAll(Ready()&& &m_impl->context->w==&w))return false;
    auto const& c=*m_impl->context;bool const origin=c.origin.Matches(w);
    bool const fields=c.Equal(v,true);return origin&&fields;
}
bool NativePairedDarwinFields::EndpointDraft::MatchesTransverse(CV const& high,CV const& low)const {
    if(!EndpointAll(Ready()))return false;
    auto const& c=*m_impl->context;bool layout=true;
    for(int d=0;d<3;++d){
        auto const& h=c.saved.fields[int(EndpointPart::TransverseHigh)][d];
        auto const& l=c.saved.fields[int(EndpointPart::TransverseLow)][d];
        layout=EndpointLayout(high[d],h,false)&&EndpointLayout(low[d],l,false)&&layout;
        if(high[d])layout=h.nGrowVect().allGE(high[d]->nGrowVect())&&layout;
        if(low[d])layout=l.nGrowVect().allGE(low[d]->nGrowVect())&&layout;
    }
    if(!EndpointAll(layout))return false;
    amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
    for(int d=0;d<3;++d){
        EndpointCompare(*high[d],c.saved.fields[int(EndpointPart::TransverseHigh)][d],bad,true);
        EndpointCompare(*low[d],c.saved.fields[int(EndpointPart::TransverseLow)][d],bad,true);
    }
    return EndpointAll(invalid.dataValue()==0);
}
bool NativePairedDarwinFields::EndpointDraft::BindAcceptedBoundary(WarpX& w,EndpointView const& v){
    if(!EndpointAll(Ready()&& &m_impl->context->w==&w))return false;
    auto& c=*m_impl->context;
    bool const origin=c.origin.Matches(w,true);bool const fields=c.Equal(v,true);
    if(!origin||!fields)return false;
    c.origin.clock=c.origin.time;c.origin.step=w.getistep(0);c.origin.accepted_boundary=true;
    return true;
}
bool NativePairedDarwinFields::EndpointDraft::RedistributeAccepted(WarpX& w,
    EndpointView const& v,bool local,amrex::IntVect const& max_cells,bool sort,
    amrex::IntVect const& bins,bool deposition,amrex::IntVect const& index_type)
{
    if(!EndpointAll(Ready()&& &m_impl->context->w==&w))return false;
    auto& c=*m_impl->context;
    if(!c.Equal(v,true)||!c.origin.RedistributeAccepted(w,local,max_cells,sort,bins,deposition,index_type))return false;
    return c.Equal(v,true);
}
bool NativePairedDarwinFields::EndpointDraft::PublishAuxiliary(WarpX& w,V const& high,V const& low){
    if(!EndpointAll(Ready()&& &m_impl->context->w==&w))return false;
    auto& c=*m_impl->context;
    bool const origin=c.origin.Matches(w),fields=c.Equal(c.ViewOfSaved(),true);
    if(!origin||!fields)return false;
    auto native=w.m_fields.get_alldirs(FT::Efield_aux,0);
    bool layout=w.finestLevel()==0;
    for(int d=0;d<3;++d){
        auto const& source=c.saved.fields[int(EndpointPart::ElectricHigh)][d];
        layout=EndpointLayout(high[d],*native[d])&&EndpointLayout(low[d],*native[d])&&
            native[d]->ixType().nodeCentered()&&native[d]->nComp()==1&&
            native[d]->DistributionMap()==source.DistributionMap()&&
            native[d]->boxArray()==amrex::convert(source.boxArray(),amrex::IntVect::TheNodeVector())&&layout;
    }
    if(!EndpointAll(layout))return false;
    std::vector<MF*> destination{high[0],high[1],high[2],low[0],low[1],low[2],native[0],native[1],native[2]};
    std::vector<std::vector<EndpointRange>> ranges(destination.size());bool disjoint=true;
    for(std::size_t i=0;i<destination.size();++i)disjoint=EndpointRanges(*destination[i],ranges[i])&&disjoint;
    for(std::size_t i=0;i<ranges.size();++i)for(std::size_t j=0;j<i;++j)disjoint=!EndpointOverlap(ranges[i],ranges[j])&&disjoint;
    std::vector<EndpointRange> protected_ranges;
    for(auto const& name:w.m_fields.list()){
        auto const* f=w.m_fields.internal_get(name);
        if(std::find(destination.begin(),destination.end(),f)==destination.end())
            disjoint=EndpointRanges(*f,protected_ranges)&&disjoint;
    }
    for(auto const& f:c.saved.fields)for(auto const& x:f)disjoint=EndpointRanges(x,protected_ranges)&&disjoint;
    for(auto const& r:ranges)disjoint=!EndpointOverlap(r,protected_ranges)&&disjoint;
    if(!EndpointAll(disjoint))return false;
    auto h=std::make_unique<Field>(Clone(native)),l=std::make_unique<Field>(Clone(native));
    // Use exactly the native MC centering operator, once for each represented
    // physical input. Never center a rounded high+low reconstruction.
    View high_view{&(*h)[0],&(*h)[1],&(*h)[2]};
    View low_view{&(*l)[0],&(*l)[1],&(*l)[2]};
    w.InterpolateLevelZeroFieldToAux(high_view,c.saved.PartView(EndpointPart::ElectricHigh));
    w.InterpolateLevelZeroFieldToAux(low_view,c.saved.PartView(EndpointPart::ElectricLow));
    bool finite=true;
    for(int d=0;d<3;++d){finite=(*h)[d].is_finite(0,1,(*h)[d].nGrowVect(),true)&&finite;
        finite=(*l)[d].is_finite(0,1,(*l)[d].nGrowVect(),true)&&finite;}
    if(!EndpointAll(finite))return false;
    c.auxiliary_high=std::move(h);c.auxiliary_low=std::move(l);
    Copy(high,high_view);Copy(low,low_view);Copy(native,high_view);return true;
}
bool NativePairedDarwinFields::EndpointDraft::MatchesAuxiliary(WarpX& w,CV const& high,CV const& low)const {
    if(!EndpointAll(Ready()&& &m_impl->context->w==&w&&
        m_impl->context->auxiliary_high&&m_impl->context->auxiliary_low))return false;
    auto const& c=*m_impl->context;auto native=w.m_fields.get_alldirs(FT::Efield_aux,0);
    bool layout=true;
    for(int d=0;d<3;++d)layout=EndpointLayout(high[d],(*c.auxiliary_high)[d])&&
        EndpointLayout(low[d],(*c.auxiliary_low)[d])&&EndpointLayout(native[d],(*c.auxiliary_high)[d])&&layout;
    if(!EndpointAll(layout))return false;
    amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
    for(int d=0;d<3;++d){EndpointCompare(*high[d],(*c.auxiliary_high)[d],bad);
        EndpointCompare(*low[d],(*c.auxiliary_low)[d],bad);EndpointCompare(*native[d],(*c.auxiliary_high)[d],bad);}
    return EndpointAll(invalid.dataValue()==0);
}
NativeVacuumConstraintResult NativePairedDarwinFields::EndpointDraft::Check(WarpX& w,EndpointView const& v){
    if(!EndpointAll(Ready()&& &m_impl->context->w==&w))return {};
    return m_impl->context->Check(v);
}
bool NativePairedDarwinFields::EndpointDraft::Publish(WarpX& w,
    std::array<V,endpoint_parts> const& output,MF& node_high,MF& node_low){
    if(!EndpointAll(Ready()&& &m_impl->context->w==&w))return false;
    auto& c=*m_impl->context;
    bool layout=true;
    std::vector<MF*> destinations;
    for(int k=0;k<endpoint_parts;++k)for(int d=0;d<3;++d){
        layout=EndpointLayout(output[k][d],c.saved.fields[k][d])&&layout;
        destinations.push_back(output[k][d]);
    }
    auto electric=w.m_fields.get_alldirs(FT::Efield_fp,0),inertia=w.m_fields.get_alldirs("hybrid_E_inertial_fp",0);
    auto* native_node=w.m_fields.get("hybrid_E_inertial_nodal",0);
    for(int d=0;d<3;++d){
        layout=EndpointLayout(electric[d],c.saved.fields[int(EndpointPart::ElectricHigh)][d])&&
            EndpointLayout(inertia[d],c.saved.fields[int(EndpointPart::InertiaHigh)][d],false)&&
            c.saved.fields[int(EndpointPart::InertiaHigh)][d].nGrowVect().allGE(inertia[d]->nGrowVect())&&layout;
        destinations.push_back(electric[d]);destinations.push_back(inertia[d]);
    }
    for(auto* f:{&node_high,&node_low,native_node}){layout=EndpointLayout(f,c.saved.nodal_high)&&layout;destinations.push_back(f);}
    if(!EndpointAll(layout))return false;
    std::vector<std::vector<EndpointRange>> ranges(destinations.size());
    bool disjoint=true;
    for(std::size_t i=0;i<destinations.size();++i)disjoint=EndpointRanges(*destinations[i],ranges[i])&&disjoint;
    for(std::size_t i=0;i<ranges.size();++i)for(std::size_t j=0;j<i;++j)disjoint=!EndpointOverlap(ranges[i],ranges[j])&&disjoint;
    std::vector<EndpointRange> inputs;
    for(auto const& f:c.saved.fields)for(auto const& x:f)disjoint=EndpointRanges(x,inputs)&&disjoint;
    for(auto const& x:c.origin.fields)disjoint=EndpointRanges(*x->original,inputs)&&disjoint;
    for(auto const& r:ranges)disjoint=!EndpointOverlap(r,inputs)&&disjoint;
    if(!EndpointAll(disjoint))return false;
    auto const fresh=c.Check(c.ViewOfSaved());if(!fresh.compatible||!fresh.converged)return false;
    // All failure branches precede every publication write.
    for(int k=0;k<endpoint_parts;++k)Copy(output[k],c.saved.PartView(static_cast<EndpointPart>(k)));
    Copy(electric,c.saved.PartView(EndpointPart::ElectricHigh));
    Copy(inertia,c.saved.PartView(EndpointPart::InertiaHigh));
    MF::Copy(node_high,c.saved.nodal_high,0,0,node_high.nComp(),node_high.nGrowVect());
    MF::Copy(node_low,c.saved.nodal_low,0,0,node_low.nComp(),node_low.nGrowVect());
    MF::Copy(*native_node,c.saved.nodal_high,0,0,native_node->nComp(),native_node->nGrowVect());
    amrex::Print()<<"Native retained endpoint pair pending recovery="<<std::setprecision(17)<<fresh.residual
        <<" target="<<c.saved.recovered.Receipt().solve.target<<" plasma="<<fresh.plasma_error
        <<" div="<<fresh.divergence_error<<" null="<<fresh.null_defect
        <<" materialized="<<c.saved.materialized<<"\n";
    return true;
}
struct NativeEndpointPairProducer {
    static NativeVacuumConstraintResult Solve(WarpX& w,MF const& density,MF const& mask,
        View const& baseline,View const& reference,View const& force,View const& ell,
        View const& out,View const& out_inertia,EndpointOwner::EndpointDraft& output,
        View const* baseline_low=nullptr){
        auto const shape=w.m_fields.get_alldirs(FT::Efield_fp,0);
        bool valid=density.nComp()==1&&mask.nComp()==1&&density.boxArray()==mask.boxArray()&&
            density.DistributionMap()==mask.DistributionMap();
        for(auto const* v:{&baseline,&reference,&force,&ell,&out,&out_inertia})for(int c=0;c<3;++c)
            valid=(*v)[c]&&(*v)[c]->nComp()==1&&(*v)[c]->boxArray()==shape[c]->boxArray()&&
                (*v)[c]->DistributionMap()==shape[c]->DistributionMap()&&valid;
        if(baseline_low)for(int c=0;c<3;++c)
            valid=(*baseline_low)[c]&&(*baseline_low)[c]->nComp()==1&&
                (*baseline_low)[c]->boxArray()==shape[c]->boxArray()&&
                (*baseline_low)[c]->DistributionMap()==shape[c]->DistributionMap()&&valid;
        if(!EndpointAll(valid))return {};
        std::vector<EndpointRange> inputs;
        if(baseline_low)for(auto const* f:*baseline_low)valid=EndpointRanges(*f,inputs)&&valid;
        for(auto const* v:{&baseline,&reference,&force,&ell})for(auto const* f:*v)valid=EndpointRanges(*f,inputs)&&valid;
        valid=EndpointRanges(density,inputs)&&EndpointRanges(mask,inputs)&&valid;
        std::vector<std::vector<EndpointRange>> outputs(6);
        for(int c=0;c<3;++c){valid=EndpointRanges(*out[c],outputs[c])&&valid;
            valid=EndpointRanges(*out_inertia[c],outputs[c+3])&&valid;}
        for(std::size_t i=0;i<outputs.size();++i){valid=!EndpointOverlap(outputs[i],inputs)&&valid;
            for(std::size_t j=0;j<i;++j)valid=!EndpointOverlap(outputs[i],outputs[j])&&valid;}
        if(!EndpointAll(valid))return {};
        auto data=std::make_unique<EndpointOwner::EndpointDraft::Impl>();
        data->context=std::make_unique<EndpointSolveContext>(w,density,mask,baseline,reference,force,ell,baseline_low);
        auto& c=*data->context;if(!c.Prepare())return {};
        auto result=SolveRetainedJoined(w,*c.support,*c.core,*c.all,c.trace,V(c.reference),V(c.force),V(c.ell),out,out_inertia,&c.saved);
        if(result.compatible&&result.converged&&c.saved.ready)output.m_impl=std::move(data);
        return result;
    }
};

NativeInertiaSupportOptions NativeVacuumSupportOptions(WarpX &w) {
  auto const &model = *w.get_pointer_HybridPICModel();
  NativeInertiaSupportOptions o;
  o.coefficient = InertiaEdgePolicy::NativeEdgeCandidate;
  o.recovery = InertiaRecoveryMask::Vacuum;
  o.components = InertiaRecoveryComponents::All;
  o.charge_floor = PhysConst::q_e * model.m_n_floor;
  o.reference_charge_density = PhysConst::q_e * model.m_n0_ref;
  o.recovery_density_fraction = model.m_darwin_vacuum_recovery_density_fraction;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    o.lower[d] = w.Geom(0).isPeriodic(d) ? InitialRateBoundary::Periodic
                 : WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC
                     ? InitialRateBoundary::PMC
                     : InitialRateBoundary::PEC;
    o.upper[d] = w.Geom(0).isPeriodic(d) ? InitialRateBoundary::Periodic
                 : WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC
                     ? InitialRateBoundary::PMC
                     : InitialRateBoundary::PEC;
  }
#if defined(WARPX_DIM_RZ)
  o.lower[0] = InitialRateBoundary::Axis;
#endif
  return o;
}
NativeVacuumConstraintResult
CheckNativeVacuumAcceptedConstraint(WarpX &w, MF const &density,
                                    MF const &mask_density, View const *reference,
                                    View const *accepted_transverse) {
  NativeVacuumConstraintResult result;
  auto* endpoint=NativeEndpointPairedFields(w);
  int pending=endpoint&&(endpoint->EndpointPending()||endpoint->EndpointAccepted())?1:0,minimum=pending,maximum=pending;
  // The new selector is agreed at construction. Only the opted-in owner adds
  // this retained-representation consensus; legacy no-owner callers are exact.
  if(NativeEndpointPairSelected(w)){
    int present=endpoint?1:0;amrex::ParallelDescriptor::ReduceIntMin(present);
    if(!present)return result;
    amrex::ParallelDescriptor::ReduceIntMin(minimum);
    amrex::ParallelDescriptor::ReduceIntMax(maximum);
    if(minimum!=maximum)return result;
    if(pending){
      bool no_reference=reference==nullptr;amrex::ParallelDescriptor::ReduceBoolAnd(no_reference);
      if(!no_reference)return result;
      return endpoint->CheckEndpoint(density,mask_density,accepted_transverse);
    }
  }
  NativeInertiaSupport support(Geometry(w), w.boxArray(0), w.DistributionMap(0),
                               NativeVacuumSupportOptions(w));
  if (!support.FreezeEdges(density, mask_density))
    return result;
  Core::Options co;
  double length = 0.;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    co.pmc_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
    co.pmc_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
    length = std::max(length, w.Geom(0).ProbHi(d) - w.Geom(0).ProbLo(d));
  }
  amrex::Array<amrex::iMultiFab, 3> trace, global, empty;
  DefineTrace(w, support, co, trace, global, empty);
  JoinedGraph graph(w, trace, support.NodeWeights());
  auto electric = w.m_fields.get_alldirs(FT::Efield_fp, 0);
  auto ell = w.m_fields.get_alldirs("hybrid_E_long_fp", 0);
  auto transverse = Clone(accepted_transverse ? *accepted_transverse : electric);
  if (!accepted_transverse)
    for (int c = 0; c < 3; ++c)
      MF::Subtract(transverse[c], *ell[c], 0, 0, 1, 0);
  // Match the accepted solve's transverse PMC images. A native circuit
  // supplies its standard closed-rate reference at the restored clock.
  VectorImages(w, V(transverse), false, reference);
  MF divergence(density.boxArray(), density.DistributionMap(), 1, 0);
  w.get_pointer_fdtd_solver_fp(0)->ComputeDivE(V(transverse), divergence);
  amrex::ReduceOps<amrex::ReduceOpMax> reduce;
  amrex::ReduceData<double> data(reduce);
  using Tuple = decltype(data)::Type;
  for (amrex::MFIter mfi(divergence); mfi.isValid(); ++mfi) {
    auto f = divergence.const_array(mfi),
         v = support.NodeWeights().const_array(mfi);
    reduce.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                  return {v(i, j, k) > 0. ? std::abs(f(i, j, k)) : 0.};
                });
  }
  double error = amrex::get<0>(data.value());
  amrex::ParallelDescriptor::ReduceRealMax(error);
  result.divergence_error = length * error;
  result.null_defect =
      length * graph.Compatibility(divergence, support.NodeWeights());
  result.compatible =
      std::isfinite(result.null_defect) && result.null_defect <= 1.e-9;
  result.converged = result.compatible &&
                     std::isfinite(result.divergence_error) &&
                     result.divergence_error < 1.e-8;
  result.graph_iterations = graph.iterations;
  return result;
}
bool ApplyNativeVacuumInertia(WarpX &w, MF const &density, int component,
                              View const &rate, View const &out) {
  MF physical(density, amrex::make_alias, component, 1);
  auto const &model = *w.get_pointer_HybridPICModel();
  auto const &mask = *w.m_fields.get("hybrid_rho_vacmask_fp", 0);
  NativeInertiaSupport support(Geometry(w), w.boxArray(0), w.DistributionMap(0),
                               NativeVacuumSupportOptions(w));
  if (!support.FreezeEdges(physical, mask))
    return false;
  support.ApplyMass({rate[0], rate[1], rate[2]}, out,
                    model.m_electron_inertia_mass /
                        (PhysConst::q_e * PhysConst::q_e * model.m_n0_ref));
  return true;
}
static NativeVacuumConstraintResult
SolveNativeVacuumConstraintImpl(WarpX &w, MF const &density, MF const &mask_density,
                            View const &baseline, View const &reference,
                            View const &force, View const &ell, View const &out,
                            View const &out_inertia, View const *nores_force,
                            CircuitCoupler *coupler, amrex::Real time,
                            View const *out_reference, NativePairedDarwinFields::EndpointDraft* endpoint=nullptr,
                            View const *baseline_low=nullptr) {
  NativeVacuumConstraintResult result;
  int retained=0;
  amrex::ParmParse("endpoint_diagnostic").query("retained_joined_recovery",retained);
  // Reuse the original two selector collectives, including draft presence.
  // A null output has exactly the original mode and branch arithmetic.
  int retained_min=retained+(endpoint?2:0)+(baseline_low?4:0),retained_max=retained_min;
  amrex::ParallelDescriptor::ReduceIntMin(retained_min);
  amrex::ParallelDescriptor::ReduceIntMax(retained_max);
  if (retained_min!=retained_max || retained<0 || retained>1 || (endpoint&&!retained) ||
      (baseline_low&&!endpoint)) return result;
  if (retained) {
    bool supported=false;
#if defined(WARPX_DIM_RZ) && !defined(__FAST_MATH__) && \
    (!defined(__FINITE_MATH_ONLY__) || __FINITE_MATH_ONLY__ == 0)
    supported=sizeof(amrex::Real)==sizeof(double) && !coupler && !nores_force &&
      !out_reference && compensated_curl::Supported(w) && w.Geom(0).ProbLo(0)==0. &&
      !w.Geom(0).isPeriodic(0) && w.Geom(0).isPeriodic(1) &&
      WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC &&
      WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving && WarpX::nox==3;
    for (int d=0;d<AMREX_SPACEDIM;++d)
      supported=supported && WarpX::field_boundary_lo[d]!=FieldBoundaryType::PMC &&
        WarpX::field_boundary_hi[d]!=FieldBoundaryType::PMC;
#if !defined(AMREX_USE_GPU)
    bool const pmc=NativeEndpointPMCQualificationSelected(w);
    supported=supported || (pmc && endpoint && !coupler && !nores_force &&
        !out_reference && compensated_curl::Supported(w));
#endif
#if defined(AMREX_USE_GPU)
    // Initialization has no EndpointDraft yet. Require the captured
    // accepted-owner scope; a standalone retained solve stays closed.
    supported=supported && NativeEndpointCudaQualificationSelected(w);
#endif
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if (!supported) return result;
  }
  if(endpoint)return NativeEndpointPairProducer::Solve(w,density,mask_density,
      baseline,reference,force,ell,out,out_inertia,*endpoint,baseline_low);
  auto E = w.m_fields.get_alldirs(FT::Efield_fp, 0),
       B = w.m_fields.get_alldirs(FT::Bfield_fp, 0);
  NativeInertiaSupport support(Geometry(w), w.boxArray(0), w.DistributionMap(0),
                               NativeVacuumSupportOptions(w));
  if (!support.FreezeEdges(density, mask_density))
    return result;
  Core::Options co;
  co.relative_tolerance = 1.e-14;
  co.absolute_tolerance = 1.e-14;
  if (coupler) {
    // The native core has |K E| = (mu0/diagonal) |Cdot|. A fixed
    // field-unit floor otherwise permits a much larger current-rate defect.
    // Its RZ/3D metric weights are at least 1/16 (axis and PMC caps).
    // Reserve one tenth of the absolute physical current-rate budget for
    // the inner absolute floor; the relative tolerance remains unchanged.
    double diagonal = 0.;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
      diagonal += 4. / (w.Geom(0).CellSize(d) * w.Geom(0).CellSize(d));
    co.absolute_tolerance = .25 * PhysConst::mu0 * 1.e-11 / diagonal;
  }
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    co.pmc_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
    co.pmc_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
  }
  amrex::Array<amrex::iMultiFab, 3> trace, global, empty;
  DefineTrace(w, support, co, trace, global, empty);

  auto zero = Clone(E);
  Zero(V(zero));
  Core core(w, E, B, co), all(w, E, B, co);
  if (!core.Freeze({&support.RecoveryMask(0), &support.RecoveryMask(1),
                    &support.RecoveryMask(2)},
                   {&trace[0], &trace[1], &trace[2]}, baseline, reference) ||
      !all.Freeze({&global[0], &global[1], &global[2]},
                  {&empty[0], &empty[1], &empty[2]}, V(zero), reference))
    return result;
  if (retained) {
    if (!RetainedFreeUnionMatches(support,trace,global)) return result;
    return SolveRetainedJoined(w,support,core,all,trace,reference,force,ell,out,out_inertia);
  }
  if (nores_force) {
    for (int c = 0; c < 3; ++c) {
      if (!force[c]->is_finite(0, 1, force[c]->nGrowVect()) ||
          !(*nores_force)[c]->is_finite(0, 1, (*nores_force)[c]->nGrowVect()))
        return result;
    }
  }
  std::unique_ptr<NativeVacuumCircuitFieldRate> circuit;
  if (coupler) {
    circuit = std::make_unique<NativeVacuumCircuitFieldRate>(w, *coupler);
    std::string error;
    if (!circuit->Prepare(time,
          {&support.RecoveryMask(0), &support.RecoveryMask(1), &support.RecoveryMask(2)},
          {&trace[0], &trace[1], &trace[2]}, co, error)) {
      amrex::Print() << "Native joined circuit preparation rejected: " << error << "\n";
      return result;
    }
  }
  JoinedMatrixFree op(w, support, core, all, trace, reference,
                      nores_force ? &force : nullptr, nores_force, circuit.get());
  auto rhs = op.makeVecRHS(), solution = op.makeVecLHS(),
       action = op.makeVecRHS(), residual = op.makeVecRHS();
  result.compatible = op.RightHandSide(rhs, force, ell, result.null_defect);
  result.graph_iterations = op.graph.iterations;
  std::string observe;
  amrex::ParmParse("endpoint_diagnostic").query("vacuum_observe", observe);
  auto dump = [&](std::string const &name, MF const &field) {
    if (!observe.empty())
      amrex::VisMF::Write(field, observe + "_" + name);
  };
  if (!observe.empty()) {
    dump("density", density);
    dump("mask", mask_density);
    dump("weights", support.NodeWeights());
    dump("rhs_node", rhs.node);
    MF labels(rhs.node.boxArray(), rhs.node.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(labels); mfi.isValid(); ++mfi) {
      auto a = labels.array(mfi);
      auto b = op.graph.labels.const_array(mfi);
      amrex::ParallelFor(mfi.validbox(),
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                           a(i, j, k) = b(i, j, k);
                         });
    }
    dump("labels", labels);
    for (int c = 0; c < 3; ++c) {
      auto suffix = std::to_string(c);
      dump("baseline" + suffix, *baseline[c]);
      dump("ell" + suffix, *ell[c]);
      dump("origin" + suffix, op.origin[c]);
      dump("f0_" + suffix, *force[c]);
      dump("rhs_edge" + suffix, rhs.edge[c]);
    }
  }
  if (!result.compatible)
    return result;
  NativeVacuumPhysicalStopping stopping;
  if (!stopping.Prepare(op)) return result;
  double const initial_norm = op.norm2(rhs);
  if (!std::isfinite(initial_norm)) return result;
  double const target = std::min(std::max(1.e-11, 1.e-12*initial_norm), stopping.target);
  FlexibleGMRES<JoinedVector, JoinedMatrixFree> solver;
  solver.define(op);
  solver.setRestartLength(150);
  solver.setMaxIters(1000);
  solver.setVerbose(0);
  solver.solve(solution, rhs, 0., target);
  result.iterations = solver.getNumIters();
  result.residual = solver.getResidualNorm();
  if (solver.getStatus() != 0)
    return result;
  op.apply(action, solution);
  op.linComb(residual, 1., rhs, -1., action);
  bool residual_finite = residual.node.is_finite(0, 1, 0);
  for (auto &e : residual.edge)
    residual_finite = e.is_finite(0, 1, 0) && residual_finite;
  if (!residual_finite) return result;
  // Certify the actual fresh action, not a residual from an earlier inner call.
  result.residual = op.norm2(residual);
  if (!std::isfinite(result.residual) || result.residual > target) return result;
  for (auto &e : residual.edge)
    result.plasma_error = std::max(result.plasma_error, e.norminf(0));
  auto finished = Clone(E);
  auto x = op.ViewOf(solution);
  if (!core.Recover(V(finished), &x) || !op.Complete(V(finished), true))
    return result;
  result.divergence_error = op.FullDivergenceError(V(finished));
  bool const remainder_certified = stopping.Certify(op, residual);
  amrex::Print() << std::setprecision(17)
                 << "NATIVE_VACUUM_PHYSICAL_STOPPING kappa=" << stopping.kappa
                 << " target=" << target
                 << " underflow=" << stopping.underflow_allowance
                 << " remainder_bound=" << stopping.remainder_bound
                 << " remainder_limit=" << stopping.remainder_limit
                 << " certified=" << int(remainder_certified) << "\n";
  if (!remainder_certified) return result;
  if (circuit) {
    double current_scale = 1., vacuum_error = 0.;
    for (int c = 0; c < 3; ++c) {
      auto const& current = circuit->CurrentRate(c);
      current_scale = std::max(current_scale, current.norminf(0));
      MF::Copy(op.curl[c], current, 0, 0, 1, 0);
      for (amrex::MFIter it(op.curl[c]); it.isValid(); ++it) {
        auto value = op.curl[c].array(it);
        auto mask = support.RecoveryMask(c).const_array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
          if (!mask(i,j,k)) value(i,j,k) = 0.;
        });
      }
      vacuum_error = std::max(vacuum_error, op.curl[c].norminf(0));
    }
    result.vacuum_current_relative_error = vacuum_error / current_scale;
    if (!std::isfinite(result.vacuum_current_relative_error) ||
        result.vacuum_current_relative_error > 1.e-10) return result;
  }
  result.actions = op.actions;
  if (!observe.empty()) {
    dump("solution_node", solution.node);
    dump("residual_node", residual.node);
    dump("full_divergence", op.divergence);
    for (int c = 0; c < 3; ++c) {
      auto suffix = std::to_string(c);
      dump("solution_edge" + suffix, solution.edge[c]);
      dump("residual_edge" + suffix, residual.edge[c]);
      dump("finished" + suffix, finished[c]);
    }
    double const null_error =
        op.graph.Compatibility(op.divergence, support.NodeWeights()) *
        op.length;
    amrex::Print() << "Native joined observed null of finished div="
                   << null_error
                   << " retained residual node=" << residual.node.norminf()
                   << "\n";
  }
  if (op.failed || !std::isfinite(result.residual) ||
      !std::isfinite(result.plasma_error) || !std::isfinite(result.divergence_error) ||
      result.plasma_error >= 1.e-8 || result.divergence_error >= 1.e-8)
    return result;
  auto total = Clone(E);
  for (int c = 0; c < 3; ++c)
    MF::LinComb(total[c], 1., finished[c], 0, 1., *ell[c], 0, 0, 1, 0);
  op.ElectricRate(V(total), false);
  op.ApplyRateMass(V(finished), true);
  MF::Copy(op.potential, solution.node, 0, 0, 1, 0);
  op.potential.mult(op.length, 0, 1, 0);
  op.potential.OverrideSync(w.Geom(0).periodicity());
  op.potential.FillBoundary(w.Geom(0).periodicity());
  Gradient(w, op.potential, V(op.gradient));
  support.ApplyMass({&op.gradient[0], &op.gradient[1], &op.gradient[2]},
                    V(op.curl));
  for (int c = 0; c < 3; ++c)
    MF::Subtract(op.mass[c], op.curl[c], 0, 0, 1, 0);
  auto completed_reference = V(op.reference);
  VectorImages(w, V(finished), false, &completed_reference);
  VectorImages(w, V(op.mass), false);
  // Publish only after every output has been checked collectively. In
  // particular max reductions must not silently hide a NaN component.
  bool finite = true;
  for (int c = 0; c < 3; ++c) {
    finite = finished[c].is_finite(0, 1, finished[c].nGrowVect()) && finite;
    finite = op.mass[c].is_finite(0, 1, op.mass[c].nGrowVect()) && finite;
    finite = op.reference[c].is_finite(0, 1, op.reference[c].nGrowVect()) && finite;
  }
  if (!finite) return result;
  Copy(out, V(finished));
  Copy(out_inertia, V(op.mass));
  if (out_reference) Copy(*out_reference, completed_reference);
  result.converged = true;
  return result;
}

NativeVacuumConstraintResult
SolveNativeVacuumConstraint(WarpX &w, MF const &density, MF const &mask_density,
                            View const &baseline, View const &reference,
                            View const &force, View const &ell, View const &out,
                            View const &out_inertia, View const *nores_force,
                            NativePairedDarwinFields::EndpointDraft* endpoint) {
  return SolveNativeVacuumConstraintImpl(w, density, mask_density, baseline,
      reference, force, ell, out, out_inertia, nores_force, nullptr, 0., nullptr, endpoint);
}

NativeVacuumConstraintResult
SolveNativeVacuumPairedEndpoint(WarpX& w,MF const& density,MF const& mask_density,
    View const& high,View const& low,View const& reference,View const& force,
    View const& ell,View const& out,View const& out_inertia,
    NativePairedDarwinFields::EndpointDraft& endpoint)
{
  return SolveNativeVacuumConstraintImpl(w,density,mask_density,high,reference,
      force,ell,out,out_inertia,nullptr,nullptr,0.,nullptr,&endpoint,&low);
}
NativeVacuumConstraintResult
SolveNativeVacuumCircuitConstraint(WarpX &w, CircuitCoupler &coupler,
    amrex::Real time, MF const &density, MF const &mask_density,
    View const &baseline, View const &force, View const &ell,
    View const &out, View const &out_inertia, View const &out_reference,
    View const *nores_force) {
  NativeVacuumConstraintResult rejected;
  auto const shape = w.m_fields.get_alldirs(FT::Efield_fp, 0);
  int presence = nores_force ? 1 : 0;
  amrex::ParallelDescriptor::ReduceIntSum(presence);
  if (presence != 0 && presence != amrex::ParallelDescriptor::NProcs()) return rejected;
  auto const &scalar = *w.m_fields.get(FT::rho_fp, 0);
  bool valid = std::isfinite(time) && w.finestLevel() == 0;
  for (auto const *f : {&density, &mask_density})
    valid = f->nComp() == 1 && f->boxArray() == scalar.boxArray() &&
        f->DistributionMap() == scalar.DistributionMap() &&
        f->nGrowVect().allGE(amrex::IntVect(1)) && valid;
  std::vector<View const *> inputs{&baseline, &force, &ell};
  if (nores_force) inputs.push_back(nores_force);
  std::vector<View const *> outputs{&out, &out_inertia, &out_reference};
  for (auto const *views : {&inputs, &outputs})
    for (auto const *view : *views)
      for (int c = 0; c < 3; ++c) {
        auto const *f = (*view)[c];
        valid = f && f->nComp() == 1 && f->boxArray() == shape[c]->boxArray() &&
            f->DistributionMap() == shape[c]->DistributionMap() &&
            f->nGrowVect() == shape[c]->nGrowVect() && valid;
      }
  std::vector<MF const *> destinations;
  for (auto const *view : outputs)
    for (auto const *f : *view) {
      for (auto const *prior : destinations) valid = (f != prior) && valid;
      destinations.push_back(f);
      for (auto const *input : inputs)
        for (auto const *value : *input) valid = (f != value) && valid;
      valid = (f != &density && f != &mask_density) && valid;
    }
  amrex::ParallelDescriptor::ReduceBoolAnd(valid);
  if (!valid) return rejected;
  // Distinct MultiFab objects can still be aliases of the same allocation.
  // Inspect allocation metadata only; never copy device field values to host.
  struct Range { std::uintptr_t begin, end; bool output; };
  std::vector<Range> ranges;
  auto allocation = [&](MF const &f, bool output) {
    for (amrex::MFIter it(f); it.isValid(); ++it) {
      auto const &fab = f[it];
      auto const begin = reinterpret_cast<std::uintptr_t>(fab.dataPtr());
      ranges.push_back({begin, begin + fab.size() * sizeof(amrex::Real), output});
    }
  };
  for (auto const *view : inputs)
    for (auto const *f : *view) allocation(*f, false);
  allocation(density, false);
  allocation(mask_density, false);
  for (auto const *view : outputs)
    for (auto const *f : *view) allocation(*f, true);
  std::sort(ranges.begin(), ranges.end(),
      [](Range const &left, Range const &right) { return left.begin < right.begin; });
  for (std::size_t i = 0; i < ranges.size(); ++i)
    for (std::size_t j = i + 1; j < ranges.size() && ranges[j].begin < ranges[i].end; ++j)
      valid = !(ranges[i].output || ranges[j].output) && valid;
  amrex::ParallelDescriptor::ReduceBoolAnd(valid);
  if (!valid) return rejected;
  for (auto const *view : inputs)
    for (auto const *f : *view)
      valid = f->is_finite(0, 1, f->nGrowVect()) && valid;
  valid = density.is_finite(0, 1, density.nGrowVect()) && valid;
  valid = mask_density.is_finite(0, 1, mask_density.nGrowVect()) && valid;
  if (!valid) return rejected;
  auto zero_reference = Clone(shape);
  Zero(V(zero_reference));
  return SolveNativeVacuumConstraintImpl(w, density, mask_density, baseline,
      V(zero_reference), force, ell, out, out_inertia, nores_force, &coupler,
      time, &out_reference);
}

NativeVacuumConstraintResult
CorrectNativeVacuumLongitudinal(WarpX &w, MF const &density, int component,
                                amrex::Real interval, View const &raw,
                                View const &held, MF &potential_delta,
                                View const &field_delta) {
  AMREX_ALWAYS_ASSERT(interval > 0. && std::isfinite(interval));
  NativeVacuumConstraintResult result;
  MF physical(density, amrex::make_alias, component, 1);
  NativeInertiaSupport support(Geometry(w), w.boxArray(0), w.DistributionMap(0),
                               NativeVacuumSupportOptions(w));
  if (!support.FreezeEdges(physical,
                           *w.m_fields.get("hybrid_rho_vacmask_fp", 0)))
    return result;
  Core::Options co;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    co.pmc_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
    co.pmc_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
  }
  amrex::Array<amrex::iMultiFab, 3> trace, global, empty;
  DefineTrace(w, support, co, trace, global, empty);
  auto const &model = *w.get_pointer_HybridPICModel();
  double const beta =
      PhysConst::epsilon_0 * model.m_electron_inertia_mass /
      (PhysConst::q_e * PhysConst::q_e * model.m_n0_ref * interval * interval);
  bool auxiliary = false;
  amrex::ParmParse("endpoint_diagnostic")
      .query("vacuum_schur_full_grid", auxiliary);
  if (auxiliary) {
    LongitudinalSchurOptions o;
    o.compatible_yee = true;
    o.output_ghosts = potential_delta.nGrowVect();
    o.max_semicoarsening_levels = model.m_darwin_poisson_semicoarsening;
    o.semicoarsening_direction =
        model.m_darwin_poisson_semicoarsening_direction;
    amrex::ParmParse pp("implicit_evolve");
    pp.query("darwin_schur_relative_tolerance", o.relative_tolerance);
    pp.query("darwin_schur_absolute_tolerance", o.absolute_tolerance);
    pp.query("darwin_schur_max_iterations", o.max_iterations);
    pp.query("darwin_schur_restart_length", o.restart_length);
    pp.query("darwin_schur_pc_cycles", o.preconditioner_cycles);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      auto map = [](InitialRateBoundary type) {
        switch (type) {
        case InitialRateBoundary::Axis:
          return LongitudinalBoundary::Axis;
        case InitialRateBoundary::PEC:
          return LongitudinalBoundary::PEC;
        case InitialRateBoundary::PMC:
          return LongitudinalBoundary::PMC;
        default:
          return LongitudinalBoundary::Periodic;
        }
      };
      auto so = NativeVacuumSupportOptions(w);
      o.lower[d] = map(so.lower[d]);
      o.upper[d] = map(so.upper[d]);
    }
    auto coefficient = Clone(raw);
    for (int c = 0; c < 3; ++c)
      for (amrex::MFIter mfi(coefficient[c]); mfi.isValid(); ++mfi) {
        auto a = coefficient[c].array(mfi);
        auto kap = support.KappaEdge(c).const_array(mfi);
        auto p = trace[c].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                             a(i, j, k) = p(i, j, k) ? beta * kap(i, j, k) : 0.;
                           });
      }
    DarwinLongitudinalSchur auxiliary_op(w.Geom(0), w.boxArray(0),
                                         w.DistributionMap(0), o);
    if (!auxiliary_op.FreezeEdges(
            {&coefficient[0], &coefficient[1], &coefficient[2]}))
      return result;
    auto correction = auxiliary_op.Correct({raw[0], raw[1], raw[2]},
                                           {held[0], held[1], held[2]});
    result.compatible = true;
    result.converged = correction.converged;
    result.iterations = correction.iterations;
    result.residual = correction.residual;
    if (!result.converged)
      return result;
    MF::Copy(potential_delta, auxiliary_op.CorrectionPotential(), 0, 0, 1,
             potential_delta.nGrowVect());
    auto field = auxiliary_op.CorrectionField();
    for (int c = 0; c < 3; ++c)
      MF::Copy(*field_delta[c], *field[c], 0, 0, 1,
               field_delta[c]->nGrowVect());
    return result;
  }
  VacuumFiniteSchur op(w, support, trace, beta);
  auto rhs = op.makeVecRHS(), solution = op.makeVecLHS(),
       action = op.makeVecRHS();
  auto difference = Clone(raw);
  for (int c = 0; c < 3; ++c)
    MF::Subtract(difference[c], *held[c], 0, 0, 1, 0);
  VectorImages(w, V(difference), false);
  w.get_pointer_fdtd_solver_fp(0)->ComputeDivE(V(difference), rhs);
  rhs.mult(-1., 0, 1, 0);
  result.null_defect = op.graph.Compatibility(rhs, support.NodeWeights());
  result.compatible =
      std::isfinite(result.null_defect) && result.null_defect <= 1.e-9;
  std::string observe;
  amrex::ParmParse("endpoint_diagnostic")
      .query("vacuum_schur_observe", observe);
  if (!observe.empty()) {
    static int counter = 0;
    auto prefix = observe + "_" + std::to_string(counter++) + "_";
    auto dump = [&](std::string const &key, MF const &f) {
      amrex::VisMF::Write(f, prefix + key);
    };
    dump("rhs", rhs);
    dump("weights", support.NodeWeights());
    dump("rho", physical);
    dump("mask", *w.m_fields.get("hybrid_rho_vacmask_fp", 0));
    auto labels = op.makeVecLHS();
    for (amrex::MFIter mfi(labels); mfi.isValid(); ++mfi) {
      auto a = labels.array(mfi);
      auto b = op.graph.labels.const_array(mfi);
      amrex::ParallelFor(mfi.validbox(),
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                           a(i, j, k) = b(i, j, k);
                         });
    }
    dump("labels", labels);
    auto et = Clone(raw);
    for (int c = 0; c < 3; ++c) {
      auto tail = std::to_string(c);
      auto const &a =
          *w.m_fields.get("hybrid_A_fp", ablastr::fields::Direction{c}, 0);
      auto const &a0 =
          *w.m_fields.get("hybrid_A_old_fp", ablastr::fields::Direction{c}, 0);
      dump("A" + tail, a);
      dump("Aold" + tail, a0);
      dump("rawEL" + tail, *raw[c]);
      dump("heldEL" + tail, *held[c]);
      dump("Ei" + tail, *w.m_fields.get("hybrid_E_inertial_fp",
                                        ablastr::fields::Direction{c}, 0));
      MF::LinComb(et[c], -1. / interval, a, 0, 1. / interval, a0, 0, 0, 1, 0);
      dump("ET" + tail, et[c]);
    }
    VectorImages(w, V(et), false);
    w.get_pointer_fdtd_solver_fp(0)->ComputeDivE(V(et), labels);
    dump("divET", labels);
  }
  if (!result.compatible)
    return result;
  op.graph.Restrict(rhs);
  FlexibleGMRES<MF, VacuumFiniteSchur> solver;
  solver.define(op);
  solver.setRestartLength(100);
  solver.setMaxIters(600);
  solver.setVerbose(0);
  solver.solve(solution, rhs, 1.e-12, 1.e-12);
  result.iterations = solver.getNumIters();
  result.residual = solver.getResidualNorm();
  if (solver.getStatus() != 0)
    return result;
  op.FullAction(action, solution);
  MF::Subtract(action, rhs, 0, 0, 1, 0);
  result.divergence_error =
      op.graph.Compatibility(action, support.NodeWeights());
  result.actions = op.actions;
  op.Canonical(solution);
  MF::Copy(potential_delta, solution, 0, 0, 1, 0);
  potential_delta.setBndry(0.);
  potential_delta.OverrideSync(w.Geom(0).periodicity());
  potential_delta.FillBoundary(w.Geom(0).periodicity());
  Gradient(w, potential_delta, field_delta);
  VectorImages(w, field_delta, false);
  result.converged = true;
  return result;
}
#include "NativeVacuumFiniteOracle.H"
} // namespace warpx::darwin
