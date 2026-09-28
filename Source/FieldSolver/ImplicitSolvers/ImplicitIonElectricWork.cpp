/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ImplicitIonElectricWork.H"
#include "IonElectricWorkReference.H"
#include "Particles/Gather/FieldGather.H"
#include <AMReX_Reduce.H>
#include <algorithm>
#include <set>
#include <tuple>
namespace warpx::thermal {
IonElectricWorkResult
MeasureImplicitIonElectricWork (
    amrex::Geometry const& geometry, IonElectricWorkFields const& fields,
    std::vector<IonElectricWorkSpecies> const& species,
    std::vector<IonElectricWorkParticles> const& particles, amrex::Real dt,
    IonElectricGatherContract contract)
{
    IonElectricWorkResult result;
    result.gather_contract=contract;
    bool ok=std::isfinite(dt) && dt>0 &&
        contract!=IonElectricGatherContract::Unspecified &&
        fields.filled_ghosts.allGE(amrex::IntVect(0));
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)
    ok=false;
#endif
    auto const* reference=fields.component[0];
    ok=ok && reference;
    bool const has_total=fields.total[0]!=nullptr;
    result.has_total_field=has_total;
    bool const has_remainder=fields.remainder[0]!=nullptr;
    result.has_remainder_field=has_remainder;
    auto validate=[&](amrex::MultiFab const* f) {
        return f && reference && f->nComp()==1 &&
            f->ixType().toIntVect()==amrex::IntVect::TheNodeVector() &&
            f->boxArray()==reference->boxArray() &&
            f->DistributionMap()==reference->DistributionMap() &&
            f->nGrowVect()==reference->nGrowVect() &&
            f->nGrowVect().allGE(fields.filled_ghosts);
    };
    for (int d=0;d<3;++d) {
        ok=ok && validate(fields.component[d]) &&
            ((fields.total[d]!=nullptr)==has_total);
        if (has_total) { ok=ok && validate(fields.total[d]); }
        ok=ok && ((fields.remainder[d]!=nullptr)==has_remainder);
        if (has_remainder) { ok=ok && validate(fields.remainder[d]); }
    }
    std::set<std::string> names;
    for (auto const& s:species) {
        ok=ok && !s.name.empty() && names.insert(s.name).second &&
            std::isfinite(s.mass) && s.mass>0 &&
            std::isfinite(s.charge) && s.charge!=0;
    }
    std::set<std::tuple<std::size_t,int,int>> tiles;
    for (auto const& t:particles) {
        ok=ok && t.species<species.size() && t.count>=0 && reference &&
            t.grid>=0 && t.grid<(reference ? reference->size() : 0) &&
            tiles.emplace(t.species,t.grid,t.tile).second;
        if (reference && t.grid>=0 && t.grid<reference->size()) {
            ok=ok && reference->DistributionMap()[t.grid]==amrex::ParallelDescriptor::MyProc();
        }
        if (t.count==0) { continue; }
        ok=ok && t.idcpu && t.weight;
        for (int d=0;d<3;++d) {
            ok=ok && t.gather_position[d] && t.old_momentum[d] && t.midpoint_momentum[d];
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(ok);
    if (!ok) { return result; }
    amrex::XDim3 inverse{1,1,1};
    inverse.x=geometry.InvCellSize(0);
#ifdef WARPX_DIM_RZ
    inverse.z=geometry.InvCellSize(1);
#elif defined(WARPX_DIM_3D)
    inverse.y=geometry.InvCellSize(1);
    inverse.z=geometry.InvCellSize(2);
#endif
    result.species.resize(species.size());
    result.diagnostics.resize(species.size());
    static_assert(IonElectricWorkComponent::Count==14 && IonElectricWorkDiagnostic::Count==8);
    for (std::size_t s=0;s<species.size();++s) {
        using Sum=amrex::ReduceOpSum;
        amrex::ReduceOps<Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum,Sum> op;
        using R=amrex::Real;
        amrex::ReduceData<R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,amrex::Long,amrex::Long> data(op);
        using Tuple=decltype(data)::Type;
        op.eval(1,data,[] AMREX_GPU_DEVICE(int)->Tuple {
            return {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        });
        auto const mass=species[s].mass, charge=species[s].charge;
        for (auto const& t:particles) {
            if (t.species!=s || t.count==0) { continue; }
            amrex::GpuArray<amrex::Array4<R const>,3> component{}, total{}, remainder{};
            auto const fab_box=(*reference)[t.grid].box();
            auto const lo=amrex::lbound(fab_box);
            auto const lo_index=fab_box.smallEnd();
            auto const filled=amrex::grow(reference->boxArray()[t.grid],fields.filled_ghosts);
            auto const filled_lo=filled.smallEnd(), filled_hi=filled.bigEnd();
            for (int d=0;d<3;++d) {
                component[d]=(*fields.component[d])[t.grid].const_array();
                total[d]=has_total ? (*fields.total[d])[t.grid].const_array() : component[d];
                remainder[d]=has_remainder ? (*fields.remainder[d])[t.grid].const_array() : component[d];
            }
            // Same native lower-corner convention: local shape coordinates are
            // positive, including at negative-index physical ghost cells.
            amrex::XDim3 origin{0,0,0};
            origin.x=geometry.ProbLo(0)+(fab_box.smallEnd(0)-geometry.Domain().smallEnd(0))/inverse.x;
#ifdef WARPX_DIM_RZ
            origin.z=geometry.ProbLo(1)+(fab_box.smallEnd(1)-geometry.Domain().smallEnd(1))/inverse.z;
#elif defined(WARPX_DIM_3D)
            origin.y=geometry.ProbLo(1)+(fab_box.smallEnd(1)-geometry.Domain().smallEnd(1))/inverse.y;
            origin.z=geometry.ProbLo(2)+(fab_box.smallEnd(2)-geometry.Domain().smallEnd(2))/inverse.z;
#endif
            auto const nodal=amrex::IndexType::TheNodeType();
            op.eval(t.count,data,[=] AMREX_GPU_DEVICE(long p)->Tuple {
                bool good=amrex::ConstParticleIDWrapper{t.idcpu[p]}.is_valid() &&
                    std::isfinite(t.weight[p]) && t.weight[p]>=0;
                IonWorkPoint position{},old_u{},mid_u{},e{},all_e{},remainder_e{},ignored_b{};
                for (int d=0;d<3;++d) {
                    position[d]=t.gather_position[d][p];
                    old_u[d]=t.old_momentum[d][p];
                    mid_u[d]=t.midpoint_momentum[d][p];
                    good=good && std::isfinite(position[d]) &&
                        std::isfinite(old_u[d]) && std::isfinite(mid_u[d]);
                }
                amrex::GpuArray<R,AMREX_SPACEDIM> coordinate{};
#ifdef WARPX_DIM_RZ
                coordinate={(std::sqrt(position[0]*position[0]+position[1]*position[1])-origin.x)*inverse.x,
                            (position[2]-origin.z)*inverse.z};
#elif defined(WARPX_DIM_3D)
                coordinate={(position[0]-origin.x)*inverse.x,
                            (position[1]-origin.y)*inverse.y,(position[2]-origin.z)*inverse.z};
#endif
                Compute_shape_factor<3> shape;
                for (int d=0;d<AMREX_SPACEDIM;++d) {
                    R const q=coordinate[d];
                    good=good && std::isfinite(q) && q>=1 && q<filled_hi[d]-lo_index[d]+1;
                    if (good) {
                        R weights[4];
                        int const first=lo_index[d]+shape(weights,q);
                        good=first>=filled_lo[d] && first+3<=filled_hi[d];
                    }
                }
                if (!good) { return {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1}; }
                amrex::GpuArray<amrex::GpuArray<double,2>,AMREX_SPACEDIM> domain{};
                amrex::GpuArray<amrex::GpuArray<bool,2>,AMREX_SPACEDIM> crop{};
                // Native routine; B slots reuse the input only because the
                // E-only diagnostic discards B. No alternate gather stencil.
                auto gather=[&](auto const& f,IonWorkPoint& out) {
                    ignored_b={0,0,0};
                    return doGatherShapeNImplicit(
                        position[0],position[1],position[2],position[0],position[1],position[2],
                        out[0],out[1],out[2],ignored_b[0],ignored_b[1],ignored_b[2],
                        f[0],f[1],f[2],f[0],f[1],f[2],
                        nodal,nodal,nodal,nodal,nodal,nodal,
                        inverse,origin,domain,crop,lo,1,3,CurrentDepositionAlgo::Direct);
                };
                good=bool(gather(component,e));
                if (has_total) { good=good && bool(gather(total,all_e)); }
                if (has_remainder) { good=good && bool(gather(remainder,remainder_e)); }
                auto const v=ComputeIonElectricWork(mass,charge,t.weight[p],dt,
                    old_u,mid_u,e,all_e,has_total);
                auto const ref=ComputeIonElectricWorkReference(
                    mass,charge,t.weight[p],dt,old_u,mid_u,e);
                auto const rw=ComputeIonElectricWork(mass,charge,t.weight[p],dt,
                    old_u,mid_u,remainder_e,all_e,false);
                auto const rb=ComputeIonElectricWorkReference(
                    mass,charge,t.weight[p],dt,old_u,mid_u,remainder_e);
                using C=IonElectricWorkComponent;
                IonElectricWorkDiagnostics const diagnostic{
                    ref.old_kinetic_energy,ref.endpoint_kinetic_energy,
                    ref.momentum_roundoff_reference,ref.component_cauchy_work_bound,
                    rw[C::ComponentWork],rw[C::ComponentWorkAbs],
                    rw[C::NonrelativisticWork],rb.component_cauchy_work_bound};
                for (auto x:v) { good=good && std::isfinite(x); }
                for (auto x:diagnostic) { good=good && std::isfinite(x); }
                if (!good) { return {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1}; }
                return {v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],
                        v[8],v[9],v[10],v[11],v[12],v[13],
                        diagnostic[0],diagnostic[1],diagnostic[2],diagnostic[3],
                        diagnostic[4],diagnostic[5],diagnostic[6],diagnostic[7],1,0};
            });
        }
        auto const sums=data.value();
        constexpr int work_count=IonElectricWorkComponent::Count;
        std::array<R,work_count+IonElectricWorkDiagnostic::Count> reduced{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((reduced[I]=amrex::get<I>(sums)),...);
        }(std::make_index_sequence<work_count+IonElectricWorkDiagnostic::Count>{});
        amrex::ParallelDescriptor::ReduceRealSum(reduced.data(),reduced.size());
        std::copy_n(reduced.begin(),work_count,result.species[s].begin());
        std::copy_n(reduced.begin()+work_count,IonElectricWorkDiagnostic::Count,
                    result.diagnostics[s].begin());
        result.particles+=amrex::get<22>(sums);
        result.invalid_particles+=amrex::get<23>(sums);
    }
    amrex::ParallelDescriptor::ReduceLongSum(result.particles);
    amrex::ParallelDescriptor::ReduceLongSum(result.invalid_particles);
    result.valid=result.invalid_particles==0;
    if (!result.valid) { result.species.clear(); result.diagnostics.clear(); }
    return result;
}
} // namespace warpx::thermal
