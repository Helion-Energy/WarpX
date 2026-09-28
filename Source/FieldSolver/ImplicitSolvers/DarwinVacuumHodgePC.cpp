/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "DarwinVacuumHodgePC.H"
#include "DarwinRZGreenSolver.H"
#include "DarwinRZYeeGreenSolver.H"
#include "Circuit/Coils/DeviceReduction.H"
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include "EmbeddedBoundary/Enabled.H"
#include "DarwinVacuumAffineResponse.H"
#include "NonlinearSolvers/FlexibleGMRES.H"
#include "WarpX.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace warpx::darwin {
namespace {
using MF = amrex::MultiFab;
using Field = amrex::Array<MF,3>;
using View = DarwinVacuumHodgePC::View;
using Real = amrex::Real;
View V(Field& f) { return {&f[0],&f[1],&f[2]}; }
View CV(Field const& f) { return {const_cast<MF*>(&f[0]),const_cast<MF*>(&f[1]),const_cast<MF*>(&f[2])}; }
int Dimension(int c) {
#if defined(WARPX_DIM_RZ)
    return c==0 ? 0 : c==2 ? 1 : -1;
#else
    return c;
#endif
}
#include "NativeVacuumGraph.H"
bool Overlap(MF const& a, MF const& b) {
    // Record storage addresses first: AMReX deliberately forbids nested
    // MFIter instances. These are host allocation metadata, not field copies.
    std::vector<std::pair<std::uintptr_t,std::uintptr_t>> ranges;
    for(amrex::MFIter it(a);it.isValid();++it) {
        auto const& x=a[it];
        auto begin=reinterpret_cast<std::uintptr_t>(x.dataPtr());
        ranges.emplace_back(begin,begin+static_cast<std::uintptr_t>(x.size()*sizeof(Real)));
    }
    bool bad=false;
    for(amrex::MFIter it(b);it.isValid();++it) {
        auto const& y=b[it];
        auto begin=reinterpret_cast<std::uintptr_t>(y.dataPtr());
        auto end=begin+static_cast<std::uintptr_t>(y.size()*sizeof(Real));
        for(auto const& range:ranges)bad=bad||(range.first<end&&begin<range.second);
    }
    amrex::ParallelDescriptor::ReduceBoolOr(bad);
    return bad;
}
}
struct DarwinVacuumHodgePC::Impl {
    using RT=Real;
    WarpX& w;
    View layout, magnetic;
    Options options;
    amrex::Geometry geometry;
    bool ready=false,failed=false;
    Statistics stats;
    amrex::Array<amrex::iMultiFab,3> charged,vacuum;
    amrex::Array<std::unique_ptr<amrex::iMultiFab>,3> edge_owner;
    std::unique_ptr<amrex::iMultiFab> node_owner;
    Field edge_metric, weighted, scratch, grad, curl;
    Field auxiliary_weight, auxiliary_rhs, auxiliary_solution;
    Field pc_action, pc_residual, pc_correction;
    amrex::Array<std::unique_ptr<amrex::MLEBNodeFDLaplacian>,3> auxiliary_operator;
    amrex::Array<std::unique_ptr<amrex::MLMG>,3> auxiliary_solver;
    std::unique_ptr<DarwinRZGreenSolver> auxiliary_green;
    std::unique_ptr<DarwinRZYeeGreenSolver> m_yee_green;
    amrex::Array<amrex::iMultiFab, 3> m_interface_index;
    amrex::Gpu::DeviceVector<amrex::Long> m_interface_ids;
    amrex::Gpu::DeviceVector<Real> m_interface_rhs, m_interface_lu;
    amrex::Gpu::DeviceVector<Real> m_interface_reduce_scratch;
    amrex::Gpu::DeviceVector<int> m_interface_pivots;
    MF node_measure,node_weight,phi,node_scratch;
    std::unique_ptr<JoinedGraph> graph;
    std::unique_ptr<DarwinVacuumAffineResponse> core;
    struct Block { Field edge; MF node; bool leased=false; };
    std::vector<std::unique_ptr<Block>> pool;
    struct Vec {
        Impl* owner=nullptr; int index=-1;
        Vec(Impl* p,int i):owner(p),index(i){}
        Vec(Vec const&)=delete;Vec& operator=(Vec const&)=delete;
        Vec(Vec&& other) noexcept:owner(other.owner),index(other.index){other.owner=nullptr;}
        Vec& operator=(Vec&& other) noexcept {
            if(this!=&other){if(owner)owner->pool[index]->leased=false;owner=other.owner;index=other.index;other.owner=nullptr;}return *this;
        }
        ~Vec(){if(owner)owner->pool[index]->leased=false;}
        Block& block()const{return *owner->pool[index];}
    };
    Impl(WarpX& sim,View const& e,View const& b,Options o):w(sim),layout(e),magnetic(b),options(o),geometry(sim.Geom(0)) {}
    void Define(Field& f,int ghosts) {
        for(int c=0;c<3;++c) { f[c].define(layout[c]->boxArray(),layout[c]->DistributionMap(),1,ghosts);f[c].setVal(0.); }
    }
    bool Layout(MF const& f,MF const& ref)const {
        return f.nComp()==1&&f.boxArray()==ref.boxArray()&&f.DistributionMap()==ref.DistributionMap();
    }
    void Check(View const& x)const {
        bool valid=true;for(int c=0;c<3;++c)valid=valid&&x[c]&&Layout(*x[c],*layout[c]);
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);AMREX_ALWAYS_ASSERT(valid);
    }
    void CheckNode(MF const& x)const {
        bool valid=Layout(x,node_measure);amrex::ParallelDescriptor::ReduceBoolAnd(valid);AMREX_ALWAYS_ASSERT(valid);
    }
    void Distinct(View const& out,View const& in)const {
        Check(out);Check(in);
        for(auto* a:out)for(auto* b:in)AMREX_ALWAYS_ASSERT(!Overlap(*a,*b));
        for(int c=0;c<3;++c)for(int q=0;q<c;++q)AMREX_ALWAYS_ASSERT(!Overlap(*out[c],*out[q]));
    }
    bool Build(Mask const& p,Mask const& v,std::string& error) {
        // Bind rank-local controls before any selector-dependent collective.
        // A valid but different interface cap must not split the setup protocol.
        amrex::Array<int, 4> minimum{
            int(options.use_yee_green_interface_inverse),
            int(options.use_auxiliary_preconditioner),
            int(options.use_green_auxiliary_inverse), options.max_interface_dofs};
        auto maximum = minimum;
        amrex::ParallelDescriptor::ReduceIntMin(minimum.data(), int(minimum.size()));
        amrex::ParallelDescriptor::ReduceIntMax(maximum.data(), int(maximum.size()));
        if (minimum != maximum) {
            error = "Hodge preconditioner selectors and interface cap must agree on every rank";
            return false;
        }
        if (options.use_yee_green_interface_inverse) {
            bool supported = false;
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
            supported = options.use_auxiliary_preconditioner && !options.use_green_auxiliary_inverse
                && options.max_interface_dofs > 0 && options.max_interface_dofs <= 4096
                && w.maxLevel() == 0 && !EB::enabled()
                && geometry.ProbLo(0) == 0. && geometry.Domain().smallEnd() == amrex::IntVect(0)
                && !geometry.isPeriodic(0) && !geometry.isPeriodic(1)
                && geometry.Domain().length(0) > 1 && geometry.Domain().length(1) > 1
                && WarpX::field_boundary_lo[0] == FieldBoundaryType::None
                && WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC
                && WarpX::field_boundary_lo[1] == FieldBoundaryType::PMC
                && WarpX::field_boundary_hi[1] == FieldBoundaryType::PMC;
#ifdef AMREX_USE_GPU
            supported = supported && (amrex::ParallelDescriptor::NProcs() == 1 ||
                                      amrex::ParallelDescriptor::UseGpuAwareMpi());
#endif
#endif
            amrex::ParallelDescriptor::ReduceBoolAnd(supported);
            if (!supported) {
                error = "Yee interface PC requires single-level no-EB RZ/FFT, regular axis, "
                    "radial PEC and two nonperiodic PMC caps, GPU-aware MPI, interface cap1..4096, "
                    "and nodal Green auxiliary disabled";
                return false;
            }
        }
        if(options.use_green_auxiliary_inverse) {
            bool supported=false;
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
            supported=options.use_auxiliary_preconditioner && w.maxLevel()==0 && !EB::enabled()
                && geometry.ProbLo(0)==0. && geometry.Domain().smallEnd()==amrex::IntVect(0)
                && !geometry.isPeriodic(0) && !geometry.isPeriodic(1)
                && geometry.Domain().length(0)>1 && geometry.Domain().length(1)>1
                && WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC;
            for(auto bc:{WarpX::field_boundary_lo[1],WarpX::field_boundary_hi[1]})
                supported=supported&&(bc==FieldBoundaryType::PEC||bc==FieldBoundaryType::PMC);
#ifdef AMREX_USE_GPU
            supported=supported&&(amrex::ParallelDescriptor::NProcs()==1||
                                  amrex::ParallelDescriptor::UseGpuAwareMpi());
#endif
#endif
            amrex::ParallelDescriptor::ReduceBoolAnd(supported);
            if(!supported){error="Green auxiliary inverse requires single-level no-EB nonperiodic RZ with PEC radial wall, PEC/PMC caps, FFT, and GPU-aware MPI when distributed";return false;}
        }
        bool ok=std::isfinite(options.b)&&options.b>0.&&std::isfinite(options.length)&&options.length>0.
            &&std::isfinite(options.relative_tolerance)&&options.relative_tolerance>=0.
            &&std::isfinite(options.absolute_tolerance)&&options.absolute_tolerance>=0.
            &&options.max_iterations>0&&options.restart_length>0;
        bool anchored=false;
        amrex::Long nodes=1;
        for(int d=0;d<AMREX_SPACEDIM;++d) {
            anchored=anchored||WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC||WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC;
            nodes*=geometry.Domain().length(d)+1;
        }
        ok=ok&&anchored&&nodes<std::numeric_limits<int>::max();
#if defined(WARPX_DIM_RZ)
        ok=ok&&geometry.ProbLo(0)==0.&&WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC;
#elif !defined(WARPX_DIM_3D)
        ok=false;
#endif
        for(int c=0;c<3;++c)ok=ok&&p[c]&&v[c]&&p[c]->nComp()==1&&v[c]->nComp()==1
            &&p[c]->boxArray()==layout[c]->boxArray()&&v[c]->boxArray()==layout[c]->boxArray()
            &&p[c]->DistributionMap()==layout[c]->DistributionMap()&&v[c]->DistributionMap()==layout[c]->DistributionMap();
        amrex::ParallelDescriptor::ReduceBoolAnd(ok);
        if(!ok){error="Hodge PC requires finite options, native layouts and an admitted scalar-PEC-anchored topology";return false;}
        auto lo=geometry.Domain().smallEnd(),hi=geometry.Domain().bigEnd()+amrex::IntVect(1);
        amrex::GpuArray<int,AMREX_SPACEDIM> dl{},dh{},per{},pmc_low{},pmc_high{};
        for(int d=0;d<AMREX_SPACEDIM;++d){dl[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC;dh[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC;per[d]=geometry.isPeriodic(d);pmc_low[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;pmc_high[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;}
        int invalid=0;
        for(int c=0;c<3;++c) {
            auto type=layout[c]->ixType().toIntVect();
            amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(*p[c]);it.isValid();++it) {
                auto a=p[c]->const_array(it),b=v[c]->const_array(it);
                op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T {
                    amrex::IntVect q(AMREX_D_DECL(i,j,k));bool fixed=false;
                    for(int d=0;d<AMREX_SPACEDIM;++d)fixed=fixed||(type[d]&&((dl[d]&&q[d]==lo[d])||(dh[d]&&q[d]==hi[d])));
#if defined(WARPX_DIM_RZ)
                    fixed=fixed||(c==1&&i==0);
#endif
                    return {int((a(q)!=0&&a(q)!=1)||(b(q)!=0&&b(q)!=1)||a(q)+b(q)!=(fixed?0:1))};
                });
            }
            invalid=std::max(invalid,amrex::get<0>(data.value()));
        }
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        if(invalid){error="P/V masks must be disjoint binary raw-support rows with exact physical traces";return false;}
        for(int c=0;c<3;++c) {
            charged[c].define(p[c]->boxArray(),p[c]->DistributionMap(),1,1);charged[c].setVal(0);
            vacuum[c].define(v[c]->boxArray(),v[c]->DistributionMap(),1,1);vacuum[c].setVal(0);
            amrex::iMultiFab::Copy(charged[c],*p[c],0,0,1,0);charged[c].FillBoundary(geometry.periodicity());
            amrex::iMultiFab::Copy(vacuum[c],*v[c],0,0,1,0);vacuum[c].FillBoundary(geometry.periodicity());
            edge_owner[c]=layout[c]->OwnerMask(geometry.periodicity());
        }
        amrex::Long count_v=0;
        for(int c=0;c<3;++c) {
            amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<amrex::Long> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(vacuum[c]);it.isValid();++it){auto mask=vacuum[c].const_array(it),own=edge_owner[c]->const_array(it);op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{return {amrex::Long(mask(i,j,k)&&own(i,j,k))};});}
            count_v+=amrex::get<0>(data.value());
        }
        amrex::ParallelDescriptor::ReduceLongSum(count_v);
        if(count_v==0){error="No vacuum rows: caller must retain the existing positive-density path";return false;}
        auto nba=amrex::convert(w.boxArray(0),amrex::IntVect(1));auto const& dm=w.DistributionMap(0);
        node_measure.define(nba,dm,1,0);node_weight.define(nba,dm,1,0);phi.define(nba,dm,1,1);node_scratch.define(nba,dm,1,0);
        node_owner=node_measure.OwnerMask(geometry.periodicity());
        for(amrex::MFIter it(node_measure);it.isValid();++it) {
            auto m=node_measure.array(it);auto own=node_owner->const_array(it);auto weight=node_weight.array(it);
            amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){
                amrex::IntVect q(AMREX_D_DECL(i,j,k));Real z=1.;
                for(int d=0;d<AMREX_SPACEDIM;++d){if((dl[d]&&q[d]==lo[d])||(dh[d]&&q[d]==hi[d]))z=0.;if(!per[d]&&(q[d]==lo[d]||q[d]==hi[d]))z*=.5;}
#if defined(WARPX_DIM_RZ)
                z*=i==0?Real(.25):Real(i)-(i==hi[0]?Real(.5):Real(0));
#endif
                m(q)=z;weight(q)=own(q)?z:0.;
            });
        }
        graph=std::make_unique<JoinedGraph>(w,charged,node_weight);
        if(graph->roots.size()>1){error="More than one floating charged component is outside the first Hodge PC scope";return false;}
        stats.label_iterations=graph->iterations;stats.floating_components=int(graph->roots.size());
        Define(edge_metric,1);Define(weighted,1);Define(scratch,0);Define(grad,0);Define(curl,0);
        for(int c=0;c<3;++c) {
            auto type=layout[c]->ixType().toIntVect();
            for(amrex::MFIter it(edge_metric[c]);it.isValid();++it) {
                auto m=edge_metric[c].array(it);
                amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){
                    amrex::IntVect q(AMREX_D_DECL(i,j,k));Real z=1.;
#if defined(WARPX_DIM_RZ)
                    z=Real(i)+(type[0]?Real(0):Real(.5));if(type[0]&&i==0)z=.125;
#endif
                    for(int d=0;d<AMREX_SPACEDIM;++d)if(type[d]&&((pmc_low[d]&&q[d]==lo[d])||(pmc_high[d]&&q[d]==hi[d])))z*=.5;
                    m(q)=z;
                });
            }
            edge_metric[c].FillBoundary(geometry.periodicity());
        }
        DarwinVacuumAffineResponse::Options co;
        for(int d=0;d<AMREX_SPACEDIM;++d){co.pmc_lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;co.pmc_hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;}
        core=std::make_unique<DarwinVacuumAffineResponse>(w,layout,magnetic,co);
        for(auto& f:scratch)f.setVal(0.);
        if(!core->Freeze({&vacuum[0],&vacuum[1],&vacuum[2]},{&charged[0],&charged[1],&charged[2]},V(scratch),{nullptr,nullptr,nullptr})){error="Native compensated-curl freeze failed";return false;}
        if(!core->ApplyCompensatedMaskedCurl(V(curl),V(scratch))){error="Compensated native curl stencil is unavailable";return false;}
        for(int k=0;k<2*options.restart_length+5;++k){auto block=std::make_unique<Block>();Define(block->edge,0);block->node.define(nba,dm,1,0);block->node.setVal(0.);pool.push_back(std::move(block));}
        amrex::Long vc=0,nc=0;
        for(int c=0;c<3;++c){amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<amrex::Long> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(vacuum[c]);it.isValid();++it){auto mask=vacuum[c].const_array(it),own=edge_owner[c]->const_array(it);op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{return {amrex::Long(mask(i,j,k)&&own(i,j,k))};});}vc+=amrex::get<0>(data.value());}
        {amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<amrex::Long> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(node_weight);it.isValid();++it){auto label=graph->labels.const_array(it);auto weight=node_weight.const_array(it);op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{return {amrex::Long(label(i,j,k)<0&&weight(i,j,k)>0.)};});}nc=amrex::get<0>(data.value());}
        amrex::ParallelDescriptor::ReduceLongSum(vc);amrex::ParallelDescriptor::ReduceLongSum(nc);
        stats.vacuum_edges=vc;stats.potential_dofs=nc+stats.floating_components;
        auto bytes=[](MF const& f){std::size_t n=0;for(amrex::MFIter it(f);it.isValid();++it)n+=f[it].box().numPts()*sizeof(Real);return n;};
        for(auto const* fs:{&edge_metric,&weighted,&scratch,&grad,&curl})for(auto const& f:*fs)stats.local_owned_bytes+=bytes(f);
        for(auto const* f:{&node_measure,&node_weight,&phi,&node_scratch})stats.local_owned_bytes+=bytes(*f);
        for(auto const& q:pool){stats.local_owned_bytes+=bytes(q->node);for(auto const& f:q->edge)stats.local_owned_bytes+=bytes(f);}
        ready = true;
        if (options.use_yee_green_interface_inverse) {
            if (!BuildGreenInterface(error)) {
                ready = false;
                return false;
            }
        } else if (options.use_auxiliary_preconditioner) {
            BuildAuxiliary();
        }
        error.clear();
        return true;
    }
    void BuildAuxiliary() {
        Define(pc_action,0);Define(pc_residual,0);Define(pc_correction,0);
        auto const nba=node_measure.boxArray();auto const dm=node_measure.DistributionMap();
        auto const lo=geometry.Domain().smallEnd(),hi=geometry.Domain().bigEnd()+amrex::IntVect(1);
        auto const periodic=geometry.isPeriodicArray();
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
        if(options.use_green_auxiliary_inverse)
            auxiliary_green=std::make_unique<DarwinRZGreenSolver>(geometry,
                WarpX::field_boundary_lo[1]==FieldBoundaryType::PMC,
                WarpX::field_boundary_hi[1]==FieldBoundaryType::PMC, true,
                DarwinRZGreenSolver::BoundaryFamily::ElectricAuxiliary);
#endif
        for(int c=0;c<3;++c) {
            auxiliary_weight[c].define(nba,dm,1,0);
            auxiliary_rhs[c].define(nba,dm,1,0);auxiliary_rhs[c].setVal(0.);
            auxiliary_solution[c].define(nba,dm,1,1);auxiliary_solution[c].setVal(0.);
            amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> low{},high{};
            amrex::GpuArray<int,AMREX_SPACEDIM> fixed_low{},fixed_high{};
            for(int d=0;d<AMREX_SPACEDIM;++d) {
                bool const normal=Dimension(c)==d;
                for(int side=0;side<2;++side) {
                    auto const bc=side?WarpX::field_boundary_hi[d]:WarpX::field_boundary_lo[d];
                    bool fixed=(bc==FieldBoundaryType::PEC&&!normal)||(bc==FieldBoundaryType::PMC&&normal);
#if defined(WARPX_DIM_RZ)
                    // AMReX requires Neumann as its RZ axis descriptor. Its
                    // native alpha/r^2 vector term fixes r/theta at the axis.
                    if(d==0&&side==0)fixed=false;
#endif
                    (side?fixed_high[d]:fixed_low[d])=fixed;
                    (side?high[d]:low[d])=periodic[d]?amrex::LinOpBCType::Periodic:
                        fixed?amrex::LinOpBCType::Dirichlet:amrex::LinOpBCType::Neumann;
                }
            }
            for(amrex::MFIter it(auxiliary_weight[c]);it.isValid();++it) {
                auto a=auxiliary_weight[c].array(it);
                amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k) {
                    amrex::IntVect q(AMREX_D_DECL(i,j,k));Real weight=1.;
                    for(int d=0;d<AMREX_SPACEDIM;++d) {
                        if((fixed_low[d]&&q[d]==lo[d])||(fixed_high[d]&&q[d]==hi[d]))weight=0.;
                        if(!periodic[d]&&(q[d]==lo[d]||q[d]==hi[d]))weight*=.5;
                    }
#if defined(WARPX_DIM_RZ)
                    weight*=i==0?Real(.25):Real(i);
                    if(i==0&&c<2)weight=0.;
#endif
                    a(q)=weight;
                });
            }
            if(auxiliary_green)continue;
            amrex::LPInfo info;
            auxiliary_operator[c]=std::make_unique<amrex::MLEBNodeFDLaplacian>(
                amrex::Vector<amrex::Geometry>{geometry},amrex::Vector<amrex::BoxArray>{w.boxArray(0)},
                amrex::Vector<amrex::DistributionMapping>{w.DistributionMap(0)},info);
#if defined(WARPX_DIM_RZ)
            auxiliary_operator[c]->setRZ(true);
            auxiliary_operator[c]->setAlpha(c<2?1.:0.);
#endif
            auxiliary_operator[c]->setSigma({AMREX_D_DECL(1.,1.,1.)});
            auxiliary_operator[c]->setDomainBC(low,high);
            auxiliary_solver[c]=std::make_unique<amrex::MLMG>(*auxiliary_operator[c]);
            auxiliary_solver[c]->setVerbose(0);auxiliary_solver[c]->setBottomVerbose(0);
            auxiliary_solver[c]->setFixedIter(1);auxiliary_solver[c]->setMaxIter(1);
            auxiliary_solver[c]->setBottomSolver(amrex::BottomSolver::smoother);
            auxiliary_solver[c]->setFinalSmooth(8); // Exact bottom smoother count tested in R06.
            auxiliary_solver[c]->prepareForSolve(amrex::Vector<MF*>{&auxiliary_solution[c]},
                                                   amrex::Vector<MF const*>{&auxiliary_rhs[c]});
        }
        // Owned real arrays only. Hierarchy and communication bytes are
        // separately reported by the fixture/build allocator receipts.
        for(auto const* fields:{&auxiliary_weight,&auxiliary_rhs,&auxiliary_solution,&pc_action,&pc_residual,&pc_correction})
            for(auto const& field:*fields)for(amrex::MFIter it(field);it.isValid();++it)
                stats.local_owned_bytes+=field[it].size()*sizeof(Real);
    }
#include "NativeVacuumGreenInterface.H"
    void CheckAuxiliary(View const& x)const {
        AMREX_ALWAYS_ASSERT(ready && options.use_auxiliary_preconditioner &&
                            !options.use_yee_green_interface_inverse);
        for(auto* f:x){AMREX_ALWAYS_ASSERT(f);CheckNode(*f);}
    }
    void ProlongAuxiliary(View const& out,View const& in) {
        Check(out);CheckAuxiliary(in);for(auto* a:out)for(auto* b:in)AMREX_ALWAYS_ASSERT(!Overlap(*a,*b));
        for(int c=0;c<3;++c) {
            auto& node=auxiliary_solution[c];node.setBndry(0.);MF::Copy(node,*in[c],0,0,1,0);
            for(amrex::MFIter it(node);it.isValid();++it){auto a=node.array(it);auto m=auxiliary_weight[c].const_array(it);
                amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){if(m(i,j,k)==0.)a(i,j,k)=0.;});}
            node.OverrideSync(geometry.periodicity());node.FillBoundary(geometry.periodicity());
            int const d=Dimension(c);auto off=d<0?amrex::IntVect(0):amrex::IntVect::TheDimensionVector(d);
            for(amrex::MFIter it(*out[c]);it.isValid();++it){auto a=out[c]->array(it);auto n=node.const_array(it);auto v=vacuum[c].const_array(it);
                amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){amrex::IntVect q(AMREX_D_DECL(i,j,k));a(q)=v(q)?(d<0?n(q):.5*(n(q)+n(q+off))):0.;});}
            out[c]->OverrideSync(geometry.periodicity());
        }
    }
    void RestrictAuxiliary(View const& out,View const& in) {
        CheckAuxiliary(out);Check(in);for(auto* a:out)for(auto* b:in)AMREX_ALWAYS_ASSERT(!Overlap(*a,*b));
        for(int c=0;c<3;++c)for(int q=0;q<c;++q)AMREX_ALWAYS_ASSERT(!Overlap(*out[c],*out[q]));
        for(int c=0;c<3;++c) {
            weighted[c].setVal(0.);MF::Copy(weighted[c],*in[c],0,0,1,0);weighted[c].OverrideSync(geometry.periodicity());
            for(amrex::MFIter it(weighted[c]);it.isValid();++it){auto a=weighted[c].array(it);auto m=edge_metric[c].const_array(it);auto v=vacuum[c].const_array(it);
                amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=v(i,j,k)?a(i,j,k)*m(i,j,k):0.;});}
            weighted[c].FillBoundary(geometry.periodicity());
            int const d=Dimension(c);auto off=d<0?amrex::IntVect(0):amrex::IntVect::TheDimensionVector(d);
            for(amrex::MFIter it(*out[c]);it.isValid();++it){auto a=out[c]->array(it);auto edge=weighted[c].const_array(it);auto m=auxiliary_weight[c].const_array(it);
                amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){amrex::IntVect q(AMREX_D_DECL(i,j,k));a(q)=m(q)>0.?(d<0?edge(q):.5*(edge(q)+edge(q-off)))/m(q):0.;});}
            out[c]->OverrideSync(geometry.periodicity());
        }
    }
    void HodgeCompletion(View const& out,View const& in) {
        Curl(out,in);Adjoint(node_scratch,in);Gradient(V(grad),node_scratch);
        for(int c=0;c<3;++c){MF::Saxpy(*out[c],1.,grad[c],0,0,1,0);out[c]->mult(options.length*options.length,0,1,0);}
        ++stats.hodge_completion_actions;
    }
    void AuxiliaryPrecond(View const& out,View const& rhs) {
        // Fixed multiplicative cycle: one damped edge-Jacobi update,
        // one native nodal vector-Poisson V-cycle (or its optional cached
        // exact Green inverse), then two edge-Jacobi updates.
        // All edges, including the nodal-transfer alternating null, are kept.
        Real const diagonal=2.*options.length*options.length*core->Diagonal();
        for(int c=0;c<3;++c){MF::Copy(*out[c],*rhs[c],0,0,1,0);out[c]->mult(1./diagonal,0,1,0);}
        HodgeCompletion(V(pc_action),out);
        for(int c=0;c<3;++c)MF::LinComb(pc_residual[c],1.,*rhs[c],0,-1.,pc_action[c],0,0,1,0);
        RestrictAuxiliary(V(auxiliary_rhs),V(pc_residual));
        for(int c=0;c<3;++c) {
            // Pinned MLEBNodeFDLaplacian is div grad - alpha/r^2.
            // Negate the RHS to approximate the positive Hodge inverse.
            auxiliary_rhs[c].mult(-1./(options.length*options.length),0,1,0);
            auxiliary_solution[c].setVal(0.);
            if(auxiliary_green) {
                auxiliary_green->SolveComponent(auxiliary_solution[c],auxiliary_rhs[c],c);
                ++stats.green_auxiliary_solves;
            } else {
                auxiliary_solver[c]->solve({&auxiliary_solution[c]},{&auxiliary_rhs[c]},0.,0.);
                ++stats.auxiliary_cycles;
            }
        }
        ProlongAuxiliary(V(pc_correction),V(auxiliary_solution));
        for(int c=0;c<3;++c)MF::Saxpy(*out[c],1.,pc_correction[c],0,0,1,0);
        for(int sweep=0;sweep<2;++sweep) {
            HodgeCompletion(V(pc_action),out);
            for(int c=0;c<3;++c){MF::LinComb(pc_residual[c],1.,*rhs[c],0,-1.,pc_action[c],0,0,1,0);MF::Saxpy(*out[c],1./diagonal,pc_residual[c],0,0,1,0);}
        }
    }
    void Canonical(MF& out,MF const& in) {
        AMREX_ALWAYS_ASSERT(ready);CheckNode(out);CheckNode(in);
        MF::Copy(out,in,0,0,1,0);out.OverrideSync(geometry.periodicity());
        for(int root:graph->roots) {
            amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum> op;amrex::ReduceData<Real,Real> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(out);it.isValid();++it){auto a=out.const_array(it),m=node_weight.const_array(it);auto l=graph->labels.const_array(it);op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{Real z=l(i,j,k)==root?m(i,j,k):0.;return {z*a(i,j,k),z};});}
            auto q=data.value();Real total[]{amrex::get<0>(q),amrex::get<1>(q)};amrex::ParallelDescriptor::ReduceRealSum(total,2);AMREX_ALWAYS_ASSERT(total[1]>0.);
            Real mean=total[0]/total[1];
            for(amrex::MFIter it(out);it.isValid();++it){auto a=out.array(it);auto l=graph->labels.const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){if(l(i,j,k)==root)a(i,j,k)=mean;});}
        }
        for(amrex::MFIter it(out);it.isValid();++it){auto a=out.array(it);auto l=graph->labels.const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){if(l(i,j,k)==0)a(i,j,k)=0.;});}
        out.setBndry(0.);out.OverrideSync(geometry.periodicity());out.FillBoundary(geometry.periodicity());
    }
    void Gradient(View const& out,MF const& in) {
        AMREX_ALWAYS_ASSERT(ready);Check(out);CheckNode(in);
        for(auto* f:out)AMREX_ALWAYS_ASSERT(!Overlap(*f,in));
        Canonical(phi,in);auto dx=geometry.CellSizeArray();
        for(int c=0;c<3;++c){int d=Dimension(c);out[c]->setVal(0.);if(d<0)continue;auto off=amrex::IntVect::TheDimensionVector(d);Real idx=1./dx[d];
            for(amrex::MFIter it(*out[c]);it.isValid();++it){auto a=out[c]->array(it);auto p=phi.const_array(it);auto m=vacuum[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){amrex::IntVect q(AMREX_D_DECL(i,j,k));a(q)=m(q)?(p(q+off)-p(q))*idx:0.;});}
            out[c]->OverrideSync(geometry.periodicity());}
        ++stats.gradient_actions;
    }
    void Adjoint(MF& out,View const& in) {
        AMREX_ALWAYS_ASSERT(ready);Check(in);CheckNode(out);for(auto* f:in)AMREX_ALWAYS_ASSERT(!Overlap(out,*f));
        for(int c=0;c<3;++c){weighted[c].setVal(0.);MF::Copy(weighted[c],*in[c],0,0,1,0);weighted[c].OverrideSync(geometry.periodicity());
            for(amrex::MFIter it(weighted[c]);it.isValid();++it){auto a=weighted[c].array(it);auto m=edge_metric[c].const_array(it);auto mask=vacuum[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=mask(i,j,k)?a(i,j,k)*m(i,j,k):0.;});}weighted[c].FillBoundary(geometry.periodicity());}
        auto dx=geometry.CellSizeArray();
        for(amrex::MFIter it(node_scratch);it.isValid();++it){auto outa=node_scratch.array(it);auto m=node_measure.const_array(it);amrex::GpuArray<amrex::Array4<Real const>,AMREX_SPACEDIM> f{};
            for(int d=0;d<AMREX_SPACEDIM;++d){
#if defined(WARPX_DIM_RZ)
                int c=d==0?0:2;
#else
                int c=d;
#endif
                f[d]=weighted[c].const_array(it);}
            amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){amrex::IntVect q(AMREX_D_DECL(i,j,k));Real z=0.;for(int d=0;d<AMREX_SPACEDIM;++d){auto off=amrex::IntVect::TheDimensionVector(d);z+=(f[d](q-off)-f[d](q))/dx[d];}outa(q)=m(q)>0.?z/m(q):0.;});}
        Canonical(out,node_scratch);++stats.adjoint_actions;
    }
    void Curl(View const& out,View const& in) {
        AMREX_ALWAYS_ASSERT(ready);Distinct(out,in);AMREX_ALWAYS_ASSERT(core->ApplyCompensatedMaskedCurl(out,in));
        for(auto* f:out)f->mult(core->Diagonal(),0,1,0);++stats.curl_actions;
    }
    Real DotNode(MF const& a,MF const& b)const {
        AMREX_ALWAYS_ASSERT(ready);CheckNode(a);CheckNode(b);amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<Real> data(op);using T=decltype(data)::Type;
        for(amrex::MFIter it(a);it.isValid();++it){auto x=a.const_array(it),y=b.const_array(it),m=node_weight.const_array(it);op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{return {x(i,j,k)*y(i,j,k)*m(i,j,k)};});}
        Real r=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(r);return r;
    }
    Real DotEdge(View const& a,View const& b)const {
        AMREX_ALWAYS_ASSERT(ready);Check(a);Check(b);Real r=0.;
        for(int c=0;c<3;++c){amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<Real> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(*a[c]);it.isValid();++it){auto x=a[c]->const_array(it),y=b[c]->const_array(it),m=edge_metric[c].const_array(it);auto v=vacuum[c].const_array(it),o=edge_owner[c]->const_array(it);op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{return {v(i,j,k)&&o(i,j,k)?x(i,j,k)*y(i,j,k)*m(i,j,k):0.};});}r+=amrex::get<0>(data.value());}
        amrex::ParallelDescriptor::ReduceRealSum(r);return r;
    }
    void Restrict(View const& fields)const {
        for(int c=0;c<3;++c)for(amrex::MFIter it(*fields[c]);it.isValid();++it){auto a=fields[c]->array(it);auto m=vacuum[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){if(!m(i,j,k))a(i,j,k)=0.;});}
    }
    void Augmented(View const& out,MF& scalar,View const& in,MF const& potential) {
        AMREX_ALWAYS_ASSERT(ready);Distinct(out,in);CheckNode(scalar);CheckNode(potential);AMREX_ALWAYS_ASSERT(!Overlap(scalar,potential));
        for(auto* f:out){AMREX_ALWAYS_ASSERT(!Overlap(*f,potential));AMREX_ALWAYS_ASSERT(!Overlap(*f,scalar));}for(auto* f:in)AMREX_ALWAYS_ASSERT(!Overlap(*f,scalar));
        for(int c=0;c<3;++c)MF::Copy(scratch[c],*in[c],0,0,1,0);Restrict(V(scratch));
        Curl(out,V(scratch));Gradient(V(grad),potential);Real l=options.length;
        for(int c=0;c<3;++c){out[c]->mult(l*l,0,1,0);MF::Saxpy(*out[c],l,grad[c],0,0,1,0);}
        Adjoint(scalar,V(scratch));scalar.mult(l,0,1,0);++stats.augmented_actions;
    }
    Vec makeVecLHS(){for(int k=0;k<int(pool.size());++k)if(!pool[k]->leased){pool[k]->leased=true;return Vec(this,k);}amrex::Abort("Hodge PC work pool exhausted");return Vec(this,0);}
    Vec makeVecRHS(){return makeVecLHS();}
    void setToZero(Vec& a){for(auto& f:a.block().edge)f.setVal(0.);a.block().node.setVal(0.);}
    void assign(Vec& a,Vec const& b){for(int c=0;c<3;++c)MF::Copy(a.block().edge[c],b.block().edge[c],0,0,1,0);MF::Copy(a.block().node,b.block().node,0,0,1,0);}
    void scale(Vec& a,Real f){for(auto& q:a.block().edge)q.mult(f,0,1,0);a.block().node.mult(f,0,1,0);}
    void increment(Vec& a,Vec const& b,Real f){for(int c=0;c<3;++c)MF::Saxpy(a.block().edge[c],f,b.block().edge[c],0,0,1,0);MF::Saxpy(a.block().node,f,b.block().node,0,0,1,0);}
    void linComb(Vec& a,Real f,Vec const& b,Real g,Vec const& c){for(int d=0;d<3;++d)MF::LinComb(a.block().edge[d],f,b.block().edge[d],0,g,c.block().edge[d],0,0,1,0);MF::LinComb(a.block().node,f,b.block().node,0,g,c.block().node,0,0,1,0);}
    Real dotProduct(Vec const& a,Vec const& b){return DotEdge(CV(a.block().edge),CV(b.block().edge))+DotNode(a.block().node,b.block().node);}
    Real norm2(Vec const& a){Real r=dotProduct(a,a);if(failed||r<0.)return std::numeric_limits<Real>::quiet_NaN();return std::isfinite(r)?std::sqrt(r):r;}
    void apply(Vec& a,Vec const& b){Augmented(V(a.block().edge),a.block().node,CV(b.block().edge),b.block().node);}
    void precond(Vec& a,Vec const& b) {
        if (options.use_yee_green_interface_inverse) {
            GreenInterfacePrecond(V(a.block().edge), CV(b.block().edge));
        } else if(options.use_auxiliary_preconditioner)AuxiliaryPrecond(V(a.block().edge),CV(b.block().edge));
        else {assign(a,b);Real d=options.length*options.length*core->Diagonal();for(auto& f:a.block().edge)f.mult(1./d,0,1,0);}
        Canonical(a.block().node,b.block().node);
    }
    Result Solve(View const& out,MF& potential,View const& d,View const& h) {
        AMREX_ALWAYS_ASSERT(ready);Check(out);Check(d);Check(h);CheckNode(potential);Distinct(out,d);Distinct(out,h);
        for(auto* f:out)AMREX_ALWAYS_ASSERT(!Overlap(*f,potential));
        for(auto* f:d)AMREX_ALWAYS_ASSERT(!Overlap(*f,potential));
        for(auto* f:h)AMREX_ALWAYS_ASSERT(!Overlap(*f,potential));
        auto rhs=makeVecRHS(),solution=makeVecLHS();
        for(int c=0;c<3;++c){MF::Copy(rhs.block().edge[c],*d[c],0,0,1,0);rhs.block().edge[c].mult(-1.,0,1,0);}Restrict(V(rhs.block().edge));
        Adjoint(rhs.block().node,h);rhs.block().node.mult(options.b/options.length,0,1,0);
        Result result;failed=false;result.initial=norm2(rhs);result.target=std::max(options.absolute_tolerance,options.relative_tolerance*result.initial);
        FlexibleGMRES<Vec,Impl> solver;solver.define(*this);solver.setMaxIters(options.max_iterations);solver.setRestartLength(options.restart_length);solver.setVerbose(0);
        solver.solve(solution,rhs,options.relative_tolerance,options.absolute_tolerance);
        result.status=solver.getStatus();result.iterations=solver.getNumIters();result.residual=solver.getResidualNorm();result.finite=std::isfinite(norm2(solution))&&std::isfinite(result.residual);result.converged=result.finite&&result.status==0&&result.residual<=result.target;
        stats.linear_iterations+=result.iterations;
        if(result.finite){for(int c=0;c<3;++c){MF::Copy(*out[c],solution.block().edge[c],0,0,1,0);out[c]->mult(options.length*options.length/options.b,0,1,0);}MF::Copy(potential,solution.block().node,0,0,1,0);potential.mult(options.length,0,1,0);}
        return result;
    }
};
DarwinVacuumHodgePC::DarwinVacuumHodgePC(WarpX& w,View const& e,View const& b,Options o):m_impl(std::make_unique<Impl>(w,e,b,o)){}
DarwinVacuumHodgePC::~DarwinVacuumHodgePC()=default;
bool DarwinVacuumHodgePC::Prepare(Mask const& p,Mask const& v,std::string& error){auto n=std::make_unique<Impl>(m_impl->w,m_impl->layout,m_impl->magnetic,m_impl->options);if(!n->Build(p,v,error))return false;m_impl.swap(n);return true;}
bool DarwinVacuumHodgePC::Ready()const noexcept{return m_impl->ready;}
void DarwinVacuumHodgePC::Invalidate()noexcept{m_impl->ready=false;}
DarwinVacuumHodgePC::Statistics const& DarwinVacuumHodgePC::Stats()const{return m_impl->stats;}
amrex::iMultiFab const& DarwinVacuumHodgePC::Labels()const{AMREX_ALWAYS_ASSERT(Ready());return m_impl->graph->labels;}
MF const& DarwinVacuumHodgePC::NodeMetric()const{AMREX_ALWAYS_ASSERT(Ready());return m_impl->node_weight;}
void DarwinVacuumHodgePC::CanonicalPotential(MF& out,MF const& in){m_impl->Canonical(out,in);}
void DarwinVacuumHodgePC::Gradient(View const& out,MF const& in){m_impl->Gradient(out,in);}
void DarwinVacuumHodgePC::Adjoint(MF& out,View const& in){m_impl->Adjoint(out,in);}
void DarwinVacuumHodgePC::Curl(View const& out,View const& in){m_impl->Curl(out,in);}
Real DarwinVacuumHodgePC::EdgeDot(View const& a,View const& b)const{return m_impl->DotEdge(a,b);}
Real DarwinVacuumHodgePC::PotentialDot(MF const& a,MF const& b)const{return m_impl->DotNode(a,b);}
void DarwinVacuumHodgePC::ProlongAuxiliary(View const& out,View const& in){m_impl->ProlongAuxiliary(out,in);}
void DarwinVacuumHodgePC::RestrictAuxiliary(View const& out,View const& in){m_impl->RestrictAuxiliary(out,in);}
MF const& DarwinVacuumHodgePC::AuxiliaryMetric(int c)const{AMREX_ALWAYS_ASSERT(Ready() && m_impl->options.use_auxiliary_preconditioner && !m_impl->options.use_yee_green_interface_inverse && c>=0 && c<3);return m_impl->auxiliary_weight[c];}
void DarwinVacuumHodgePC::ApplyHodgeInverse(View const& out, View const& in)
{
    AMREX_ALWAYS_ASSERT(Ready() && m_impl->options.use_yee_green_interface_inverse);
    m_impl->GreenInterfacePrecond(out, in);
}
void DarwinVacuumHodgePC::ApplyAugmented(View const& out,MF& scalar,View const& in,MF const& phi){m_impl->Augmented(out,scalar,in,phi);}
DarwinVacuumHodgePC::Result DarwinVacuumHodgePC::Solve(View const& out,MF& phi,View const& d,View const& h){return m_impl->Solve(out,phi,d,h);}
} // namespace warpx::darwin
