/* Field-only primitive fixture. No OneStep, residual or acceptance route. */
#include "FieldSolver/ImplicitSolvers/DarwinVacuumHodgePC.H"
#include "FieldSolver/ImplicitSolvers/NativeInertiaSupport.H"
#include "FieldSolver/ImplicitSolvers/NativeVacuumConstraint.H"
#include "Initialization/WarpXInit.H"
#include "../native_retained_vacuum_fixture/NativeBytes.H"
#include "ReferenceRows.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
using namespace warpx::darwin;
using MF=amrex::MultiFab;
using Field=amrex::Array<MF,3>;
using View=DarwinVacuumHodgePC::View;
using R=amrex::Real;
using FT=warpx::fields::FieldType;
View V(Field& f){return {&f[0],&f[1],&f[2]};}
Field Clone(View const& layout){Field out;for(int c=0;c<3;++c){out[c].define(layout[c]->boxArray(),layout[c]->DistributionMap(),1,0);out[c].setVal(0.);}return out;}
void Zero(Field& a){for(auto& f:a)f.setVal(0.);}
void Scale(Field& a,R scale){for(auto& f:a)f.mult(scale,0,1,0);}
void Copy(Field& a,Field const& b){for(int c=0;c<3;++c)MF::Copy(a[c],b[c],0,0,1,0);}
R Error(Field const& a,Field const& b){R x=0.;for(int c=0;c<3;++c){MF q(a[c].boxArray(),a[c].DistributionMap(),1,0);MF::LinComb(q,1.,a[c],0,-1.,b[c],0,0,1,0);x=std::max(x,q.norminf(0));}return x;}
RawState State(WarpX& w){auto out=ParticleBytes(w);for(auto const& name:w.m_fields.list()){auto* f=w.m_fields.internal_get(name);for(amrex::MFIter it(*f);it.isValid();++it)out["field/"+name+"/"+std::to_string(it.index())]=ReadBytes((*f)[it].dataPtr(),(*f)[it].size());}out["time"]=ReadScalar(w.gett_new(0));out["step"]=ReadScalar(w.getistep(0));return out;}
void PointValue(Field& field,Point p,R value){
    for(amrex::MFIter it(field[p.c]);it.isValid();++it){auto a=field[p.c].array(it);amrex::IntVect q(AMREX_D_DECL(p.i,p.j,p.k));if(it.validbox().contains(q))amrex::ParallelFor(1,[=]AMREX_GPU_DEVICE(int){a(q)=value;});}
}
R ReadPoint(MF const& f,Point p,amrex::Geometry const& g){
    auto owner=f.OwnerMask(g.periodicity());R value=0.;
    for(amrex::MFIter it(f);it.isValid();++it){amrex::IntVect q(AMREX_D_DECL(p.i,p.j,p.k));if(!it.validbox().contains(q))continue;auto a=f.const_array(it);auto o=owner->const_array(it);amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<R> data(op);using T=decltype(data)::Type;op.eval(amrex::Box(q,q),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{return {o(i,j,k)?a(i,j,k):0.};});value+=amrex::get<0>(data.value());}
    amrex::ParallelDescriptor::ReduceRealSum(value);return value;
}
void Dump(std::ofstream& stream,Field const& field,amrex::Geometry const& g,char const* name,int column){
    for(int row=0;row<int(electric_points.size());++row){auto p=electric_points[row];R value=ReadPoint(field[p.c],p,g);if(amrex::ParallelDescriptor::IOProcessor())stream<<name<<' '<<column<<' '<<row<<' '<<value<<'\n';}
}
void Check(R difference,R bound,char const* label){
    amrex::Print()<<std::setprecision(17)<<"HODGE_IDENTITY "<<label<<" error="<<difference<<" bound="<<bound<<'\n';
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(difference)&&std::isfinite(bound)&&difference<=bound,label);
}
void Run(WarpX& w){
#if !defined(WARPX_DIM_RZ)
    amrex::Abort("R01 source-symbol runtime fixture is RZ only; common primitive 3D compile is separate");
#else
    auto const& g=w.Geom(0);
    AMREX_ALWAYS_ASSERT(g.Domain().length(0)==12&&g.Domain().length(1)==6);
    AMREX_ALWAYS_ASSERT(g.ProbLo(0)==0.&&g.ProbHi(0)==.5&&g.ProbLo(1)==-.5&&g.ProbHi(1)==.5);
    AMREX_ALWAYS_ASSERT(!g.isPeriodic(0)&&!g.isPeriodic(1)&&WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC&&WarpX::field_boundary_lo[1]==FieldBoundaryType::PMC&&WarpX::field_boundary_hi[1]==FieldBoundaryType::PMC);
    bool axis=false,auxiliary=false;amrex::ParmParse fixture("fixture");fixture.query("axis",axis);fixture.query("auxiliary",auxiliary);
    auto old=State(w);auto e=w.m_fields.get_alldirs(FT::Efield_fp,0),b=w.m_fields.get_alldirs(FT::Bfield_fp,0);
    auto cyl=amrex::Geometry(g.Domain(),g.ProbDomain(),1,g.isPeriodic());
    auto options=NativeVacuumSupportOptions(w);NativeInertiaSupport support(cyl,w.boxArray(0),w.DistributionMap(0),options);
    auto& raw=*w.m_fields.get(FT::rho_fp,0);MF rho(raw,amrex::make_alias,0,1);
    AMREX_ALWAYS_ASSERT(support.FreezeEdges(rho,rho));
    amrex::Array<amrex::iMultiFab,3> P,Vmask;
    auto lo=g.Domain().smallEnd(),hi=g.Domain().bigEnd()+amrex::IntVect(1);
    amrex::GpuArray<int,AMREX_SPACEDIM> dl{},dh{};
    for(int q=0;q<AMREX_SPACEDIM;++q){dl[q]=WarpX::field_boundary_lo[q]==FieldBoundaryType::PEC;dh[q]=WarpX::field_boundary_hi[q]==FieldBoundaryType::PEC;}
    for(int c=0;c<3;++c){P[c].define(e[c]->boxArray(),e[c]->DistributionMap(),1,0);Vmask[c].define(e[c]->boxArray(),e[c]->DistributionMap(),1,0);auto type=e[c]->ixType().toIntVect();
        for(amrex::MFIter it(P[c]);it.isValid();++it){auto p=P[c].array(it),v=Vmask[c].array(it);auto physical=support.PhysicalSupport(c).const_array(it);
            amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){amrex::IntVect q(AMREX_D_DECL(i,j,k));bool fixed=c==1&&i==0;for(int d=0;d<AMREX_SPACEDIM;++d)fixed=fixed||(type[d]&&((dl[d]&&q[d]==lo[d])||(dh[d]&&q[d]==hi[d])));p(q)=!fixed&&!axis&&physical(q);v(q)=!fixed&&(axis||!physical(q));});}
    }
    DarwinVacuumHodgePC::Options o;o.b=std::pow(.5*w.getdt(0),2)/(PhysConst::epsilon_0*PhysConst::mu0);o.length=1.;o.max_iterations=2000;o.restart_length=100;o.relative_tolerance=1.e-12;o.absolute_tolerance=0.;o.use_auxiliary_preconditioner=auxiliary;
    DarwinVacuumHodgePC pc(w,e,b,o);std::string error;bool const prepared=pc.Prepare({&P[0],&P[1],&P[2]},{&Vmask[0],&Vmask[1],&Vmask[2]},error);
    amrex::Print()<<"HODGE_PREPARE "<<prepared<<" "<<error<<'\n';AMREX_ALWAYS_ASSERT(prepared);
    auto const& stats=pc.Stats();amrex::Print()<<"HODGE_SUPPORT V="<<stats.vacuum_edges<<" W="<<stats.potential_dofs<<" floating="<<stats.floating_components<<" labels="<<stats.label_iterations<<" real_work_bytes_local="<<stats.local_owned_bytes<<'\n';
    AMREX_ALWAYS_ASSERT(axis?(stats.vacuum_edges==233&&stats.potential_dofs==84&&stats.floating_components==0):(stats.vacuum_edges==100&&stats.potential_dofs==29&&stats.floating_components==1));
    MF scalar(pc.NodeMetric().boxArray(),pc.NodeMetric().DistributionMap(),1,0),canonical(scalar.boxArray(),scalar.DistributionMap(),1,0),adj(scalar.boxArray(),scalar.DistributionMap(),1,0),node_action(scalar.boxArray(),scalar.DistributionMap(),1,0),solution_phi(scalar.boxArray(),scalar.DistributionMap(),1,0);
    auto x=Clone(e),y=Clone(e),gx=Clone(e),kx=Clone(e),ky=Clone(e),action=Clone(e),saved=Clone(e),d=Clone(e),h=Clone(e),solution=Clone(e);
    for(amrex::MFIter it(scalar);it.isValid();++it){auto a=scalar.array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=.125+.015625*i+.0078125*j*j;});}
    pc.CanonicalPotential(canonical,scalar);
    for(int c=0;c<3;++c)for(amrex::MFIter it(x[c]);it.isValid();++it){auto a=x[c].array(it),q=y[c].array(it);auto mask=Vmask[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=mask(i,j,k)?(.125*(c+1)+.015625*i-.03125*j+((i+j)%2?.0625:-.0625)):0.;q(i,j,k)=mask(i,j,k)?(.25+.0078125*i*j+.03125*c):0.;});}
    auto input_bytes=FieldBytes({&x[0],&x[1],&x[2],&y[0],&y[1],&y[2],&scalar,&canonical});
    pc.Gradient(V(gx),canonical);pc.Adjoint(adj,V(x));R lhs=pc.EdgeDot(V(gx),V(x)),rhs=pc.PotentialDot(canonical,adj);R const eps=std::numeric_limits<R>::epsilon();
    Check(std::abs(lhs-rhs),512*eps*(std::sqrt(pc.EdgeDot(V(gx),V(gx))*pc.EdgeDot(V(x),V(x)))+std::sqrt(pc.PotentialDot(canonical,canonical)*pc.PotentialDot(adj,adj))),"weighted_adjoint");
    pc.Curl(V(kx),V(x));pc.Curl(V(ky),V(y));lhs=pc.EdgeDot(V(x),V(ky));rhs=pc.EdgeDot(V(kx),V(y));
    Check(std::abs(lhs-rhs),512*eps*(std::sqrt(pc.EdgeDot(V(x),V(x))*pc.EdgeDot(V(ky),V(ky)))+std::sqrt(pc.EdgeDot(V(y),V(y))*pc.EdgeDot(V(kx),V(kx)))),"weighted_curl_symmetry");
    R const energy=pc.EdgeDot(V(x),V(kx));AMREX_ALWAYS_ASSERT(energy>0.);amrex::Print()<<"HODGE_ENERGY "<<std::setprecision(17)<<energy<<'\n';
    pc.Curl(V(action),V(gx));R diag=0.;for(int q=0;q<AMREX_SPACEDIM;++q)diag+=4./std::pow(g.CellSize(q),2);
    Check(std::sqrt(pc.EdgeDot(V(action),V(action))),512*eps*diag*std::sqrt(pc.EdgeDot(V(gx),V(gx))),"native_gradient_curl_null");
    pc.ApplyAugmented(V(action),node_action,V(x),canonical);Copy(saved,action);pc.ApplyAugmented(V(kx),adj,V(y),scalar);pc.ApplyAugmented(V(action),node_action,V(x),canonical);
    AMREX_ALWAYS_ASSERT(Error(action,saved)==0.);
    AMREX_ALWAYS_ASSERT(input_bytes==FieldBytes({&x[0],&x[1],&x[2],&y[0],&y[1],&y[2],&scalar,&canonical}));
    if(!axis){
    Copy(d,action);Scale(d,-1.);Copy(h,x);Scale(h,o.length*o.length/o.b);
    auto result=pc.Solve(V(solution),solution_phi,V(d),V(h));
    amrex::Print()<<std::setprecision(17)<<"HODGE_SOLVE status="<<result.status<<" converged="<<result.converged<<" finite="<<result.finite<<" iterations="<<result.iterations<<" initial="<<result.initial<<" residual="<<result.residual<<" target="<<result.target<<'\n';
    AMREX_ALWAYS_ASSERT(result.converged);
    for(int c=0;c<3;++c)amrex::VisMF::Write(solution[c],"SOLUTION_PHYSICAL_E_"+std::to_string(c));
    amrex::VisMF::Write(solution_phi,"SOLUTION_PHYSICAL_PHI");
    Scale(solution,o.b/(o.length*o.length));solution_phi.mult(1./o.length,0,1,0);
    pc.ApplyAugmented(V(kx),adj,V(solution),solution_phi);
    for(int c=0;c<3;++c){amrex::VisMF::Write(solution[c],"SOLUTION_U_"+std::to_string(c));amrex::VisMF::Write(kx[c],"ACTION_EDGE_"+std::to_string(c));amrex::VisMF::Write(action[c],"RHS_EDGE_"+std::to_string(c));}
    amrex::VisMF::Write(solution_phi,"SOLUTION_PSI");amrex::VisMF::Write(adj,"ACTION_NODE");amrex::VisMF::Write(node_action,"RHS_NODE");amrex::VisMF::Write(pc.NodeMetric(),"NODE_METRIC");
    for(int c=0;c<3;++c)MF::Subtract(kx[c],action[c],0,0,1,0);MF::Subtract(adj,node_action,0,0,1,0);
    R const fresh=std::sqrt(pc.EdgeDot(V(kx),V(kx))+pc.PotentialDot(adj,adj));
    Check(fresh,result.target,"fresh_augmented_full_rhs");
    MF::Subtract(solution_phi,canonical,0,0,1,0);
    amrex::Print()<<"HODGE_SOLUTION electric_gap="<<Error(solution,x)<<" phi_gap="<<solution_phi.norminf(0)<<'\n';
    }
    std::ofstream stream;if(amrex::ParallelDescriptor::IOProcessor()){stream.open("NATIVE_COLUMNS.txt");stream<<std::setprecision(17);}
    if(axis) {
        for(int col=0;col<int(electric_points.size());++col){Zero(y);auto q=electric_points[col];bool trace=false;for(auto n:trace_indices)trace=trace||n==col;if(trace)continue;PointValue(y,q,1.);pc.Curl(V(ky),V(y));Dump(stream,ky,g,"K_ALL",col);}
    } else {
    for(int col=0;col<int(empty_indices.size());++col){Zero(y);PointValue(y,electric_points[empty_indices[col]],1.);pc.Curl(V(ky),V(y));Dump(stream,ky,g,"K",col);}
    }
    if(!axis)for(int col:charged_coupling_indices){Zero(y);PointValue(y,electric_points[col],1.);pc.Curl(V(ky),V(y));Dump(stream,ky,g,"K_P",col);}
    for(int col=0;col<29;++col){scalar.setVal(0.);for(int row=0;row<int(scalar_points.size());++row){auto p=scalar_points[row];R value=potential_basis[row*29+col];for(amrex::MFIter it(scalar);it.isValid();++it){amrex::IntVect q(AMREX_D_DECL(p.i,p.j,p.k));if(it.validbox().contains(q)){auto a=scalar.array(it);amrex::ParallelFor(1,[=]AMREX_GPU_DEVICE(int){a(q)=value;});}}}pc.Gradient(V(gx),scalar);Dump(stream,gx,g,"G",col);}
    if(axis) {
        scalar.setVal(0.);for(amrex::MFIter it(scalar);it.isValid();++it){amrex::IntVect point(AMREX_D_DECL(0,2,0));if(it.validbox().contains(point)){auto value=scalar.array(it);amrex::ParallelFor(1,[=]AMREX_GPU_DEVICE(int){value(point)=1.;});}}
        pc.Gradient(V(gx),scalar);pc.Curl(V(kx),V(gx));Check(std::sqrt(pc.EdgeDot(V(kx),V(kx))),512*eps*diag*std::sqrt(pc.EdgeDot(V(gx),V(gx))),"axis_gradient_curl_null");Dump(stream,gx,g,"G_AXIS",0);
    }
    if(auxiliary) {
        Field nodes,restricted;
        for(int c=0;c<3;++c){nodes[c].define(scalar.boxArray(),scalar.DistributionMap(),1,0);restricted[c].define(scalar.boxArray(),scalar.DistributionMap(),1,0);
            for(amrex::MFIter it(nodes[c]);it.isValid();++it){auto value=nodes[c].array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){value(i,j,k)=.125*(c+1)+.015625*i+.0078125*j*j;});}}
        pc.ProlongAuxiliary(V(gx),V(nodes));pc.RestrictAuxiliary(V(restricted),V(x));R left=pc.EdgeDot(V(gx),V(x)),right=0.,absolute=0.;
        for(int c=0;c<3;++c){auto owner=nodes[c].OwnerMask(g.periodicity());amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum> op;amrex::ReduceData<R,R> data(op);using Pair=decltype(data)::Type;
            for(amrex::MFIter it(nodes[c]);it.isValid();++it){auto v=nodes[c].const_array(it),r=restricted[c].const_array(it),m=pc.AuxiliaryMetric(c).const_array(it);auto own=owner->const_array(it);
                op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->Pair{R q=own(i,j,k)?v(i,j,k)*r(i,j,k)*m(i,j,k):0.;return {q,std::abs(q)};});}
            auto values=data.value();R local[]{amrex::get<0>(values),amrex::get<1>(values)};amrex::ParallelDescriptor::ReduceRealSum(local,2);right+=local[0];absolute+=local[1];}
        Check(std::abs(left-right),512*eps*(std::abs(left)+absolute),"auxiliary_transfer_adjoint");
    }
    // Rejected re-prepare is transactional; old action/readiness remains usable.
    amrex::Array<amrex::iMultiFab,3> badP,badV;
    for(int c=0;c<3;++c){badP[c].define(P[c].boxArray(),P[c].DistributionMap(),1,0);badV[c].define(P[c].boxArray(),P[c].DistributionMap(),1,0);amrex::iMultiFab::Copy(badP[c],P[c],0,0,1,0);amrex::iMultiFab::Copy(badV[c],Vmask[c],0,0,1,0);badP[c].plus(0,0,1,0);}
    badV[0].setVal(2);AMREX_ALWAYS_ASSERT(!pc.Prepare({&badP[0],&badP[1],&badP[2]},{&badV[0],&badV[1],&badV[2]},error)&&pc.Ready());
    pc.Curl(V(ky),V(x));
    bool equal=old==State(w);amrex::ParallelDescriptor::ReduceBoolAnd(equal);AMREX_ALWAYS_ASSERT(equal);
    auto final=pc.Stats();amrex::Print()<<"HODGE_PRIMITIVE_PASS exact_registry_particles_rng=1 exact_ABA=1 bad_mask_rejected=1 action_calls="<<final.augmented_actions<<" K_calls="<<final.curl_actions<<" G_calls="<<final.gradient_actions<<" GT_calls="<<final.adjoint_actions<<" auxiliary_cycles="<<final.auxiliary_cycles<<" hodge_actions="<<final.hodge_completion_actions<<" linear_iterations="<<final.linear_iterations<<'\n';
    pc.Invalidate();AMREX_ALWAYS_ASSERT(!pc.Ready());
#endif
}
int main(int argc,char** argv){warpx::initialization::initialize_external_libraries(argc,argv);{auto& w=WarpX::GetInstance();w.InitData();Run(w);WarpX::Finalize();}warpx::initialization::finalize_external_libraries();}
