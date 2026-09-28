/* Native stored-base continuity regression; no particle advance. */
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "FieldSolver/ImplicitSolvers/MassMatrixDensityProjection.H"
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <iomanip>
#include <limits>
using MF=amrex::MultiFab;
using warpx::fields::FieldType;
using ablastr::fields::Direction;
amrex::Real Difference(MF const&a,MF const&b){MF f(a.boxArray(),a.DistributionMap(),1,a.nGrowVect());MF::LinComb(f,1.,a,0,-1.,b,0,0,1,a.nGrowVect());return f.norminf();}
int main(int argc,char**argv){warpx::initialization::initialize_external_libraries(argc,argv);{
 auto&w=WarpX::GetInstance();w.InitData();w.HybridPICInitializeRhoJandB();auto&reg=w.m_fields;auto const jp=reg.get_alldirs(FieldType::current_fp,0);auto const&rho=*reg.get(FieldType::rho_fp,0);int const component=0; // InitData stores accepted rho here; midpoint component is not deposited yet.
 MassMatrixDensityProjection projection;projection.Capture(rho,component,jp);
 std::array<MF,3> work,delta;ablastr::fields::VectorField view{};for(int c=0;c<3;++c){work[c].define(jp[c]->boxArray(),jp[c]->DistributionMap(),1,jp[c]->nGrowVect());delta[c].define(jp[c]->boxArray(),jp[c]->DistributionMap(),1,jp[c]->nGrowVect());work[c].setVal(0.);delta[c].setVal(0.);view[c]=&work[c];}
 MF base(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect()),out(base.boxArray(),base.DistributionMap(),1,base.nGrowVect()),expected(base.boxArray(),base.DistributionMap(),1,base.nGrowVect()),first(base.boxArray(),base.DistributionMap(),1,base.nGrowVect()),div(base.boxArray(),base.DistributionMap(),1,0);MF::Copy(base,rho,component,0,1,base.nGrowVect());AMREX_ALWAYS_ASSERT(base.norminf()>.1);projection.CaptureCurrentIncrementAndCompose(view);for(int c=0;c<3;++c)AMREX_ALWAYS_ASSERT(Difference(work[c],*jp[c])==0.);projection.ApplyCurrentIncrement(w,0,out,0,1.);AMREX_ALWAYS_ASSERT(Difference(out,base)==0.);
 auto const dx=w.Geom(0).CellSizeArray(),lo=w.Geom(0).ProbLoArray();auto const type=delta[2].ixType().toIntVect();for(amrex::MFIter m(delta[2]);m.isValid();++m){auto a=delta[2].array(m);amrex::ParallelFor(m.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){amrex::Real const z=lo[1]+(j+.5*(1-type[1]))*dx[1];a(i,j,k)=1.e-14*std::sin(2.*3.141592653589793*z);});}
 auto evaluate=[&](amrex::Real scale){for(int c=0;c<3;++c){MF::Copy(work[c],delta[c],0,0,1,work[c].nGrowVect());work[c].mult(scale,0,1,work[c].nGrow());}projection.CaptureCurrentIncrementAndCompose(view);projection.ApplyCurrentIncrement(w,0,out,0,1.);};
 evaluate(1.);MF::Copy(first,out,0,0,1,out.nGrowVect());ablastr::fields::VectorField retained{&delta[0],&delta[1],&delta[2]};AMREX_ALWAYS_ASSERT(projection.CurrentIncrementReady());AMREX_ALWAYS_ASSERT(Difference(work[2],*jp[2])==0.);MassMatrixDensityProjection::ComputeDivergence(w,0,retained,div);MF::LinComb(expected,1.,base,0,-1.,div,0,0,1,0);
 amrex::Real const unprojected_oracle_error=Difference(out,expected);
 // The independent native-divergence oracle must honor the represented
 // nonnegative charge mask and choose one periodic/nodal image before adding.
 for(amrex::MFIter m(div);m.isValid();++m){auto a=div.array(m);auto b=base.const_array(m);amrex::ParallelFor(m.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=b(i,j,k)>0?std::max(-b(i,j,k),-a(i,j,k)):0.;});}
 div.OverrideSync(w.Geom(0).periodicity());MF::LinComb(expected,1.,base,0,1.,div,0,0,1,0);
 amrex::Print()<<std::setprecision(17)<<"DENSITY_ORACLE unprojected_error="<<unprojected_oracle_error<<" native_projected_error="<<Difference(out,expected)<<"\n";
 AMREX_ALWAYS_ASSERT(Difference(out,expected)==0.);amrex::Real const change=Difference(out,base);AMREX_ALWAYS_ASSERT(change>1.e-14);
 evaluate(-2.);evaluate(1.);AMREX_ALWAYS_ASSERT(Difference(out,first)==0.);projection.InvalidateCurrentIncrement();AMREX_ALWAYS_ASSERT(!projection.CurrentIncrementReady());bool stale=false;amrex::ParmParse("density_test").query("stale",stale);if(stale)projection.ApplyCurrentIncrement(w,0,out,0,1.);projection.Capture(rho,component,jp);AMREX_ALWAYS_ASSERT(!projection.CurrentIncrementReady());evaluate(1.);AMREX_ALWAYS_ASSERT(Difference(out,first)==0.);
 amrex::Print()<<std::setprecision(17)<<"DENSITY_INCREMENT_PASS zero_base_exact=1 represented_current_change=0 retained_current="<<delta[2].norminf()<<" recovered_density_change="<<change<<" ABA_exact=1 rebase_invalidates=1 no_advance=1\n";
 bool nonfinite=false;amrex::ParmParse("density_test").query("nonfinite",nonfinite);
 if(nonfinite){for(int c=0;c<3;++c)work[c].setVal(c==2?std::numeric_limits<amrex::Real>::quiet_NaN():0.);projection.CaptureCurrentIncrementAndCompose(view);projection.ApplyCurrentIncrement(w,0,out,0,1.);bool propagated=!out.is_finite(0,1,0);amrex::Print()<<"DENSITY_NONFINITE propagated="<<propagated<<"\n";AMREX_ALWAYS_ASSERT_WITH_MESSAGE(propagated,"Nonfinite MM increments must reach residual rejection");}
 WarpX::Finalize();}warpx::initialization::finalize_external_libraries();}
