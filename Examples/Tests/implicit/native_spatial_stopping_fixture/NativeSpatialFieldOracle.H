/* Independent periodic Fourier and native metric oracle, fixture only. */
#ifndef NATIVE_SPATIAL_FIELD_ORACLE_H_
#define NATIVE_SPATIAL_FIELD_ORACLE_H_
#include "NativeBytes.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/ImplicitSolvers/DarwinInitialRateSchur.H"
#include <AMReX_Reduce.H>
#include <limits>
inline void SpatialFieldOracle(WarpX& w){
#if defined(WARPX_DIM_3D)
    using F=amrex::MultiFab;using A=std::array<F,3>;using R=amrex::Real;
    auto const& g=w.Geom(0);auto const dx=g.CellSizeArray();R const dv=dx[0]*dx[1]*dx[2];
    auto ev=w.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp,0),bv=w.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
    std::vector<F const*> native;for(auto* p:ev){native.push_back(p);}for(auto* p:bv){native.push_back(p);}
    auto before=ParticleBytes(w);auto fields=FieldBytes(native);A potential,magnetic,current,zero;
    ablastr::fields::VectorField av,b,cv;std::vector<F const*> input;
    for(int c=0;c<3;++c){potential[c].define(ev[c]->boxArray(),ev[c]->DistributionMap(),1,ev[c]->nGrowVect());
        current[c].define(ev[c]->boxArray(),ev[c]->DistributionMap(),1,ev[c]->nGrowVect());
        zero[c].define(ev[c]->boxArray(),ev[c]->DistributionMap(),1,ev[c]->nGrowVect());zero[c].setVal(0.);
        magnetic[c].define(bv[c]->boxArray(),bv[c]->DistributionMap(),1,bv[c]->nGrowVect());
        av[c]=&potential[c];b[c]=&magnetic[c];cv[c]=&current[c];input.push_back(av[c]);}
    auto load=[&](int mode){for(int c=0;c<3;++c){potential[c].setVal(0.);R const dx0=dx[0],L=g.ProbLength(0);
        for(amrex::MFIter mfi(potential[c]);mfi.isValid();++mfi){auto const f=potential[c].array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                if(c==1&&mode==0){f(i,j,k)=std::sin(2.*MathConst::pi*i*dx0/L);}
                if(c==1&&mode==1){f(i,j,k)=1.;}
                if(c==0&&mode==2){f(i,j,k)=2.*std::sin(MathConst::pi*dx0/L)/dx0*std::cos(2.*MathConst::pi*(i+.5)*dx0/L);}
            });}potential[c].OverrideSync(g.periodicity());potential[c].FillBoundary(g.periodicity());}};
    auto curl=[&](){for(int c=0;c<3;++c){magnetic[c].setVal(0.);current[c].setVal(0.);}
        w.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(b,av,w.GetEBUpdateBFlag()[0],0);
        for(auto& f:magnetic){f.OverrideSync(g.periodicity());f.FillBoundary(g.periodicity());}
        w.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(cv,b,w.GetEBUpdateEFlag()[0],0);
        for(auto& f:current){f.OverrideSync(g.periodicity());f.FillBoundary(g.periodicity());}};
    auto dot=[&](F const& a,F const& c){auto owner=a.OwnerMask(g.periodicity());
        amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<R> data(op);using T=typename decltype(data)::Type;
        for(amrex::MFIter mfi(a);mfi.isValid();++mfi){auto const x=a.const_array(mfi),y=c.const_array(mfi);auto const m=owner->const_array(mfi);
            op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T{return{m(i,j,k)?dv*x(i,j,k)*y(i,j,k):0.};});}
        R v=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(v);return v;};
    load(0);auto source=FieldBytes(input);curl();R const lambda=4.*std::pow(std::sin(MathConst::pi/g.Domain().length(0)),2)/(dx[0]*dx[0]*PhysConst::mu0);
    R error=0.,benergy=0.,work=0.;for(int c=0;c<3;++c){F diff(current[c].boxArray(),current[c].DistributionMap(),1,0);
        F::LinComb(diff,1.,current[c],0,-lambda,potential[c],0,0,1,0);error=std::max(error,diff.norm0());
        benergy+=dot(magnetic[c],magnetic[c])/PhysConst::mu0;work+=dot(potential[c],current[c]);}
    R const round=4096.*std::numeric_limits<R>::epsilon();
    AMREX_ALWAYS_ASSERT(error<=round*lambda&&std::abs(benergy-work)<=round*(std::abs(benergy)+std::abs(work))&&FieldBytes(input)==source);
    auto output=FieldBytes({&current[0],&current[1],&current[2],&magnetic[0],&magnetic[1],&magnetic[2]});
    warpx::darwin::InitialRateSchurOptions o;o.compatible_yee=true;o.relative_tolerance=1.e-12;o.output_ghosts=potential[0].nGrowVect();
    for(int d=0;d<3;++d){o.lower[d]=o.upper[d]=warpx::darwin::InitialRateBoundary::Periodic;}
    warpx::darwin::DarwinInitialRateSchur projection(g,w.boxArray(0),w.DistributionMap(0),o);
    F unit(amrex::convert(w.boxArray(0),amrex::IntVect(1)),w.DistributionMap(0),1,1);unit.setVal(1.);AMREX_ALWAYS_ASSERT(projection.Freeze(unit));
    R projection_error=0.;
    for(int mode=0;mode<3;++mode){load(mode);auto in=FieldBytes(input);auto result=projection.Correct({av[0],av[1],av[2]},{&zero[0],&zero[1],&zero[2]});AMREX_ALWAYS_ASSERT(result.converged);
        for(int c=0;c<3;++c){F diff(current[c].boxArray(),current[c].DistributionMap(),1,0);
            F::LinComb(diff,1.,*projection.CorrectionField()[c],0,mode==2?-1.:0.,potential[c],0,0,1,0);
            R const normalized=diff.norm0()/std::max(R(1),potential[c].norm0());projection_error=std::max(projection_error,normalized);}
        AMREX_ALWAYS_ASSERT(FieldBytes(input)==in);
    }
    AMREX_ALWAYS_ASSERT(projection_error<=32.e-12+round);
    load(0);curl();AMREX_ALWAYS_ASSERT(FieldBytes({&current[0],&current[1],&current[2],&magnetic[0],&magnetic[1],&magnetic[2]})==output);
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==before&&FieldBytes(native)==fields);
    amrex::Print()<<std::setprecision(17)<<"NATIVE_SPATIAL_FIELD_ORACLE_PASS eigen_error="<<error<<" eigen_bound="<<round*lambda<<" curl_work="<<benergy-work<<" curl_bound="<<round*(std::abs(benergy)+std::abs(work))<<" projection_error="<<projection_error<<" purity=exact ABA=exact\n";
#endif
}
#endif
