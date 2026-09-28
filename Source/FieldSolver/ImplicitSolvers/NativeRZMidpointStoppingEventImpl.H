/* Internal implementation, included only by NativeRZSpatialStoppingEvent.cpp.
 * Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
namespace {
constexpr char const* midpoint_energy_name="hybrid_electron_energy_fp";
R MidpointStepBound(NativeRZSpatialStoppingEvent::Impl const& p,
    WarpXSolverVec const& state,WarpXSolverVec const& direction)
{
    auto const& x=state.getMultiFabBlock(midpoint_energy_name,0);
    auto const& v=direction.getMultiFabBlock(midpoint_energy_name,0);
    amrex::ReduceOps<amrex::ReduceOpMin> op;amrex::ReduceData<R> data(op);
    using T=typename decltype(data)::Type;
    for(amrex::MFIter mfi(x);mfi.isValid();++mfi){
        auto const a=x.const_array(mfi),d=v.const_array(mfi),u=p.u0.const_array(mfi);
        op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
            R const old=u(i,j,k),end=old+a(i,j,k),step=d(i,j,k);
            if(!std::isfinite(end)||!std::isfinite(step)||old<=0.||end<=0.)return {R(0)};
            return {step<0.?amrex::min(R(1),.99*end/(-step)):R(1)};
        });
    }
    R result=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealMin(result);
    return std::isfinite(result)?amrex::max(R(0),amrex::min(R(1),result)):R(0);
}
struct MidpointPhysical final:ThermalNonlinearOperator {
    NativeRZSpatialStoppingEvent::Impl& p;
    explicit MidpointPhysical(NativeRZSpatialStoppingEvent::Impl& x):p(x){}
    bool Residual(WarpXSolverVec& out,WarpXSolverVec const& x,int,bool) override {
        return p.MidpointResidual(out,x);
    }
    R StepBound(WarpXSolverVec const& x,WarpXSolverVec const& v)const override {
        return MidpointStepBound(p,x,v);
    }
    bool Freeze(WarpXSolverVec const& x,int,bool use_pc,ThermalSolveResult& statistics) override {
        if(!use_pc)return true;
        ++statistics.residual_evaluations;return p.FreezeMidpointPC(x);
    }
    bool Precondition(WarpXSolverVec& out,WarpXSolverVec const& rhs) override {
        return p.ApplyMidpointPC(out,rhs);
    }
    void RestoreInput(WarpXSolverVec const&) override {p.ready=false;}
};
}

bool NativeRZSpatialStoppingEvent::Impl::InitializeMidpoint()
{
    midpoint_receipt={};midpoint_captured=false;
    auto const& o=midpoint_options.nonlinear;
    // Field reductions always execute, even with rank-local invalid options.
    bool const finite_energy=u0.is_finite();R const minimum_energy=u0.min(0);
    bool valid=std::isfinite(o.absolute_tolerance)&&o.absolute_tolerance>0.&&
        o.absolute_tolerance<=1.e-12&&o.relative_tolerance==0.&&
        std::isfinite(o.linear_relative_tolerance)&&o.linear_relative_tolerance>0.&&
        o.linear_relative_tolerance<=1.e-5&&o.probe_relative_size==1.e-5&&
        !o.adaptive_forcing&&o.max_newton_iterations==32&&o.max_linear_iterations==200&&
        o.restart_length==60&&o.max_backtracks==10&&
        o.block_relative_tolerances.empty()&&o.block_absolute_tolerances.empty()&&
        o.block_reference_norms.empty()&&finite_energy&&minimum_energy>0.&&
        (!midpoint_options.analytic_temperature_column||o.use_preconditioner)&&
        (midpoint_options.initial_guess==NativeStoppingThermalOptions::InitialGuess::Zero||
         midpoint_options.initial_guess==NativeStoppingThermalOptions::InitialGuess::PrivateHalfStep);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid)return Reject("midpoint source requires positive U and fixed 1e-12-or-tighter absolute block targets");
    R low[6]={o.absolute_tolerance,o.linear_relative_tolerance,R(o.use_preconditioner),
        R(o.probe_rhs_scale_floor),R(midpoint_options.initial_guess==NativeStoppingThermalOptions::InitialGuess::PrivateHalfStep),
        R(midpoint_options.analytic_temperature_column)};
    R high[6];std::copy(low,low+6,high);
    amrex::ParallelDescriptor::ReduceRealMin(low,6);amrex::ParallelDescriptor::ReduceRealMax(high,6);
    for(int i=0;i<6;++i)valid=valid&&low[i]==high[i];
    if(!Collective(valid))return Reject("midpoint source options differ across ranks");
    // Circuit ON needs its own tangent/root qualification; preserve the
    // established circuit path when this numerical option is disabled.
    if(!Collective(!midpoint_options.analytic_temperature_column||!circuit))
        return Reject("analytic temperature PC with circuit is not qualified");
    R mass_per_charge=0.;for(auto const& s:species)if(s.charge>0.)mass_per_charge=std::max(mass_per_charge,s.mass/s.charge);
    midpoint_energy_scale=u0.norm0()+.5*reference_rho*mass_per_charge*
        options.proper_speed_cap*options.proper_speed_cap;
    if(!std::isfinite(midpoint_energy_scale)||midpoint_energy_scale<=0.)return Reject("midpoint source energy scale");
    midpoint_receipt.energy_scale=midpoint_energy_scale;
    ThermalMomentOptions mo;
    mo.number_density_floor=context.Options().number_density_floor;
    mo.active_density_floor=mo.number_density_floor;
    mo.gamma=sim.get_pointer_HybridPICModel()->m_gamma;
    mo.nodal_ghosts=t0.nGrow();
#if defined(WARPX_DIM_RZ)
    mo.verboncoeur_axis_correction=sim.UseVerboncoeurAxisCorrection();
#endif
    mo.boundary[0]={MomentBoundary::Axis,MomentBoundary::PEC};
    mo.current_boundary[0]={MomentBoundary::Axis,MomentBoundary::PMC};
    auto const z=context.Geometry().isPeriodic(1)?MomentBoundary::Periodic:MomentBoundary::PMC;
    mo.boundary[1]={z,z};
    midpoint_moments=std::make_unique<KineticThermalMoments>(context.Geometry(),context.Cells(),context.Distribution(),mo);
    for(auto* a:{&midpoint_energy,&midpoint_increment,&midpoint_heat,&midpoint_residual}){
        a->define(u0.boxArray(),u0.DistributionMap(),1,u0.nGrowVect());a->setVal(0.);
    }
    std::vector<WarpXSolverVec::MultiFabBlockSpec> specs;
    for(int c=0;c<3;++c){
        midpoint_current_names[c]=std::string(NativeAcceptedStoppingContext::AcceptedCurrentName)+
            "[dir="+static_cast<std::string>(ablastr::fields::Direction{c})+"]";
        valid=valid&&sim.m_fields.has(midpoint_current_names[c],0);
    }
    if(!Collective(valid))return Reject("midpoint current layout templates unavailable");
    for(int c=0;c<3;++c){
        valid=valid&&sim.m_fields.get(midpoint_current_names[c],0)==binding.electron_current[c];
        specs.push_back({midpoint_current_names[c],current_scale});
    }
    valid=valid&&sim.m_fields.get(midpoint_energy_name,0)==&energy;
    if(!Collective(valid))return Reject("midpoint source layout ownership mismatch");
    specs.push_back({midpoint_energy_name,midpoint_energy_scale});
    midpoint_solution.Define(&sim,"Efield_fp","none",specs,psi_scale);
    midpoint_solution.zero();midpoint_trial=Vector();midpoint_spatial_residual=Vector();
    if(o.use_preconditioner){
        for(auto* v:{&midpoint_pc_base,&midpoint_pc_direction,&midpoint_pc_plus,&midpoint_pc_minus,
            &midpoint_pc_fplus,&midpoint_pc_fminus,&midpoint_pc_response}){
            v->Define(midpoint_solution);v->zero();
        }
    }
    if(midpoint_options.analytic_temperature_column&&!InitializeMidpointTemperaturePC())return false;
    midpoint_pc_frozen=false;midpoint_captured=true;return true;
}

bool NativeRZSpatialStoppingEvent::Impl::DefineMidpointState(WarpXSolverVec& x)const
{
    if(circuit&&!Collective(midpoint_captured&&!committed&&!closed&&
        !x.IsDefined()&&CircuitContextMatches()))return false;
    if(!midpoint_captured||x.IsDefined())return false;
    x.Define(midpoint_solution);x.zero();return true;
}

bool NativeRZSpatialStoppingEvent::Impl::MidpointResidual(WarpXSolverVec& out,WarpXSolverVec const& x)
{
    if(circuit&&!Collective(midpoint_captured&&!committed&&!closed&&
        x.IsDefined()&&out.IsDefined()&&CircuitContextMatches()))
        return Reject("stale circuit source residual context");
    if(!midpoint_captured||committed||closed||!x.IsDefined()||!out.IsDefined())return false;
    x.assertSameType(midpoint_solution);out.assertSameType(midpoint_solution);
    auto const& increment=x.getMultiFabBlock(midpoint_energy_name,0);
    bool finite=increment.is_finite();
    for(int c=0;c<3;++c){
        // Enter every field collective even when a different local block is bad.
        bool const fp=x.getArrayVec()[0][c]->is_finite();
        bool const fj=x.getMultiFabBlock(midpoint_current_names[c],0).is_finite();
        finite=finite&&fp&&fj;
    }
    if(!Collective(finite))return false;
    Copy(midpoint_increment,increment);Copy(midpoint_energy,u0);Copy(u1,u0);
    amrex::MultiFab::Saxpy(midpoint_energy,.5,increment,0,0,1,0);
    amrex::MultiFab::Add(u1,increment,0,0,1,0);
    u1.FillBoundary(context.Geometry().periodicity());
    if(!Collective(u1.is_finite()&&u1.min(0)>0.))return false;
    KineticThermalStateView thermal{rho0};thermal.energy=&midpoint_energy;
    if(!midpoint_moments->Evaluate(thermal))return false;
    for(int c=0;c<3;++c){
        amrex::MultiFab::Copy(midpoint_trial.a[c],*x.getArrayVec()[0][c],0,0,1,0);
        midpoint_trial.a[c].mult(1./psi_scale,0,1,0);
        amrex::MultiFab::Copy(midpoint_trial.a[c+3],x.getMultiFabBlock(midpoint_current_names[c],0),0,0,1,0);
        midpoint_trial.a[c+3].mult(1./current_scale,0,1,0);
    }
    if(!Evaluate(midpoint_trial,midpoint_spatial_residual))return false;
    ledger.spatial_residual=VectorMax(midpoint_spatial_residual);
    for(int c=0;c<3;++c){
        amrex::MultiFab::Copy(*out.getArrayVec()[0][c],midpoint_spatial_residual.a[c],0,0,1,0);
        out.getArrayVec()[0][c]->mult(psi_scale,0,1,0);
        auto& mean_out=out.getMultiFabBlock(midpoint_current_names[c],0);
        amrex::MultiFab::Copy(mean_out,midpoint_spatial_residual.a[c+3],0,0,1,0);
        mean_out.mult(current_scale,0,1,0);
    }
    heat_map->RestrictNodalScalar(heat,0,midpoint_heat);
    amrex::MultiFab::LinComb(midpoint_residual,1.,increment,0,-1.,midpoint_heat,0,0,1,0);
    amrex::MultiFab::Copy(out.getMultiFabBlock(midpoint_energy_name,0),midpoint_residual,0,0,1,0);
    midpoint_receipt.maximum_residual=midpoint_residual.norm0()/midpoint_energy_scale;
    return Collective(midpoint_residual.is_finite()&&std::isfinite(midpoint_receipt.maximum_residual));
}

bool NativeRZSpatialStoppingEvent::Impl::FreezeMidpointPC(WarpXSolverVec const& base)
{
    midpoint_pc_frozen=false;
    midpoint_pc_temperature_frozen=false;
    if(midpoint_options.analytic_temperature_column&&!MidpointTemperatureContextValid(false))
        return Reject("analytic temperature PC context changed");
    if(!midpoint_captured||committed||closed||!midpoint_options.nonlinear.use_preconditioner)return false;
    midpoint_pc_base.Copy(base);
    if(!MidpointResidual(midpoint_pc_response,base))return false;
    amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> op;
    amrex::ReduceData<R,R,R> data(op);using Tuple=typename decltype(data)::Type;
    R const dt=options.interval;
    for(auto const& s:species){if(s.charge==0.)continue;
        for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
            auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());auto const* point=s.points.at(key).data();
            bool const fast=s.fast;
            op.eval(pti.numParticles(),data,[=] AMREX_GPU_DEVICE(long n)->Tuple{
                auto const p=point[n];R const stop=fast?-std::expm1(-p.rate*dt):0.;
                return {p.weight*p.charge*stop,p.weight*p.mass*stop,
                    p.weight*p.charge*p.charge/p.mass*(1.-.5*stop)};
            });
        }
    }
    auto const sums_pc=data.value();R sums[3]={amrex::get<0>(sums_pc),amrex::get<1>(sums_pc),amrex::get<2>(sums_pc)};
    amrex::ParallelDescriptor::ReduceRealSum(sums,3);
    auto& receipt=midpoint_receipt;
    receipt.pc_beta=sums[0]/(reference_rho*volume);
    receipt.pc_electron_drag=PhysConst::q_e*sums[1]/(PhysConst::m_e*reference_rho*volume);
    receipt.pc_ion_response=PhysConst::m_e*sums[2]/(PhysConst::q_e*reference_rho*volume);
    R const beta=receipt.pc_beta,A=receipt.pc_electron_drag,zeta=receipt.pc_ion_response;
    R const c=1.-.5*beta,d=1.+.5*A,det=c*c+zeta*d;
    bool const valid=std::isfinite(beta)&&std::isfinite(A)&&std::isfinite(zeta)&&
        beta>=0.&&A>=0.&&zeta>0.&&c>0.&&std::isfinite(det)&&det>0.;
    if(!Collective(valid))return false;
    if(midpoint_options.analytic_temperature_column&&!FreezeMidpointTemperaturePC())return false;
    ++receipt.pc_freezes;midpoint_pc_frozen=true;return true;
}
bool NativeRZSpatialStoppingEvent::Impl::ApplyMidpointPC(WarpXSolverVec& out,WarpXSolverVec const& rhs)
{
    if(midpoint_options.analytic_temperature_column&&!MidpointTemperatureContextValid(true)){
        midpoint_pc_temperature_frozen=false;midpoint_pc_frozen=false;
        return Reject("analytic temperature PC context changed");
    }
    if(!midpoint_pc_frozen||committed||closed)return false;
    ++midpoint_receipt.pc_applications;
    rhs.assertSameType(midpoint_solution);out.assertSameType(midpoint_solution);
    midpoint_pc_direction.zero();
    auto const& rhs_u=rhs.getMultiFabBlock(midpoint_energy_name,0);
    amrex::MultiFab::Copy(midpoint_pc_direction.getMultiFabBlock(midpoint_energy_name,0),rhs_u,0,0,1,0);
    R const unorm=midpoint_pc_direction.blockNorms().back();
    midpoint_pc_response.zero();
    if(!std::isfinite(unorm))return false;
    if(unorm>0.&&midpoint_options.analytic_temperature_column){
        if(!ApplyMidpointTemperatureColumn(rhs_u)){
            midpoint_pc_temperature_frozen=false;midpoint_pc_frozen=false;
            return Reject("analytic temperature PC action declined");
        }
    }else if(unorm>0.){
        R step=midpoint_options.nonlinear.probe_relative_size*
            std::max(R(1),midpoint_pc_base.blockNorms().back())/unorm;
        midpoint_pc_plus.Copy(midpoint_pc_direction);midpoint_pc_plus.scale(step);
        midpoint_pc_minus.Copy(midpoint_pc_plus);midpoint_pc_minus.scale(-1.);
        R const bound=std::min(MidpointStepBound(*this,midpoint_pc_base,midpoint_pc_plus),
            MidpointStepBound(*this,midpoint_pc_base,midpoint_pc_minus));
        if(!std::isfinite(bound)||bound<=0.)return false;
        if(bound<1.)step*=.5*bound;
        if(!std::isfinite(step)||step<=0.)return false;
        midpoint_pc_plus.linComb(1.,midpoint_pc_base,step,midpoint_pc_direction);
        midpoint_pc_minus.linComb(1.,midpoint_pc_base,-step,midpoint_pc_direction);
        if(!MidpointResidual(midpoint_pc_fplus,midpoint_pc_plus)||
           !MidpointResidual(midpoint_pc_fminus,midpoint_pc_minus))return false;
        midpoint_pc_response.linComb(.5/step,midpoint_pc_fplus,-.5/step,midpoint_pc_fminus);
        midpoint_receipt.pc_temperature_probes+=2;
    }
    // The native field equation has an identity longitudinal row and separate
    // transverse current/curl row. Never apply the stiff transverse inverse
    // to P_L. Use the existing boundary-compatible unit-kappa projector on
    // private right-hand sides; physical variable mass/curl remain in Jv.
    R const ps=psi_scale,js=current_scale;
    for(int component=0;component<3;++component){
        auto& rf=*midpoint_pc_direction.getArrayVec()[0][component];
        auto& rj=midpoint_pc_direction.getMultiFabBlock(midpoint_current_names[component],0);
        amrex::MultiFab::LinComb(rf,1./ps,*rhs.getArrayVec()[0][component],0,
            -1./ps,*midpoint_pc_response.getArrayVec()[0][component],0,0,1,0);
        amrex::MultiFab::LinComb(rj,1./js,rhs.getMultiFabBlock(midpoint_current_names[component],0),0,
            -1./js,midpoint_pc_response.getMultiFabBlock(midpoint_current_names[component],0),0,0,1,0);
        rf.OverrideSync(context.Geometry().periodicity());rf.FillBoundary(context.Geometry().periodicity());
        rj.OverrideSync(context.Geometry().periodicity());rj.FillBoundary(context.Geometry().periodicity());
    }
    for(int block=0;block<2;++block){
        CV input{};
        for(int component=0;component<3;++component)input[component]=block==0?
            midpoint_pc_direction.getArrayVec()[0][component]:
            &midpoint_pc_direction.getMultiFabBlock(midpoint_current_names[component],0);
        auto const result=projection->Correct(input,Const(zero));
        ++midpoint_receipt.pc_projection_calls;
        midpoint_receipt.pc_projection_iterations+=result.iterations;
        midpoint_receipt.pc_projection_residual=std::max(midpoint_receipt.pc_projection_residual,result.residual);
        if(!result.converged)return false;
        auto const correction=projection->CorrectionField();
        auto& target=block==0?midpoint_pc_plus:midpoint_pc_minus;
        for(int component=0;component<3;++component)
            amrex::MultiFab::Copy(*target.getArrayVec()[0][component],*correction[component],0,0,1,0);
    }
    out.Copy(rhs);
    R const beta=midpoint_receipt.pc_beta,A=midpoint_receipt.pc_electron_drag,zeta=midpoint_receipt.pc_ion_response;
    R const c=1.-.5*beta,d=1.+.5*A,b=A+beta,s=(c*c+zeta*d)/d;
    // k=1 is the frozen reference-mass approximation, not a change to the
    // actual edge mass in the residual or physical endpoint/work maps.
    if(!Collective(std::isfinite(s)&&s>0.&&std::isfinite(d)&&d>0.))return false;
    int const high=context.Geometry().Domain().bigEnd(0)+1;
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(int component=0;component<3;++component){
        auto& mean_out=out.getMultiFabBlock(midpoint_current_names[component],0);
        for(amrex::MFIter mfi(*out.getArrayVec()[0][component]);mfi.isValid();++mfi){
            auto const p=out.getArrayVec()[0][component]->array(mfi),m=mean_out.array(mfi);
            auto const rp=midpoint_pc_direction.getArrayVec()[0][component]->const_array(mfi);
            auto const rm=midpoint_pc_direction.getMultiFabBlock(midpoint_current_names[component],0).const_array(mfi);
            auto const lp=midpoint_pc_plus.getArrayVec()[0][component]->const_array(mfi);
            auto const lm=midpoint_pc_minus.getArrayVec()[0][component]->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                bool const axis=component==1&&i==0,conductor=component!=0&&i==high;
                R const pv=(axis||conductor)?rp(i,j,k):lp(i,j,k)+
                    (rp(i,j,k)-lp(i,j,k)+(b/d)*(rm(i,j,k)-lm(i,j,k)))/s;
                R const mv=axis?rm(i,j,k):(rm(i,j,k)+.5*c*pv)/d;
                if(!std::isfinite(pv)||!std::isfinite(mv)){amrex::HostDevice::Atomic::Add(invalid,1);p(i,j,k)=0.;m(i,j,k)=0.;return;}
                p(i,j,k)=ps*pv;m(i,j,k)=js*mv;
            });
        }
    }
    amrex::Gpu::synchronize();if(!Collective(bad.dataValue()==0))return false;
    bool finite=true;
    for(int component=0;component<3;++component){bool const fp=out.getArrayVec()[0][component]->is_finite();
        bool const fm=out.getMultiFabBlock(midpoint_current_names[component],0).is_finite();finite=finite&&fp&&fm;}
    return Collective(finite);
}

bool NativeRZSpatialStoppingEvent::Impl::SolveMidpoint()
{
    // A pure residual/ABA diagnostic may precede a new solve on the same
    // captured physical origin. Work() contains additive background receipts;
    // discard only those result accumulators, never the captured constraints.
    ledger.magnetic_initial_energy=0.;ledger.evaluations=0.;ledger.projection_residual=0.;
    if(circuit){circuit->evaluations=0;circuit->work={};}
    background_ledger.magnetic_self_work=0.;background_ledger.magnetic_cross_work=0.;
    background_ledger.magnetic_cross_scale=0.;background_ledger.initial_constraint_work=0.;
    background_ledger.initial_displacement_work=0.;background_ledger.initial_conductor_work=0.;
    midpoint_receipt.predictor_preparations=0;midpoint_receipt.predictor_failures=0;
    midpoint_receipt.pc_freezes=0;midpoint_receipt.pc_temperature_probes=0;
    midpoint_receipt.pc_temperature_actions=0;midpoint_receipt.pc_temperature_projection_calls=0;
    midpoint_receipt.pc_temperature_projection_iterations=0;
    midpoint_receipt.pc_temperature_freeze_point_visits=0;midpoint_receipt.pc_temperature_action_point_visits=0;
    midpoint_receipt.pc_temperature_projection_residual=0.;midpoint_pc_temperature_frozen=false;
    midpoint_receipt.pc_projection_calls=0;midpoint_receipt.pc_projection_iterations=0;midpoint_receipt.pc_applications=0;
    midpoint_receipt.physical_projection_calls=0;midpoint_receipt.physical_projection_iterations=0;
    midpoint_receipt.predictor_native_evaluations=0;midpoint_receipt.predictor_linear_iterations=0;
    midpoint_receipt.pc_projection_residual=0.;midpoint_pc_frozen=false;
    midpoint_solution.zero();
    if(midpoint_options.initial_guess==NativeStoppingThermalOptions::InitialGuess::PrivateHalfStep){
        ++midpoint_receipt.predictor_preparations;
        auto half=options;half.interval*=.5;
        std::unique_ptr<NativeRZSpatialStoppingEvent> guess;
        if(background)guess=std::make_unique<NativeRZSpatialStoppingEvent>(sim,context,temperature,energy,
            half,destinations,background_destinations);
        else guess=std::make_unique<NativeRZSpatialStoppingEvent>(sim,context,temperature,energy,half,destinations);
        if(guess->Prepare()){
            for(int c=0;c<3;++c){
                amrex::MultiFab::Copy(*midpoint_solution.getArrayVec()[0][c],*guess->ElectricImpulse()[c],0,0,1,0);
                midpoint_solution.getArrayVec()[0][c]->mult(2.,0,1,0);
                amrex::MultiFab::LinComb(midpoint_solution.getMultiFabBlock(midpoint_current_names[c],0),
                    1.,*guess->CandidateElectronCurrent()[c],0,-1.,j0[c],0,0,1,0);
            }
            amrex::MultiFab::LinComb(midpoint_solution.getMultiFabBlock(midpoint_energy_name,0),
                2.,guess->CandidateEnergy(),0,-2.,u0,0,0,1,0);
        }else{++midpoint_receipt.predictor_failures;}
        midpoint_receipt.predictor_native_evaluations=guess->Ledger().evaluations;
        midpoint_receipt.predictor_linear_iterations=guess->Ledger().linear_iterations;
        if(!Unchanged(false))return Reject("private midpoint predictor mutated input");
    }
    MidpointPhysical physical(*this);
    midpoint_receipt.nonlinear=SolveThermalSystem(physical,midpoint_solution,midpoint_options.nonlinear);
    auto const& n=midpoint_receipt.nonlinear;
    ledger.iterations=n.newton_iterations;ledger.linear_iterations=n.linear_iterations;
    if(n.status!=ThermalSolveStatus::Converged||!n.final_evaluation_attempted||
       !n.final_evaluation_valid||!n.final_residual_finite)return Reject("midpoint source nonlinear solve");
    if(ledger.spatial_residual>1.e-12||midpoint_receipt.maximum_residual>1.e-12)
        return Reject("midpoint source native maximum row");
    return true;
}

bool NativeRZSpatialStoppingEvent::Impl::MidpointWork()
{
    auto const& g=context.Geometry();auto& r=midpoint_receipt;
    r.signed_residual=Integral(midpoint_residual,g);
    amrex::MultiFab tmp(u0.boxArray(),u0.DistributionMap(),1,0);
    amrex::MultiFab::Copy(tmp,midpoint_residual,0,0,1,0);tmp.abs(0,1,0);
    r.l1_residual=Integral(tmp,g);
    r.restricted_heat=Integral(midpoint_heat,g);
    amrex::MultiFab::LinComb(tmp,1.,u1,0,-1.,u0,0,0,1,0);
    r.delivered_heat=Integral(tmp,g);
    R const increment=Integral(midpoint_increment,g);
    r.heat_map_error=r.restricted_heat-ledger.heat;
    r.storage_error=r.delivered_heat-increment;
    r.residual_identity_error=r.delivered_heat-r.restricted_heat-r.signed_residual-r.storage_error;
    r.residual_bound=std::nextafter(volume*midpoint_energy_scale*1.e-12,std::numeric_limits<R>::infinity());
    R positive=0.;
    for(auto const* q:{&u0,&u1,&midpoint_increment,&midpoint_heat,&midpoint_residual}){
        amrex::MultiFab::Copy(tmp,*q,0,0,1,0);tmp.abs(0,1,0);positive+=Integral(tmp,g);
    }
    R const count=static_cast<R>(context.Geometry().Domain().numPts())+ledger.contribution_count;
    R const operations=64.+16.*count;
    if(!std::isfinite(positive)||!std::isfinite(operations)||operations*epsilon>=.01)
        return Reject("midpoint thermal arithmetic scope");
    r.roundoff_bound=std::nextafter(4.*Roundoff(operations)*positive+
        operations*std::numeric_limits<R>::denorm_min(),std::numeric_limits<R>::infinity());
    bool const valid=std::isfinite(r.delivered_heat)&&std::isfinite(r.signed_residual)&&
        std::isfinite(r.residual_identity_error)&&std::isfinite(r.roundoff_bound)&&
        r.l1_residual<=r.residual_bound+r.roundoff_bound&&
        std::abs(r.residual_identity_error)<=r.roundoff_bound&&
        std::abs(r.delivered_heat-ledger.heat-r.heat_map_error-r.storage_error-r.signed_residual)<=r.roundoff_bound;
    return Collective(valid)?true:Reject("midpoint thermal residual-to-ledger identity");
}

struct NativeRZMidpointStoppingEvent::Impl {
    std::unique_ptr<NativeRZSpatialStoppingEvent::Impl> physical;
};
NativeRZMidpointStoppingEvent::NativeRZMidpointStoppingEvent(WarpX& w,NativeAcceptedStoppingContext const& c,
    amrex::MultiFab const& t,amrex::MultiFab& u,SpatialStoppingOptions const& o,SpatialStoppingFields const& f,
    NativeStoppingThermalOptions const& options):m_impl(std::make_unique<Impl>())
{
    m_impl->physical=std::make_unique<NativeRZSpatialStoppingEvent::Impl>(w,c,t,u,o,f);
    m_impl->physical->midpoint_thermal=true;m_impl->physical->midpoint_options=options;
}
NativeRZMidpointStoppingEvent::NativeRZMidpointStoppingEvent(WarpX& w,NativeAcceptedStoppingContext const& c,
    amrex::MultiFab const& t,amrex::MultiFab& u,SpatialStoppingOptions const& o,SpatialStoppingFields const& f,
    RZStoppingBackground const& bg,NativeStoppingThermalOptions const& options)
    :NativeRZMidpointStoppingEvent(w,c,t,u,o,f,options)
{
    m_impl->physical->background=true;m_impl->physical->background_destinations=bg;
}
NativeRZMidpointStoppingEvent::NativeRZMidpointStoppingEvent(WarpX& w,NativeAcceptedStoppingContext const& c,
    amrex::MultiFab const& t,amrex::MultiFab& u,SpatialStoppingOptions const& o,SpatialStoppingFields const& f,
    RZStoppingBackground const& bg,NativeStoppingCircuitBinding const& circuit,
    NativeStoppingThermalOptions const& options):NativeRZMidpointStoppingEvent(w,c,t,u,o,f,bg,options)
{
    m_impl->physical->circuit=std::make_unique<NativeRZSpatialStoppingEvent::Impl::CircuitState>();
    m_impl->physical->circuit->binding=circuit;
}
NativeRZMidpointStoppingEvent::~NativeRZMidpointStoppingEvent()=default;
bool NativeRZMidpointStoppingEvent::Capture()
{
    auto& p=*m_impl->physical;
    if(p.committed||p.closed||p.midpoint_captured)return p.Reject("midpoint source context already captured or closed");
    p.ready=false;p.ledger={};p.background_ledger={};
    if(!Collective(p.Scope()))return p.Reject("unsupported midpoint stopping scope");
    return p.Capture()&&p.InitializeMidpoint();
}
bool NativeRZMidpointStoppingEvent::Prepare(){return Capture()&&Solve();}
bool NativeRZMidpointStoppingEvent::DefineState(WarpXSolverVec& x)const{return m_impl->physical->DefineMidpointState(x);}
bool NativeRZMidpointStoppingEvent::Residual(WarpXSolverVec& out,WarpXSolverVec const& x)
{
    auto& p=*m_impl->physical;p.ready=false;return p.MidpointResidual(out,x);
}
bool NativeRZMidpointStoppingEvent::Solve()
{
    auto& p=*m_impl->physical;p.ready=false;
    if(p.circuit&&!Collective(p.midpoint_captured&&!p.committed&&!p.closed&&p.CircuitContextMatches()))
        return p.Reject("stale circuit source solve context");
    if(!p.midpoint_captured||p.committed||p.closed)return p.Reject("midpoint source context unavailable");
    if(!p.SolveMidpoint()||!p.Work()||!p.MidpointWork())return false;
    if(!p.Unchanged(false))return p.Reject("midpoint source preparation mutated accepted input");
    p.ready=true;p.failure="none";return true;
}
bool NativeRZMidpointStoppingEvent::FreezePreconditioner(WarpXSolverVec const& x)
{
    auto& p=*m_impl->physical;p.ready=false;
    if(p.circuit&&!Collective(p.midpoint_captured&&!p.committed&&!p.closed&&
        x.IsDefined()&&p.CircuitContextMatches()))return p.Reject("stale circuit source PC freeze context");
    return p.FreezeMidpointPC(x);
}
bool NativeRZMidpointStoppingEvent::Precondition(WarpXSolverVec& out,WarpXSolverVec const& rhs)
{
    auto& p=*m_impl->physical;p.ready=false;
    if(p.circuit&&!Collective(p.midpoint_captured&&!p.committed&&!p.closed&&
        out.IsDefined()&&rhs.IsDefined()&&p.CircuitContextMatches()))return p.Reject("stale circuit source PC apply context");
    return p.ApplyMidpointPC(out,rhs);
}
bool NativeRZMidpointStoppingEvent::CopySolution(WarpXSolverVec& x)const
{
    auto const& p=*m_impl->physical;
    if(p.circuit&&!Collective(p.ready&&p.midpoint_captured&&!p.committed&&!p.closed&&p.CircuitFinalActionMatches()))return false;
    if(!p.ready)return false;
    if(!x.IsDefined())x.Define(p.midpoint_solution);
    x.Copy(p.midpoint_solution);return true;
}
bool NativeRZMidpointStoppingEvent::Prepared()const{
    auto const& p=*m_impl->physical;
    return p.ready&&(!p.circuit||(p.midpoint_captured&&!p.committed&&!p.closed&&p.CircuitFinalActionMatches()));
}
NativeStoppingCircuitWork const& NativeRZMidpointStoppingEvent::CircuitWork()const{
    auto const& p=*m_impl->physical;AMREX_ALWAYS_ASSERT(p.circuit);return p.circuit->work;
}
NativeStoppingCircuitPublication const& NativeRZMidpointStoppingEvent::CircuitPublication()const{
    auto const& p=*m_impl->physical;AMREX_ALWAYS_ASSERT(p.circuit);return p.circuit->publication;
}
char const* NativeRZMidpointStoppingEvent::Failure()const{return m_impl->physical->failure;}
RZSpatialStoppingLedger const& NativeRZMidpointStoppingEvent::Ledger()const{return m_impl->physical->ledger;}
NativeStoppingThermalReceipt const& NativeRZMidpointStoppingEvent::ThermalReceipt()const{return m_impl->physical->midpoint_receipt;}
amrex::MultiFab const& NativeRZMidpointStoppingEvent::CandidateEnergy()const{return m_impl->physical->u1;}
CV NativeRZMidpointStoppingEvent::CandidateElectronCurrent()const{return Const(m_impl->physical->j1);}
CV NativeRZMidpointStoppingEvent::ElectricImpulse()const{return Const(m_impl->physical->impulse);}
CV NativeRZMidpointStoppingEvent::MeanElectronCurrent()const{return Const(m_impl->physical->mean);}
RZStoppingBackgroundLedger const& NativeRZMidpointStoppingEvent::BackgroundLedger()const{return m_impl->physical->background_ledger;}
CV NativeRZMidpointStoppingEvent::CandidatePotential()const{return Const(m_impl->physical->background?m_impl->physical->full_a:m_impl->physical->a1);}
CV NativeRZMidpointStoppingEvent::CandidateMagnetic()const{return Const(m_impl->physical->background?m_impl->physical->full_b:m_impl->physical->b1);}
CV NativeRZMidpointStoppingEvent::CandidateDisplacement()const{return Const(m_impl->physical->background?m_impl->physical->full_d:m_impl->physical->d1);}
CV NativeRZMidpointStoppingEvent::CandidateConductorCurrent()const{return Const(m_impl->physical->background?m_impl->physical->full_w:m_impl->physical->wall);}
CV NativeRZMidpointStoppingEvent::StoppingVelocity()const{return Const(m_impl->physical->ve);}
CV NativeRZMidpointStoppingEvent::StoppingImpulseDensity()const{return m_impl->physical->reaction->ConjugateMomentumDensity();}
CV NativeRZMidpointStoppingEvent::ElectricAuxiliary()const{return Const(m_impl->physical->aux);}
CV NativeRZMidpointStoppingEvent::ElectronReactionImpulse()const{return Const(m_impl->physical->reaction_b);}
CV NativeRZMidpointStoppingEvent::CandidateIonCurrent()const{return Const(m_impl->physical->i1);}
CV NativeRZMidpointStoppingEvent::InitialIonCurrent()const{return Const(m_impl->physical->i0);}
amrex::MultiFab const& NativeRZMidpointStoppingEvent::MidpointTemperature()const{return m_impl->physical->midpoint_moments->NodalTemperature();}
amrex::MultiFab const& NativeRZMidpointStoppingEvent::HeatDensity()const{return m_impl->physical->heat;}
amrex::MultiFab const& NativeRZMidpointStoppingEvent::EnergyIncrement()const{return m_impl->physical->midpoint_increment;}
amrex::MultiFab const& NativeRZMidpointStoppingEvent::EnergyResidual()const{return m_impl->physical->midpoint_residual;}
bool NativeRZMidpointStoppingEvent::CommitOnce(FV const& j,InvalidationHook h){return m_impl->physical->Publish(j,h);}
bool NativeRZMidpointStoppingEvent::CommitCircuitOnce(FV const& j,InvalidationHook h){return m_impl->physical->Publish(j,h,true);}
bool NativeRZMidpointStoppingEvent::AppendDisplacementCertificate(
    warpx::darwin::NativeLongitudinalProducerCertificate const& old,
    warpx::darwin::NativeLongitudinalProducerCertificate& updated) const
{
    auto const& s=*m_impl->physical;
    bool supported=s.committed&&!s.closed&&s.displacement_potential&&
        s.background_destinations.longitudinal_certificate.get()==&old&&&old==&updated;
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported)return false;
    return updated.AppendEvent(*s.displacement_potential,Const(s.d1));
}
bool NativeRZMidpointStoppingEvent::Rollback(InvalidationHook h){return RollbackRZEvent(*m_impl->physical,h);}
void NativeRZMidpointStoppingEvent::Finalize(){m_impl->physical->closed=true;m_impl->physical->ready=false;}
