/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Private opt-in joined consumer. Included after the legacy native helpers.
// No accepted field, particle, history or source publication occurs here.
namespace retained_joined_arithmetic = warpx::ohm::compensated;
struct JoinedRetainedMatrixFree : JoinedMatrixFree
{
    using Pair = retained_joined_arithmetic::Pair;
    Core::RetainedRecovery recovered;
    Field field_low, total_low, rate_low, current_low, ea_low, curl_low, mass_low, row_low;
    MF divergence_low;
    compensated_curl::Coefficients coefficients;
    std::string observation, capture_phase;
    int affine_recoveries = 0, responses = 0, affine_rates = 0, linear_rates = 0;
    double row_roundoff = 0.;

    JoinedRetainedMatrixFree(WarpX& warp, NativeInertiaSupport& sup, Core& recovery,
        Core& global, amrex::Array<amrex::iMultiFab,3> const& mask, View const& ref)
        : JoinedMatrixFree(warp,sup,recovery,global,mask,ref),
          field_low(Clone(V(field))), total_low(Clone(V(total))), rate_low(Clone(V(rate))),
          current_low(Clone(V(current))), ea_low(Clone(V(ea))), curl_low(Clone(V(curl))),
          mass_low(Clone(V(mass))), row_low(Clone(V(field))),
          coefficients(compensated_curl::NativeCoefficients(warp))
    {
        for (auto* f : {&field_low,&total_low,&rate_low,&current_low,&ea_low,
                        &curl_low,&mass_low,&row_low}) Zero(V(*f));
        divergence_low.define(divergence.boxArray(),divergence.DistributionMap(),1,0);
        divergence_low.setVal(0.);
        amrex::ParmParse("endpoint_diagnostic").query("vacuum_observe",observation);
    }
    void Dump(std::string const& name, MF const& value) const
    {
        if (!observation.empty()) amrex::VisMF::Write(value,observation+"_retained_"+name);
    }
    void DumpFields(std::string const& phase) const
    {
        if (observation.empty()) return;
        for (int c=0;c<3;++c) {
            auto const suffix=std::to_string(c);
            Dump(phase+"_field_hi"+suffix,field[c]);
            Dump(phase+"_field_lo"+suffix,field_low[c]);
            Dump(phase+"_rate_hi"+suffix,rate[c]);
            Dump(phase+"_rate_lo"+suffix,rate_low[c]);
            Dump(phase+"_current_hi"+suffix,current[c]);
            Dump(phase+"_current_lo"+suffix,current_low[c]);
            Dump(phase+"_mass_hi"+suffix,mass[c]);
            Dump(phase+"_mass_lo"+suffix,mass_low[c]);
            Dump(phase+"_row_lo"+suffix,row_low[c]);
        }
    }
    bool Finite(View const& high, View const& low)
    {
        bool valid=true;
        for (int c=0;c<3;++c) {
            bool const h=high[c]->is_finite(0,1,0);
            bool const l=low[c]->is_finite(0,1,0);
            valid=h && l && valid;
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        failed=!valid || failed;
        return !failed;
    }
    bool Recover(bool affine, View const* increment=nullptr)
    {
        Core::RetainedReceipt receipt;
        if (affine) { ++affine_recoveries; receipt=core.RecoverRetained(recovered,increment); }
        else {
            AMREX_ALWAYS_ASSERT(increment);
            ++responses; receipt=core.ResponseRetained(recovered,*increment);
        }
        if (!receipt.ready) {
            failed=true;
            amrex::Print()<<"Native retained joined recovery declined affine="<<affine
                <<" failure="<<int(receipt.solve.failure)<<" iterations="<<receipt.solve.iterations
                <<" residual="<<std::setprecision(17)<<receipt.solve.residual
                <<" target="<<receipt.solve.target<<"\n";
            return false;
        }
        capture_phase=affine?(affine_recoveries==1?"origin":"final_pair"):
            (responses==1?"response_first":"");
        for (int c=0;c<3;++c) {
            field[c].setVal(0.);field_low[c].setVal(0.);
            MF::Copy(field[c],recovered.High(c),0,0,1,0);
            MF::Copy(field_low[c],recovered.Low(c),0,0,1,0);
        }
        if (affine || responses==1)
            amrex::Print()<<"Native retained joined recovery affine="<<affine
                <<" iterations="<<receipt.solve.iterations<<" pair="<<std::setprecision(17)
                <<receipt.assembled_residual<<" target="<<receipt.solve.target
                <<" high_compensated="<<receipt.materialized_compensated
                <<" high_ordinary="<<receipt.materialized_ordinary
                <<" discarded_action="<<receipt.discarded_action<<"\n";
        return true;
    }
    // The high evaluation owns all affine B/convective/cylindrical terms.
    // Native linear_only removes precisely those terms for the low action.
    bool Rates(bool affine, View const* ell=nullptr)
    {
        for (int c=0;c<3;++c) {
            for (amrex::MFIter it(total[c]);it.isValid();++it) {
                auto h=total[c].array(it),l=total_low[c].array(it);
                auto x=field[c].const_array(it),y=field_low[c].const_array(it);
                auto e=ell?(*ell)[c]->const_array(it):amrex::Array4<RT const>{};
                bool const add=ell!=nullptr;
                amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const v=add?retained_joined_arithmetic::Add({x(i,j,k),y(i,j,k)},
                        {e(i,j,k),0.}):Pair{x(i,j,k),y(i,j,k)};
                    h(i,j,k)=v.hi;l(i,j,k)=v.lo;
                });
            }
        }
        VectorImages(w,V(total),true);VectorImages(w,V(total_low),true);
        w.InterpolateLevelZeroFieldToAux(V(ea),V(total));
        w.InterpolateLevelZeroFieldToAux(V(ea_low),V(total_low));
        warpx::particles::EvaluateNativeInstantaneousIonRate(w,V(ea),V(ba),V(current),
                                                            V(rate),!affine);
        if (affine) ++affine_rates; else ++linear_rates;
        warpx::particles::EvaluateNativeInstantaneousIonRate(w,V(ea_low),V(ba),
                                                            V(current_low),V(rate_low),true);
        ++linear_rates;
        if (!capture_phase.empty()) for (int c=0;c<3;++c) {
            auto const suffix=std::to_string(c);
            Dump(capture_phase+"_particle_total_hi"+suffix,total[c]);
            Dump(capture_phase+"_particle_total_lo"+suffix,total_low[c]);
            Dump(capture_phase+"_gather_hi"+suffix,ea[c]);
            Dump(capture_phase+"_gather_lo"+suffix,ea_low[c]);
        }
        bool const rate_valid=Finite(V(rate),V(rate_low));
        bool const current_valid=Finite(V(current),V(current_low));
        return rate_valid && current_valid;
    }
    bool RateMass()
    {
        if (!core.ApplyRetainedGlobalCurl(V(curl),V(curl_low),recovered)) {
            failed=true;return false;
        }
        RT const factor=-all.Diagonal()/PhysConst::mu0, scale=scale_si;
        if (!capture_phase.empty()) for (int c=0;c<3;++c) {
            auto const suffix=std::to_string(c);
            Dump(capture_phase+"_global_curl_hi"+suffix,curl[c]);
            Dump(capture_phase+"_global_curl_lo"+suffix,curl_low[c]);
            Dump(capture_phase+"_kappa"+suffix,support.KappaEdge(c));
        }
        for (int c=0;c<3;++c) {
            for (amrex::MFIter it(mass[c]);it.isValid();++it) {
                auto h=mass[c].array(it),l=mass_low[c].array(it);
                auto ch=curl[c].const_array(it),cl=curl_low[c].const_array(it);
                auto rh=rate[c].const_array(it),rl=rate_low[c].const_array(it);
                auto m=support.KappaEdge(c).const_array(it);
                amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    namespace a=retained_joined_arithmetic;
                    auto const current_rate=a::Multiply({ch(i,j,k),cl(i,j,k)},factor);
                    auto const net=a::Add(current_rate,a::Negate(a::Sum(rh(i,j,k),rl(i,j,k))));
                    // Match the native represented coefficient scale*kappa,
                    // then retain the product and current-rate cancellation.
                    auto const value=a::Multiply(net,scale*m(i,j,k));
                    h(i,j,k)=value.hi;l(i,j,k)=value.lo;
                });
            }
        }
        VectorImages(w,V(mass),false);VectorImages(w,V(mass_low),false);
        return Finite(V(mass),V(mass_low));
    }
    void GradientMass(MF const& node)
    {
        MF::Copy(potential,node,0,0,1,0);potential.mult(length,0,1,0);
        potential.OverrideSync(w.Geom(0).periodicity());potential.FillBoundary(w.Geom(0).periodicity());
        Gradient(w,potential,V(gradient));
        support.ApplyMass({&gradient[0],&gradient[1],&gradient[2]},V(curl));
    }
    // Only this final row boundary returns to the legacy single outer vector.
    void Assemble(View const& output, bool rhs, View const* force=nullptr,
                  View const* ell=nullptr)
    {
        for (int c=0;c<3;++c) {
            for (amrex::MFIter it(*output[c]);it.isValid();++it) {
                auto out=output[c]->array(it),lo=row_low[c].array(it);
                auto eh=field[c].const_array(it),el=field_low[c].const_array(it);
                auto mh=mass[c].const_array(it),ml=mass_low[c].const_array(it);
                auto gm=curl[c].const_array(it);
                auto f=force?(*force)[c]->const_array(it):amrex::Array4<RT const>{};
                auto e=ell?(*ell)[c]->const_array(it):amrex::Array4<RT const>{};
                bool const physical=force!=nullptr;
                amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    namespace a=retained_joined_arithmetic;
                    auto value=a::Add({eh(i,j,k),el(i,j,k)},a::Negate({mh(i,j,k),ml(i,j,k)}));
                    if (rhs) value=a::Add(a::Sum(f(i,j,k),-e(i,j,k)),a::Negate(value));
                    else {
                        value=a::Add(value,{gm(i,j,k),0.});
                        if (physical) value=a::Add(value,a::Sum(e(i,j,k),-f(i,j,k)));
                    }
                    out(i,j,k)=value.hi;lo(i,j,k)=value.lo;
                });
            }
        }
        RestrictEdges(output);RestrictEdges(V(row_low));
        for (auto const& f:row_low) row_roundoff=std::max(row_roundoff,f.norminf(0));
    }
    void DivPair(MF& output, RT scalar)
    {
        Copy(V(total),V(field));Copy(V(total_low),V(field_low));
        VectorImages(w,V(total),false);VectorImages(w,V(total_low),false);
#if defined(WARPX_DIM_RZ)
        auto const coeff=coefficients;
        for (amrex::MFIter it(output);it.isValid();++it) {
            auto h=output.array(it),l=divergence_low.array(it);
            compensated_curl::PairRead const f{
                {total[0].const_array(it),total[1].const_array(it),total[2].const_array(it)},
                {total_low[0].const_array(it),total_low[1].const_array(it),total_low[2].const_array(it)}};
            amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                namespace a=retained_joined_arithmetic;
                RT const r=coeff.rmin+i*coeff.dr;
                auto const radial=r!=0.?compensated_curl::RadialWeighted(f,0,i,j,k,false,r,coeff):
                    compensated_curl::Quotient(a::Multiply(f(0,i,j,k),4.),coeff.dr);
                auto const axial=compensated_curl::Difference(f,2,1,i,j,k,false,coeff.inverse[2]);
                auto const value=a::Multiply(a::Add(radial,axial),scalar);
                h(i,j,k)=value.hi;l(i,j,k)=value.lo;
            });
        }
#elif defined(WARPX_DIM_3D)
        auto const coeff=coefficients;
        for (amrex::MFIter it(output);it.isValid();++it) {
            auto h=output.array(it),l=divergence_low.array(it);
            compensated_curl::PairRead const f{
                {total[0].const_array(it),total[1].const_array(it),total[2].const_array(it)},
                {total_low[0].const_array(it),total_low[1].const_array(it),total_low[2].const_array(it)}};
            amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                namespace a=retained_joined_arithmetic;
                auto const x=compensated_curl::Difference(f,0,0,i,j,k,false,coeff.inverse[0]);
                auto const y=compensated_curl::Difference(f,1,1,i,j,k,false,coeff.inverse[1]);
                auto const z=compensated_curl::Difference(f,2,2,i,j,k,false,coeff.inverse[2]);
                auto const value=a::Multiply(a::Add(a::Add(x,y),z),scalar);
                h(i,j,k)=value.hi;l(i,j,k)=value.lo;
            });
        }
#else
        amrex::ignore_unused(output,scalar);failed=true;
#endif
        bool const hi=output.is_finite(0,1,0),lo=divergence_low.is_finite(0,1,0);
        failed=!hi || !lo || failed;
    }
    double DivergenceError()
    {
        DivPair(divergence,1.);
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<double> data(op);
        using Tuple=decltype(data)::Type;
        for (amrex::MFIter it(divergence);it.isValid();++it) {
            auto h=divergence.const_array(it),l=divergence_low.const_array(it),
                 v=support.NodeWeights().const_array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
                return {v(i,j,k)>0.?std::abs(h(i,j,k))+std::abs(l(i,j,k)):0.};
            });
        }
        double value=amrex::get<0>(data.value(op));amrex::ParallelDescriptor::ReduceRealMax(value);
        return failed?std::numeric_limits<double>::quiet_NaN():length*value;
    }
    double RowError(View const& hi)
    {
        if (!Finite(hi,V(row_low))) return std::numeric_limits<double>::quiet_NaN();
        double value=0.;
        for (int c=0;c<3;++c) value=std::max(value,hi[c]->norminf(0)+row_low[c].norminf(0));
        return value;
    }
    void apply(JoinedVector& out, JoinedVector const& in)
    {
        ++actions;
        auto increment=ViewOf(in);
        if (!Recover(false,&increment) || !Rates(false) || !RateMass()) return;
        GradientMass(in.node);Assemble(V(out.edge),false);
        DivPair(out.node,length);
        if (responses==1) {
            Dump("response_first_div_before_projection",out.node);
            Dump("response_first_potential",potential);
            for (int c=0;c<3;++c) {
                Dump("response_first_row"+std::to_string(c),out.edge[c]);
                Dump("response_first_gradient_mass"+std::to_string(c),curl[c]);
            }
        }
        if (!graph.ProjectAndRestrict(out.node,support.NodeWeights())) failed=true;
        if (responses==1) Dump("response_first_div_after_projection",out.node);
        if (responses==1) { DumpFields("response_first");Dump("response_first_div_lo",divergence_low); }
    }
    bool RightHandSide(JoinedVector& rhs, View const& force, View const& ell,
                       double& compatibility)
    {
        if (!Recover(true) || !Rates(true,&ell) || !RateMass()) return false;
        Copy(V(origin),V(field));
        Assemble(V(rhs.edge),true,&force,&ell);DivPair(rhs.node,-length);
        compatibility=graph.Compatibility(rhs.node,support.NodeWeights())+
            graph.Compatibility(divergence_low,support.NodeWeights());
        DumpFields("origin");Dump("origin_rhs_node",rhs.node);Dump("origin_div_lo",divergence_low);
        for (int c=0;c<3;++c) Dump("origin_rhs_edge"+std::to_string(c),rhs.edge[c]);
        if (failed || !std::isfinite(compatibility) || compatibility>1.e-9) return false;
        return graph.ProjectAndRestrict(rhs.node,support.NodeWeights());
    }
};

bool RetainedFreeUnionMatches(NativeInertiaSupport const& support,
    amrex::Array<amrex::iMultiFab,3> const& trace,
    amrex::Array<amrex::iMultiFab,3> const& global)
{
    int bad=0;
    for (int c=0;c<3;++c) {
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);
        using Tuple=decltype(data)::Type;
        for (amrex::MFIter it(global[c]);it.isValid();++it) {
            auto g=global[c].const_array(it),t=trace[c].const_array(it),
                 m=support.RecoveryMask(c).const_array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
                return {int(bool(g(i,j,k))!=(bool(t(i,j,k)) || bool(m(i,j,k))))};
            });
        }
        bad=std::max(bad,amrex::get<0>(data.value(op)));
    }
    amrex::ParallelDescriptor::ReduceIntMax(bad);return bad==0;
}

// Endpoint-only snapshot. The old null-output path neither allocates this
// storage nor changes its high-only publication test. Recovery ownership is
// transferred only after the original materialization measurement completes.
struct RetainedEndpointSave {
    using Owner=NativePairedDarwinFields;
    using Part=Owner::EndpointPart;
    std::array<Field,Owner::endpoint_parts> fields;
    MF node,nodal_high,nodal_low;
    Field original_trace; // Numeric affine input; no serialized Core receipt.
    Core::RetainedRecovery recovered;
    NativeVacuumConstraintResult result;
    double pair_null=0.;
    bool materialized=false,ready=false;
    View PartView(Part k){return V(fields[static_cast<int>(k)]);}
    bool Capture(JoinedRetainedMatrixFree& op,JoinedVector const& solution,
                 View const& ell) {
        auto& w=op.w;
        original_trace=Clone(op.ViewOf(solution));
        for(auto& f:fields){f=Clone(V(op.field));Zero(V(f));}
        Copy(PartView(Part::TransverseHigh),V(op.field));
        Copy(PartView(Part::TransverseLow),V(op.field_low));
        auto th=PartView(Part::TransverseHigh),tl=PartView(Part::TransverseLow);
        VectorImages(w,th,false);VectorImages(w,tl,false);
        auto eh=PartView(Part::ElectricHigh),el=PartView(Part::ElectricLow);
        auto mh=PartView(Part::InertiaHigh),ml=PartView(Part::InertiaLow);
        auto dh=PartView(Part::ElectronRateHigh),dl=PartView(Part::ElectronRateLow);
        auto ih=PartView(Part::IonRateHigh),il=PartView(Part::IonRateLow);
        auto global_h=Clone(V(op.field)),global_l=Clone(V(op.field));
        Zero(V(global_h));Zero(V(global_l));
        if(!op.core.ApplyRetainedGlobalCurl(V(global_h),V(global_l),op.recovered))return false;
        auto const factor=-op.all.Diagonal()/PhysConst::mu0;
        for(int c=0;c<3;++c){
            MF::Copy(*ih[c],op.rate[c],0,0,1,0);MF::Copy(*il[c],op.rate_low[c],0,0,1,0);
            for(amrex::MFIter it(*eh[c]);it.isValid();++it){
                auto e=eh[c]->array(it),l=el[c]->array(it),m=mh[c]->array(it),n=ml[c]->array(it);
                auto d=dh[c]->array(it),q=dl[c]->array(it);
                auto x=th[c]->const_array(it),y=tl[c]->const_array(it),z=ell[c]->const_array(it);
                auto a=op.mass[c].const_array(it),b=op.mass_low[c].const_array(it),g=op.curl[c].const_array(it);
                auto ch=global_h[c].const_array(it),cl=global_l[c].const_array(it);
                auto rh=op.rate[c].const_array(it),rl=op.rate_low[c].const_array(it);
                amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    namespace ar=retained_joined_arithmetic;
                    auto const total=ar::Add({x(i,j,k),y(i,j,k)},{z(i,j,k),0.});
                    auto const inertia=ar::Add({a(i,j,k),b(i,j,k)},{-g(i,j,k),0.});
                    auto const current=ar::Add(ar::Multiply({ch(i,j,k),cl(i,j,k)},factor),
                        ar::Negate(ar::Sum(rh(i,j,k),rl(i,j,k))));
                    e(i,j,k)=total.hi;l(i,j,k)=total.lo;m(i,j,k)=inertia.hi;n(i,j,k)=inertia.lo;
                    d(i,j,k)=current.hi;q(i,j,k)=current.lo;
                });
            }
        }
        VectorImages(w,eh,true);VectorImages(w,el,true);
        for(auto const& v:{mh,ml,dh,dl,ih,il})VectorImages(w,v,false);
        auto const& native=*w.m_fields.get("hybrid_E_inertial_nodal",0);
        for(auto* f:{&nodal_high,&nodal_low}){
            f->define(native.boxArray(),native.DistributionMap(),native.nComp(),native.nGrowVect());f->setVal(0.);
        }
        CopyNativeEndpointCurrentMirror(w,mh,nodal_high);
        CopyNativeEndpointCurrentMirror(w,ml,nodal_low);
        node.define(solution.node.boxArray(),solution.node.DistributionMap(),1,solution.node.nGrowVect());
        MF::Copy(node,solution.node,0,0,1,node.nGrowVect());
        bool finite=nodal_high.is_finite(0,nodal_high.nComp(),nodal_high.nGrowVect(),true)&&
            nodal_low.is_finite(0,nodal_low.nComp(),nodal_low.nGrowVect(),true);
        for(auto const& f:fields)for(auto const& v:f)
            finite=v.is_finite(0,1,v.nGrowVect(),true)&&finite;
        amrex::ParallelDescriptor::ReduceBoolAnd(finite);return finite;
    }
};

NativeVacuumConstraintResult SolveRetainedJoined(WarpX& w, NativeInertiaSupport& support,
    Core& core, Core& all, amrex::Array<amrex::iMultiFab,3> const& trace,
    View const& reference, View const& force, View const& ell,
    View const& out, View const& out_inertia,
    RetainedEndpointSave* endpoint=nullptr)
{
    NativeVacuumConstraintResult result;
    JoinedRetainedMatrixFree op(w,support,core,all,trace,reference);
    auto rhs=op.makeVecRHS(),solution=op.makeVecLHS(),action=op.makeVecRHS(),residual=op.makeVecRHS();
    result.compatible=op.RightHandSide(rhs,force,ell,result.null_defect);
    result.graph_iterations=op.graph.iterations;
    if (!result.compatible) return result;
    // The physical PMC rows include every omitted floating gauge. Preserve
    // the setup-derived conversion and fixed remainder budget on this path;
    // the existing periodic retained solve keeps its original stopping call.
    bool physical_stopping=false;
#if !defined(AMREX_USE_GPU)
    for(int d=0;d<AMREX_SPACEDIM;++d)physical_stopping=physical_stopping ||
        WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC ||
        WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
#endif
    NativeVacuumPhysicalStopping stopping;
    double target=0.;
    if(physical_stopping){
        if(!stopping.Prepare(op))return result;
        double const initial=op.norm2(rhs);if(!std::isfinite(initial))return result;
        target=std::min(std::max(1.e-11,1.e-12*initial),stopping.target);
    }
    FlexibleGMRES<JoinedVector,JoinedRetainedMatrixFree> solver;
    solver.define(op);solver.setRestartLength(150);solver.setMaxIters(1000);solver.setVerbose(0);
    if(physical_stopping)solver.solve(solution,rhs,0.,target);
    else solver.solve(solution,rhs,1.e-12,1.e-11);
    result.iterations=solver.getNumIters();result.residual=solver.getResidualNorm();
    if (solver.getStatus()!=0) return result;
    op.apply(action,solution);op.linComb(residual,1.,rhs,-1.,action);
    bool finite=residual.node.is_finite(0,1,0);
    for (auto const& e:residual.edge) {
        finite=e.is_finite(0,1,0) && finite;
        result.plasma_error=std::max(result.plasma_error,e.norminf(0));
    }
    if (!finite || op.failed) return result;
    if(physical_stopping){
        result.residual=op.norm2(residual);
        if(!std::isfinite(result.residual)||result.residual>target)return result;
    }
    op.Dump("final_outer_residual_node",residual.node);
    for (int c=0;c<3;++c) op.Dump("final_outer_residual_edge"+std::to_string(c),residual.edge[c]);
    auto trace_increment=op.ViewOf(solution);
    if (!op.Recover(true,&trace_increment) || !op.Rates(true,&ell) || !op.RateMass()) return result;
    op.GradientMass(solution.node);op.Assemble(V(action.edge),false,&force,&ell);
    result.plasma_error=std::max(result.plasma_error,op.RowError(V(action.edge)));
    result.divergence_error=op.DivergenceError();result.actions=op.actions;
    if(physical_stopping){
        bool const certified=stopping.Certify(op,residual,&op.divergence_low);
        amrex::Print()<<std::setprecision(17)<<"NATIVE_VACUUM_RETAINED_PHYSICAL_STOPPING kappa="
            <<stopping.kappa<<" target="<<target<<" remainder_bound="<<stopping.remainder_bound
            <<" remainder_limit="<<stopping.remainder_limit<<" certified="<<int(certified)<<"\n";
        if(!certified)return result;
    }
    double const pair_null=op.length*(op.graph.Compatibility(op.divergence,support.NodeWeights())+
        op.graph.Compatibility(op.divergence_low,support.NodeWeights()));
    op.DumpFields("final_pair");
    op.Dump("final_pair_div_hi",op.divergence);op.Dump("final_pair_div_lo",op.divergence_low);
    op.Dump("final_potential",op.potential);
    for (int c=0;c<3;++c) {
        op.Dump("final_pair_row"+std::to_string(c),action.edge[c]);
        op.Dump("final_gradient_mass"+std::to_string(c),op.curl[c]);
    }
    amrex::Print()<<"Native retained joined paired final plasma="<<std::setprecision(17)
        <<result.plasma_error<<" divergence="<<result.divergence_error<<" null="<<pair_null
        <<" outer_residual="<<result.residual<<" outer_iterations="<<result.iterations<<"\n";
    if (op.failed || !std::isfinite(result.residual) || !std::isfinite(result.plasma_error) ||
        !std::isfinite(result.divergence_error) || !std::isfinite(pair_null) ||
        result.plasma_error>=1.e-8 || result.divergence_error>=1.e-8 || pair_null>1.e-9) return result;

    if(endpoint && !endpoint->Capture(op,solution,ell))return result;

    // Measure the actual single-field publication, using the unchanged ordinary
    // global curl, native affine rate and native divergence. No paired inertia
    // is allowed to escape alongside a high-only electric field.
    auto finished=Clone(V(op.field)),paired_mass=Clone(V(op.mass));
    for (int c=0;c<3;++c)
        MF::LinComb(op.total[c],1.,finished[c],0,1.,*ell[c],0,0,1,0);
    op.ElectricRate(V(op.total),false);op.ApplyRateMass(V(finished),true);
    op.GradientMass(solution.node);
    double materialized_plasma=0.;
    for (int c=0;c<3;++c) {
        MF::Subtract(op.mass[c],op.curl[c],0,0,1,0);
        MF::LinComb(action.edge[c],1.,finished[c],0,1.,*ell[c],0,0,1,0);
        MF::Subtract(action.edge[c],*force[c],0,0,1,0);
        MF::Subtract(action.edge[c],op.mass[c],0,0,1,0);
    }
    op.RestrictEdges(V(action.edge));
    for (int c=0;c<3;++c) {
        bool const row=action.edge[c].is_finite(0,1,0);
        finite=row && finite;
        materialized_plasma=std::max(materialized_plasma,action.edge[c].norminf(0));
        op.Dump("materialized_mass"+std::to_string(c),op.mass[c]);
        op.Dump("materialized_rate"+std::to_string(c),op.rate[c]);
        op.Dump("materialized_current"+std::to_string(c),op.current[c]);
        op.Dump("materialized_row"+std::to_string(c),action.edge[c]);
        op.Dump("paired_rate_mass"+std::to_string(c),paired_mass[c]);
    }
    double const materialized_div=op.FullDivergenceError(V(finished));
    double const materialized_null=op.length*op.graph.Compatibility(op.divergence,support.NodeWeights());
    op.Dump("materialized_div",op.divergence);
    auto const& receipt=op.recovered.Receipt();
    bool const materialized=finite && receipt.single_field_within_target &&
        std::isfinite(materialized_plasma) && materialized_plasma<1.e-8 &&
        std::isfinite(materialized_div) && materialized_div<1.e-8 &&
        std::isfinite(materialized_null) && materialized_null<=1.e-9 && !op.failed;
    amrex::Print()<<"Native retained joined final pair_plasma="<<std::setprecision(17)
        <<result.plasma_error<<" pair_div="<<result.divergence_error<<" pair_null="<<pair_null
        <<" high_plasma="<<materialized_plasma<<" high_div="<<materialized_div
        <<" high_null="<<materialized_null<<" high_recovery="<<receipt.materialized_ordinary
        <<" recovery_target="<<receipt.solve.target<<" materialized="<<materialized
        <<" affine_recoveries="<<op.affine_recoveries<<" responses="<<op.responses
        <<" affine_rates="<<op.affine_rates<<" linear_rates="<<op.linear_rates
        <<" row_low_max="<<op.row_roundoff<<"\n";
    if(endpoint) {
        // This is a distinct represented endpoint result, never a relabeling
        // of the failed single-field receipt. Caller outputs remain unchanged.
        endpoint->recovered=std::move(op.recovered);
        endpoint->result=result;endpoint->result.null_defect=pair_null;
        endpoint->result.converged=true;endpoint->pair_null=pair_null;
        endpoint->materialized=materialized;endpoint->ready=true;
        return endpoint->result;
    }
    if (!materialized) return result;
    VectorImages(w,V(finished),false);VectorImages(w,V(op.mass),false);
    for (int c=0;c<3;++c) {
        bool const e=finished[c].is_finite(0,1,finished[c].nGrowVect());
        bool const m=op.mass[c].is_finite(0,1,op.mass[c].nGrowVect());
        finite=e && m && finite;
    }
    if (!finite) return result;
    Copy(out,V(finished));Copy(out_inertia,V(op.mass));
    result.plasma_error=materialized_plasma;result.divergence_error=materialized_div;
    result.null_defect=materialized_null;result.converged=true;
    return result;
}
