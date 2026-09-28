/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL
 * Private implementation included only by NativeRZSpatialStoppingEvent.cpp. */
namespace {
// The signed linear counterpart of KineticThermalMoments::FillScalarImages.
// Keep native ordering and cell/node reflection offsets; no positivity clamp.
void MidpointTemperatureImages(amrex::MultiFab& data,amrex::Geometry const& geometry)
{
    data.OverrideSync(geometry.periodicity());data.FillBoundary(geometry.periodicity());
    auto const domain=amrex::convert(geometry.Domain(),data.ixType());
    auto const lo=domain.smallEnd(),hi=domain.bigEnd();
    auto const nodal=data.ixType().toIntVect();auto const periodic=geometry.isPeriodicArray();
    for(amrex::MFIter mfi(data);mfi.isValid();++mfi){auto const a=data.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            int src[3]={i,j,k};bool image=false;
            for(int d=0;d<AMREX_SPACEDIM;++d){
                if(!periodic[d]&&(src[d]<lo[d]||src[d]>hi[d])){
                    src[d]=src[d]<lo[d]?2*lo[d]-src[d]-(1-nodal[d]):
                        2*hi[d]-src[d]+(1-nodal[d]);image=true;
                }
            }
            if(image)a(i,j,k)=a(src[0],src[1],src[2]);
        });
    }
}
}

bool NativeRZSpatialStoppingEvent::Impl::MidpointTemperatureContextValid(bool frozen) const
{
    bool valid=context.Valid()&&context.Generation()==generation&&!committed&&!closed;
    if(frozen)valid=valid&&midpoint_pc_temperature_frozen&&
        midpoint_pc_temperature_generation==context.Generation();
    for(auto const& s:species){if(!s.fast||s.charge==0.)continue;
        auto const& particles=s.pc->GetParticles(0);
        valid=valid&&particles.size()==s.temperature_points.size()&&
            s.temperature_points.size()==s.temperature_impulses.size();
        for(auto const& [key,tile]:particles){
            auto const point=s.temperature_points.find(key);auto const imp=s.temperature_impulses.find(key);
            valid=valid&&point!=s.temperature_points.end()&&imp!=s.temperature_impulses.end();
            if(point!=s.temperature_points.end()&&imp!=s.temperature_impulses.end())
                valid=valid&&point->second.size()==std::size_t(tile.numParticles())&&
                    imp->second.size()==std::size_t(tile.numParticles())&&tile.numNeighborParticles()==0;
        }
    }
    return Collective(valid);
}

bool NativeRZSpatialStoppingEvent::Impl::InitializeMidpointTemperaturePC()
{
#if !defined(WARPX_DIM_RZ)
    return Reject("analytic temperature PC requires native RZ scope");
#else
    bool const valid=context.Valid()&&context.Generation()==generation&&
        midpoint_options.nonlinear.use_preconditioner;
    if(!Collective(valid))return Reject("analytic temperature PC initial context");
    auto const& cell=midpoint_moments->NumberDensity();
    for(auto* f:{&midpoint_pc_temperature_density,&midpoint_pc_temperature_cell}){
        f->define(cell.boxArray(),cell.DistributionMap(),1,cell.nGrowVect());f->setVal(0.);
    }
    auto const& node=midpoint_moments->NodalTemperature();
    midpoint_pc_temperature_node.define(node.boxArray(),node.DistributionMap(),1,node.nGrowVect());
    midpoint_pc_temperature_node.setVal(0.);
    for(int c=0;c<3;++c){
        AllocateLike(midpoint_pc_temperature_current[c],i1[c]);
        AllocateLike(midpoint_pc_temperature_electron[c],j1[c]);
        AllocateLike(midpoint_pc_temperature_sum[c],delta[c]);
        midpoint_pc_temperature_current[c].setVal(0.);
        midpoint_pc_temperature_electron[c].setVal(0.);midpoint_pc_temperature_sum[c].setVal(0.);
    }
    // Same immutable gamma used when InitializeMidpoint constructed its
    // physical KineticThermalMoments context, before any residual evaluation.
    midpoint_pc_temperature_gamma=sim.get_pointer_HybridPICModel()->m_gamma;
    midpoint_pc_temperature_points=0;
    for(auto& s:species){if(!s.fast||s.charge==0.)continue;
        for(auto const& [key,tile]:s.pc->GetParticles(0)){
            s.temperature_points[key].resize(tile.numParticles());
            s.temperature_impulses[key].resize(tile.numParticles());
            midpoint_pc_temperature_points+=tile.numParticles();
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(midpoint_pc_temperature_points);
    midpoint_pc_temperature_reaction=std::make_unique<NativeStoppingReactionTranspose>(context);
    midpoint_pc_temperature_bad.resize(1);midpoint_pc_temperature_frozen=false;
    return true;
#endif
}

bool NativeRZSpatialStoppingEvent::Impl::FreezeMidpointTemperaturePC()
{
#if !defined(WARPX_DIM_RZ)
    return false;
#else
    midpoint_pc_temperature_frozen=false;
    if(!MidpointTemperatureContextValid(false))return Reject("analytic temperature PC context changed");
    Copy(midpoint_pc_temperature_density,midpoint_moments->NumberDensity());
    auto* invalid=midpoint_pc_temperature_bad.data();
    amrex::ParallelFor(1,[=] AMREX_GPU_DEVICE(int){*invalid=0;});
    auto const& g=context.Geometry();auto const inverse=g.InvCellSizeArray(),plo=g.ProbLoArray();
    auto const magnetic=sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
    CV bfield{magnetic[0],magnetic[1],magnetic[2]};
    R const dt=options.interval,inv_c2=1./(PhysConst::c*PhysConst::c);
    R const rate_floor=1.e-3*PhysConst::q_e/PhysConst::kb;
    for(auto& s:species){if(!s.fast||s.charge==0.)continue;
        for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
            auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());
            auto const* point=s.points.at(key).data();auto* saved=s.temperature_points.at(key).data();
            auto const* angle=s.old.at(key).GetStructOfArrays().GetRealData(PIdx::theta).data();
            auto gather=context.GatherView(pti,bfield);
            for(int c=0;c<3;++c)gather.velocity[c]=ve[c].const_array(pti);
            auto const t=midpoint_moments->NodalTemperature().const_array(pti);
            amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                auto const p=point[n];auto const velocity=gather(p.position[0],p.position[1],p.position[2]);
                R const te=ablastr::particles::doGatherScalarFieldNodal(
                    p.position[0],p.position[1],p.position[2],t,inverse,plo);
                bool ok=velocity.valid&&std::isfinite(te)&&te>rate_floor&&
                    std::isfinite(p.rate)&&p.rate>=0.;
                R const theta=angle[n],cm=std::cos(-theta),sm=std::sin(-theta);
                V const local{p.minus[0]*cm-p.minus[1]*sm,p.minus[0]*sm+p.minus[1]*cm,p.minus[2]};
                R const gb=std::sqrt(1.+(local[0]*local[0]+local[1]*local[1]+local[2]*local[2])*inv_c2);
                R const fac=std::exp(-p.rate*dt),factor=1.5*dt*p.rate*fac/te;
                V v{},dv{};
                for(int c=0;c<3;++c){R const relative=local[c]/gb-velocity.collision_frame[c];
                    v[c]=velocity.collision_frame[c]+relative*fac;dv[c]=relative*factor;}
                R const denominator=1.-Dot(v,v)*inv_c2;
                ok=ok&&denominator>0.&&std::isfinite(denominator)&&std::isfinite(factor);
                if(!ok){amrex::HostDevice::Atomic::Add(invalid,1);return;}
                R const ga=1./std::sqrt(denominator),coupling=ga*ga*ga*Dot(v,dv)*inv_c2;
                V du{};for(int c=0;c<3;++c)du[c]=ga*dv[c]+coupling*v[c];
                R const co=std::cos(theta),si=std::sin(theta);
                MidpointTemperaturePoint result;result.position=p.position;
                result.weight=p.weight;result.mass=p.mass;result.charge=p.charge;
                result.proper_derivative={du[0]*co-du[1]*si,du[0]*si+du[1]*co,du[2]};
                // Final current uses the represented endpoint proper velocity.
                R const gn=Gamma(p.next),term=Dot(p.next,result.proper_derivative)*inv_c2/(gn*gn*gn);
                for(int c=0;c<3;++c){
                    result.velocity_derivative[c]=result.proper_derivative[c]/gn-p.next[c]*term;
                    ok=ok&&std::isfinite(result.proper_derivative[c])&&std::isfinite(result.velocity_derivative[c]);
                }
                if(!ok){amrex::HostDevice::Atomic::Add(invalid,1);return;}
                saved[n]=result;
            });
        }
    }
    amrex::Gpu::synchronize();int bad=0;
    amrex::Gpu::dtoh_memcpy(&bad,invalid,sizeof(int));
    bool const density_finite=midpoint_pc_temperature_density.is_finite();
    R const minimum=midpoint_pc_temperature_density.min(0);
    if(!Collective(bad==0&&density_finite&&minimum>0.&&
        std::isfinite(midpoint_pc_temperature_gamma)&&midpoint_pc_temperature_gamma>1.))
        return Reject("analytic temperature PC requires finite smooth rate above floor");
    midpoint_receipt.pc_temperature_freeze_point_visits+=midpoint_pc_temperature_points;
    midpoint_pc_temperature_generation=context.Generation();midpoint_pc_temperature_frozen=true;
    return true;
#endif
}

bool NativeRZSpatialStoppingEvent::Impl::ApplyMidpointTemperatureColumn(amrex::MultiFab const& direction)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(direction);return false;
#else
    // Every collective executes on every rank before a local failure is combined.
    if(!MidpointTemperatureContextValid(true))return false;
    bool const finite_direction=direction.is_finite();
    if(!Collective(finite_direction))return false;
    ++midpoint_receipt.pc_temperature_actions;
    midpoint_receipt.pc_temperature_action_point_visits+=midpoint_pc_temperature_points;
    auto const& g=context.Geometry();R const gm1=midpoint_pc_temperature_gamma-1.;
    for(amrex::MFIter mfi(midpoint_pc_temperature_cell);mfi.isValid();++mfi){
        auto const u=direction.const_array(mfi),n=midpoint_pc_temperature_density.const_array(mfi);
        auto const t=midpoint_pc_temperature_cell.array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            t(i,j,k)=gm1*(.5*u(i,j,k))/(PhysConst::kb*n(i,j,k));
        });
    }
    MidpointTemperatureImages(midpoint_pc_temperature_cell,g);
    for(amrex::MFIter mfi(midpoint_pc_temperature_node);mfi.isValid();++mfi){
        auto const t=midpoint_pc_temperature_cell.const_array(mfi);auto const node=midpoint_pc_temperature_node.array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            R const base=t(i,j,k);R sum=0.;
            for(int bits=0;bits<4;++bits)sum+=t(i-(bits&1),j-((bits>>1)&1),k)-base;
            node(i,j,k)=base+sum/4.;
        });
    }
    MidpointTemperatureImages(midpoint_pc_temperature_node,g);
    bool const finite_cell=midpoint_pc_temperature_cell.is_finite();
    bool const finite_node=midpoint_pc_temperature_node.is_finite();
    if(!Collective(finite_cell&&finite_node))return false;
    for(auto& f:midpoint_pc_temperature_current)f.setVal(0.);
    if(!midpoint_pc_temperature_reaction->Begin())return false;
    auto* invalid=midpoint_pc_temperature_bad.data();
    amrex::ParallelFor(1,[=] AMREX_GPU_DEVICE(int){*invalid=0;});
    auto const inverse=g.InvCellSizeArray(),plo=g.ProbLoArray();
    for(auto& s:species){if(!s.fast||s.charge==0.)continue;
        for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
            auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());
            auto const* points=s.temperature_points.at(key).data();
            auto* tuples=s.temperature_impulses.at(key).data();
            auto const temperature_direction=midpoint_pc_temperature_node.const_array(pti);
            auto box=pti.tilebox();box.grow(sim.get_ng_depos_J());
            auto const origin=WarpX::LowerCorner(box,0,0.),inv=WarpX::InvCellSize(0);
            auto const lower=amrex::lbound(box);
            auto const jx=midpoint_pc_temperature_current[0].array(pti),jy=midpoint_pc_temperature_current[1].array(pti),jz=midpoint_pc_temperature_current[2].array(pti);
            amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                auto const p=points[n];R const dt=ablastr::particles::doGatherScalarFieldNodal(
                    p.position[0],p.position[1],p.position[2],temperature_direction,inverse,plo);
                StoppingIonImpulse tuple;tuple.position=p.position;tuple.weight=p.weight;
                V velocity{};bool ok=std::isfinite(dt);
                for(int c=0;c<3;++c){tuple.impulse[c]=p.mass*(p.proper_derivative[c]*dt);
                    velocity[c]=p.velocity_derivative[c]*dt;
                    ok=ok&&std::isfinite(tuple.impulse[c])&&std::isfinite(velocity[c]);}
                if(!ok){amrex::HostDevice::Atomic::Add(invalid,1);return;}
                tuples[n]=tuple;
                R const xp=p.position[0],yp=p.position[1],rr=std::sqrt(xp*xp+yp*yp);
                V x{(rr-origin.x)*inv.x,0.,(p.position[2]-origin.z)*inv.z};
                V v{(xp*velocity[0]+yp*velocity[1])/rr*inv.x,0.,velocity[2]*inv.z};
                R const vt=(-yp*velocity[0]+xp*velocity[1])/rr;
                int const first_r=int(x[0])-1,first_z=int(x[2])-1;
                for(int iz=first_z;iz<=first_z+3;++iz)for(int ir=first_r;ir<=first_r+3;++ir){
                    V value{},unused{};
                    bool const valid=warpx::particles::EsirkepovCurrentRate<3,2>(x,v,V{},amrex::GpuArray<int,3>{ir,0,iz},vt,0.,value,unused);
                    if(!valid){amrex::HostDevice::Atomic::Add(invalid,1);continue;}
                    int const ii=ir+lower.x,jj=iz+lower.y;
                    R const a=p.charge*p.weight*inv.y*inv.z,b=p.charge*p.weight*inv.x*inv.y*inv.z,c=p.charge*p.weight*inv.x*inv.y;
                    if(ir<first_r+3)amrex::HostDevice::Atomic::Add(&jx(ii,jj,0),a*value[0]);
                    amrex::HostDevice::Atomic::Add(&jy(ii,jj,0),b*value[1]);
                    if(iz<first_z+3)amrex::HostDevice::Atomic::Add(&jz(ii,jj,0),c*value[2]);
                }
            });
        }
    }
    amrex::Gpu::synchronize();int bad=0;amrex::Gpu::dtoh_memcpy(&bad,invalid,sizeof(int));
    if(!Collective(bad==0)){midpoint_pc_temperature_reaction->Invalidate();return false;}
    // No host read or synchronization per tile. All tuples are complete before
    // the single collective preflight above; enqueue native transpose deposits.
    for(auto& s:species){if(!s.fast||s.charge==0.)continue;
        for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
            auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());
            midpoint_pc_temperature_reaction->DepositTile(
                pti,s.temperature_impulses.at(key).data(),pti.numParticles());
        }
    }
    if(!midpoint_pc_temperature_reaction->Finish())return false;
    auto& di=midpoint_pc_temperature_current;auto& dj=midpoint_pc_temperature_electron;
    sim.ApplyInverseVolumeScalingToCurrentDensity(&di[0],&di[1],&di[2],0);
    for(auto& f:di)ablastr::utils::communication::SumBoundary(f,0,1,f.nGrowVect(),f.nGrowVect(),false,g.periodicity());
    sim.ApplyJfieldBoundary(0,&di[0],&di[1],&di[2],PatchType::fine);Sync(di,g);
    auto const ri=midpoint_pc_temperature_reaction->InertiaImpulse();
    for(int c=0;c<3;++c){
        for(amrex::MFIter mfi(dj[c]);mfi.isValid();++mfi){
            auto const a=dj[c].array(mfi);auto const reaction_value=ri[c]->const_array(mfi),coefficient=mass[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                a(i,j,k)=(c==1&&i==0)?0.:reaction_value(i,j,k)/coefficient(i,j,k);
            });
        }
    }
    Sync(dj,g);
    for(int c=0;c<3;++c)amrex::MultiFab::LinComb(midpoint_pc_temperature_sum[c],1.,di[c],0,1.,dj[c],0,0,1,0);
    Sync(midpoint_pc_temperature_sum,g);
    bool finite=true;
    for(int c=0;c<3;++c){bool const current_finite=midpoint_pc_temperature_sum[c].is_finite();finite=finite&&current_finite;}
    if(!Collective(finite))return false;
    auto const result=projection->Correct(Const(midpoint_pc_temperature_sum),Const(zero));
    ++midpoint_receipt.pc_temperature_projection_calls;
    midpoint_receipt.pc_temperature_projection_iterations+=result.iterations;
    midpoint_receipt.pc_temperature_projection_residual=std::max(midpoint_receipt.pc_temperature_projection_residual,result.residual);
    if(!result.converged)return false;
    auto const longitudinal=projection->CorrectionField();
    int const high=g.Domain().bigEnd(0)+1;R const scale=psi_scale/current_scale;
    for(int c=0;c<3;++c){
        auto& f=*midpoint_pc_response.getArrayVec()[0][c];
        auto& mean_response=midpoint_pc_response.getMultiFabBlock(midpoint_current_names[c],0);
        for(amrex::MFIter mfi(f);mfi.isValid();++mfi){
            auto const out=f.array(mfi),out_mean=mean_response.array(mfi);
            auto const total=midpoint_pc_temperature_sum[c].const_array(mfi),lp=longitudinal[c]->const_array(mfi),electron=dj[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                bool const axis=c==1&&i==0,conductor=c!=0&&i==high;
                out(i,j,k)=(axis||conductor)?0.:scale*(total(i,j,k)-lp(i,j,k));
                out_mean(i,j,k)=axis?0.:-.5*electron(i,j,k);
            });
        }
        bool const ff=f.is_finite(),fj=mean_response.is_finite();finite=finite&&ff&&fj;
    }
    return Collective(finite);
#endif
}
