/* Diagnostic prototype. Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL. */
#include "NativeInstantaneousIonIncrement.H"
#include "DarwinVacuumJointSolve.H"
#include "ImplicitParticleEndpointAudit.H"
#include "Particles/Deposition/EsirkepovAxialCurrentIncrement.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/Deposition/EsirkepovInstantaneousCurrentIncrement.H"
#include "Particles/Deposition/EsirkepovAdjacentInstantaneousCurrentIncrement.H"
#include "Particles/Pusher/NativeBorisImpulse.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuAtomic.H>
#include <algorithm>
#include <cmath>
namespace warpx::particles {
namespace {
template<bool Adjacent>
bool DepositInstantaneousIonIncrement(WarpX& sim,
    ablastr::fields::VectorField const& base,ablastr::fields::VectorField const& delta)
{
    using R=amrex::Real;using P=amrex::ParticleReal;
    using Joint=warpx::thermal::DarwinVacuumJointSolve;
    auto const axial_lease=Joint::CaptureAxialProducerLease(sim);
#if defined(WARPX_DIM_RZ)
    constexpr int Dim=2;
#elif defined(WARPX_DIM_3D)
    constexpr int Dim=3;
#else
    amrex::Abort("Correlated instantaneous current supports RZ/Cartesian3D only");
    constexpr int Dim=3;
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxLevel()==0&&!EB::enabled()&&
        !sim.getdo_moving_window()&&WarpX::grid_type==GridType::Staggered&&
        WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov&&
        WarpX::nox==3&&WarpX::ncomps==1&&!WarpX::use_filter&&
        !WarpX::do_shared_mem_current_deposition,
        "Correlated instantaneous current requires native unfiltered single-level cubic Esirkepov");
    auto const primary=sim.m_fields.get_alldirs(warpx::fields::FieldType::current_fp,0);
    for(int c=0;c<3;++c){
        AMREX_ALWAYS_ASSERT(base[c]!=delta[c]);
        for(auto const* v:{&base,&delta}){
            auto* f=(*v)[c];
            AMREX_ALWAYS_ASSERT(f&&f!=primary[c]&&f->boxArray()==primary[c]->boxArray()&&
                f->DistributionMap()==primary[c]->DistributionMap()&&f->nComp()==1&&
                f->nGrowVect().allGE(sim.get_ng_depos_J()));f->setVal(0.);
        }
    }
    if(!axial_lease.valid)return false;
    bool metadata_ok=true;
    auto const inverse=WarpX::InvCellSize(0);auto const& geom=sim.Geom(0);
    auto const plo=geom.ProbLoArray(),phi=geom.ProbHiArray();
    amrex::GpuArray<int,AMREX_SPACEDIM> periodic{};
    for(int d=0;d<AMREX_SPACEDIM;++d)periodic[d]=geom.isPeriodic(d);
    amrex::Gpu::DeviceScalar<int> failure(0);int* bad=failure.dataPtr();
    for(auto const& name:sim.GetPartContainer().GetSpeciesNames()){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
        R const q=pc.getCharge();if(q==0.)continue;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dynamic_cast<PhysicalParticleContainer*>(&pc)&&
            !dynamic_cast<SubcycledParticleContainer*>(&pc)&&!pc.DoFieldIonization()&&
            !pc.do_not_deposit&&!pc.HasiAttrib("nsuborbits")&&
            pc.HasiAttrib("diagnostic_impulse_valid"),
            "Correlated instantaneous current requires fresh native unsplit impulse records");
        auto const axial=warpx::implicit::MakeAxialEndpointBoundary(geom,pc.GetParticleBoundaryData(),
            WarpX::field_boundary_lo,WarpX::field_boundary_hi,axial_lease.enabled);
        if(axial_lease.enabled){
            bool local=axial.enabled;auto const names=pc.GetRealSoANames();
            for(auto const* name:{"implicit_final_gather_x","implicit_final_gather_y","implicit_final_gather_z"})
                local=local&&std::find(names.begin(),names.end(),name)!=names.end();
            if(!local){metadata_ok=false;continue;}
        }
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){
            amrex::GpuArray<P const*,3> const x0{pti.GetAttribs("x_n").dataPtr(),pti.GetAttribs("y_n").dataPtr(),pti.GetAttribs("z_n").dataPtr()};
            amrex::GpuArray<P const*,3> const u0{pti.GetAttribs("ux_n").dataPtr(),pti.GetAttribs("uy_n").dataPtr(),pti.GetAttribs("uz_n").dataPtr()};
            amrex::GpuArray<P const*,3> const chord{pti.GetAttribs("esirkepov_chord_x").dataPtr(),pti.GetAttribs("esirkepov_chord_y").dataPtr(),pti.GetAttribs("esirkepov_chord_z").dataPtr()};
            amrex::GpuArray<P const*,3> const kick{pti.GetAttribs("diagnostic_du_x").dataPtr(),pti.GetAttribs("diagnostic_du_y").dataPtr(),pti.GetAttribs("diagnostic_du_z").dataPtr()};
            auto const* valid=pti.GetiAttribs("diagnostic_impulse_valid").dataPtr();
            auto const* chord_valid=pti.GetiAttribs("esirkepov_chord_valid").dataPtr();
            auto const* weight=pti.GetAttribs(PIdx::w).dataPtr();
            amrex::GpuArray<P const*,3> gather{};warpx::implicit::EndpointTileContract axial_support;
            if(axial_lease.enabled){
                gather={pti.GetAttribs("implicit_final_gather_x").dataPtr(),pti.GetAttribs("implicit_final_gather_y").dataPtr(),pti.GetAttribs("implicit_final_gather_z").dataPtr()};
                axial_support=warpx::implicit::MakeNativeAxialCurrentContract(sim,pti.tilebox(),pti.index());
            }
            bool const axial_enabled=axial_lease.enabled;
            int const cap_low=geom.Domain().smallEnd(AMREX_SPACEDIM-1),cap_high=geom.Domain().bigEnd(AMREX_SPACEDIM-1)+1;

            auto tile=pti.tilebox();tile.grow(sim.get_ng_depos_J());
            auto const origin=WarpX::LowerCorner(tile,0,0.);auto const lo=amrex::lbound(tile);
            amrex::GpuArray<amrex::Array4<R>,3> const b{base[0]->array(pti),base[1]->array(pti),base[2]->array(pti)},d{delta[0]->array(pti),delta[1]->array(pti),delta[2]->array(pti)};
            amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long p){
                if(!valid[p]||!chord_valid[p]||!std::isfinite(weight[p])||weight[p]<0.){
                    amrex::Gpu::Atomic::Max(bad,1);return;
                }
                amrex::GpuArray<double,3> const old{x0[0][p],x0[1][p],x0[2][p]},dx{chord[0][p],chord[1][p],chord[2][p]},u{u0[0][p],u0[1][p],u0[2][p]},du{kick[0][p],kick[1][p],kick[2][p]};
                amrex::GpuArray<double,3> v{},dv{};
                if(!RelativisticVelocityIncrement(u,du,v,dv)){
                    amrex::Gpu::Atomic::Max(bad,1);return;
                }
                EsirkepovInstantaneousPointIncrement point{};
                bool ok=true,axial_image=false;int cap_knot=0;
#if defined(WARPX_DIM_RZ)
                RZPointIncrement rz{};
                ok=CylindricalPointIncrement({old[0],old[1]},{dx[0],dx[1]},rz);
                R const r=rz.radius,co=rz.basis[0],si=rz.basis[1],dc=rz.basis_delta[0],ds=rz.basis_delta[1];
                R const vr=(old[0]*v[0]+old[1]*v[1])/r,vt=(-old[1]*v[0]+old[0]*v[1])/r;
                R const dvr=co*dv[0]+si*dv[1]+v[0]*dc+v[1]*ds+dc*dv[0]+ds*dv[1];
                R const dvt=-si*dv[0]+co*dv[1]-v[0]*ds+v[1]*dc-ds*dv[0]+dc*dv[1];
                bool const z_inside=old[2]+dx[2]>=plo[1]&&old[2]+dx[2]<phi[1];
                if(axial_enabled&&!periodic[1]&&!z_inside){
                    warpx::implicit::AxialEndpointImage image;
                    axial_image=warpx::implicit::endpoint_detail::axialCurrentFits(
                        {old[0]+dx[0],old[1]+dx[1],old[2]+dx[2]},
                        {u[0]+du[0],u[1]+du[1],u[2]+du[2]},
                        {gather[0][p],gather[1][p],gather[2][p]},axial_support,axial,image);
                    if(axial_image)cap_knot=(image.side==0?cap_low:cap_high)-lo.y;
                }
                ok=ok&&r>=plo[0]&&r<phi[0]&&r+rz.radial_delta>=plo[0]&&r+rz.radial_delta<phi[0]&&
                    (periodic[1]||(old[2]>=plo[1]&&old[2]<phi[1]&&(z_inside||axial_image)));
                point.position={EsirkepovIncrement{(r-origin.x)*inverse.x,rz.radial_delta*inverse.x},{0,0},{(old[2]-origin.z)*inverse.z,dx[2]*inverse.z}};
                point.velocity={EsirkepovIncrement{vr*inverse.x,dvr*inverse.x},{0,0},{v[2]*inverse.z,dv[2]*inverse.z}};
                point.azimuthal_velocity={vt,dvt};
#else
                double const origin_array[3]={origin.x,origin.y,origin.z},inverse_array[3]={inverse.x,inverse.y,inverse.z};
                for(int c=0;c<3;++c){
                    ok=ok&&(periodic[c]||(old[c]>=plo[c]&&old[c]<phi[c]&&old[c]+dx[c]>=plo[c]&&old[c]+dx[c]<phi[c]));
                    point.position[c]={(old[c]-origin_array[c])*inverse_array[c],dx[c]*inverse_array[c]};
                    point.velocity[c]={v[c]*inverse_array[c],dv[c]*inverse_array[c]};
                }
#endif
                // Validate support and integer range BEFORE converting starts.
                amrex::GpuArray<EsirkepovIncrement,3> check{};
                CubicAdjacentPointSupport support{};
                bool use_axial_pair=false;CubicAxialPointSupport axial_union_support{};
                if constexpr(Adjacent){
                    if(!ok||!PrepareCubicAdjacentPointSupport<Dim>(point,support)||
                       !EsirkepovAdjacentInstantaneousCurrentPair<Dim>(point,{0,0,0},check)){
                        amrex::Gpu::Atomic::Max(bad,1);return;
                    }
                }else{
                    bool supported=ok&&EsirkepovInstantaneousCurrentPair<3,Dim>(point,{0,0,0},check);
#if defined(WARPX_DIM_RZ)
                    if(ok&&!supported&&axial_enabled){
                        warpx::implicit::AxialEndpointImage image;
                        bool const path=warpx::implicit::endpoint_detail::axialCurrentFits(
                            {old[0]+dx[0],old[1]+dx[1],old[2]+dx[2]},
                            {u[0]+du[0],u[1]+du[1],u[2]+du[2]},
                            {gather[0][p],gather[1][p],gather[2][p]},axial_support,axial,image,false);
                        bool const one=path&&OneAxialKnotSupport(point,axial_union_support);
                        long long const knot=static_cast<long long>(axial_union_support.knot)+lo.y;
                        bool const admitted=one&&(image.reflected?
                            (axial_image&&axial_union_support.knot==cap_knot):
                            (z_inside&&knot>cap_low&&knot<cap_high));
                        if(admitted)cap_knot=axial_union_support.knot;
                        use_axial_pair=admitted&&
                            EsirkepovAxialInstantaneousCurrentPair(point,cap_knot,{0,0,0},check);
                        supported=use_axial_pair;
                    }
#else
                    amrex::ignore_unused(axial_image,cap_knot);
#endif
                    if(!supported){amrex::Gpu::Atomic::Max(bad,1);return;}
                }
                int const ir=Adjacent?support.lower[0]:int(point.position[0].value)-1;
                int const iz=Adjacent?support.lower[2]:(use_axial_pair?axial_union_support.axial_first:int(point.position[2].value)-1);
                int const jy=Dim==3?(Adjacent?support.lower[1]:int(point.position[1].value)-1):0;
                int const ih=Adjacent?support.upper[0]:ir+3;
                int const kh=Adjacent?support.upper[2]:(use_axial_pair?axial_union_support.axial_last:iz+3);
                int const jh=Dim==3?(Adjacent?support.upper[1]:jy+3):0;
                amrex::GpuArray<R,3> const scale{q*weight[p]*inverse.y*inverse.z,
                    q*weight[p]*(Dim==3?inverse.x*inverse.z:inverse.x*inverse.y*inverse.z),q*weight[p]*inverse.x*inverse.y};
                for(int k=iz;k<=kh;++k)for(int j=jy;j<=jh;++j)for(int i=ir;i<=ih;++i){
                    amrex::GpuArray<EsirkepovIncrement,3> pair{};
                    bool pair_ok=false;
                    if constexpr(Adjacent){pair_ok=EsirkepovAdjacentInstantaneousCurrentPair<Dim>(point,{i,j,k},pair);}
                    else{pair_ok=use_axial_pair?
                        EsirkepovAxialInstantaneousCurrentPair(point,cap_knot,{i,j,k},pair):
                        EsirkepovInstantaneousCurrentPair<3,Dim>(point,{i,j,k},pair);}
                    if(!pair_ok){amrex::Gpu::Atomic::Max(bad,1);return;}
                    int const ii=i+lo.x,jj=Dim==3?j+lo.y:k+lo.y,kk=Dim==3?k+lo.z:0;
                    for(int c=0;c<3;++c){
                        if((c==0&&i==ih)||(c==2&&k==kh)||(Dim==3&&c==1&&j==jh))continue;
                        if(!b[c].contains(ii,jj,kk)||!d[c].contains(ii,jj,kk)){amrex::Gpu::Atomic::Max(bad,1);return;}
                        if constexpr(Adjacent){
                            if(!std::isfinite(scale[c])||!std::isfinite(scale[c]*pair[c].value)||
                               !std::isfinite(scale[c]*pair[c].delta)){
                                amrex::Gpu::Atomic::Max(bad,1);return;
                            }
                        }
                        amrex::Gpu::Atomic::AddNoRet(&b[c](ii,jj,kk),scale[c]*pair[c].value);
                        amrex::Gpu::Atomic::AddNoRet(&d[c](ii,jj,kk),scale[c]*pair[c].delta);
                    }
                }
            });
        }
    }
    int failed=failure.dataValue()||!metadata_ok||!Joint::AxialProducerLeaseCurrentLocal(sim,axial_lease);amrex::ParallelDescriptor::ReduceIntMax(failed);
    if(failed){for(auto const* v:{&base,&delta})for(auto* f:*v)f->setVal(0.);return false;}
    for(auto const* v:{&base,&delta}){
#if defined(WARPX_DIM_RZ)
        sim.ApplyInverseVolumeScalingToCurrentDensity((*v)[0],(*v)[1],(*v)[2],0);
#endif
        for(auto* f:*v)ablastr::utils::communication::SumBoundary(*f,0,1,f->nGrowVect(),f->nGrowVect(),WarpX::do_single_precision_comms,geom.periodicity());
        sim.ApplyJfieldBoundary(0,(*v)[0],(*v)[1],(*v)[2],PatchType::fine);
        for(auto* f:*v){f->OverrideSync(geom.periodicity());f->FillBoundary(geom.periodicity());}
    }
    return true;
}
} // namespace

bool DepositNativeInstantaneousIonIncrement(WarpX& sim,
    ablastr::fields::VectorField const& base,ablastr::fields::VectorField const& delta)
{
    return DepositInstantaneousIonIncrement<false>(sim,base,delta);
}

bool DepositNativeAdjacentInstantaneousIonIncrement(WarpX& sim,
    ablastr::fields::VectorField const& base,ablastr::fields::VectorField const& delta)
{
    return DepositInstantaneousIonIncrement<true>(sim,base,delta);
}
} // namespace warpx::particles
