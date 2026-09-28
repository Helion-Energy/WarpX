/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ThermalEnergyBalance.H"
#include "Utils/WarpXConst.H"
#include <AMReX_MFIter.H>
#include <AMReX_Reduce.H>
#include <array>
#include <cmath>
namespace warpx::thermal {
ThermalEnergyBalance MeasureThermalEnergyBalance(
    const EulerianThermalStage& stage, const amrex::MultiFab& energy)
{
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
    using Real=amrex::Real;
    auto const& geom=stage.Geometry();
    auto const dx=geom.CellSizeArray();
#if defined(WARPX_DIM_RZ)
    Real const rlo=geom.ProbLo(0);
#endif
    Real const dt=stage.Options().dt;
    Real const theta=stage.Options().theta, gm1=stage.Options().gamma-1;
    AMREX_ALWAYS_ASSERT(energy.boxArray()==stage.OldEnergy().boxArray() &&
        energy.DistributionMap()==stage.OldEnergy().DistributionMap());
    amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
        amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
        amrex::ReduceOpSum,amrex::ReduceOpSum> op;
    amrex::ReduceData<Real,Real,Real,Real,Real,Real,Real,Real> data(op);
    using Tuple=decltype(data)::Type;
    for (amrex::MFIter mfi(energy);mfi.isValid();++mfi) {
        auto const u=energy.const_array(mfi),old=stage.OldEnergy().const_array(mfi);
        auto const q=stage.EvaluatedSource().const_array(mfi);
        auto const qr=stage.FaceFlux(0).const_array(mfi),qz=stage.FaceFlux(1).const_array(mfi);
        auto const fr=stage.FaceAdvectionFlux(0).const_array(mfi),fz=stage.FaceAdvectionFlux(1).const_array(mfi);
        auto const vr=stage.FaceVelocity(0).const_array(mfi),vz=stage.FaceVelocity(1).const_array(mfi);
#if defined(WARPX_DIM_3D)
        auto const q2=stage.FaceFlux(2).const_array(mfi);
        auto const f2=stage.FaceAdvectionFlux(2).const_array(mfi);
        auto const v2=stage.FaceVelocity(2).const_array(mfi);
#endif
        op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
#if defined(WARPX_DIM_RZ)
            Real const r=rlo+(i+0.5)*dx[0],rl=rlo+i*dx[0],rh=rl+dx[0];
            Real const volume=2*MathConst::pi*r*dx[0]*dx[1];
            Real const conductive=-dt*((rh*qr(i+1,j,k)-rl*qr(i,j,k))/(r*dx[0])+
                (qz(i,j+1,k)-qz(i,j,k))/dx[1]);
            Real const advective=-dt*((rh*fr(i+1,j,k)-rl*fr(i,j,k))/(r*dx[0])+
                (fz(i,j+1,k)-fz(i,j,k))/dx[1]);
            Real const compressive=-dt*gm1*u(i,j,k)*((rh*vr(i+1,j,k)-rl*vr(i,j,k))/(r*dx[0])+
                (vz(i,j+1,k)-vz(i,j,k))/dx[1]);
#else
            Real const volume=dx[0]*dx[1]*dx[2];
            Real const conductive=-dt*((qr(i+1,j,k)-qr(i,j,k))/dx[0]+
                (qz(i,j+1,k)-qz(i,j,k))/dx[1]+(q2(i,j,k+1)-q2(i,j,k))/dx[2]);
            Real const advective=-dt*((fr(i+1,j,k)-fr(i,j,k))/dx[0]+
                (fz(i,j+1,k)-fz(i,j,k))/dx[1]+(f2(i,j,k+1)-f2(i,j,k))/dx[2]);
            Real const compressive=-dt*gm1*u(i,j,k)*((vr(i+1,j,k)-vr(i,j,k))/dx[0]+
                (vz(i,j+1,k)-vz(i,j,k))/dx[1]+(v2(i,j,k+1)-v2(i,j,k))/dx[2]);
#endif
            Real const delta=(u(i,j,k)-old(i,j,k))/theta;
            Real const source=dt*q(i,j,k);
            Real const defect=delta-source-conductive-advective-compressive;
            return {old(i,j,k)*volume,(old(i,j,k)+delta)*volume,source*volume,
                conductive*volume,advective*volume,compressive*volume,defect*volume,
                std::abs(defect)*volume};
        });
    }
    auto const t=data.value();
    std::array<Real,8> v{amrex::get<0>(t),amrex::get<1>(t),amrex::get<2>(t),
        amrex::get<3>(t),amrex::get<4>(t),amrex::get<5>(t),amrex::get<6>(t),amrex::get<7>(t)};
    amrex::ParallelDescriptor::ReduceRealSum(v.data(),int(v.size()));
    for (Real x:v) { AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(x),"Nonfinite thermal energy receipt"); }
    return {v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]};
#else
    amrex::ignore_unused(stage,energy);
    amrex::Abort("Thermal energy receipt requires an RZ or Cartesian 3D stage");
    return {};
#endif
}
} // namespace warpx::thermal
