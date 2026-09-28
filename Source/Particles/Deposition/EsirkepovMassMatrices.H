/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ESIRKEPOV_MASS_MATRICES_H_
#define WARPX_ESIRKEPOV_MASS_MATRICES_H_

#include "Particles/Deposition/MassMatricesDeposition.H"

#include <AMReX_ConstexprFor.H>

// RZ m=0 implementation of the current response to a frozen magnetic-force
// matrix and a full-order nodal momentum gather. It differentiates the actual
// Esirkepov endpoint shapes, including their cross terms. The resulting bands
// are nonsymmetric: the energy-adjoint diagonal-band completion is inapplicable.
#if defined(WARPX_DIM_RZ)
namespace esirkepov_mm {

template <int Order>
struct Shape
{
    double weights[Order+1]{};
    int first;
    AMREX_GPU_HOST_DEVICE explicit Shape (double x)
        : first(Compute_shape_factor<Order>{}(weights, x)) {}
    AMREX_GPU_HOST_DEVICE double operator() (int i) const
    {
        int const slot = i-first;
        return slot >= 0 && slot <= Order ? weights[slot] : 0.0;
    }
};

template <int Order>
struct Derivative
{
    Shape<Order-1> plus, minus;
    AMREX_GPU_HOST_DEVICE explicit Derivative (double x) : plus(x+0.5), minus(x-0.5) {}
    AMREX_GPU_HOST_DEVICE double operator() (int i) const { return plus(i)-minus(i); }
};

/** Build fully stored current-row / nodal-electric-column mass matrices.
 * Coordinate arrays refer to the physical nonlinear midpoint orbit. E gather
 * weights and magnetic rotation are held, as in the existing MM approximation;
 * the deposition response includes the induced endpoint displacement.
 */
template <int Order>
void Deposit (
    GetParticlePosition<PIdx> const& position,
    amrex::ParticleReal const* xn, amrex::ParticleReal const* yn,
    amrex::ParticleReal const* zn, amrex::ParticleReal const* weight,
    amrex::ParticleReal const* uxn, amrex::ParticleReal const* uyn,
    amrex::ParticleReal const* uzn, amrex::ParticleReal const* ux,
    amrex::ParticleReal const* uy, amrex::ParticleReal const* uz,
    amrex::GpuArray<amrex::Array4<amrex::Real>,9> const matrices,
    amrex::GpuArray<amrex::Array4<amrex::Real const>,3> const magnetic,
    amrex::GpuArray<amrex::IndexType,3> const magnetic_type,
    long const count, int const width, amrex::Real const dt,
    amrex::XDim3 const dinv, amrex::XDim3 const xyzmin, amrex::Dim3 const lo,
    amrex::Real const charge, amrex::Real const mass)
{
    int const radius = (width-1)/2;
    // The particle response needs enough registers that 256-thread blocks
    // admit only one resident block on common GPUs. Smaller blocks allow more
    // active warps without changing the deposition or its particle state.
    amrex::For<128>(count, [=] AMREX_GPU_DEVICE(long ip) {
        amrex::ParticleReal x, y, z;
        position(ip,x,y,z);
        amrex::Real const r = std::hypot(x,y);
        amrex::Real const ro = std::hypot(xn[ip],yn[ip]);
        amrex::Real const xe = 2.0*x-xn[ip], ye = 2.0*y-yn[ip];
        amrex::Real const re = std::hypot(xe,ye), ze = 2.0*z-zn[ip];
        amrex::Real const ravg = 0.5*(ro+re);
        amrex::Real const cs = r > 0 ? x/r : 1.0;
        amrex::Real const sn = r > 0 ? y/r : 0.0;
        amrex::ParticleReal bx = 0, by = 0, bz = 0;
        doDirectGatherVectorField<Order,Order>(x,y,z,bx,by,bz,
            magnetic[0],magnetic[1],magnetic[2],
            magnetic_type[0],magnetic_type[1],magnetic_type[2],
            dinv,xyzmin,lo,1);
        amrex::ParticleReal response[9];
        auto const gi = GetImplicitGammaInverse(uxn[ip],uyn[ip],uzn[ip],ux[ip],uy[ip],uz[ip]);
        setMassMatricesKernels(charge,mass,dt,gi,cs,sn,ux[ip],uy[ip],uz[ip],bx,by,bz,
            response[0],response[1],response[2],response[3],response[4],response[5],
            response[6],response[7],response[8]);
        Shape<Order> const sr0((ro-xyzmin.x)*dinv.x), sr1((re-xyzmin.x)*dinv.x);
        Shape<Order> const sz0((zn[ip]-xyzmin.z)*dinv.z), sz1((ze-xyzmin.z)*dinv.z);
        Shape<Order> const gr((r-xyzmin.x)*dinv.x), gz((z-xyzmin.z)*dinv.z);
        Derivative<Order> const dr1((re-xyzmin.x)*dinv.x), dz1((ze-xyzmin.z)*dinv.z);
        int const ir0 = std::min(sr0.first,sr1.first);
        int const ir1 = std::max(sr0.first,sr1.first)+Order;
        int const iz0 = std::min(sz0.first,sz1.first);
        int const iz1 = std::max(sz0.first,sz1.first)+Order;
        amrex::Real const qvol = charge*weight[ip]*dinv.x*dinv.y*dinv.z;
        amrex::Real const vt = ravg > 0 ? (x*uy[ip]-y*ux[ip])*gi/ravg : 0.0;
        amrex::Real dre[3], dze[3], dvt[3];
        for (int e = 0; e < 3; ++e) {
            amrex::Real const dvx = cs*response[e]-sn*response[3+e];
            amrex::Real const dvy = sn*response[e]+cs*response[3+e];
            dre[e] = re > 0 ? dt*(xe*dvx+ye*dvy)/re : 0.0;
            dze[e] = dt*response[6+e];
            dvt[e] = ravg > 0 ? (xn[ip]*dvy-yn[ip]*dvx-0.5*vt*dre[e])/ravg : 0.0;
        }
        amrex::Real sum_r = 0, sum_dr = 0;
        for (int i = ir0; i <= ir1; ++i) {
            sum_r += sr0(i)-sr1(i);
            sum_dr += dr1(i);
            amrex::Real sum_z = 0, sum_dz = 0;
            for (int j = iz0; j <= iz1; ++j) {
                sum_z += sz0(j)-sz1(j);
                sum_dz += dz1(j);
                amrex::Real row[9]{};
                amrex::Real const wt = (sr1(i)*sz1(j)+sr0(i)*sz0(j))/3.0
                    +(sr1(i)*sz0(j)+sr0(i)*sz1(j))/6.0;
                for (int e = 0; e < 3; ++e) {
                    if (i < ir1) {
                        row[e] = qvol/(dt*dinv.x)*(
                            -sum_dr*dre[e]*dinv.x*0.5*(sz0(j)+sz1(j))
                            +sum_r*0.5*dz1(j)*dze[e]*dinv.z);
                    }
                    row[3+e] = qvol*(dvt[e]*wt+vt*(
                        dr1(i)*dre[e]*dinv.x*(sz1(j)/3.0+sz0(j)/6.0)
                        +dz1(j)*dze[e]*dinv.z*(sr1(i)/3.0+sr0(i)/6.0)));
                    if (j < iz1) {
                        row[6+e] = qvol/(dt*dinv.z)*(
                            0.5*dr1(i)*dre[e]*dinv.x*sum_z
                            -0.5*(sr0(i)+sr1(i))*sum_dz*dze[e]*dinv.z);
                    }
                }
                int const di0 = gr.first-i+radius;
                int const dj0 = gz.first-j+radius;
                AMREX_ALWAYS_ASSERT(di0 >= 0 && di0+Order < width
                    && dj0 >= 0 && dj0+Order < width);
                // Keep the matrix block and gather offsets compile-time
                // constants. This avoids dynamically indexed per-thread arrays
                // in the scatter and reuses each current response across its
                // full electric gather stencil.
                amrex::constexpr_for<0,9>([&] (auto block) {
                    constexpr int q = block;
                    if ((q < 3 && i == ir1) || (q >= 6 && j == iz1)) { return; }
                    amrex::constexpr_for<0,Order+1>([&] (auto a) {
                        amrex::constexpr_for<0,Order+1>([&] (auto b) {
                            int const component = di0+a+width*(dj0+b);
                            amrex::Real const gather = gr.weights[a]*gz.weights[b];
                            amrex::Gpu::Atomic::AddNoRet(
                                &matrices[q](lo.x+i,lo.y+j,0,component),row[q]*gather);
                        });
                    });
                });
            }
        }
    });
}
} // namespace esirkepov_mm
#endif // WARPX_DIM_RZ
#endif
