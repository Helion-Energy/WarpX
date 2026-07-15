/* Copyright 2026 WarpX neutral-DSMC extension
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 *
 * Thermal (diffuse) particle re-emission from embedded boundaries, for
 * boundary.particle_eb = thermal: a fully accommodating wall model for
 * neutral-gas DSMC. Specular reflection is handled separately by
 * boundary.particle_eb = reflecting (ParticleBoundaryProcess::Reflect).
 */
#ifndef WARPX_PARTICLEREFLECTATEB_H_
#define WARPX_PARTICLEREFLECTATEB_H_

#include "EmbeddedBoundary/DistanceToEB.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Initialization/SampleGaussianFluxDistribution.H"
#include "Utils/WarpXConst.H"

#include <ablastr/particles/NodalFieldGather.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particle.H>
#include <AMReX_Random.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_TypeTraits.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <type_traits>

/**
 * \brief Re-emit particles thermally from the embedded boundary (fully
 * accommodating diffuse wall, boundary.particle_eb = thermal).
 *
 * Detects particles that have crossed into the EB region (signed distance < 0)
 * and, rather than absorbing them, re-emits them from a wall Maxwellian. For
 * each such particle (following the bisection approach of
 * ParticleBoundaryProcess::Reflect and
 * ParticleBoundaryBuffer::FindEmbeddedBoundaryIntersection):
 * 1. Bisect along the particle trajectory to find the exact contact point on
 *    the EB surface (where the signed distance phi = 0).
 * 2. Compute the surface normal at that contact point (not at the penetrated
 *    position), which handles curved EB surfaces correctly.
 * 3. Re-emit from the wall Maxwellian: a half-Maxwellian flux distribution
 *    along the inward normal and a full Maxwellian in the tangential plane,
 *    with thermal speed uth (the same convention and sampling primitives as
 *    the domain thermal boundary, see thermalize_boundary_particle).
 * 4. Advance the particle from the contact point with the new velocity for
 *    the remaining fraction of the timestep.
 * The particle id is kept valid (the particle is not removed), except for the
 * corner safeguards below, which absorb rather than mis-emit:
 * (a) the particle did not start the step in the gas region (sharp corner,
 *     tunneling through a thin feature, bad initialization): there is no wall
 *     crossing to bracket;
 * (b) the level-set normal at the contact point is degenerate (ill-defined at
 *     a sharp corner);
 * (c) the re-emitted particle lands back inside the solid (sharp concave
 *     corner).
 *
 * \param pc particle container to operate on
 * \param distance_to_eb signed-distance MultiFabs (one per level)
 * \param mass particle species mass
 * \param uth wall thermal speed normalized by c: sqrt(kB*T_wall/m)/c, parsed
 *            per species from boundary.<species_name>.u_th (the same input
 *            the domain thermal boundary condition uses)
 * \param dt_lev time step at each refinement level
 */
template <class PC, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
thermalReflectParticlesAtEB (PC& pc,
                             ablastr::fields::MultiLevelScalarField const& distance_to_eb,
                             amrex::ParticleReal mass,
                             amrex::ParticleReal uth,
                             amrex::Vector<amrex::Real> const& dt_lev)
{
    using namespace amrex::literals;
    BL_PROFILE("thermalReflectParticlesAtEB");

    for (int lev = 0; lev <= pc.finestLevel(); ++lev)
    {
        const auto plo = pc.Geom(lev).ProbLoArray();
        const auto dxi = pc.Geom(lev).InvCellSizeArray();
        const amrex::Real dt = dt_lev[lev];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
        {
            const auto getPosition = GetParticlePosition<PIdx>(pti);
            const auto setPosition = SetParticlePosition<PIdx>(pti);
            auto& tile = pti.GetParticleTile();
            auto ptd = tile.getParticleTileData();
            auto& soa = pti.GetStructOfArrays();
            const auto np = tile.numParticles();
            auto phi = (*distance_to_eb[lev])[pti].array();
            amrex::ParticleReal* const AMREX_RESTRICT ux = soa.GetRealData(PIdx::ux).data();
            amrex::ParticleReal* const AMREX_RESTRICT uy = soa.GetRealData(PIdx::uy).data();
            amrex::ParticleReal* const AMREX_RESTRICT uz = soa.GetRealData(PIdx::uz).data();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (const int ip, amrex::RandomEngine const& engine) noexcept
            {
                if (!amrex::ParticleIDWrapper{ptd.m_idcpu[ip]}.is_valid()) { return; }

                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                int i, j, k;
                amrex::Real W[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    xp, yp, zp, plo, dxi, i, j, k, W);
                amrex::Real const phi_value = ablastr::particles::interp_field_nodal(i, j, k, W, phi);
                if (phi_value >= 0.0_rt) { return; }  // particle still in the gas region

                const amrex::ParticleReal ux_ip = ux[ip];
                const amrex::ParticleReal uy_ip = uy[ip];
                const amrex::ParticleReal uz_ip = uz[ip];

                // Corner safeguard (a): reconstruct the start-of-step position (the position
                // push moved the particle by v*dt with u unchanged). If the particle did not
                // start the step in the gas region (sharp corner, tunneling through a thin
                // feature, bad initialization), there is no wall crossing to bracket -> absorb
                // it (otherwise it can accumulate in the solid over long runs).
                {
                    amrex::ParticleReal xs = xp, ys = yp, zs = zp;
                    UpdatePosition(xs, ys, zs, ux_ip, uy_ip, uz_ip, -dt, mass);
                    int is, js, ks;
                    amrex::Real Ws[AMREX_SPACEDIM][2];
                    ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                        xs, ys, zs, plo, dxi, is, js, ks, Ws);
                    if (ablastr::particles::interp_field_nodal(is, js, ks, Ws, phi) < 0.0_rt) {
                        amrex::ParticleIDWrapper{ptd.m_idcpu[ip]}.make_invalid();
                        return;
                    }
                }

                // Bisect along the trajectory to find the fraction of dt to walk backward
                // from the end-of-step position to the exact contact point on the EB
                // surface (phi = 0). This handles curved surfaces correctly, unlike
                // mirroring across the local tangent plane (x_new = x - 2*phi*n), which is
                // only exact for planar walls.
                const auto dt_fraction = static_cast<amrex::Real>(amrex::bisect(0.0, 1.0,
                    [=] (amrex::Real dt_frac) {
                        amrex::ParticleReal xt = xp, yt = yp, zt = zp;
                        UpdatePosition(xt, yt, zt, ux_ip, uy_ip, uz_ip, -dt_frac*dt, mass);
                        int bi, bj, bk;
                        amrex::Real bW[AMREX_SPACEDIM][2];
                        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                            xt, yt, zt, plo, dxi, bi, bj, bk, bW);
                        return ablastr::particles::interp_field_nodal(bi, bj, bk, bW, phi);
                    }));

                // Contact point on the EB surface
                amrex::ParticleReal x_c = xp, y_c = yp, z_c = zp;
                UpdatePosition(x_c, y_c, z_c, ux_ip, uy_ip, uz_ip, -dt_fraction*dt, mass);

                // Unit surface normal at the contact point, in 3D Cartesian
                // coordinates for every dimensionality (interp_normal
                // normalizes internally and rotates radial normals to the
                // particle azimuth).
                auto const n = DistanceToEB::interp_normal(x_c, y_c, z_c, plo, dxi, phi);
                // Corner safeguard (b): degenerate level-set gradient at a sharp corner
                // makes the internally-normalized normal non-finite -> absorb.
                if (!(std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2]))) {
                    amrex::ParticleIDWrapper{ptd.m_idcpu[ip]}.make_invalid();
                    return;
                }

                // Diffuse (thermal) re-emission from the wall Maxwellian, in the
                // orthonormal frame (n, t1, t2): half-Maxwellian flux along the
                // inward normal, full Maxwellian tangentially. Valid for every
                // dimensionality since both the normal and the momenta are 3D
                // Cartesian (in 2D the out-of-plane direction is one of the
                // tangentials; in RZ the normal is pre-rotated to the azimuth).
                const auto cc = static_cast<amrex::ParticleReal>(PhysConst::c);
                amrex::ParticleReal e0 = 0._prt, e1 = 0._prt, e2 = 0._prt;
                if (std::abs(n[0]) <= std::abs(n[1]) && std::abs(n[0]) <= std::abs(n[2])) {
                    e0 = 1._prt;
                } else if (std::abs(n[1]) <= std::abs(n[2])) {
                    e1 = 1._prt;
                } else {
                    e2 = 1._prt;
                }
                const amrex::ParticleReal edn = e0*n[0] + e1*n[1] + e2*n[2];
                amrex::ParticleReal t10 = e0 - edn*n[0];
                amrex::ParticleReal t11 = e1 - edn*n[1];
                amrex::ParticleReal t12 = e2 - edn*n[2];
                const amrex::ParticleReal t1n =
                    1._prt/std::sqrt(t10*t10 + t11*t11 + t12*t12);
                t10 *= t1n; t11 *= t1n; t12 *= t1n;
                const amrex::ParticleReal t20 = n[1]*t12 - n[2]*t11;
                const amrex::ParticleReal t21 = n[2]*t10 - n[0]*t12;
                const amrex::ParticleReal t22 = n[0]*t11 - n[1]*t10;

                const amrex::ParticleReal un =
                    cc*static_cast<amrex::ParticleReal>(generateGaussianFluxDist(0._rt, uth, engine));
                const amrex::ParticleReal v1 =
                    cc*static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, uth, engine));
                const amrex::ParticleReal v2 =
                    cc*static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, uth, engine));
                ux[ip] = un*n[0] + v1*t10 + v2*t20;
                uy[ip] = un*n[1] + v1*t11 + v2*t21;
                uz[ip] = un*n[2] + v1*t12 + v2*t22;

                // Advance from the contact point with the new velocity for the remaining
                // fraction of the timestep.
                xp = x_c; yp = y_c; zp = z_c;
                UpdatePosition(xp, yp, zp, ux[ip], uy[ip], uz[ip], dt_fraction*dt, mass);
                // Corner safeguard (c): if the re-emitted particle is STILL inside the solid
                // (sharp concave corner where the post-emission advance re-enters the wall),
                // absorb it rather than let it accumulate in the body.
                {
                    int ir, jr, kr;
                    amrex::Real Wr[AMREX_SPACEDIM][2];
                    ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                        xp, yp, zp, plo, dxi, ir, jr, kr, Wr);
                    if (ablastr::particles::interp_field_nodal(ir, jr, kr, Wr, phi) < 0.0_rt) {
                        amrex::ParticleIDWrapper{ptd.m_idcpu[ip]}.make_invalid();
                        return;
                    }
                }
                setPosition(ip, xp, yp, zp);
            });
        }
    }
}

#endif // WARPX_PARTICLEREFLECTATEB_H_
