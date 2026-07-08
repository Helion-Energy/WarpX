/* Copyright 2026 WarpX neutral-DSMC extension
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 *
 * Reflect particles off embedded boundaries (specular + diffuse/thermal),
 * a Maxwell accommodation wall model for neutral-gas DSMC. WarpX ships only
 * absorbing EB particle interaction; this adds reflection so that interior
 * solid geometry (orifice plates, tubes) conserves the gas it should.
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
 * \brief Reflect particles off the embedded boundary (Maxwell accommodation model).
 *
 * Detects particles that have crossed into the EB region (signed distance < 0) and,
 * rather than absorbing them, reflects them back into the domain. For each such
 * particle (following the bisection approach of PR #6588 and
 * ParticleBoundaryBuffer::FindEmbeddedBoundaryIntersection):
 * 1. Bisect along the particle trajectory to find the exact contact point on the
 *    EB surface (where the signed distance phi = 0).
 * 2. Compute the surface normal at that contact point (not at the penetrated
 *    position), which handles curved EB surfaces correctly.
 * 3. Update the velocity at the contact point: with probability `accommodation`
 *    the reflection is diffuse (re-emitted from a wall Maxwellian: a
 *    half-Maxwellian flux along the inward normal, full Maxwellian in the
 *    tangential plane); otherwise it is specular (mirror the normal velocity
 *    component).
 * 4. Advance the particle from the contact point with the new velocity for the
 *    remaining fraction of the timestep.
 * The particle id is kept valid (the particle is not removed).
 *
 * \param pc particle container to operate on
 * \param distance_to_eb signed-distance MultiFabs (one per level)
 * \param mass particle species mass
 * \param uth wall thermal speed normalized by c: sqrt(kB*T_wall/m)/c (used for diffuse)
 * \param accommodation probability of diffuse vs specular reflection, in [0,1]
 * \param dt_lev time step at each refinement level
 */
template <class PC, std::enable_if_t<amrex::IsParticleContainer<PC>::value, int> foo = 0>
void
reflectParticlesAtEB (PC& pc,
                      ablastr::fields::MultiLevelScalarField const& distance_to_eb,
                      amrex::ParticleReal mass,
                      amrex::ParticleReal uth,
                      amrex::ParticleReal accommodation,
                      amrex::Vector<amrex::Real> const& dt_lev)
{
    using namespace amrex::literals;
    BL_PROFILE("reflectParticlesAtEB");

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

                // Surface normal at the contact point
                int in, jn, kn;
                amrex::Real Wn[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x_c, y_c, z_c, plo, dxi, in, jn, kn, Wn);
                int ic, jc, kc;
                amrex::Real Wc[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::CELL>(
                    x_c, y_c, z_c, plo, dxi, ic, jc, kc, Wc);
                amrex::RealVect n = DistanceToEB::interp_normal(in, jn, kn, Wn, ic, jc, kc, Wc, phi, dxi);
                // Corner safeguard (b): degenerate normal (ill-defined at a sharp corner) -> absorb.
                if (DistanceToEB::dot_product(n, n) < amrex::Real(1.e-20)) {
                    amrex::ParticleIDWrapper{ptd.m_idcpu[ip]}.make_invalid();
                    return;
                }
                DistanceToEB::normalize(n);  // unit normal pointing from the wall into the gas

                const bool diffuse = (accommodation > 0._prt) &&
                    ( accommodation >= 1._prt || amrex::Random(engine) < accommodation );
                const auto cc = static_cast<amrex::ParticleReal>(PhysConst::c);

#if defined(WARPX_DIM_3D)
                if (!diffuse) {
                    const amrex::ParticleReal un = ux[ip]*n[0] + uy[ip]*n[1] + uz[ip]*n[2];
                    ux[ip] -= 2._prt*un*n[0];
                    uy[ip] -= 2._prt*un*n[1];
                    uz[ip] -= 2._prt*un*n[2];
                } else {
                    // orthonormal frame (n, t1, t2) for diffuse re-emission
                    amrex::RealVect e = ( std::abs(n[0]) <= std::abs(n[1]) && std::abs(n[0]) <= std::abs(n[2]) )
                        ? amrex::RealVect{1.0_rt, 0.0_rt, 0.0_rt}
                        : ( (std::abs(n[1]) <= std::abs(n[2]))
                              ? amrex::RealVect{0.0_rt, 1.0_rt, 0.0_rt}
                              : amrex::RealVect{0.0_rt, 0.0_rt, 1.0_rt} );
                    const amrex::Real edn = DistanceToEB::dot_product(e, n);
                    amrex::RealVect t1 = {e[0]-edn*n[0], e[1]-edn*n[1], e[2]-edn*n[2]};
                    DistanceToEB::normalize(t1);
                    const amrex::RealVect t2 = {n[1]*t1[2]-n[2]*t1[1],
                                                n[2]*t1[0]-n[0]*t1[2],
                                                n[0]*t1[1]-n[1]*t1[0]};
                    const amrex::ParticleReal un = cc*static_cast<amrex::ParticleReal>(generateGaussianFluxDist(0._rt, uth, engine));
                    const amrex::ParticleReal v1 = cc*static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, uth, engine));
                    const amrex::ParticleReal v2 = cc*static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, uth, engine));
                    ux[ip] = un*n[0] + v1*t1[0] + v2*t2[0];
                    uy[ip] = un*n[1] + v1*t1[1] + v2*t2[1];
                    uz[ip] = un*n[2] + v1*t1[2] + v2*t2[2];
                }

#elif defined(WARPX_DIM_XZ)
                if (!diffuse) {
                    const amrex::ParticleReal un = ux[ip]*n[0] + uz[ip]*n[1];
                    ux[ip] -= 2._prt*un*n[0];
                    uz[ip] -= 2._prt*un*n[1];
                } else {
                    const amrex::ParticleReal un = cc*static_cast<amrex::ParticleReal>(generateGaussianFluxDist(0._rt, uth, engine));
                    const amrex::ParticleReal vt = cc*static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, uth, engine));
                    const amrex::ParticleReal vy = cc*static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, uth, engine));
                    ux[ip] = un*n[0] - vt*n[1];   // in-plane tangential t = (-n[1], n[0])
                    uz[ip] = un*n[1] + vt*n[0];
                    uy[ip] = vy;                   // out-of-plane component
                }

#elif defined(WARPX_DIM_RZ)
                // EB is not supported in RZ at runtime; this branch only needs to compile.
                // The normal is (n_r, n_z): rotate it to the particle azimuth at the
                // contact point to work with the Cartesian momentum components.
                const amrex::ParticleReal r = std::sqrt(x_c*x_c + y_c*y_c);
                const amrex::ParticleReal cr = (r > 0._prt) ? x_c/r : 1._prt;
                const amrex::ParticleReal sr = (r > 0._prt) ? y_c/r : 0._prt;
                amrex::ParticleReal ur = ux[ip]*cr + uy[ip]*sr;
                const amrex::ParticleReal ut = -ux[ip]*sr + uy[ip]*cr;
                if (!diffuse) {
                    const amrex::ParticleReal un = ur*n[0] + uz[ip]*n[1];
                    ur -= 2._prt*un*n[0];
                    uz[ip] -= 2._prt*un*n[1];
                } else {
                    const amrex::ParticleReal un = cc*static_cast<amrex::ParticleReal>(generateGaussianFluxDist(0._rt, uth, engine));
                    const amrex::ParticleReal vt = cc*static_cast<amrex::ParticleReal>(amrex::RandomNormal(0._rt, uth, engine));
                    ur = un*n[0] - vt*n[1];
                    uz[ip] = un*n[1] + vt*n[0];
                }
                ux[ip] = ur*cr - ut*sr;
                uy[ip] = ur*sr + ut*cr;

#elif defined(WARPX_DIM_1D_Z)
                if (!diffuse) {
                    uz[ip] -= 2._prt*(uz[ip]*n[0])*n[0];
                } else {
                    const amrex::ParticleReal s = (n[0] >= 0._rt) ? 1._prt : -1._prt;
                    uz[ip] = s*cc*static_cast<amrex::ParticleReal>(generateGaussianFluxDist(0._rt, uth, engine));
                }
#endif
                // Advance from the contact point with the new velocity for the remaining
                // fraction of the timestep.
                xp = x_c; yp = y_c; zp = z_c;
                UpdatePosition(xp, yp, zp, ux[ip], uy[ip], uz[ip], dt_fraction*dt, mass);
                // Corner safeguard (c): if the reflected particle is STILL inside the solid
                // (sharp concave corner where the post-reflection advance re-enters the wall),
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
