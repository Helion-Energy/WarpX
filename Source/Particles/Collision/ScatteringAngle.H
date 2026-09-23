/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_SCATTERING_ANGLE_H_
#define WARPX_PARTICLES_COLLISION_SCATTERING_ANGLE_H_

#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>

/** @file
 *
 * Anisotropic scattering-angle sampling for collisions: the Okhrimovskyy
 * distribution, and the rotation of a direction through a given polar and
 * azimuthal angle about itself.
 */

namespace ScatteringAngle {

    /** \brief Sample the cosine of the scattering angle from the screened-Coulomb
     * distribution of Okhrimovskyy, Bogaerts & Gijbels, Phys. Rev. E 65, 037402 (2002),
     * whose differential cross section is proportional to 1/(1 - xi cos(chi))^2.
     *
     * The cumulative distribution inverts in closed form, so one uniform deviate gives
     * one angle. xi = 0 is isotropic, xi -> 1 forward-peaked and xi < 0 backward-peaked.
     *
     * @param[in] xi anisotropy parameter, -1 < xi < 1
     * @param[in] r uniform random number in [0, 1)
     * @return cos(chi), clamped to [-1, 1]
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    double sampleOkhrimovskyyCosChi (double const xi, double const r)
    {
        const double cos_chi = 1.0 - 2.0*r*(1.0 - xi) / (1.0 + xi*(1.0 - 2.0*r));
        return amrex::min(amrex::max(cos_chi, -1.0), 1.0);
    }

    /** \brief Return the unit vector at polar angle chi and azimuth phi about the
     * direction (nx, ny, nz), which must be a unit vector.
     *
     * This is the rotation of Perez et al., Phys. Plasmas 19, 083104 (2012), Eq. 12 (used
     * inline in UpdateMomentumPerezElastic.H), up to the origin of the azimuth: both span
     * the plane perpendicular to n with the spherical unit vectors (theta_hat, -phi_hat),
     * in the opposite order, which for a uniform azimuth gives the same distribution. It
     * differs in three deliberate ways: it works on a unit vector so that the caller sets
     * the outgoing magnitude (inelastic collisions change it), it runs in double precision
     * whatever the build precision, and it changes basis at |nz| >= 0.9 rather than only
     * when the perpendicular component underflows, so the basis stays well conditioned
     * for every direction.
     *
     * @param[in] nx, ny, nz reference direction (unit vector)
     * @param[in] cos_chi cosine of the polar angle measured from the reference direction
     * @param[in] phi azimuthal angle about the reference direction
     * @param[out] x, y, z the rotated unit vector
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void rotateAboutDirection (double const nx, double const ny, double const nz,
                               double const cos_chi, double const phi,
                               double& x, double& y, double& z)
    {
        const double sin_chi = std::sqrt(amrex::max(1.0 - cos_chi*cos_chi, 0.0));
        // Build an orthonormal pair (e1, e2) perpendicular to n, crossing n with whichever
        // Cartesian axis is furthest from it so that the cross product is well conditioned.
        double e1x, e1y, e1z;
        if (std::abs(nz) < 0.9) {
            // e1 = n x z_hat
            e1x = ny;
            e1y = -nx;
            e1z = 0.0;
        } else {
            // e1 = n x x_hat
            e1x = 0.0;
            e1y = nz;
            e1z = -ny;
        }
        const double e1_norm = std::sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
        e1x /= e1_norm;
        e1y /= e1_norm;
        e1z /= e1_norm;
        // e2 = n x e1
        const double e2x = ny*e1z - nz*e1y;
        const double e2y = nz*e1x - nx*e1z;
        const double e2z = nx*e1y - ny*e1x;

        const double a = sin_chi*std::cos(phi);
        const double b = sin_chi*std::sin(phi);
        x = cos_chi*nx + a*e1x + b*e2x;
        y = cos_chi*ny + a*e1y + b*e2y;
        z = cos_chi*nz + a*e1z + b*e2z;
    }

    /** \brief Draw a scattering direction relative to an incident direction: the
     * azimuth phi, uniform on [0, 2 pi), then cos(chi) from the Okhrimovskyy
     * distribution. Two deviates, in that order.
     *
     * @param[in] xi anisotropy parameter, -1 < xi < 1 (0 is isotropic)
     * @param[in] engine the random engine
     * @param[out] cos_chi cosine of the polar angle from the incident direction
     * @param[out] phi azimuthal angle about the incident direction
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void drawOkhrimovskyyAngles (double const xi, amrex::RandomEngine const& engine,
                                 double& cos_chi, double& phi)
    {
        phi = 2.0 * MathConst::pi * amrex::Random(engine);
        cos_chi = sampleOkhrimovskyyCosChi(xi, amrex::Random(engine));
    }

    /** \brief Scatter a velocity through an angle drawn from the Okhrimovskyy
     * distribution about its own direction, giving it the magnitude vp.
     *
     * With xi == 0 this is exactly ParticleUtils::RandomizeVelocity: the same deviates,
     * in the same order, give the same vector, so an isotropic table leaves results and
     * the random stream unchanged. Otherwise it draws the same two deviates in the same
     * order (azimuth, then polar angle). A zero input velocity has no direction to
     * scatter about and falls back to isotropic.
     *
     * @param[in,out] ux, uy, uz velocity before (in) and after (out) scattering
     * @param[in] vp velocity magnitude after scattering
     * @param[in] xi anisotropy parameter, -1 < xi < 1
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ScatterVelocityOkhrimovskyy ( amrex::ParticleReal& ux, amrex::ParticleReal& uy,
                                       amrex::ParticleReal& uz,
                                       const amrex::ParticleReal vp,
                                       const amrex::ParticleReal xi,
                                       amrex::RandomEngine const& engine )
    {
        const auto ux_d = static_cast<double>(ux);
        const auto uy_d = static_cast<double>(uy);
        const auto uz_d = static_cast<double>(uz);
        const double u_norm = std::sqrt(ux_d*ux_d + uy_d*uy_d + uz_d*uz_d);
        if (xi == 0 || !(u_norm > 0.0)) {
            ParticleUtils::RandomizeVelocity(ux, uy, uz, vp, engine);
            return;
        }
        double cos_chi, phi;
        drawOkhrimovskyyAngles(xi, engine, cos_chi, phi);
        double x, y, z;
        rotateAboutDirection(ux_d/u_norm, uy_d/u_norm, uz_d/u_norm, cos_chi, phi, x, y, z);
        ux = static_cast<amrex::ParticleReal>(x * vp);
        uy = static_cast<amrex::ParticleReal>(y * vp);
        uz = static_cast<amrex::ParticleReal>(z * vp);
    }

}

#endif // WARPX_PARTICLES_COLLISION_SCATTERING_ANGLE_H_
