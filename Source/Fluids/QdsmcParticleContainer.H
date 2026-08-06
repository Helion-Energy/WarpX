/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Acciarri, Prabhat Kumar (Helion Energy Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_QDSMCPARTICLECONTAINER_H_
#define WARPX_QDSMCPARTICLECONTAINER_H_

#include "QdsmcParticleContainer_fwd.H"

#include <AMReX_AmrCoreFwd.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_ParIter.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>


/**
 * @brief Indexing for the QDSMC fictitious-particle SoA storage.
 *
 * QDSMC particles are one-per-node Lagrangian markers used to transport the
 * electron entropy K_e by the electron fluid velocity V_e over one PIC step
 * (see Belyaev et al., Phys. Plasmas 31, 012902 (2024), Sec. III.A). Homing
 * the markers at the nodes of the (nodal) QDSMC grid fields makes the
 * at-rest gather/deposit round trip the identity; cell-center homes would
 * smooth the fields by a [1/2,1/2]^dim box filter every step (a numerical
 * diffusion ~ dx^2/(4*dt) per direction, independent of dt).
 *
 * Layout: AMReX's `ParticleContainerPureSoA\<NReal=nattribs, NInt=0\>` treats the
 * first AMREX_SPACEDIM real attributes as the AMReX particle positions used
 * for tile-assignment and `Redistribute()`. The remaining attributes are
 * user data.
 *
 *   Position attributes (AMReX-tracked, dim-dependent slot count):
 *     1D Z : [z]               -> AMReX position 0
 *     XZ/RZ: [x, z]            -> AMReX positions 0, 1
 *     3D   : [x, y, z]         -> AMReX positions 0, 1, 2
 *
 *   User attributes (always present, 3D semantic regardless of field dim):
 *     x_node, y_node, z_node : home position (owning grid node)
 *     vx, vy, vz             : electron-fluid velocity gathered at home (3 components)
 *     entropy                : weighted entropy K_e * N_e carried by this particle
 *     np_real                : weight N_e (electrons in this node's dual cell)
 *
 * The home positions and velocities are always stored as 3 components even when
 * the field dimension is lower. In 2D the y-home is set to 0 at init; in 1D
 * both x-home and y-home are 0. The gathers and scatters consume (x, y, z) as
 * a 3-vector and the underlying nodal shape weights route them to the right
 * field axes per dim (e.g. r = sqrt(x^2 + y^2) for RZ).
 */
struct QdsmcPIdx
{
    enum
    {
        // AMReX position slots (the first AMREX_SPACEDIM entries).
#if !defined(WARPX_DIM_1D_Z)
        x,
#endif
#if defined(WARPX_DIM_3D)
        y,
#endif
        z,
        // Home position (3D, the owning node; gather target at start of step).
        x_node, y_node, z_node,
        // Electron-fluid velocity gathered at home (3 components, always).
        vx, vy, vz,
        // QDSMC transported scalars: weighted entropy and electron count.
        entropy, np_real,
        // Limited (MC) slopes of the transported scalars, per GRID dimension
        // (d0/d1/d2; unused above AMREX_SPACEDIM stay 0), in physical units
        // [attr / m]. Loaded by SetK and consumed by the gradient-corrected
        // deposit; zeroed by ResetParticles.
        entropy_g0, entropy_g1, entropy_g2,
        np_g0, np_g1, np_g2,
        nattribs
    };
};


/**
 * @brief Container for QDSMC fictitious electron-energy-equation particles.
 *
 * One particle is initialized per grid node; the particle's velocity
 * is set to V_e (gathered at home), its entropy slot to K_e * N_e, its weight
 * slot to N_e, and the particle is then pushed by one PIC step. After the
 * push, entropy and weight are scattered back to the grid; the new K_e is
 * recovered as (deposited entropy) / (deposited weight), and the electron
 * temperature is then computed from K_e and the updated ion-derived n_e via
 * the polytropic relation.
 *
 * Inherits from `ParticleContainerPureSoA` so all attributes live in
 * struct-of-arrays storage; positions are the first AMREX_SPACEDIM slots.
 */
class QdsmcParticleContainer
    : public amrex::ParticleContainerPureSoA<QdsmcPIdx::nattribs, 0>
{
public:
    static constexpr int NStructReal = 0;
    static constexpr int NStructInt = 0;
    static constexpr int NReal = QdsmcPIdx::nattribs;
    static constexpr int NInt = 0;

    explicit QdsmcParticleContainer (amrex::AmrCore* amr_core);
    ~QdsmcParticleContainer() override = default;

    QdsmcParticleContainer ( QdsmcParticleContainer const &)             = delete;
    QdsmcParticleContainer& operator= ( QdsmcParticleContainer const & ) = delete;
    QdsmcParticleContainer ( QdsmcParticleContainer&& )                  = default;
    QdsmcParticleContainer& operator= ( QdsmcParticleContainer&& )       = default;

    using iterator = amrex::ParIterSoA<QdsmcPIdx::nattribs, 0>;
    using const_iterator = amrex::ParConstIterSoA<QdsmcPIdx::nattribs, 0>;

    /**
     * @brief Initialize one particle per grid node.
     *
     * Node ownership is unique: box-seam nodes belong to the lower box, and
     * in periodic directions the domain's top node (the image of the bottom
     * one) gets no marker. At a non-periodic domain-top node the particle
     * position is pulled 1e-6*dx inside the boundary so Redistribute cannot
     * delete it; the home record keeps the same pulled-in position.
     *
     * For dim D < 3 the missing position components of the home are set to 0.
     * Particle velocity, entropy and np_real are all set to 0; subsequent
     * SetV/SetK calls will populate them.
     */
    void InitParticles (int lev);

    /**
     * @brief Gather a 3-component vector field (e.g. V_e) at the particle's
     *        home position into the particle's velocity attributes.
     *
     * The field staggering and dim are inferred from AMREX_SPACEDIM; the
     * underlying kernel uses ablastr::particles::compute_weights\<NODE\>.
     */
    void SetV (int lev,
               const amrex::MultiFab & Ux,
               const amrex::MultiFab & Uy,
               const amrex::MultiFab & Uz);

    /**
     * @brief Gather K_e and n_e at the particle's home position and store as
     *        (entropy = K_e * N_e, np_real = N_e).
     */
    void SetK (int lev,
               const amrex::MultiFab & Kfield,
               const amrex::MultiFab & rhofield);

    /**
     * @brief Two-stage midpoint (RK2) velocity evaluation: re-gather the
     *        velocity field at the trajectory midpoint
     *        x_mid = home + (dt/2) * v, where v is the velocity currently
     *        stored on the particle (from SetV), and overwrite vx/vy/vz
     *        with the midpoint sample. A subsequent PushX(dt) then realizes
     *        x_new = home + dt * v(x_mid).
     *
     *        The midpoint excursion is at most half the one-cell CFL bound,
     *        so the gather stays within the home tile's ghost coverage and
     *        no Redistribute is needed. In non-periodic directions x_mid is
     *        clamped just inside the domain (same guard as PushX).
     */
    void GatherVAtMidpoint (int lev, amrex::Real dt,
                            amrex::MultiFab const* eb_dist,
                            const amrex::MultiFab & Ux,
                            const amrex::MultiFab & Uy,
                            const amrex::MultiFab & Uz);

    /**
     * @brief Advance the particle position by x = x_home + v*dt (forward Euler).
     *        In non-periodic directions the new position is clamped just
     *        inside the domain so Redistribute() cannot delete the marker
     *        (the carried entropy accumulates at the boundary nodes instead).
     *        Calls Redistribute() at the end so particles end up in the
     *        correct tile for their new position.
     */
    /** \param[in] eb_dist nodal signed distance to the embedded boundary
     *        (nullptr = no EB marker handling). With it, markers homed at
     *        covered nodes (distance <= 0) never move — V_e is not physical
     *        inside the conductor — and markers pushed into the conductor
     *        are specularly mirrored back across the level set,
     *        x -> x - 2 phi(x) n(x) (the dsmc-eb-reflection semantics at
     *        sub-cell hop scale); a mirror that lands covered again (deep
     *        staircase corner) falls back to the home position. The same
     *        mirror guards the midpoint velocity sample. */
    void PushX (int lev, amrex::Real dt,
                amrex::MultiFab const* eb_dist = nullptr);

    /**
     * @brief Scatter the per-particle entropy (K_e * N_e) onto Kfield via
     *        linear shape-factor weighted atomic add.
     *        Kfield is zeroed at the start of this call.
     *
     * @param gradient_corrected deposit w * (A + 1/2 G . (x_dest - x_p))
     *        instead of w * A, using the limited slopes loaded by SetK. The
     *        correction sums to zero exactly (the hat's weighted mean is the
     *        particle position), so conservation is unchanged, and it
     *        cancels the hat kernel's second moment, making the remap exact
     *        through quadratics (2nd-order spatial transport). The MC
     *        limiter keeps deposited values within the local neighbor range
     *        (positivity).
     */
    void DepositK (int lev, amrex::MultiFab & Kfield,
                   bool gradient_corrected = false);

    /**
     * @brief Scatter the per-particle weight np_real/V_cell (= n_e) onto
     *        Field via linear shape-factor weighted atomic add.
     *        Field is zeroed at the start of this call.
     *        See DepositK for the gradient_corrected option.
     */
    void DepositField (int lev, amrex::MultiFab & Field,
                       bool gradient_corrected = false);

    /**
     * @brief Reset particle position to home, zero velocity / entropy / weight.
     *        Calls Redistribute() afterward.
     */
    void ResetParticles (int lev);

private:
    /**
     * @brief Scatter one real attribute of every marker onto a nodal
     *        MultiFab with the order-1 charge-deposition kernel:
     *        field += scale * attribute * S(x). The field is zeroed first
     *        and guard-cell deposits are summed back in afterward.
     *
     * @param slope_attr0 index of the first of AMREX_SPACEDIM consecutive
     *        slope attributes for the half-gradient-corrected deposit, or
     *        -1 for the plain point deposit.
     */
    void DepositScalar (int lev, int attr, int slope_attr0,
                        amrex::Real scale, amrex::MultiFab & field);
};

#endif // WARPX_QDSMCPARTICLECONTAINER_H_
