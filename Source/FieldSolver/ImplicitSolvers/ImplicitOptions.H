#ifndef IMPLICIT_OPTIONS_H_
#define IMPLICIT_OPTIONS_H_

#include <AMReX_REAL.H>

struct ImplicitOptions
{
    int max_particle_iterations = 21;
    amrex::ParticleReal particle_tolerance = static_cast<amrex::ParticleReal>(1.0e-10);
    bool print_unconverged_particle_details = false;
    bool use_mass_matrices_jacobian = false;
    bool use_mass_matrices_pc = false;
    bool linear_stage_of_jfnk = false;
    bool evolve_suborbit_particles_only = false;
    /** Specularly reflect particles at the r-max domain boundary inside the
     *  implicit per-particle Picard iteration (RZ geometry with a reflecting
     *  particle boundary at r-hi only). */
    bool reflect_particles_at_rmax = false;
    /** Deposit the mass matrices inside the push-and-deposit pass (same tile
     *  loop, same particle data and gather fields as the current deposit)
     *  instead of a separate pass over all particles afterwards. Set by the
     *  implicit solver on the nonlinear residual evaluations that deposit J
     *  when implicit_evolve.fused_mass_matrices_deposit is on. */
    bool deposit_mass_matrices = false;
};

#endif
