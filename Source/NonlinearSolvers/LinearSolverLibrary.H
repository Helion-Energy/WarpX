#ifndef LINEAR_SOLVER_LIBRARY_H_
#define LINEAR_SOLVER_LIBRARY_H_

#include "FlexibleGMRES.H"  // IWYU pragma: export
#include "AMReXGMRES_Wrapper.H"  // IWYU pragma: export
#include "PETScKSP_Wrapper.H"  // IWYU pragma: export

#include <AMReX_Enum.H>

/**
  * \brief struct to select the linear solver for implicit schemes
  */
AMREX_ENUM (LinearSolverType,
    amrex_gmres, amrex_fgmres, petsc_ksp
);

#endif
