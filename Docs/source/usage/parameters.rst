.. _running-cpp-parameters:

Inputs: Parameter List
======================

This section describes the list of parameters that can be set in the WarpX inputs file.

Examples of inputs files can be found in the :ref:`Examples <usage-examples>` section.

.. tip::

   If you enjoy AI/LLM/agentic workflows, see our :ref:`AI (LLM)-Assisted Input File Design <ai_input_design>` section, too.

.. note::

   WarpX's input parameters are read via AMReX's `ParmParse <https://amrex-codes.github.io/amrex/docs_html/Basics.html#parmparse>`__.

.. note::

   The AMReX parser (see :ref:`running-cpp-parameters-parser`) is used for the right-hand side of all input parameters that consist of one or more integers or floats. Expressions like :pp:param:`<species_name>.density_max = "0.1+2.3"` and expressions that include user-defined constants are accepted.

.. _running-cpp-parameters-parser:

Parsers and constants
---------------------

WarpX uses AMReX's math parser that reads expressions in the input file.
It can be used in all input parameters that consist of one or more integers or floats.
Integer input parameters expecting boolean, 0 or 1, are not parsed.
Note that when multiple values are expected, the expressions are space delimited.
For integer input values, the expressions are evaluated as real numbers and the final result rounded to the nearest integer.
See `this section <https://amrex-codes.github.io/amrex/docs_html/Basics.html#parser>`__ of the AMReX documentation for a complete list of functions supported by the math parser.

WarpX constants
^^^^^^^^^^^^^^^

WarpX provides a few pre-defined constants that can be used for any input parameter that consists of one or more floats.

=============  ==================================
``q_e``        Elementary charge (C)
``m_e``        Electron mass (kg)
``m_p``        Proton mass (kg)
``m_u``        Unified atomic mass unit (kg)
``epsilon0``   Vacuum permittivity (F/m)
``mu0``        Vacuum permeability (H/m)
``clight``     Vacuum speed of light (m/s)
``kb``         Boltzmann's constant (J/K)
``hbar``       Reduced Planck constant (J*s)
``pi``         Mathematical constant :math:`\pi`
=============  ==================================

The numerical values of these constants are set in `Source/ablastr/constant.H <https://github.com/BLAST-WarpX/warpx/blob/development/Source/ablastr/constant.H>`__.

User-defined constants
^^^^^^^^^^^^^^^^^^^^^^

Users can define their own constants in the inputs file.
These constants can be used for any input parameter that consists of one or more integers or floats.
User-defined constant names can contain only letters, numbers and the character ``_``.
The name of each constant has to begin with a letter.
The following names are used by WarpX, and cannot be used as user-defined constants: ``x``, ``y``, ``z``, ``X``, ``Y``, ``t``.
The values of the constants can include the predefined WarpX constants listed above as well as other user-defined constants.
For example:

* ``my_constants.a0 = 3.0``
* ``my_constants.z_plateau = 150e-6``
* ``my_constants.n0 = 1e22``
* ``my_constants.wp = sqrt(n0*q_e**2/(epsilon0*m_e))``

Spatial coordinates
^^^^^^^^^^^^^^^^^^^

For profiles that depend on spatial coordinates (e.g., the plasma momentum distribution or the laser field, see below :ref:`Particle initialization <running-cpp-parameters-particle>` and :ref:`Laser initialization <running-cpp-parameters-laser>`), the parser interprets some variables as spatial coordinates.
These are specified in the input parameter, i.e., ``density_function(x,y,z)`` and ``field_function(X,Y,t)``.

The parser reads Python-style expressions between double quotes.
For example, ``"a0*x**2 * (1-y*1e2) * (x>0)"`` is a valid expression, where ``a0`` is a user-defined constant (see above) and ``x`` and ``y`` are spatial coordinates.
The names are case sensitive.
The factor ``(x>0)`` equals ``1`` where ``x>0`` and ``0`` where ``x<=0``.
It allows the user to define functions by intervals.
Alternatively, the expression above can be written as ``if(x>0, a0*x**2 * (1-y*1e2), 0)``.

Time intervals
^^^^^^^^^^^^^^

WarpX can parse time step interval expressions of the form ``start:stop:period``, e.g., ``1:2:3, 4::, 5:6, :, ::10``.
A comma is used as a separator between groups of intervals, which we call slices.
The resulting time intervals are the `union set <https://en.wikipedia.org/wiki/Union_(set_theory)>`__ of all given slices.
White spaces are ignored.
A single slice can have 0, 1 or 2 colons ``:``, just as `NumPy slices <https://numpy.org/doc/stable/reference/generated/numpy.s_.html>`__, but with inclusive upper bound for ``stop``:

* With no colon, the given value is the period.

* With 1 colon, the given string is of the type ``start:stop``.

* With 2 colons, the given string is of the type ``start:stop:period``.

Any value that is not given is set to default.
Default is ``0`` for the start, ``std::numeric_limits<int>::max()`` for the stop, and ``1`` for the period.
For the syntax with 1 or 2 colons, having values in the string is optional (e.g., ``::5``, ``100 ::10``, and ``100 :`` are all valid).

All values can be expressions that are parsed in the same way as other integer input parameters.

Here are some examples of valid time interval expressions and their meaning:

* ``something_intervals = 50``: do something at time steps 0, 50, 100, 150, etc. (equivalent to ``something_intervals = ::50``).

* ``something_intervals = 300:600:100``: do something at time steps 300, 400, 500 and 600.

* ``something_intervals = 300::50``: do something at time steps 300, 350, 400, 450, etc.

* ``something_intervals = 105:108,205:208``: do something at time steps 105, 106, 107, 108, 205, 206, 207 and 208 (equivalent to ``something_intervals = 105 : 108 : , 205 : 208 :``).

* ``something_intervals = :`` or  ``something_intervals = ::``: do something at every time step.

* ``something_intervals = 167:167,253:253,275:425:50`` do something at time steps 167, 253, 275, 325, 375 and 425.

This is similar to the Python slicing syntax, except that the stop is inclusive (e.g., ``0:100`` contains 100) and that no colon means that the given value is the period.

Note that if a given period is zero or negative, the corresponding slice is disregarded.
For example, ``something_intervals = -1`` deactivates ``something`` and ``something_intervals = ::-1,100:1000:25`` is equivalent to ``something_intervals = 100:1000:25``.


Simulation Time
---------------

.. pp:param:: max_step
    :type: ``integer``

    The number of PIC cycles to perform.

.. pp:param:: stop_time
    :type: ``float``
    :unit: seconds

    The maximum physical time of the simulation. Can be provided instead of :pp:param:`max_step`. If both
    :pp:param:`max_step` and :pp:param:`stop_time` are provided, both criteria are used and the simulation stops
    when the first criterion is hit.

    Note: in boosted-frame simulations, :pp:param:`stop_time` refers to the time in the boosted frame.

.. pp:param:: warpx.zmax_plasma_to_compute_max_step
    :type: ``float``
    :optional:

    Can be useful when running in a boosted frame. If specified, automatically
    calculates the number of iterations required in the boosted frame for the
    lower ``z`` end of the simulation domain to reach
    :pp:param:`warpx.zmax_plasma_to_compute_max_step` (typically the plasma end,
    given in the lab frame). The value of :pp:param:`max_step` is overwritten, and
    printed to standard output. Currently only works if the Lorentz boost and
    the moving window are along the z direction.

.. pp:param:: warpx.compute_max_step_from_btd
    :type: ``integer``
    :default: 0
    :optional:

    Can be useful when computing back-transformed diagnostics.  If specified,
    automatically calculates the number of iterations required in the boosted
    frame for all back-transformed diagnostics to be completed. If :pp:param:`max_step`,
    :pp:param:`stop_time`, or :pp:param:`warpx.zmax_plasma_to_compute_max_step` are not specified,
    or the current values of :pp:param:`max_step` and/or :pp:param:`stop_time` are too low to fill
    all BTD snapshots, the values of :pp:param:`max_step` and/or :pp:param:`stop_time` are
    overwritten with the new values and printed to standard output.


.. _running-cpp-parameters-overall:

Overall simulation parameters
-----------------------------

.. pp:param:: authors
    :type: ``string``
    :comment: e.g. ``"Jane Doe <jane@example.com>, Jimmy Joe <jimmy@example.com>"``

    Authors of an input file / simulation setup.
    When provided, this information is added as metadata to (openPMD) output files.

.. pp:param:: warpx.used_inputs_file
    :type: ``string``
    :default: ``warpx_used_inputs``

    Name of a file that WarpX writes to archive the used inputs.
    The context of this file will contain an exact copy of all explicitly and implicitly used inputs parameters, including those :ref:`extended and overwritten from the command line <usage_run>`.

.. pp:param:: warpx.gamma_boost
    :type: ``float``

    The Lorentz factor of the boosted frame in which the simulation is run. (The corresponding Lorentz transformation is assumed to be along :pp:param:`warpx.boost_direction`.)
    For more practical guidance on setting up boosted-frame simulations, refer to the :ref:`FAQ: What do I need to know about using the boosted frame? <faq_boosted_frame>`.

    When using this parameter, the input parameters are interpreted as in the
    lab-frame and automatically converted to the boosted frame.
    (See the corresponding documentation of each input parameters for exceptions.)

.. pp:param:: warpx.boost_direction
    :type: string
    :comment: ``x``, ``y`` or ``z``

    The direction of the Lorentz-transform for boosted-frame simulations
    (The direction ``y`` cannot be used in 2D simulations.)

.. pp:param:: warpx.random_seed
    :type: ``string`` or ``int`` > 0
    :optional:

    If provided :pp:param:`warpx.random_seed = random`, the random seed will be determined
    using ``std::random_device`` and ``std::clock()``,
    thus every simulation run produces different random numbers.
    If provided :pp:param:`warpx.random_seed = n`, and it is required that ``n > 0``,
    the random seed for each MPI rank is ``(mpi_rank+1) * n``,
    where ``mpi_rank`` starts from 0.
    ``n = 1`` and :pp:param:`warpx.random_seed = default`
    produce the default random seed.
    Note that when GPU threading is used,
    one should not expect to obtain the same random numbers,
    even if a fixed :pp:param:`warpx.random_seed` is provided.

.. pp:param:: algo.evolve_scheme
    :type: ``string``
    :default: ``explicit``

    Specifies the evolve scheme used by WarpX.

    * ``explicit``: Use an explicit solver, such as the standard FDTD or PSATD

    * ``theta_implicit_em``: Use a :math:`\theta`-implicit electromagnetic solver.

      - **Time-biasing parameter:**
        The fields (:math:`\textbf{E}` & :math:`\textbf{B}`) used to advance the system are computed at time :math:`t^{n+\theta}`: :math:`\mathbf{E}^{n+\theta}=\left(1-\theta\right)\mathbf{E}^n + \theta\mathbf{E}^{n+1}`, where :math:`\theta\in[0.5,1.0]`.

        - ``implicit_evolve.theta`` (``float``, default: 0.5)
        - :math:`\theta = 0.5`: Exact energy conservation.
        - :math:`\theta = 1.0`: Maximal damping of high-k modes.

      - **Field gather and current depositions:**
        Exact energy conservation requires matching gather and deposition.
        The following depositions support this:

        - :pp:param:`algo.current_deposition = direct`
        - :pp:param:`algo.current_deposition = villasenor`
        - :pp:param:`algo.current_deposition = esirkepov` (Not compatible with ``implicit_evolve.use_mass_matrices_jacobian = true``.)

      - **Numerical stability:**

        - Robust to finite-grid instability (does not require cells that resolve the plasma Debye length).
        - Numerically stable for large :math:`\Delta t` (does not require resolving the plasma period or satisfying the CFL condition for light waves).
        - Practical limits on :math:`\Delta t` set by solver efficiency, number of particle cell crossings, and physics resolution.

      - **Nonlinear solvers:**
        Advancing the implicit system in time requires solving a nonlinear system. The nonlinear solver options are ``picard`` and ``newton``.

        - ``implicit_evolve.nonlinear_solver`` (``string``, default: None)

        - ``implicit_evolve.nonlinear_solver = picard``: Use a Picard iteration method. Requires small time steps; often non-convergent for large time steps.

          - ``picard.verbose`` (``bool``, default: true)
          - ``picard.require_convergence`` (``bool``, default: true)
          - ``picard.max_iterations`` (``int``, default: 100)
          - ``picard.relative_tolerance`` (``float``, default: 1.0e-6)
          - ``picard.absolute_tolerance`` (``float``, default: 0.0)
          - ``picard.diagnostic_file`` (``string``, default: None)
          - ``picard.diagnostic_interval`` (``int``, default: 1)

        - ``implicit_evolve.nonlinear_solver = newton``: Use a PS-JFNK method. Required for large time steps, but efficiency often relies on preconditioning and/or using ``implicit_evolve.use_mass_matrices_jacobian = true``.

          - ``newton.verbose`` (``bool``, default: true)
          - ``newton.linear_solver`` (``string``, default: "gmres") Other excepted value, "petsc_ksp".
          - ``newton.require_convergence`` (``bool``, default: true).
            When ``false``, a Newton step that reaches ``max_iterations``
            without converging — or whose residual-decreasing line search
            stagnates (for operators that request one) — is accepted with
            a warning instead of aborting the run.
          - ``newton.max_frozen_steps`` (``int``, default: 10).
            With ``require_convergence = false``, a line search that
            stagnates at iteration 0 returns the state unchanged: the time
            step is a no-op, and without time-dependent drives every later
            solve repeats identically — the simulation silently freezes
            while time advances. After this many *consecutive*
            zero-progress solves the run aborts. Raise it only when
            time-dependent sources can change the residual between steps.

            Operators that project Newton directions onto admissibility
            bounds (currently the theta-implicit MHD solver, whose
            positivity/temperature floors clamp descending direction
            components at floor-resident cells) report every projection as
            ``Newton: projected N direction components onto admissibility
            bounds (mass <n>, electron_energy <n>, ion_energy <n>)`` with
            per-block counts. When the clamped (pinned) residual rows come
            to dominate the norm — a persistent active set, e.g. a
            floor-resident band under a sustained drain — the full-norm
            Armijo line search can stagnate even though every free
            component still has a productive update, because the test
            grades each trial on rows whose variables the projection just
            froze. Before counting such an iteration-0 stagnation as a
            frozen step, the solver retries the same backtracking ladder
            against the *free-subspace* residual norm (excluding the pinned
            components); an accepted rescue prints ``Newton: accepted
            free-subspace step = <s> (pinned defect = <d>, <N> pinned)``,
            where the pinned defect is the residual L2 norm over the
            pinned components alone. Only if even the free norm cannot be
            reduced does the frozen-step counter advance (the warning and
            the eventual abort message carry the pinned defect and the
            per-block pinned counts). The rescue never runs on the normal
            acceptance path and never with ``require_convergence = true``
            (which aborts on the first full-norm stagnation, as before);
            with an empty active set the behavior is bit-identical to the
            plain Armijo search. A free residual at round-off relative to
            the full norm (below :math:`10^{-12}` of it) counts as no free
            dynamics: the rescue is skipped so a fully frozen state still
            trips the freeze guard instead of silently advancing time on
            noise-level acceptances. The pinned defect is reported, not
            repaired: if it grows without bound in production, the
            documented escalation is a floor-consistency relaxation source
            (mass/energy creation at floor cells), deliberately not
            implemented here.
          - ``newton.max_iterations`` (``int``, default: 100)
          - ``newton.relative_tolerance`` (``float``, default: 1.0e-6)
          - ``newton.absolute_tolerance`` (``float``, default: 0.0)
          - ``newton.diagnostic_file`` (``string``, default: None)
          - ``newton.diagnostic_interval`` (``int``, default: 1)
          - ``newton.adaptive_forcing`` (``bool``, default: false).
            When ``true``, the linear-solve relative tolerance of each Newton iteration is set adaptively by the inexact-Newton forcing prescription of Chacón & Knoll, JCP 188 (2003) 577: :math:`\zeta_A = \gamma\,(\|F_k\|/\|F_{k-1}\|)^\alpha`, safeguarded from volatile decreases by :math:`\gamma \zeta_{k-1}^\alpha`, capped at ``newton.forcing_max``, and floored at :math:`\gamma\,\epsilon_t/\|F_k\|` so the final iteration is not oversolved. Loose tolerances far from the solution cut linear iterations while preserving superlinear Newton convergence.
          - ``newton.forcing_alpha`` (``float``, default: 1.5)
          - ``newton.forcing_gamma`` (``float``, default: 0.9)
          - ``newton.forcing_max`` (``float``, default: 0.5)

          - The PS-JFNK solver uses GMRES to solve the linear system at each nonlinear iteration:

          - ``gmres.verbose_int`` (``int``, default: 2)
          - ``gmres.restart_length`` (``int``, default: 30)
          - ``gmres.max_iterations`` (``int``, default: 1000)
          - ``gmres.relative_tolerance`` (``float``, default: 1.0e-4)
          - ``gmres.absolute_tolerance`` (``float``, default: 0.0)

      - **PS-JFNK solver specific options:**
        The PS-JFNK solver (``implicit_evolve.nonlinear_solver = newton``) has a variety of additional parameters and options.

        - At each iteration in the PS-JFNK process, each particle is self-consistently updated for fixed :math:`\textbf{E}` and :math:`\textbf{B}` on the grid using a Picard method. The options for this Picard solve are set by:

          - ``implicit_evolve.max_particle_iterations`` (``integer``, default: 21)
          - ``implicit_evolve.particle_tolerance`` (``float``, default: 1.e-10)
          - ``implicit_evolve.particle_suborbits`` (``bool``, default: false)
          - ``implicit_evolve.print_unconverged_particle_details`` (``bool``, default: false)

        - ``implicit_evolve.use_mass_matrices_jacobian`` (``bool``, default: false).
          When ``true``, the plasma current density is computed using the mass matrices during the linear stage of PS-JFNK, replacing direct particle calculations. This can enable large speed ups for simulations with many particles.

          - ``implicit_evolve.skip_particle_picard_init`` (``bool``, default: false).
            When ``true`` and ``implicit_evolve.use_mass_matrices_jacobian = true``, the full Picard update of the particles is skipped on the initial Newton step, and only a single iteration is performed.
            This can enhance the overall efficiency of the Newton solver.
            Default is true if ``implicit_evolve.particle_suborbits = true``.

        - ``implicit_evolve.use_mass_matrices_pc`` (``bool``, default: false).
          When ``true``, the plasma response is captured in the preconditioner.
          Requires use of a preconditioner (``jacobian.pc_type = pc_curl_curl_mlmg``, ``pc_petsc``, or ``pc_jacobi``).

        - ``implicit_evolve.mass_matrices_pc_width`` (``integer``, default: 0).
          If using ``jacobian.pc_type = pc_petsc``, this parameter specifies the width of the mass matrices included in the preconditioner.
          In most cases, a width of 1 is sufficient for good GMRES performance.

        - ``jacobian.pc_type`` (``string``, default: None). A preconditioner can be used to minimize the number of linear GMRES iterations. There are four options:

          - ``jacobian.pc_type = pc_curl_curl_mlmg``: Use the AMReX MLMG solver for the curl curl formulation of Maxwell's equations. This preconditioner solves the following equation:

            .. math::

               \nabla \times \left( \alpha\nabla\times\textbf{E} \right) + \boldsymbol{\beta}\cdot\textbf{E} = \textbf{b},

            where :math:`\alpha=\theta^2\Delta t^2c^2` is a scalar and :math:`\boldsymbol{\beta}` is a diagonal matrix that scales the components of :math:`\textbf{E}`.

              - Default: :math:`\boldsymbol\beta = \mathbb{I}`, giving implicit Maxwell equations, suitable for time steps that under-resolve light waves (:math:`c\Delta t > 1/\sqrt{\left(\sum_i1/\Delta x_i^2\right)}`).
              - ``implicit_evolve.use_mass_matrices_pc = true``: :math:`\boldsymbol\beta` also includes plasma response via the diagonal mass matrices, enabling time steps that under-resolve the plasma period (:math:`\omega_{pe}\Delta t > 1`).

            - ``pc_curl_curl_mlmg.verbose`` (``bool``, default: true)
            - ``pc_curl_curl_mlmg.bottom_verbose`` (``bool``, default: false)
            - ``pc_curl_curl_mlmg.agglomeration`` (``bool``, default: true)
            - ``pc_curl_curl_mlmg.consolidation`` (``bool``, default: true)
            - ``pc_curl_curl_mlmg.max_iter`` (``int``, default: 10)
            - ``pc_curl_curl_mlmg.max_coarsening_level`` (``int``, default: 30)
            - ``pc_curl_curl_mlmg.relative_tolerance`` (``float``, default: 1.0e-4)
            - ``pc_curl_curl_mlmg.absolute_tolerance`` (``float``, default: 1.0e-16)

          - ``jacobian.pc_type = pc_mhd_block``: Use the prototype
            physics-based block preconditioner for
            ``algo.evolve_scheme = theta_implicit_mhd``. In the small-flow
            approximation, it uses an electric predictor,
            three-component momentum-wave Schur solve, and electric
            corrector. With :math:`h=\theta\Delta t`, define

            .. math::

               d_\eta=\frac{h\eta_*}{\mu_0},\qquad
               d_H=\frac{h|\boldsymbol B_*|}{\mu_0\rho_{q,*}} .

            The Hall-aware scalar field operator is

            .. math::

               \mathcal P_E
               =
               \mathbb I+
               \sqrt{d_\eta^2+d_H^2}
               \nabla\times\nabla\times .

            The predictor is
            :math:`\delta\boldsymbol E^*=\mathcal P_E^{-1}\boldsymbol b_E`.
            The Hall term uses a high-wavenumber spectral-magnitude
            approximation to the skew whistler block. Hall-off operation sets
            :math:`d_H=0` and recovers the resistive operator.
            With Hall physics disabled, the domain-average theta-state field
            :math:`\boldsymbol B_*` enters the coupled momentum solve

            .. math::

               \left[
                  \mathbb I
                  -h^2c_s^2\nabla\nabla\cdot
                  +\frac{h^2}{\mu_0\rho_*}
                   \mathcal W(\boldsymbol B_*)
               \right]\delta\boldsymbol M
               =
               \boldsymbol r_M ,

            where

            .. math::

               \mathcal W(\boldsymbol B_*)\boldsymbol M
               =
               \boldsymbol B_*\times
               \nabla\times\nabla\times
               (\boldsymbol M\times\boldsymbol B_*).

            The right-hand side :math:`\boldsymbol r_M` contains the pressure
            predictor and the Lorentz response to
            :math:`\delta\boldsymbol E^*`. Density and electron energy are
            corrected from :math:`\nabla\cdot\delta\boldsymbol M`, followed by

            .. math::

               \delta\boldsymbol E
               =
               \mathcal P_E^{-1}
               \left[
                  \boldsymbol b_E
                  -\frac{\delta\boldsymbol M}{\rho_*}
                   \times\boldsymbol B_*
               \right].

            The acoustic and magnetic coefficients are
            domain-global/reference scalars, not spatial coefficient fields.
            The magnetic Schur action approximates
            :math:`\mathcal P_E^{-1}` by the identity, while its staggered
            predictor and corrector retain the resistive solve. With Hall
            physics and evolving ions, this wave Schur is disabled because
            that identity approximation loses the stiff Hall response. The
            implementation instead applies the acoustic fluid block followed
            by the ideal momentum-to-field correction. When ideal coupling is
            disabled or the mean magnetic field is negligible, it uses the
            acoustic block without that correction.

            With ``implicit_mhd.evolve_ion_fluid = false``, the preconditioner
            instead treats the ion density and momentum rows, and
            approximately the electron-energy row, as identities. It applies
            the Hall-aware field corrector without the ion-wave Schur solve.

            With ``implicit_mhd.fluid_flux = hlld`` (the conservative-form
            recast, where face-staggered :math:`\boldsymbol B^{n+\theta}` is
            the JFNK array block), the preconditioner instead composes three
            pieces, each engaged only where it measurably pays — on resolved
            states the residuals are smooth, and preconditioner content only
            injects structure that GMRES must then resolve against the
            Jacobian's rough grid-scale HLLD response:

            1. **Per-direction signal diffusion** (opt-in): every
               cell-centered block (:math:`\rho`, the three momentum
               components, :math:`U_e`, and the ion-closure energies
               :math:`E_i` or :math:`U_\parallel,U_\perp`) is preconditioned
               by one stacked Helmholtz solve,

               .. math::

                  \left[\mathbb I
                    + \theta\Delta t\, S_d\, \tfrac{h_d}{2}\,(-\partial_d^2)\right]
                  \delta U = b_U ,

               where :math:`h_d` is the cell size per direction and
               :math:`S_d = |u_d|_* + \sqrt{c_{s,*}^2 + v_{A,*}^2}` the
               per-direction reference signal speed built from the domain RMS
               of the instantaneous velocity component,
               ``implicit_mhd.reference_magnetic_field``,
               ``implicit_mhd.reference_ion_pressure``, and the instantaneous
               mean electron enthalpy, reduced by
               ``pc_mhd_block.signal_cfl_floor`` resolved cells per implicit
               interval. This approximates the
               :math:`O(\theta\Delta t\,|S|/h)` HLLD upwind dissipation, but
               HLLD dissipates per characteristic channel while one scalar
               coefficient per block diffuses them all; on every benchmarked
               regime it degraded plain GMRES, so it is disabled by default
               (``pc_mhd_block.signal_diffusion_scale = 0``) pending
               per-channel reference speeds. In RZ the Helmholtz is assembled
               in cylindrical form.

            2. **RZ (m = 0) momentum wave Schur**: in RZ the momentum block
               is additionally composed with the constant-coefficient
               cylindrical ideal-MHD wave operator
               :math:`\mathbb I - h^2 c_{s,*}^2 \nabla (\nabla\cdot)
               + h^2/(\mu_0 \rho_*)\, W_{B}` (:math:`h = \theta\Delta t`,
               with the **input** ``implicit_mhd.reference_magnetic_field``
               magnitude along :math:`\hat z`: the domain mean is near zero
               for reversed-field states, and :math:`W_B` is quadratic in
               :math:`B` so the sign structure is immaterial), discretized
               with the residual's cylindrical stencil conventions
               (r-weighted flux divergences and the :math:`-M_r/r^2` hoop
               term) on every multigrid level and smoothed with the
               per-cell block-Jacobi pattern. It engages when the reference
               fast CFL :math:`\theta\Delta t\,c_{f,*}/h_d` reaches
               ``pc_mhd_block.wave_cfl_threshold``.

            3. **Triangular Faraday corrector**: the Faraday block is the
               identity (up to the gated resistive block below) plus
               the ideal-induction coupling
               :math:`\delta B = b_B + \theta\Delta t\,
               \nabla\times(\delta\boldsymbol M/\rho_* \times
               \boldsymbol B_{cc})`, evaluated with the solver's staggered
               curl conventions (1D transverse faces; RZ face forms with the
               r-weighted corner-EMF divergence for :math:`B_z` and a zero
               azimuthal edge value on the axis) from the cell-centered
               magnetic field frozen at the preconditioner update. It
               requires either an active wave Schur or a sub-unity reference
               Alfvén CFL: correcting from an identity momentum block at
               stiff coupling amplifies the neglected
               :math:`J_{MB} J_{BM}` product quadratically.

            4. **Resistive block**: the :math:`\theta_r`-implicit resistive
               term of the B residual,
               :math:`\delta B + \theta_r \Delta t\, \nabla\times
               (\eta_\text{field}/\mu_0\, \nabla\times \delta B)` with
               :math:`\eta_\text{field}` the density-keyed field resistivity
               (including the ``implicit_mhd.vacuum_resistivity_diffusivity``
               boost) frozen at the preconditioner update, is approximately
               inverted by a fixed-count Chebyshev iteration on the exact
               discrete operator :math:`\mathbb I + \theta_r \Delta t\,
               \nabla\times(\eta_\text{field}/\mu_0\, \nabla\times\,\cdot\,)`,
               applied matrix-free on the residual's Yee staggering with
               :math:`\eta_\text{field}` evaluated at the electric-field
               locations from the residual's own interpolated density and
               tangential E zeroed on non-periodic domain boundary planes
               (exact at PEC walls and the axis). Iterating on the exact
               coupled operator is essential: every per-component scalar
               Helmholtz surrogate tried instead measurably degraded
               restarted GMRES on stiff RZ vacuum-eta states by 10–100x
               relative to no preconditioner, because the resistive
               curl-curl is dominated by its curl-free kernel and strong
               cross-component coupling. It engages only when the
               grid-scale resistive diffusion number
               :math:`\theta_r \Delta t\, \max(\eta_\text{field})/\mu_0 /
               \min(\Delta x)^2` reaches
               ``pc_mhd_block.resistive_threshold``; below it the block is
               the exact identity at zero cost. It is composed after the
               Faraday corrector (the consistent triangular order).
               ``pc_mhd_block.resistive_solver = direct`` replaces the
               Chebyshev iteration with an exact sparse factorization of
               the frozen operator (NVIDIA cuDSS; CUDA builds configured
               with ``-DWarpX_CUDSS=ON``): the matrix rows are assembled
               once per preconditioner update from the same stencil the
               matrix-free path applies, and every application is then a
               single forward/backward substitution instead of the
               :math:`\lceil 2\sqrt{\lambda_\text{max}}\rceil` stencil
               sweeps of the automatic Chebyshev count. Builds with the
               cuDSS backend default to ``direct``; all other builds
               default to the portable ``chebyshev`` iteration.
               ``pc_mhd_block.resistive_solver = banded`` instead factors
               the same frozen operator as a block-banded LU along z
               (blocks dense over the components and, in RZ, the full
               radial line; bandwidth 1, or 2 with the hyper-resistive
               chain, doubled by the folded ring ordering that treats a
               periodic z exactly), with the factorization and sweeps
               resident on the device and no external dependency. It is
               exact up to the selectable
               ``pc_mhd_block.banded_precision`` and is currently
               restricted to single-rank runs.

               When the Ohm law carries
               ``hybrid_pic_model.plasma_hyper_resistivity``, the frozen
               chain :math:`-\theta_r \Delta t\, \nabla\times(\eta_H\,
               \nabla^2(\nabla\times\,\cdot\,))/\mu_0` joins the block
               operator through the same stencil emission (with
               :math:`\eta_H` frozen at the residual's interpolated
               arguments), and the activation gate and Chebyshev interval
               gain the corresponding :math:`k_\text{max}^4`-scale
               bounds.

               When the Ohm law carries the Hall term
               (``fluid_flux = central`` with
               ``hybrid_pic_model.include_hall_term = true``) and
               ``pc_mhd_block.include_hall_mhd_coupling`` is on (the
               default), the whistler rows :math:`+\theta\Delta t\,
               \nabla\times((\nabla\times\,\cdot\,/\mu_0)\times
               \boldsymbol{Hb})` join the same emission, linearized about
               the frozen Hall coefficient vector
               :math:`\boldsymbol{Hb} = \boldsymbol B/\rho_q` on the
               electric-field staggerings (the B-form of the Hall-MHD
               field block of Chacón, JCP 526 (2025) 113789) at the
               REACTIVE :math:`\theta\Delta t` staging of the ideal EMF
               (the resistive rows keep :math:`\theta_r\Delta t`). The
               grid whistler number :math:`\theta\Delta t\,
               \max|\boldsymbol{Hb}|/\mu_0/\min(\Delta x)^2` joins the
               activation gate. The whistler coupling is first-order and
               skew between the B components -- its spectrum is the
               complex conjugate pairs :math:`1 \pm i\,\theta\Delta t\,
               D_H k^2`, outside any real Chebyshev interval -- so the
               Hall rows require ``pc_mhd_block.resistive_solver =
               banded`` or ``direct`` (asserted), whose exact LU inverts
               the non-symmetric frozen block directly;
               ``pc_mhd_block.resistive_validate_assembly`` checks the
               assembled Hall rows against the matrix-free operator to
               roundoff exactly like the resistive rows.

               When the Ohm law additionally carries the electron-inertia
               term (``hybrid_pic_model.include_electron_inertia``) and
               ``pc_mhd_block.include_electron_inertia_coupling`` is on
               (the default), the frozen inertia response
               :math:`+s\,\nabla\times((C_i/\mu_0)\,\nabla\times\,\cdot\,)`
               with :math:`C_i = m_{e,\mathrm{eff}}/(e\rho_q)` (=
               :math:`\mu_0 d_e^2` at the Ohm-floored charge density) and
               the dt-INDEPENDENT stencil factor :math:`s = (2\theta+1)/2`
               (three-point) or 1 (two-point) joins the same emission as
               an exact stencil twin of the resistive rows -- the
               push-through of the theta-stage :math:`\partial\vec{J}_e/
               \partial t` at frozen density and ion current
               (:math:`\delta\vec{J}_e = \nabla\times\delta\vec{B}/\mu_0`).
               The block is SPD and BOUNDS the skew whistler part: the
               frozen symbol becomes the capped branch
               :math:`(1 + s k^2 d_e^2) \pm i\,\theta\Delta t\,D_H k^2`,
               and the grid inertia number :math:`s\,(d_e/\Delta x)^2`
               joins the activation gate. The inertia rows require the
               same exact inverses as the Hall rows (asserted) and are
               covered by ``resistive_validate_assembly`` identically.

            Boundaries the recast residual manages itself are mapped per
            component to preconditioner-only linear-operator types matching
            the residual's fluid ghosts: ``pec`` (a reflecting wall) becomes
            Neumann for scalars and tangential momentum and homogeneous
            Dirichlet for the normal momentum, while ``open`` (zero-gradient
            fluid outflow) and ``none`` (the RZ axis and outflow ends) become
            homogeneous Neumann; the nonlinear residual keeps the true
            boundary coupling. The hlld form supports any ion closure,
            any resistivity, and uses ``pc_mhd_block.fluid_iterations``
            fixed MLMG cycles for the stacked and wave solves.

            All MLMG solves use zero initial guesses, fixed cycle counts, and
            fixed bottom smoothing. This keeps the operation stationary to
            solver roundoff for standard right-preconditioned GMRES; MLMG can
            still stop early at its internal machine-precision residual
            threshold.

            With the E-based state
            (``implicit_mhd.fluid_flux = legacy_e_centered``),
            the prototype requires one Cartesian, periodic AMR level,
            ``warpx.grid_type = staggered``, and
            zero ``hybrid_pic_model.plasma_hyper_resistivity``. Resistivity
            may be constant or time dependent; density, current, and
            per-species dependence is rejected because the field block uses a
            scalar reference value. Hall and electron-pressure physics may be
            enabled, and ions may evolve or be frozen. In RZ only the hlld
            Stage-1 form is available (the E-based operators have no
            cylindrical metric). The Hall preconditioner
            is currently an isotropic, constant-coefficient whistler-magnitude
            surrogate; it is not the full skew electron subsystem and does not
            retain multidimensional :math:`k k_\parallel` anisotropy. It does
            not yet include the electron-pressure Ohm derivative, background
            advection,
            :math:`\boldsymbol J_*\times\delta\boldsymbol B` for a nonuniform
            reference field, Rusanov transport, spatial wave coefficients, or
            the Joule-heating derivative. Strongly nonuniform and
            pressure-floor-active states are outside this scalar-coefficient
            prototype's robustness claim.

            The wave Schur complement uses compact centered pure and mixed
            second derivatives, whereas the centered nonlinear flux and
            staggered Lorentz/induction paths compose wider derivatives and
            interpolations. They agree for smooth, well-resolved modes but
            differ near the grid scale, including at the centered acoustic
            operator's checkerboard null mode. This option targets smooth
            resistive, acoustic, Alfvén, and magnetosonic dynamics, not
            discontinuous flow.

            - ``pc_mhd_block.verbose`` (``bool``, default: false)
            - ``pc_mhd_block.bottom_verbose`` (``bool``, default: false)
            - ``pc_mhd_block.include_ideal_mhd_coupling`` (``bool``,
              default: true): include the domain-mean ideal magnetic/Lorentz
              coupling. Hall-off evolving-ion solves use the momentum-wave
              Schur block. Hall-active evolving-ion solves use an acoustic
              block followed by a triangular momentum-to-field correction.
              If false, use the acoustic block without either ideal coupling.
            - ``pc_mhd_block.include_hall_mhd_coupling`` (``bool``,
              default: true): when Hall physics is enabled, include its
              spectral-magnitude coefficient in the field curl--curl block
              (E-based state) or the exact frozen whistler rows in the
              recast B block (see point 4 above; requires
              ``resistive_solver = banded`` or ``direct``).
              This changes only the preconditioner, not the nonlinear Hall
              residual.
            - ``pc_mhd_block.include_electron_inertia_coupling`` (``bool``,
              default: true): when the electron-inertia Ohm term is
              enabled, include its frozen SPD mass in the recast B block
              (see point 4 above; requires ``resistive_solver = banded``
              or ``direct``). This changes only the preconditioner, not
              the nonlinear inertia residual.
            - ``pc_mhd_block.field_iterations`` (``int``, default: 2):
              fixed MLMG iteration count for the resistive/Hall-magnitude
              curl--curl block.
            - ``pc_mhd_block.fluid_iterations`` (``int``, default: 2):
              fixed MLMG iteration count for the Hall-off three-component
              wave block, or for the scalar acoustic block.
            - ``pc_mhd_block.resistive_iterations`` (``int``, default: 0):
              hlld only; Chebyshev applications of the exact resistive
              curl-curl per preconditioner application. The default 0
              selects :math:`\lceil 2\sqrt{\lambda_\text{max}}\rceil`
              automatically, which contracts the inner iteration to
              :math:`\sim e^{-4}` independent of the stiffness; each
              application costs one stencil sweep, far below the full
              residual evaluation an outer Krylov iteration costs.
            - ``pc_mhd_block.resistive_threshold`` (``float``, default: 1.0):
              hlld only; grid-scale resistive diffusion number at which the
              resistive block engages (below it the block is the exact
              identity).
            - ``pc_mhd_block.resistive_solver`` (``string``, default:
              ``direct`` on CUDA builds configured with
              ``-DWarpX_CUDSS=ON``, ``chebyshev`` otherwise): hlld only;
              inner solver of the resistive block. ``chebyshev`` is the
              portable fixed-count iteration; ``direct`` factorizes the
              frozen operator exactly with NVIDIA cuDSS once per
              preconditioner update, so it is the default wherever that
              backend is available (requires a CUDA build configured
              with ``-DWarpX_CUDSS=ON``; explicitly requesting it on
              other builds aborts with a descriptive message).
              ``banded`` factorizes the frozen operator as a block-banded
              LU along z on the device (portable on CPU and CUDA/HIP
              builds, no external dependency; single-rank runs only;
              periodic z is treated exactly through a folded ring
              ordering at twice the bandwidth). On single-rank runs the
              ``direct`` refreeze and apply are device-resident (see
              :pp:param:`implicit_mhd.resistive_direct_device_assembly`).
            - ``pc_mhd_block.banded_precision`` (``string``, default:
              ``double``): hlld only; storage precision of the banded
              factorization and sweeps. ``double`` matches the
              exact-inverse behavior of ``direct``; ``single`` buys the
              large FP32/FP64 throughput ratio of workstation GPUs but is
              only safe at mild stiffness (the unpivoted elimination at
              production condition numbers exhausts FP32 headroom).
            - ``pc_mhd_block.banded_refreeze`` (``string``, default:
              ``step``): hlld only; ``step`` factors the banded operator
              once per time step and reuses the factorization across the
              step-internal Newton iterations (the standard
              frozen-coefficient preconditioner lag; the factorization
              dominates the banded cost), ``newton`` refactors at every
              preconditioner update.
            - ``pc_mhd_block.resistive_refreeze`` (``string``, default:
              empty): shared refreeze-cadence override honored by BOTH
              resistive backends. Empty (the default) preserves each
              branch's legacy cadence (``banded`` follows
              ``banded_refreeze``; ``direct`` refactors at every
              preconditioner update); ``step`` or ``newton`` forces
              that cadence on whichever resistive solver is active —
              ``step`` on the ``direct``/cuDSS path buys the
              banded-style once-per-step frozen-coefficient lag at
              cuDSS factor cost. When set, it takes precedence over
              ``banded_refreeze``.
            - ``pc_mhd_block.resistive_validate_assembly`` (``bool``,
              default: false): hlld only; at every active preconditioner
              update, assemble the direct solver's sparse rows and check
              them to roundoff against the matrix-free operator on a
              deterministic pseudo-random vector, aborting on mismatch
              (available on every build; used by the CI assembly tests).
              When the device value refresh is active (see
              :pp:param:`implicit_mhd.resistive_direct_device_assembly`)
              the device- and host-assembled CSR value arrays are
              additionally compared bitwise (:math:`\le 4` ULP asserted).
            - ``pc_mhd_block.wave_relaxation`` (``float``, default: 0.5):
              weighted block-Jacobi factor for the three-component wave
              smoother;
              must be in :math:`(0,0.5]`.
            - ``pc_mhd_block.include_wave_schur`` (``bool``, default: true):
              hlld/RZ only; allow the cylindrical momentum wave Schur.
            - ``pc_mhd_block.wave_cfl_threshold`` (``float``, default: 1.0):
              hlld/RZ only; reference fast CFL at which the momentum wave
              Schur engages.
            - ``pc_mhd_block.signal_diffusion_scale`` (``float``,
              default: 0.0): hlld only; overall scale of the per-direction
              signal-diffusion Helmholtz on the cell-centered blocks
              (0 disables it, the default).
            - ``pc_mhd_block.signal_cfl_floor`` (``float``, default: 1.0):
              hlld only; resolved cells per implicit interval subtracted
              from the per-direction reference signal speed before the
              diffusion coefficient is formed.
            - ``pc_mhd_block.max_coarsening_level`` (``int``, default: 30)
            - ``pc_mhd_block.agglomeration`` (``bool``, default: true)
            - ``pc_mhd_block.consolidation`` (``bool``, default: true)

          - ``jacobian.pc_type = pc_jacobi``: Use the Point-Jacobi method. This preconditioner only captures the plasma response via the diagonal mass matrices.

            - ``pc_jacobi.verbose`` (``bool``, default: true)
            - ``pc_jacobi.max_iter`` (``int``, default: 10)
            - ``pc_jacobi.relative_tolerance`` (``float``, default: 1.0e-4)
            - ``pc_jacobi.absolute_tolerance`` (``float``, default: 1.0e-16)

          - ``jacobian.pc_type = pc_petsc``: Use the PETSc solver.

            - ``pc_petsc.type`` (``string``, default: "asm")
            - ``pc_petsc.asm_overlap`` (``int``, default: 0)
            - ``pc_petsc.sub_type`` (``string``, default: "ilu")
            - ``pc_petsc.ilu_factor_levels`` (``int``, default: 2)
            - ``pc_petsc.hypre_type`` (``string``, default: "euclid")
            - ``pc_petsc.euclid_factor_levels`` (``int``, default: 2)

      - **References:** (WarpX includes relativistic extensions not discussed in references.)

        - `Angus et al., On numerical energy conservation for an implicit particle-in-cell method coupled with a binary Monte-Carlo algorithm for Coulomb collisions <https://doi.org/10.1016/j.jcp.2022.111030>`__.
        - `Angus et al., An implicit particle code with exact energy and charge conservation for electromagnetic studies of dense plasmas <https://doi.org/10.1016/j.jcp.2023.112383>`__.
        - `Angus et al., An implicit particle code with exact energy and charge conservation for studies of dense plasmas in axisymmetric geometries <https://doi.org/10.1016/j.jcp.2024.113427>`__.

    * ``theta_implicit_hybrid``: Use the :math:`\theta`-implicit hybrid-PIC
      solver. The nonlinear state contains the electric field; kinetic ions
      are pushed and deposited inside each nonlinear residual evaluation.

    * ``theta_implicit_mhd``: Use the :math:`\theta`-implicit single-ion-fluid
      MHD solver. The JFNK state contains
      :math:`(\boldsymbol E,\rho_i,\rho_i\boldsymbol u_i,U_e)`, so the ion
      continuity and momentum equations and the electron internal-energy
      equation are advanced inside the nonlinear solve instead of pushing
      particles. See :ref:`the implicit-MHD parameters
      <running-cpp-parameters-implicit-mhd>` and
      :ref:`the model description <theory-hybrid-model-implicit-mhd>`.

    * ``semi_implicit_em``: Use an approximately energy conserving semi-implicit electromagnetic solver.

      - Difference with ``theta_implicit_em`` is that light waves are treated explicit just as in the standard FDTD method. Consequently, this method has the CFL limitation :math:`c\Delta t < 1/\sqrt( \sum_i 1/\Delta x_i^2 )`.
      - Particles are treated implicitly, and all of the comments for ``theta_implicit_em`` above apply here as well (except that :math:`\theta` is fixed to 0.5).
      - The method is described in `Chen et al., A semi-implicit, energy- and charge-conserving particle-in-cell algorithm for the relativistic Vlasov-Maxwell equations <https://doi.org/10.1016/j.jcp.2020.109228>`__.


    * ``strang_implicit_spectral_em``: Use a fully implicit electromagnetic solver. All of the comments for ``theta_implicit_em``
      above apply here as well (except that :math:`\theta` is fixed to 0.5 and that charge will not be conserved).
      In this version, the advance is Strang split, with a half advance of the source free Maxwell's equation (with a spectral solver), a full advance of the particles plus longitudinal E field, and a second half advance of the source free Maxwell's equations.
      The advantage of this method is that with the Spectral advance of the fields, it is dispersionless.
      Note that exact energy convergence is achieved only with one grid block and :pp:param:`psatd.periodic_single_box_fft = 1`. Otherwise,
      the energy conservation is spoiled because of the inconsistency of the periodic assumption of the spectral solver and the
      non-periodic behavior of the individual blocks.

.. _param-electrostatic-pic:

.. pp:param:: warpx.do_electrostatic
    :type: ``string``
    :default: ``none``
    :optional:

    Specifies the electrostatic mode. When turned on, instead of updating
    the fields at each iteration with the full Maxwell equations, the fields
    are recomputed at each iteration from the Poisson equation.
    There is no limitation on the timestep in this case, but
    electromagnetic effects (e.g. propagation of radiation, lasers, etc.)
    are not captured. Several options for the electrostatic scheme are available,
    including, ``labframe``, ``labframe-electromagnetostatic``, ``labframe-effective-potential``,
    and ``relativistic``. See :ref:`here <theory-electrostatic-pic>` for details
    of each scheme.

.. pp:param:: warpx.poisson_solver
    :type: ``string``
    :default: ``multigrid``
    :optional:

    * ``multigrid``: Poisson's equation is solved using an iterative multigrid (MLMG) solver.
        See the `AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html#>`__
        for details of the MLMG solver (the default solver used with electrostatic
        simulations). The default behavior of the code is to check whether there is
        non-zero charge density in the system and if so force the MLMG solver to
        use the solution max norm when checking convergence. If there is no charge
        density, the MLMG solver will switch to using the initial guess max norm
        error when evaluating convergence and an absolute error tolerance of
        :math:`10^{-6}` :math:`\mathrm{V/m}^2` will be used (unless a different
        non-zero value is specified by the user via
        :pp:param:`warpx.self_fields_absolute_tolerance`).

    * ``fft``: Poisson's equation is solved using an Integrated Green Function method (which requires FFT calculations).
        See these references for more details :cite:t:`param-QiangPhysRevSTAB2006`, :cite:t:`param-QiangPhysRevSTAB2006err`.
        It only works in 3D and it requires the compilation flag ``-DWarpX_FFT=ON``.
        If mesh refinement is enabled, this solver only works on the coarsest level.
        On the refined patches, the Poisson equation is solved with the multigrid solver.
        In electrostatic mode, this solver requires open field boundary conditions (:pp:param:`boundary.field_lo,hi = open`).
        In electromagnetic mode, this solver can be used to initialize the species' self fields
        (:pp:param:`<species_name>.initialize_self_fields = 1`) provided that the field BCs are PML (:pp:param:`boundary.field_lo,hi = PML`).

          * ``warpx.use_2d_slices_fft_solver`` (``bool``) optional (default: 0): Select the type of Integrated Green Function solver.
            If 0, solve Poisson equation in full 3D geometry.
            If 1, solve Poisson equation in a quasi 3D geometry, neglecting the :math:`z` derivatives in the Laplacian of the Poisson equation.
            In practice, in this case, the code performs many 2D Poisson solves on all :math:`(x,y)` slices, each slice at a given :math:`z`.
            This is often a good approximation for ultra-relativistic beams propagating along the :math:`z` direction, with the relativistic solver.
            As a consequence, this solver does not need to do an FFT along the :math:`z` direction,
            and instead uses only transverse FFTs (along :math:`x` and :math:`y`) at each :math:`z` position (or :math:`z` "slice").

          * ``ablastr.nprocs_igf_fft`` (``int``) optional (default: number of MPI ranks): Number of MPI ranks used to parallelize the FFT solver.
            This can be less or equal than then number of MPI ranks that are used to run the overall simulation.
            It can be useful if the auxiliary simulation boxes fit within a single process, so to avoid extra communications.
            The auxiliary boxes are extended boxes in real and spectral space that are used to perform the necessary FFTs.
            The extended simulation box size in real space is :math:`2n_x-1, 2n_y-1, 2n_z-1` with the 3D solver, :math:`2n_x-1, 2n_y -1, n_z` with the 2D solver.
            The extended simulation box size in spectral space is :math:`n_x, 2n_y-1, 2n_z-1` with the 3D solver, :math:`n_x, 2n_y-1, n_z` with the 2D solver.

.. pp:param:: warpx.self_fields_required_precision
    :type: ``float``
    :default: 1.e-11

    The relative precision with which the electrostatic space-charge fields should
    be calculated. More specifically, the space-charge fields are
    computed with an iterative Multi-Level Multi-Grid (MLMG) solver.
    This solver can fail to reach the default precision within a reasonable time.
    This applies to the labframe electrostatic solvers (``labframe``, ``labframe-electromagnetostatic``,
    ``labframe-effective-potential``). When using ``labframe-electromagnetostatic``, this value
    is also used as the default for ``magnetostatic_solver_required_precision``.

.. pp:param:: warpx.self_fields_absolute_tolerance
    :type: ``float``
    :default: 0.0

    The absolute tolerance with which the space-charge fields should be
    calculated in units of :math:`\mathrm{V/m}^2`. More specifically, the acceptable
    residual with which the solution can be considered converged. In general
    this should be left as the default, but in cases where the simulation state
    changes very little between steps it can occur that the initial guess for
    the MLMG solver is so close to the converged value that it fails to improve
    that solution sufficiently to reach the ``self_fields_required_precision``
    value. When using ``labframe-electromagnetostatic``, this value
    is also used as the default for ``magnetostatic_solver_absolute_tolerance``.

.. pp:param:: warpx.self_fields_max_iters
    :type: ``integer``
    :default: 200

    Maximum number of iterations used for MLMG solver for space-charge
    fields calculation. In case if MLMG converges but fails to reach the desired
    ``self_fields_required_precision``, this parameter may be increased.
    This applies to the labframe electrostatic solvers (``labframe``, ``labframe-electromagnetostatic``,
    ``labframe-effective-potential``). When using ``labframe-electromagnetostatic``, this value
    is also used as the default for ``magnetostatic_solver_max_iters``.

.. pp:param:: warpx.self_fields_verbosity
    :type: ``integer``
    :default: 2

    The verbosity used for MLMG solver for space-charge fields calculation. Currently
    MLMG solver looks for verbosity levels from 0-5. A higher number results in more
    verbose output. When using ``labframe-electromagnetostatic``, this value
    is also used as the default for ``magnetostatic_solver_verbosity``.

.. pp:param:: warpx.self_fields_num_final_sweeps
    :type: ``integer``
    :default: 8

    Number of relaxation (smoothing) sweeps performed during the final smoothing
    stage of the AMReX MLMG Poisson solve for electrostatic self fields.

    Final smoothing is applied by AMReX when the smoother is used as the bottom
    solver of the multigrid solve.

    Increasing this value can improve residual reduction per MLMG iteration,
    which may help convergence, but it also increases the work performed in each
    MLMG iteration, so the most efficient value is problem-dependent.

    Must be greater than zero when specified.

.. pp:param:: warpx.magnetostatic_solver_required_precision
    :type: ``float``
    :default: value of ``self_fields_required_precision``

    The relative precision with which the magnetostatic (vector Poisson) fields should
    be calculated when using ``labframe-electromagnetostatic`` mode.
    This allows setting a different precision for the magnetostatic solver
    than for the electrostatic solver.

.. pp:param:: warpx.magnetostatic_solver_absolute_tolerance
    :type: ``float``
    :default: value of ``self_fields_absolute_tolerance``

    The absolute tolerance with which the magnetostatic fields should be
    calculated when using ``labframe-electromagnetostatic`` mode.
    This allows setting a different tolerance for the magnetostatic solver
    than for the electrostatic solver.

.. pp:param:: warpx.magnetostatic_solver_max_iters
    :type: ``integer``
    :default: value of ``self_fields_max_iters``

    Maximum number of iterations used for the magnetostatic (vector Poisson) MLMG solver
    when using ``labframe-electromagnetostatic`` mode.
    This allows setting different iteration limits for the magnetostatic solver
    than for the electrostatic solver.

.. pp:param:: warpx.magnetostatic_solver_verbosity
    :type: ``integer``
    :default: value of ``self_fields_verbosity``

    The verbosity used for the magnetostatic MLMG solver when using
    ``labframe-electromagnetostatic`` mode. Values range from 0-5, with higher
    numbers producing more verbose output.

.. pp:param:: amrex.abort_on_out_of_gpu_memory
    :type: ``0`` or ``1``
    :default: ``1`` for true

    When running on GPUs, memory that does not fit on the device will be automatically swapped to host memory when this option is set to ``0``.
    This will cause severe performance drops.
    Note that even with this set to ``1`` WarpX will not catch all out-of-memory events yet when operating close to maximum device memory.
    `Please also see the documentation in AMReX <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__.

.. pp:param:: amrex.the_arena_is_managed
    :type: ``0`` or ``1``
    :default: ``0`` for false

    When running on GPUs, device memory that is accessed from the host will automatically be transferred with managed memory.
    This is useful for convenience during development, but has sometimes severe performance and memory footprint implications if relied on (and sometimes vendor bugs).
    For all regular WarpX operations, we therefore do explicit memory transfers without the need for managed memory and thus changed the AMReX default to false.
    `Please also see the documentation in AMReX <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__.

.. pp:param:: amrex.omp_threads
    :type: ``system``, ``nosmt`` or positive integer
    :default: ``nosmt``

    An integer number can be set in lieu of the ``OMP_NUM_THREADS`` environment variable to control the number of OpenMP threads to use for the ``OMP`` compute backend on CPUs.
    By default, we use the ``nosmt`` option, which overwrites the OpenMP default of spawning one thread per logical CPU core, and instead only spawns a number of threads equal to the number of physical CPU cores on the machine.
    If set, the environment variable ``OMP_NUM_THREADS`` takes precedence over ``system`` and ``nosmt``, but not over integer numbers set in this option.


.. _running-cpp-parameters-signal:

Signal Handling
^^^^^^^^^^^^^^^

WarpX can handle Unix (Linux/macOS) `process signals <https://en.wikipedia.org/wiki/Signal_(IPC)>`__.
This can be useful to configure jobs on HPC and cloud systems to shut down cleanly when they are close to reaching their allocated walltime or to steer the simulation behavior interactively.

Allowed signal names are documented in the `C++ standard <https://en.cppreference.com/w/cpp/utility/program/SIG_types>`__ and `POSIX <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html>`__.
We follow the same naming, but remove the ``SIG`` prefix, e.g., the WarpX signal configuration name for ``SIGINT`` is ``INT``.

.. pp:param:: warpx.break_signals
    :type: array of ``string``, separated by spaces
    :optional:

    A list of signal names or numbers that the simulation should
    handle by cleanly terminating at the next timestep

.. pp:param:: warpx.checkpoint_signals
    :type: array of ``string``, separated by spaces
    :optional:

    A list of signal names or numbers that the simulation should
    handle by outputting a checkpoint at the next timestep. A
    diagnostic of type ``checkpoint`` must be configured.

.. note::

   Certain signals are only available on specific platforms, please see the links above for details.
   Typically supported on Linux and macOS are ``HUP``, ``INT``, ``QUIT``, ``ABRT``, ``USR1``, ``USR2``, ``TERM``, ``TSTP``, ``URG``, and ``IO`` among others.

   Signals to think about twice before overwriting in *interactive simulations*:
   Note that ``INT`` (interupt) is the signal that ``Ctrl+C`` sends on the terminal, which most people use to abort a process; once overwritten you need to abort interactive jobs with, e.g., ``Ctrl+\`` (``QUIT``) or sending the ``KILL`` signal.
   The ``TSTP`` (terminal stop) command is sent interactively from ``Ctrl+Z`` to temporarily send a process to sleep (until send in the background with commands such as ``bg`` or continued with ``fg``), overwriting it would thus disable that functionality.
   The signals ``KILL`` and ``STOP`` cannot be used.

   The ``FPE`` and ``ILL`` signals should not be overwritten in WarpX, as they are `controlled by AMReX <https://amrex-codes.github.io/amrex/docs_html/Debugging.html#breaking-into-debuggers>`__ for :ref:`debug workflows that catch invalid floating-point operations <debugging_warpx>`.
.. tip::

   For example, the following logic can be added to `Slurm batch scripts <https://docs.gwdg.de/doku.php?id=en:services:application_services:high_performance_computing:running_jobs_slurm:signals>`__ (`signal name to number mapping here <https://en.wikipedia.org/wiki/Signal_(IPC)#Default_action>`__) to gracefully shut down 6 min prior to walltime.
   If you have a checkpoint diagnostics in your inputs file, this automatically will write a checkpoint due to the default :pp:param:`<diag_name>.dump_last_timestep = 1` option in WarpX.

   .. code-block:: bash

      #SBATCH --signal=1@360

      srun ...                   \
        warpx.break_signals=HUP  \
        > output.txt

   For `LSF batch systems <https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=options-wa>`__, the equivalent job script lines are:

   .. code-block:: bash

      #BSUB -wa 'HUP' -wt '6'

      jsrun ...                  \
        warpx.break_signals=HUP  \
        > output.txt

.. _running-cpp-parameters-box:

Setting up the field mesh
-------------------------

.. pp:param:: amr.n_cell
    :type: ``2 integers in 2D``, ``3 integers in 3D``

    The number of grid points along each direction (on the **coarsest level**)

.. pp:param:: amr.max_level
    :type: ``integer``
    :default: ``0``

    When using mesh refinement, the number of refinement levels that will be used.

    Use 0 in order to disable mesh refinement.

.. pp:param:: amr.ref_ratio
    :type: ``integer`` per refined level
    :default: ``2``

    When using mesh refinement, this is the refinement ratio per level.
    With this option, all directions are fined by the same ratio.

.. pp:param:: amr.ref_ratio_vect
    :type: ``3 integers for x,y,z per refined level``

    When using mesh refinement, this can be used to set the refinement ratio per direction and level, relative to the previous level.

    Example: for three levels, a value of ``2 2 4 8 8 16`` refines the first level by 2-fold in x and y and 4-fold in z compared to the coarsest level (level 0/mother grid); compared to the first level, the second level is refined 8-fold in x and y and 16-fold in z.

.. pp:param:: geometry.dims
    :type: ``string``

    The dimensions of the simulation geometry.
    Supported values are ``1``, ``2``, ``3``, ``RZ``, ``RCYLINDER``, and ``RSPHERE``.

    * For ``3``, a cartesian geometry of ``x``, ``y``, ``z`` is modeled.
    * For ``2``, a cartesian geometry with the axes ``x`` and ``z`` and all physics in ``y`` is assumed to be translation symmetric.
    * For ``1``, a cartesian geometry with the axis ``z`` and the dimensions ``x`` and ``y`` are translation symmetric.
    * For ``RZ``, a cylindrical geometry with the axis ``r`` and ``z``, with an azimuthal mode decomposition, with :pp:param:`warpx.n_rz_azimuthal_modes` providing further control.
    * For ``RCYLINDER``, a cylindrical geometry with the axis ``r``, invariant in ``theta`` and ``z``.
    * For ``RSPHERE``, a spherical geometry with the axis ``r``, invariant in ``theta`` and ``phi``. The polar angle ``phi`` is relative to the ``x-y`` plane.

    Note that this value must be consistent with the :ref:`WarpX_DIMS <install-build-options>` compile-time option.
    If you installed WarpX from a :ref:`package manager <install-methods>`, then pick the right executable by name.

.. pp:param:: warpx.n_rz_azimuthal_modes
    :type: ``integer``
    :default: 1

    When using the RZ version, this is the number of azimuthal modes.
    The default is ``1``, which corresponds to a perfectly axisymmetric simulation.

.. pp:param:: geometry.prob_lo/hi
    :link_aliases:
        geometry.prob_lo
        geometry.prob_hi
    :type: ``2 floats in 2D``, ``3 floats in 3D``
    :unit: meters

    The extent of the full simulation box. This box is rectangular, and thus its
    extent is given here by the coordinates of the lower corner (:pp:param:`geometry.prob_lo`) and
    upper corner (:pp:param:`geometry.prob_hi`). The first axis of the coordinates is x
    (or r with cylindrical) and the last is z.

.. pp:param:: warpx.do_moving_window
    :type: ``integer``
    :default: 0

    Whether to use a moving window for the simulation

.. pp:param:: warpx.moving_window_dir
    :type: either ``x``, ``y`` or ``z``

    The direction of the moving window.

.. pp:param:: warpx.moving_window_v
    :type: ``float``

    The speed of moving window, in units of the speed of light
    (i.e. use ``1.0`` for a moving window that moves exactly at the speed of light)

.. pp:param:: warpx.start_moving_window_step
    :type: ``integer``
    :default: 0

    The timestep at which the moving window starts.

.. pp:param:: warpx.end_moving_window_step
    :type: ``integer``
    :default: ``-1`` for false

    The timestep at which the moving window ends.

.. pp:param:: warpx.fine_tag_lo/hi
    :link_aliases:
        warpx.fine_tag_lo
        warpx.fine_tag_hi
    :type: ``2 floats in 2D``, ``3 floats in 3D``
    :unit: meters
    :optional:

    **When using static mesh refinement with 1 level**, the extent of the refined patch.
    This patch is rectangular, and thus its extent is given here by the coordinates
    of the lower corner (:pp:param:`warpx.fine_tag_lo`) and upper corner (:pp:param:`warpx.fine_tag_hi`).

.. pp:param:: warpx.ref_patch_function(x,y,z)
    :type: ``string``
    :optional:

    A function of ``x``, ``y``, ``z`` that defines the extent of the refined patch when
    using static mesh refinement with :pp:param:`amr.max_level`>0. Note that the function can be used
    to define distinct regions for refinement, however, the refined regions should be such that
    the pml layer surrounding the patches should not overlap. For this reason, when defining
    distinct patches, please ensure that they are sufficiently separated.

.. pp:param:: warpx.refine_plasma
    :type: ``integer``
    :default: ``0``
    :optional:

    Increase the number of macro-particles that are injected "ahead" of a mesh
    refinement patch in a moving window simulation.

    Note: in development; only works with static mesh-refinement, specific
    to moving window plasma injection, and requires a single refined level.

.. pp:param:: warpx.n_current_deposition_buffer
    :type: ``integer``

    When using mesh refinement: the particles that are located inside
    a refinement patch, but within ``n_current_deposition_buffer`` cells of
    the edge of this patch, will deposit their charge and current to the
    lower refinement level, instead of depositing to the refinement patch
    itself. See the :ref:`mesh-refinement section <theory-amr>` for more details.
    If this variable is not explicitly set in the input script,
    ``n_current_deposition_buffer`` is automatically set so as to be large
    enough to hold the particle shape, on the fine grid

.. pp:param:: warpx.n_field_gather_buffer
    :type: ``integer``
    :optional:

    Default: :pp:param:`warpx.n_field_gather_buffer = n_current_deposition_buffer + 1` (one cell larger than ``n_current_deposition_buffer`` on the fine grid).

    When using mesh refinement, particles that are located inside a refinement patch, but within ``n_field_gather_buffer`` cells of the edge of the patch, gather the fields from the lower refinement level, instead of gathering the fields from the refinement patch itself.
    This avoids some of the spurious effects that can occur inside the refinement patch, close to its edge.
    See the section :ref:`Mesh refinement <theory-amr>` for more details.

.. pp:param:: warpx.do_single_precision_comms
    :type: ``integer``
    :default: 0

    Perform MPI communications for field guard regions in single precision.
    Only meaningful for ``WarpX_PRECISION=DOUBLE``.

.. pp:param:: particles.deposit_on_main_grid
    :type: ``list of strings``

    When using mesh refinement: the particle species whose name are included
    in the list will deposit their charge/current directly on the main grid
    (i.e. the coarsest level), even if they are inside a refinement patch.

.. pp:param:: particles.gather_from_main_grid
    :type: ``list of strings``

    When using mesh refinement: the particle species whose name are included
    in the list will gather their fields from the main grid
    (i.e. the coarsest level), even if they are inside a refinement patch.

.. _running-cpp-parameters-bc:

Domain Boundary Conditions
--------------------------

.. pp:param:: boundary.field_lo/hi
    :link_aliases:
        boundary.field_lo,hi
        boundary.field_lo
        boundary.field_hi
    :type: ``2 strings`` for 2D, ``3 strings`` for 3D
    :default: ``pml``

    Boundary conditions applied to fields at the lower and upper domain boundaries.
    Options are:

    * ``Periodic``: This option can be used to set periodic domain boundaries. Note that if the fields for lo in a certain dimension are set to periodic, then the corresponding upper boundary must also be set to periodic. If particle boundaries are not specified in the input file, then particles boundaries by default will be set to periodic. If particles boundaries are specified, then they must be set to periodic corresponding to the periodic field boundaries.

    * ``pml`` (default): This option can be used to add Perfectly Matched Layers (PML) around the simulation domain. See the :ref:`PML theory section <theory-bc-PML>` for more details.
      Additional pml algorithms can be explored using the parameters :pp:param:`warpx.do_pml_in_domain`, :pp:param:`warpx.pml_has_particles`, and :pp:param:`warpx.do_pml_j_damping`.

    * ``absorbing_silver_mueller``: This option can be used to set the Silver-Mueller absorbing boundary conditions. These boundary conditions are simpler and less computationally expensive than the pml, but are also less effective at absorbing the field. They only work with the Yee Maxwell solver.

    * ``damped``: This is the recommended option in the moving direction when using the spectral solver with moving window (currently only supported along z). This boundary condition applies a damping factor to the electric and magnetic fields in the outer half of the guard cells, using a sine squared profile. As the spectral solver is by nature periodic, the damping prevents fields from wrapping around to the other end of the domain when the periodicity is not desired. This boundary condition is only valid when using the spectral solver.

    * ``pec``: This option can be used to set a Perfect Electric Conductor at the simulation boundary. Please see the :ref:`PEC theory section <theory-bc-pec>` for more details. Note that PEC boundary is invalid at ``r=0`` for RZ, RCYLINDER, and RSPHERE. Please use ``none`` option. This boundary condition does not work with the spectral solver.
      There is the additional input parameter ``particles.crop_on_PEC_boundary`` which sets whether particle trajectories are cropped when particles cross PEC boundaries, defaulting to false.

    * ``pmc``: This option can be used to set a Perfect Magnetic Conductor at the simulation boundary. Please see the :ref:`PEC theory section <theory-bc-pmc>` for more details. This is equivalent to ``Neumann``. This boundary condition does not work with the spectral solver.

    * ``pec_insulator``: This option specifies a mixed perfect electric conductor and insulator boundary, where some part of the
      boundary is PEC and some is insulator. In the insulator portion, the normal fields are extrapolated and the tangential fields
      are either set to the specified value or extrapolated. The region that is insulator is specified using a spatially dependent expression with the insulator being in the area where the value of the expression is greater than zero.
      The expressions are given for the low and high boundary on each axis, as listed below. The tangential fields are specified as
      expressions that can depend on the location and time. The tangential fields are in two pairs, the electric fields and the
      magnetic fields. In each pair, if one is specified, the other will be set to zero if not also specified.
      There is the additional input parameter ``particles.crop_on_PEC_boundary`` which sets whether particle trajectories are cropped when particles cross pec_insulator boundaries, defaulting to false.

      * ``insulator.area_x_lo(y,z)``: For the lower x (or r) boundary, expression specifying the insulator location

      * ``insulator.area_x_hi(y,z)``: For the upper x (or r) boundary, expression specifying the insulator location

      * ``insulator.area_y_lo(x,z)``: For the lower y boundary, expression specifying the insulator location

      * ``insulator.area_y_hi(x,z)``: For the upper y boundary, expression specifying the insulator location

      * ``insulator.area_z_lo(x,y)``: For the lower z boundary, expression specifying the insulator location

      * ``insulator.area_z_hi(x,y)``: For the upper z boundary, expression specifying the insulator location

      * ``insulator.Ey_x_lo(y,z,t)``, ``insulator.Ez_x_lo(y,z,t)``, ``insulator.By_x_lo(y,z,t)``, ``insulator.Bz_x_lo(y,z,t)``: expressions of the tangential field values for the lower x (or r) boundary

      * ``insulator.Ey_x_hi(y,z,t)``, ``insulator.Ez_x_hi(y,z,t)``, ``insulator.By_x_hi(y,z,t)``, ``insulator.Bz_x_hi(y,z,t)``: expressions of the tangential field values for the upper x (or r) boundary

      * ``insulator.Ex_y_lo(x,z,t)``, ``insulator.Ez_y_lo(x,z,t)``, ``insulator.Bx_y_lo(x,z,t)``, ``insulator.Bz_y_lo(x,z,t)``: expressions of the tangential field values for the lower y boundary

      * ``insulator.Ex_y_hi(x,z,t)``, ``insulator.Ez_y_hi(x,z,t)``, ``insulator.Bx_y_hi(x,z,t)``, ``insulator.Bz_y_hi(x,z,t)``: expressions of the tangential field values for the upper y boundary

      * ``insulator.Ex_z_lo(x,y,t)``, ``insulator.Ey_z_lo(x,y,t)``, ``insulator.Bx_z_lo(x,y,t)``, ``insulator.By_z_lo(x,y,t)``: expressions of the tangential field values for the lower z boundary

      * ``insulator.Ex_z_hi(x,y,t)``, ``insulator.Ey_z_hi(x,y,t)``, ``insulator.Bx_z_hi(x,y,t)``, ``insulator.By_z_hi(x,y,t)``: expressions of the tangential field values for the upper z boundary

    * ``none``: No boundary condition is applied to the fields with the electromagnetic solver. This option must be used for the lower boundary, ``r=0``, with RZ, RCYLINDER, and RSPHERE.

    * ``neumann``: For the electrostatic multigrid solver, a Neumann boundary condition (with gradient of the potential equal to 0) will be applied on the specified boundary.

    * ``open``: For the electrostatic Poisson solver based on a Integrated Green Function method.
      In RZ geometry with the hybrid-PIC solver (:pp:param:`algo.maxwell_solver = hybrid`), ``open`` on the upper radial face
      and/or the axial (z) faces selects a free-space (Green's-function) boundary for the B-field advance: the ghost values
      of B on the open faces are filled with the free-space field of the interior sources through the axisymmetric
      ring-current Green's function (poloidal components, differenced from one shared flux-function table so the ghost field
      is discretely divergence-free, including the corner ghosts where open faces meet) and Ampere's law (toroidal
      component: :math:`r B_\theta` continues radially beyond r_hi, and :math:`B_\theta` continues z-invariantly at fixed
      radius beyond an open z cap, freezing the enclosed axial-current profile at its boundary-plane value). The boundary
      then carries no image currents. Supported for m = 0 only, on the r_hi/z_lo/z_hi faces in any combination (r_lo is the
      symmetry axis); open z faces require non-periodic z.
      Applied fields must be loaded through the hybrid solver's split external fields
      (:pp:param:`hybrid_pic_model.add_external_fields`); initializing the evolved B directly via
      :pp:param:`warpx.B_ext_grid_init_style` is rejected (a curl-free applied field would be erased at the open faces).
      Optional controls: ``boundary.open_bc_coarsening`` (interior linear source-binning factor of the precomputed kernel,
      default ``4``; bins near the open face are automatically graded down to single nodes under a multipole acceptance
      criterion), ``boundary.open_bc_image_sum_rtol`` (relative tolerance of the periodic-z image sum at kernel assembly,
      default ``1e-6``, must be positive), ``boundary.open_bc_max_images`` (cap on image pairs, default ``200``, must be
      at least 1).

.. pp:param:: boundary.potential_lo/hi_x/y/z
    :link_aliases:
        boundary.potential_lo_x/y/z
        boundary.potential_hi_x/y/z
        boundary.potential_lo_x
        boundary.potential_lo_y
        boundary.potential_lo_z
        boundary.potential_hi_x
        boundary.potential_hi_y
        boundary.potential_hi_z
    :default: ``0``

    Gives the value of the electric potential, in Volts, at the boundaries, for ``pec`` boundaries. With electrostatic solvers
    (i.e., with :pp:param:`warpx.do_electrostatic = ...`), this is used in order to compute the potential
    in the simulation volume at each timestep. When using other solvers (e.g. Maxwell solver),
    setting these variables will trigger an electrostatic solve at ``t=0``, to compute the initial
    electric field produced by the boundaries.

.. pp:param:: boundary.particle_lo/hi
    :link_aliases:
        boundary.particle_lo
        boundary.particle_hi
    :type: ``2 strings`` for 2D, ``3 strings`` for 3D
    :default: ``absorbing``

    Options are:

    * ``Absorbing``: Particles leaving the boundary will be deleted.

    * ``Periodic``: Particles leaving the boundary will re-enter from the opposite boundary. The field boundary condition must be consistently set to periodic and both lower and upper boundaries must be periodic.

    * ``Reflecting``: Particles leaving the boundary are reflected from the boundary back into the domain.
      When :pp:param:`boundary.reflect_all_velocities` is false, the sign of only the normal velocity is changed, otherwise the sign of all velocities are changed.

    * ``Thermal``: Particles leaving the boundary are reflected from the boundary back into the domain
      and their velocities are thermalized. The tangential velocity components are sampled from ``gaussian`` distribution
      and the component normal to the boundary is sampled from ``gaussian flux`` distribution.
      The standard deviation for these distributions should be provided for each species using
      ``boundary.<species_name>.u_th``. The same standard deviation is used to sample all components.

    * ``None``: No boundary conditions are applied to the particles.
      When using RZ, RCYLINDER, and RSPHERE, this option must be used for the lower radial boundary, the first value of :pp:param:`boundary.particle_lo`.
      This should not be used in any other cases.

.. pp:param:: boundary.reflect_all_velocities
    :type: ``bool``
    :default: ``false``
    :optional:

    For a reflecting boundary condition, this flags whether the sign of only the normal velocity is changed or all velocities.

.. pp:param:: boundary.verboncoeur_axis_correction
    :type: ``bool``
    :default: ``true``
    :optional:

    Whether to apply the Verboncoeur correction on the charge and current density on axis when using RZ, RCYLINDER, or RSPHERE.
    For nodal values (rho and Jz), the cell volume for values on axis is :math:`\pi*\Delta dr^2/4` RZ and RCYLINDER, and :math:`\pi*\Delta dr^3/8` for RSPHERE.
    In :cite:t:`param-VerboncoeurJCP2001`, it is shown that for cylindrical coordinates, using
    :math:`\pi*\Delta dr^2/3` instead will give a uniform density if the particle density is uniform.
    For spherical coordinates, using :math:`\pi*\Delta dr^3/4` similarly gives a uniform density.

Additional PML parameters
-------------------------

.. pp:param:: warpx.pml_ncell
    :type: ``int``
    :default: 10

    The depth of the PML, in number of cells.

.. pp:param:: do_similar_dm_pml
    :type: ``int``
    :default: 1

    Whether or not to use an amrex::DistributionMapping for the PML grids that is *similar* to the mother grids, meaning that the
    mapping will be computed to minimize the communication costs between the PML and the mother grids.

.. pp:param:: warpx.pml_delta
    :type: ``int``
    :default: 10

    The characteristic depth, in number of cells, over which
    the absorption coefficients of the PML increases.

.. pp:param:: warpx.do_pml_in_domain
    :type: ``int``
    :default: 0

    Whether to create the PML inside the simulation area or outside. If inside,
    it allows the user to propagate particles in PML and to use extended PML

.. pp:param:: warpx.pml_has_particles
    :type: ``int``
    :default: 0

    Whether to propagate particles in PML or not. Can only be done if PML are in simulation domain,
    i.e. if ``warpx.do_pml_in_domain = 1``.

.. pp:param:: warpx.do_pml_j_damping
    :type: ``int``
    :default: 0

    Whether to damp current in PML. Can only be used if particles are propagated in PML,
    i.e. if ``warpx.pml_has_particles = 1``.

.. pp:param:: warpx.v_particle_pml
    :type: ``float``
    :default: 1

    When :pp:param:`warpx.do_pml_j_damping = 1`, the assumed velocity of the particles to be absorbed in the PML, in units of the speed of light ``c``.

.. pp:param:: warpx.do_pml_dive_cleaning
    :type: ``bool``

    Whether to use divergence cleaning for E in the PML region.
    The value must match :pp:param:`warpx.do_pml_divb_cleaning` (either both false or both true).
    This option seems to be necessary in order to avoid strong Nyquist instabilities in 3D simulations with the PSATD solver, open boundary conditions and PML in all directions. 2D simulations and 3D simulations with open boundary conditions and PML only in one direction might run well even without divergence cleaning.
    This option is implemented only for the Cartesian PSATD solver; it is turned on by default in this case.

.. pp:param:: warpx.do_pml_divb_cleaning
    :type: ``bool``

    Whether to use divergence cleaning for B in the PML region.
    The value must match :pp:param:`warpx.do_pml_dive_cleaning` (either both false or both true).
    This option seems to be necessary in order to avoid strong Nyquist instabilities in 3D simulations with the PSATD solver, open boundary conditions and PML in all directions. 2D simulations and 3D simulations with open boundary conditions and PML only in one direction might run well even without divergence cleaning.
    This option is implemented only for the Cartesian PSATD solver; it is turned on by default in this case.

.. _running-cpp-parameters-eb:

Embedded Boundary Conditions
----------------------------

In WarpX, the embedded boundary can be defined in either of two ways:

    - **From an analytical function:**
        In that case, you will need to set the following parameter in the input file.

        .. pp:param:: warpx.eb_implicit_function
            :type: ``string``

            A function of ``x``, ``y``, ``z`` that defines the surface of the embedded
            boundary. That surface lies where the function value is 0 ;
            the physics simulation area is where the function value is negative ;
            the interior of the embedded boundary is where the function value is positive.

    - **From an STL file:**
        In that case, you will need to set the following parameters in the input file.

        .. pp:param:: eb2.stl_file
            :type: ``string``

            The path to an `STL file <https://en.wikipedia.org/wiki/STL_(file_format)>`__.
            In addition, you also need to set ``eb2.geom_type = stl``, in order for the file to be read by WarpX.
            `See the AMReX documentation for more details <https://amrex-codes.github.io/amrex/docs_html/EB.html>`__.

Whether the embedded boundary is defined with an analytical function or an STL file, you can
additionally define the electric potential at the embedded boundary with an analytical function:

.. pp:param:: warpx.eb_potential(x,y,z,t)
    :type: ``string``

    Gives the value of the electric potential, in Volts, at the surface of the embedded boundary,
    as a function of  ``x``, ``y``, ``z`` and ``t``. With electrostatic solvers (i.e., with
    :pp:param:`warpx.do_electrostatic = ...`), this is used in order to compute the potential
    in the simulation volume at each timestep. When using other solvers (e.g. Maxwell solver),
    setting this variable will trigger an electrostatic solve at ``t=0``, to compute the initial
    electric field produced by the boundaries. Note that this function is also evaluated
    inside the embedded boundary. For this reason, it is important to define
    this function in such a way that it is constant inside the embedded boundary.

.. pp:param:: boundary.particle_eb
    :type: ``string``
    :default: ``Absorbing``
    :optional:

    The boundary condition applied to the particles when they reach the surface of the embedded boundary. Options are:

    * ``Absorbing``: Particles that reach the embedded boundary are deleted. This is the default behavior.

    * ``Reflecting``: Particles that reach the embedded boundary are specularly reflected back into the simulation domain

.. _param-particle-thermalizer:

Particle thermalizer
--------------------

In simulations of the interaction between a laser and an over-dense plasma, it is not always
practical to model the entire target. In this case, the region containing the plasma may
extend all the way the domain boundary, using either an absorbing or a thermal boundary
condition for the particles. With either choice, the resulting electric field build-up at
the boundary can lead to a non-physical return current of hot electrons that can have an
effect on the plasma instabilities and laser-plasma interaction under study.

To mitigate, WarpX implements a particle thermalizing region that reduces the flux of particles
leaving the simulation domain that leads to the non-physical build-up of electric fields at the boundary. The
method used is similar to that of `Miller et al. (Phys. Plasmas 28, 112702 (2021)) <https://doi.org/10.1063/5.0065232>`__.

The user specifies a region in which particles will be thermalized, a normal direction, a temperature, and a
momentum threshold. Inside the thermalizing region, the probability that a particle will be affected increases
from 0 to 1 as :math:`\frac{1}{1-x}^{1/4}`. Particles that are affected have their momenta thermalized
using the temperature parameter ``theta`` for any direction in which their momentum component is over the threshold
(different thresholds can be set for each direction).
The parameters affecting this region are as follows:

.. pp:param:: particle_thermalizer.normal
    :type: ``string``

    The normal direction describing the thermalizer region. Allowed values are ``x``, ``y``, or ``z`` (case-insensitive). Along with the ``start`` and ``stop`` parameters below, this specifies the region in space where particles will be thermalized.
    This parameter is optional. If not specified, the thermalizer will not be applied.

.. pp:param:: particle_thermalizer.species
    :type: ``list of strings``
    :optional:

    Names of the species to which the thermalizer is applied. If not specified, the thermalizer
    is applied to all species.

.. pp:param:: particle_thermalizer.start
    :type: ``float``

    Starting coordinate (in SI units) of the thermalization region along the specified normal direction.
    This parameter is required if the thermalizer is enabled.

.. pp:param:: particle_thermalizer.end
    :type: ``float``

    Ending coordinate (in SI units) of the thermalization region along the specified normal direction.
    This parameter is required if the thermalizer is enabled.

.. pp:param:: particle_thermalizer.momentum_threshold
    :type: ``float`` or ``3 floats``

    Momentum threshold used by the thermalizer. In each direction, if a particle's normalized momentum component (e.g. :math:`\gamma \beta_x`) is above this threshold, that component will be thermalized.
    This parameter is required if the thermalizer is enabled. One or three values can be provided. In the former case, the same threshold is applied in all directions. In the latter, different thresholds
    are applied to ``x``, ``y``, and ``z`` directions.

.. pp:param:: particle_thermalizer.theta
    :type: ``float``

    Dimensionless temperature parameter (k*T/m/c^2) used to sample the thermalized particle velocities.
    This parameter is required if the thermalizer is enabled. For the selected particles, if the
    normalized momentum in any direction exceeds the threshold, the particle's momentum in that direction will be set
    to a value drawn from a Gaussian distribution with mean 0.0 and variance ``theta``.

Example::

    particle_thermalizer.normal = z
    particle_thermalizer.start = 0.0
    particle_thermalizer.end = 1.0e-6
    particle_thermalizer.momentum_threshold = 0.5
    particle_thermalizer.theta = 0.1
    particle_thermalizer.species = electrons hydrogen

.. _running-cpp-parameters-parallelization:

Distribution across MPI ranks and parallelization
-------------------------------------------------

.. pp:param:: warpx.numprocs
    :type: ``2 ints`` for 2D, ``3 ints`` for 3D
    :default: ``none``
    :optional:

    This optional parameter can be used to control the domain decomposition on the
    coarsest level. The domain will be chopped into the exact number of pieces in each
    dimension as specified by this parameter. If it's not specified, the domain
    decomposition will be determined by the parameters that will be discussed below.  If
    specified, the product of the numbers must be equal to the number of MPI processes.

.. pp:param:: amr.max_grid_size
    :type: ``integer``
    :default: ``128``
    :optional:

    Maximum allowable size of each **subdomain**
    (expressed in number of grid points, in each direction).
    Each subdomain has its own ghost cells, and can be handled by a
    different MPI rank ; several OpenMP threads can work simultaneously on the
    same subdomain.

    If ``max_grid_size`` is such that the total number of subdomains is
    **larger** that the number of MPI ranks used, than some MPI ranks
    will handle several subdomains, thereby providing additional flexibility
    for **load balancing**.

    When using mesh refinement, this number applies to the subdomains
    of the coarsest level, but also to any of the finer level.

.. pp:param:: algo.load_balance_intervals
    :type: ``string``
    :default: ``0``
    :optional:

    Using the `Time intervals`_ syntax, this string defines the timesteps at which
    WarpX should try to redistribute the work across MPI ranks, in order to have
    better load balancing.
    Use 0 to disable load_balancing.

    When performing load balancing, WarpX measures the wall time for
    computational parts of the PIC cycle. It then uses this data to decide
    how to redistribute the subdomains across MPI ranks. (Each subdomain
    is unchanged, but its owner is changed in order to have better performance.)
    This relies on each MPI rank handling several (in fact many) subdomains
    (see ``max_grid_size``).

.. pp:param:: algo.load_balance_efficiency_ratio_threshold
    :type: ``float``
    :default: ``1.1``
    :optional:

    Controls whether to adopt a proposed distribution mapping computed during a load balance.
    If the the ratio of the proposed to current distribution mapping *efficiency* (i.e.,
    average cost per MPI process; efficiency is a number in the range [0, 1]) is greater
    than the threshold value, the proposed distribution mapping is adopted.  The suggested
    range of values is :pp:param:`algo.load_balance_efficiency_ratio_threshold >= 1`, which ensures
    that the new distribution mapping is adopted only if doing so would improve the load
    balance efficiency. The higher the threshold value, the more conservative is the criterion
    for adoption of a proposed distribution; for example, with
    :pp:param:`algo.load_balance_efficiency_ratio_threshold = 1`, the proposed distribution is
    adopted *any* time the proposed distribution improves load balancing; if instead
    :pp:param:`algo.load_balance_efficiency_ratio_threshold = 2`, the proposed distribution is
    adopted only if doing so would yield a 100% to the load balance efficiency (with this
    threshold value, if the  current efficiency is ``0.45``, the new distribution would only be
    adopted if the proposed efficiency were greater than ``0.9``).

.. pp:param:: algo.load_balance_with_sfc
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    If this is ``1``: use a Space-Filling Curve (SFC) algorithm in order to
    perform load-balancing of the simulation.
    If this is ``0``: the Knapsack algorithm is used instead.

.. pp:param:: algo.load_balance_knapsack_factor
    :type: ``float``
    :default: ``1.24``
    :optional:

    Controls the maximum number of boxes that can be assigned to a rank during
    load balance when using the 'knapsack' policy for update of the distribution
    mapping; the maximum is
    ``load_balance_knapsack_factor*(average number of boxes per rank)``.
    For example, if there are 4 boxes per rank and ``load_balance_knapsack_factor=2``,
    no more than 8 boxes can be assigned to any rank.

.. pp:param:: algo.load_balance_costs_update
    :type: ``heuristic`` or ``timers``
    :default: ``timers``
    :optional:

    If this is ``heuristic``: load balance costs are updated according to a measure of
    particles and cells assigned to each box of the domain.  The cost :math:`c` is
    computed as

    .. math::

       c = n_{\text{particle}} \cdot w_{\text{particle}} + n_{\text{cell}} \cdot w_{\text{cell}},

    where
    :math:`n_{\text{particle}}` is the number of particles on the box,
    :math:`w_{\text{particle}}` is the particle cost weight factor (controlled by :pp:param:`algo.costs_heuristic_particles_wt`),
    :math:`n_{\text{cell}}` is the number of cells on the box, and
    :math:`w_{\text{cell}}` is the cell cost weight factor (controlled by :pp:param:`algo.costs_heuristic_cells_wt`).

    If this is ``timers``: costs are updated according to in-code timers.

.. pp:param:: algo.costs_heuristic_particles_wt
    :type: ``float``
    :optional:

    Particle weight factor used in ``Heuristic`` strategy for costs update; if running on GPU,
    the particle weight is set to a value determined from single-GPU tests on Summit,
    depending on the choice of solver (FDTD or PSATD) and order of the particle shape.
    If running on CPU, the default value is ``0.9``. If running on GPU, the default value is

    +----------+-----------------------+
    |          | Particle shape factor |
    +----------+-------+-------+-------+
    |          | 1     | 2     | 3     |
    +==========+=======+=======+=======+
    | FDTD/CKC | 0.599 | 0.732 | 0.855 |
    +----------+-------+-------+-------+
    | PSATD    | 0.425 | 0.595 | 0.75  |
    +----------+-------+-------+-------+

.. pp:param:: algo.costs_heuristic_cells_wt
    :type: ``float``
    :optional:

    Cell weight factor used in ``Heuristic`` strategy for costs update; if running on GPU,
    the cell weight is set to a value determined from single-GPU tests on Summit,
    depending on the choice of solver (FDTD or PSATD) and order of the particle shape.
    If running on CPU, the default value is ``0.1``. If running on GPU, the default value is

    +----------+-----------------------+
    |          | Particle shape factor |
    +----------+-------+-------+-------+
    |          | 1     | 2     | 3     |
    +==========+=======+=======+=======+
    | FDTD/CKC | 0.401 | 0.268 | 0.145 |
    +----------+-------+-------+-------+
    | PSATD    | 0.575 | 0.405 | 0.25  |
    +----------+-------+-------+-------+

.. pp:param:: warpx.do_dynamic_scheduling
    :type: ``0`` or ``1``
    :default: ``1``
    :optional:

    Whether to activate OpenMP dynamic scheduling.

.. pp:param:: warpx.roundrobin_sfc
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Whether to use AMReX's RRSFS strategy for making DistributionMapping to
    override the default space filling curve (SFC) strategy. If this is
    enabled, the round robin method is used to distribute Boxes ordered by
    SFC. This could potentially mitigate the load imbalance issue during
    initialization by avoiding putting neighboring boxes on the same
    process.

.. pp:param:: warpx.split_high_density_boxes
    :type: ``bool``
    :default: false
    :optional:

    Whether to split high density boxes during initialization. This can
    improve the potential for load balancing.

.. pp:param:: warpx.split_high_density_boxes_threshold
    :type: ``float``
    :default: 1.1
    :optional:

    Threshold used in splitting high density boxes. If a Box has more
    particles than the average number of particles per MPI process
    multiplied by this factor, we try to split this Box into smaller ones.

.. pp:param:: warpx.split_high_density_boxes_min_box_size
    :type: ``integer``
    :default: 8
    :optional:

    During splitting high density boxes, if a Box's longest side is already
    less than or equal to this number, it will not be split.


.. _running-cpp-parameters-particle:

Particle initialization
-----------------------

.. pp:param:: particles.species_names
    :type: ``strings``, separated by spaces

    The name of each species. This is then used in the rest of the input deck ;
    in this documentation we use ``<species_name>`` as a placeholder.

.. pp:param:: particles.use_fdtd_nci_corr
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Whether to activate the FDTD Numerical Cherenkov Instability corrector.
    Not currently available in the RZ, RCYLINDER, and RSPHERE configuration.

.. pp:param:: particles.rigid_injected_species
    :type: ``strings``, separated by spaces

    List of species injected using the rigid injection method. The rigid injection
    method is useful when injecting a relativistic particle beam in boosted-frame
    simulations; see the :ref:`input-output section <boosted_frame-io>` for more details.
    For species injected using this method, particles are translated along the ``+z``
    axis with constant velocity as long as their ``z`` coordinate verifies
    ``z<zinject_plane``. When ``z>zinject_plane``,
    particles are pushed in a standard way, using the specified pusher.
    (see the parameter :pp:param:`<species_name>.zinject_plane` below)

.. pp:param:: particles.do_tiling
    :type: ``bool``
    :default: ``false`` if WarpX is compiled for GPUs, ``true`` otherwise
    :optional:

    Controls whether tiling ('cache blocking') transformation is used for particles.
    Tiling should be on when using OpenMP and off when using GPUs.

.. pp:param:: <species_name>.species_type
    :type: ``string``
    :default: ``unspecified``
    :optional:

    Type of physical species.
    Currently, the accepted species are
    ``"electron"``, ``"positron"``, ``"muon"``, ``"antimuon"``, ``"photon"``, ``"neutron"``,
    ``"hydrogen1"`` (a.k.a. ``"proton"``), ``"hydrogen2"`` (a.k.a. ``"deuterium"``), ``"hydrogen3"`` (a.k.a. ``"tritium"``),
    ``"helium"``, ``"helium3"``, ``"helium4"`` (a.k.a. ``"alpha"``),
    ``"lithium"``, ``"lithium6"``, ``"lithium7"``, ``"beryllium"``, ``"beryllium9"``, ``"boron"``, ``"boron10"``, ``"boron11"``,
    ``"carbon"``, ``"carbon12"``, ``"carbon13"``, ``"carbon14"``, ``"nitrogen"``, ``"nitrogen14"``, ``"nitrogen15"``,
    ``"oxygen"``, ``"oxygen16"``, ``"oxygen17"``, ``"oxygen18"``, ``"fluorine"``, ``"fluorine19"``, ``"neon"``, ``"neon20"``,
    ``"neon21"``, ``"neon22"``, ``"aluminium"``, ``"argon"``, ``"copper"``, ``"xenon"`` and ``"gold"``.
    When an atomic element is specified (e.g. ``oxygen``), the species will be assumed to be fully ionized
    (e.g., with charge :math:`+8 e` for ``oxygen``). When only the name of an element is specified
    (e.g. ``oxygen`` instead of ``oxygen16``), the mass is a weighted average of the masses
    of the stable isotopes. When ``species_type`` is specified, ``mass`` and ``charge`` do not need to be specified.
    In that case, the mass will be taken from pre-defined values `here <https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=some>`__.
    If ``mass`` and/or ``charge`` are nonetheless specified, they will override the pre-defined values for that ``species_type``.

.. pp:param:: <species_name>.charge
    :type: ``float``
    :default: ``NaN``
    :optional:

    The charge of one *physical* particle of this species.
    If ``species_type`` is specified, the charge will be set to the physical value and ``charge`` is optional.
    When :pp:param:`<species_name>.do_field_ionization = 1`, the physical particle charge is equal to ``ionization_initial_level * charge``, so latter parameter should be equal to q_e (which is defined in WarpX as the elementary charge in coulombs).

.. pp:param:: <species_name>.mass
    :type: ``float``
    :default: ``NaN``
    :optional:

    The mass of one *physical* particle of this species.
    If ``species_type`` is specified, the mass will be set to the physical value and ``mass`` is optional.
    ``mass`` must be strictly positive. For massless species, use :pp:param:`<species_name>.species_type`. The only allowed massless species type is ``photon``.

.. pp:param:: <species_name>.xmin/ymin/zmin/xmax/ymax/zmax
    :link_aliases:
        <species_name>.xmin,ymin,zmin
        <species_name>.xmax,ymax,zmax
        <species_name>.xmin/ymin/zmin
        <species_name>.xmax/ymax/zmax
        <species_name>.xmin
        <species_name>.ymin
        <species_name>.zmin
        <species_name>.xmax
        <species_name>.ymax
        <species_name>.zmax
    :type: ``float``
    :default: unlimited
    :optional:

    When :pp:param:`<species_name>.xmin` and :pp:param:`<species_name>.xmax` are set, they delimit the region within which particles are injected.
    If periodic boundary conditions are used in direction ``i``, then the default (i.e. if the range is not specified) range will be the simulation box, :pp:param:`[geometry.prob_hi[i], geometry.prob_lo[i]]`.

.. pp:param:: <species_name>.injection_sources
    :type: ``list of strings``
    :optional:

    Names of additional injection sources. By default, WarpX assumes one injection source per species, hence all of the input
    parameters below describing the injection are parameters directly of the species. However, this option allows
    additional sources, the names of which are specified here. For each source, the name of the source is added to the
    input parameters below. For instance, with :pp:param:`<species_name>.injection_sources = source1 source2` there can be the two input
    parameters ``<species_name>.source1.injection_style`` and ``<species_name>.source2.injection_style``.
    For the parameters of each source, the parameter with the name of the source will be used.
    If it is not given, the value of the parameter without the source name will be used. This allows parameters used for all
    sources to be specified once. For example, if the ``source1`` and ``source2`` have the same value of ``uz_m``, then it can be
    set using ``<species_name>.uz_m`` instead of setting it for each source.
    Note that since by default :pp:param:`<species_name>.injection_style = none`, all injection sources can be input this way.
    Note that if a moving window is used, the bulk velocity of all of the sources must be the same since it is used when updating the window.

.. pp:param:: <species_name>.injection_style
    :type: ``string``
    :default: ``none``

    Determines how the (macro-)particles will be injected in the simulation.
    The number of particles per cell is always given with respect to the coarsest level (level 0/mother grid), even if particles are immediately assigned to a refined patch.

    The options are:

    * ``NUniformPerCell``: injection with a fixed number of evenly-spaced particles per cell.
      This requires the additional parameter :pp:param:`<species_name>.num_particles_per_cell_each_dim`.

    * ``NRandomPerCell``: injection with a fixed number of randomly-distributed particles per cell.
      This requires the additional parameter ``<species_name>.num_particles_per_cell``.

    * ``SingleParticle``: Inject a single macroparticle.
      This requires the additional parameters:

      * ``<species_name>.single_particle_pos`` (``3 doubles``, particle 3D position [meter])

      * ``<species_name>.single_particle_u`` (``3 doubles``, particle 3D normalized momentum, i.e. :math:`\gamma \beta`)

      * ``<species_name>.single_particle_weight`` ( ``double``, macroparticle weight, i.e. number of physical particles it represents)

    * ``MultipleParticles``: Inject multiple macroparticles.
      This requires the additional parameters:

      * ``<species_name>.multiple_particles_pos_x`` (list of ``doubles``, X positions of the particles [meter])

      * ``<species_name>.multiple_particles_pos_y`` (list of ``doubles``, Y positions of the particles [meter])

      * ``<species_name>.multiple_particles_pos_z`` (list of ``doubles``, Z positions of the particles [meter])

      * ``<species_name>.multiple_particles_ux`` (list of ``doubles``, X normalized momenta of the particles, i.e. :math:`\gamma \beta_x`)

      * ``<species_name>.multiple_particles_uy`` (list of ``doubles``, Y normalized momenta of the particles, i.e. :math:`\gamma \beta_y`)

      * ``<species_name>.multiple_particles_uz`` (list of ``doubles``, Z normalized momenta of the particles, i.e. :math:`\gamma \beta_z`)

      * ``<species_name>.multiple_particles_weight`` (list of ``doubles``, macroparticle weights, i.e. number of physical particles each represents)

    * ``gaussian_beam``: Inject particle beam with gaussian distribution in
      space in all directions. This requires additional parameters:

      * ``<species_name>.q_tot`` (beam charge),

      * ``<species_name>.npart_real`` (total number of real particles in the beam)

      The user must define one and only only between ``q_tot`` and ``npart_real``.
      The latter must be used for neutral species.

      * ``<species_name>.npart`` (number of macroparticles in the beam),

      * ``<species_name>.x/y/z_m`` (average position in ``x/y/z``),

      * ``<species_name>.x/y/z_rms`` (standard deviation in ``x/y/z``),

      There are additional optional parameters:

      * ``<species_name>.x/y/z_cut`` (optional, particles with ``abs(x-x_m) > x_cut*x_rms`` are not injected, same for y and z. ``<species_name>.q_tot`` is the charge of the un-cut beam, so that cutting the distribution is likely to result in a lower total charge),
      * ``<species_name>.do_symmetrize`` (optional, whether to symmetrize the beam)

      * ``<species_name>.symmetrization_order`` (order of symmetrization, default is 4, can be 4 or 8).

      If ``<species_name>.do_symmetrize`` is 0, no symmetrization occurs.  If ``<species_name>.do_symmetrize`` is 1,
      then the beam is symmetrized according to the value of ``<species_name>.symmetrization_order``.
      If set to 4, symmetrization is in the x and y direction, (x,y) (-x,y) (x,-y) (-x,-y).
      If set to 8, symmetrization is also done with x and y exchanged, (y,x), (-y,x), (y,-x), (-y,-x)).

      * ``<species_name>.focal_distance`` (optional, distance between the beam centroid and the position of the focal plane of the beam, along the direction of the beam mean velocity; space charge is ignored in the initialization of the particles)

      If ``<species_name>.focal_distance`` is specified, ``x_rms``, ``y_rms`` and ``z_rms`` are the sizes of the beam in the focal plane. Since the beam is not necessarily initialized close to its focal plane, the initial size of the beam will differ from ``x_rms``, ``y_rms``, ``z_rms``.

      Usually, in accelerator physics the operative quantities are the normalized emittances :math:`\epsilon_{x,y}` and beta functions :math:`\beta_{x,y}`.
      We assume that the beam travels along :math:`z` and we mark the quantities evaluated at the focal plane with a :math:`*`.
      Therefore, the normalized transverse emittances and beta functions are related to the focal distance :math:`f = z - z^*`, the beam sizes :math:`\sigma_{x,y}` (which in the code are ``x_rms``, ``y_rms``), the beam relativistic Lorentz factor :math:`\gamma`, and the normalized momentum spread :math:`\Delta u_{x,y}` according to the equations below (:cite:t:`param-Wiedemann2015`).

      .. math::

          \Delta u_{x,y} &= \frac{\epsilon^*_{x,y}}{\sigma^*_{x,y}},

          \sigma*_{x, y} &= \sqrt{ \frac{ \epsilon^*_{x,y} \beta^*_{x,y} }{\gamma}},

          \sigma_{x,y}(z) &= \sigma^*_{x,y} \sqrt{1 + \left( \frac{z - z^*}{\beta^*_{x,y}} \right)^2}

      * ``<species_name>.do_gaussian_beam_rotation`` (``bool``, optional) the positions of the beam particles are rotated around the beam centroid.

      If ``do_gaussian_beam_rotation = 1`` then the user needs to specify:

          * ``<species_name>.gaussian_beam_rotation_axis``: (list of 3 ``doubles``) axis around which the rotation takes place

          * ``<species_name>.gaussian_beam_rotation_angle``: (``double``) angle of rotation around the specified axis, in radians.

      * ``<species_name>.do_gaussian_beam_rotation_momenta`` (``bool``, optional) the momenta of the beam particles are also rotated using the same transformation applied to their positions. The rotation is the same as that for the positions. Momenta cannot be rotated independently; position rotation must be enabled first.

      Note that the other beam parameters (e.g. ``<species_name>.x/y/z_rms``, etc.) are used in the initialization process *before* performing the rotation.
      Therefore, the user should define the beam size, cuts, and focal distance for the beam pre-rotation, hence aligned to the Cartesian axes.

    * ``external_file``: Inject macroparticles with properties (mass, charge, position, and momentum - :math:`\gamma \beta m c`) read from an external openPMD file.
      With it users can specify the additional arguments:

      * ``<species_name>.injection_file`` (``string``) openPMD file name and

      * :pp:param:`<species_name>.charge` (``double``) optional (default is read from openPMD file) when set this will be the charge of the physical particle represented by the injected macroparticles.

      * :pp:param:`<species_name>.mass` (``double``) optional (default is read from openPMD file) when set this will be the charge of the physical particle represented by the injected macroparticles.

      * ``<species_name>.z_shift`` (``double``) optional (default is no shift) when set this value will be added to the longitudinal, ``z``, position of the particles.

      * ``<species_name>.impose_t_lab_from_file`` (``bool``) optional (default is false) only read if warpx.gamma_boost > 1., it allows to set t_lab for the Lorentz Transform as being the time stored in the openPMD file.

      Warning: ``q_tot!=0`` is not supported with the ``external_file`` injection style. If a value is provided, it is ignored and no re-scaling is done.
      The external file must include the species ``openPMD::Record`` labeled ``position`` and ``momentum`` (``double`` arrays), with dimensionality and units set via ``openPMD::setUnitDimension`` and ``setUnitSI``.
      If the external file also contains ``openPMD::Records`` for ``mass`` and ``charge`` (constant ``double`` scalars) then the species will use these, unless overwritten in the input file (see :pp:param:`<species_name>.mass`, :pp:param:`<species_name>.charge` or :pp:param:`<species_name>.species_type`).
      The ``external_file`` option is currently implemented for 2D, 3D and RZ geometries, with record components in the cartesian coordinates ``(x,y,z)`` for 3D and RZ, and ``(x,z)`` for 2D.
      For more information on the `openPMD format <https://github.com/openPMD>`__ and how to build WarpX with it, please visit :ref:`the install section <install-build-cmake>`.
      See `this file <https://github.com/BLAST-WarpX/warpx/blob/development/Examples/Tests/gaussian_beam/inputs_test_3d_focusing_gaussian_beam_from_openpmd_prepare.py>`__
      for an example of how to prepare the openPMD data file.

    * ``NFluxPerCell``: Continuously inject a flux of macroparticles from a surface. The emitting surface can be chosen to be either a plane
      defined by the user (using some of the parameters listed below), or the embedded boundary (see :ref:`Embedded Boundary Conditions <running-cpp-parameters-eb>`).
      This requires the additional parameters:

      * :pp:param:`<species_name>.flux_profile` (see the description of this parameter further below)

      * ``<species_name>.inject_from_embedded_boundary`` (``0`` or ``1``, default ``0`` ; whether to inject from the embedded boundary or from a user-specified plane.
        When injecting from the embedded boundary, the momentum distribution specified by the user along ``z`` (see e.g. ``uz_m``, ``uz_th`` below) is interpreted
        as the momentum distribution along the local normal to the embedded boundary.)

      * ``<species_name>.surface_flux_pos`` (only used when injecting from a plane, ``double``, location of the injection plane [meter])

      * ``<species_name>.flux_normal_axis`` (only used when injecting from a plane, ``x``, ``y``, or ``z`` for 3D, ``x`` or ``z`` for 2D, or ``r``, ``t``, or ``z`` for RZ, or ``r`` for RCYLINDER and RSPHERE. When ``flux_normal_axis`` is ``r`` or ``t``, the ``x`` and ``y`` components of the user-specified momentum distribution are interpreted as the ``r`` and ``t`` components respectively)

      * ``<species_name>.flux_direction`` (only used when injecting from a plane, ``-1`` or ``+1``, direction of flux relative to the plane)

      * ``<species_name>.num_particles_per_cell`` (``double``)

      * ``<species_name>.flux_tmin`` (``double``, Optional time at which the flux will be turned on. Ignored when negative.)

      * ``<species_name>.flux_tmax`` (``double``, Optional time at which the flux will be turned off. Ignored when negative.)

    * ``none``: Do not inject macro-particles (for example, in a simulation that starts with neutral, ionizable atoms, one may want to create the electrons species -- where ionized electrons can be stored later on -- without injecting electron macro-particles).

.. pp:param:: <species_name>.num_particles_per_cell_each_dim
    :type: ``3 integers in 3D, RZ, RSPHERE, 2 integers in 2D and RCYLINDER``

    With the NUniformPerCell injection style, this specifies the number of particles along each axis
    within a cell. For RZ, the three axis are radius, theta, and z and that the recommended
    number of particles per theta is at least two times the number of azimuthal modes requested.
    (It is recommended to do a convergence scan of the number of particles per theta)
    For RSPHERE, the three axis are radius, theta, and phi, and for RCYLINDER, the two axis are radius and theta.

.. pp:param:: <species_name>.random_theta
    :type: ``bool``
    :default: ``1``
    :optional:

    When using RZ, RCYLINDER, or RSPHERE geometry, particle azimuthal angles are always defined in the range :math:`(-\pi, \pi]`.

    * For :pp:param:`<species_name>.injection_style = NUniformPerCell` and this flag set to ``true``, a random azimuthal offset is applied independently in each cell. This rotates the particle distribution randomly from cell to cell while keeping angles within :math:`(-\pi, \pi]`.

    * For :pp:param:`<species_name>.injection_style = NRandomPerCell`, this flag essentially does nothing since particle positions are set randomly anyway.

.. pp:param:: <species_name>.do_splitting
    :type: ``bool``
    :default: ``0``
    :optional:

    Split particles of the species when crossing the boundary from a lower
    resolution domain to a higher resolution domain.

    Currently implemented on CPU only.

.. pp:param:: <species_name>.do_continuous_injection
    :type: ``0`` or ``1``

    Whether to inject particles during the simulation, and not only at
    initialization. This can be required with a moving window and/or when
    running in a boosted frame.

.. pp:param:: <species_name>.initialize_self_fields
    :type: ``0`` or ``1``

    Whether to calculate the space-charge fields associated with this species
    at the beginning of the simulation.
    The fields are calculated for the mean gamma of the species.

.. pp:param:: <species_name>.self_fields_required_precision
    :type: ``float``
    :default: 1.e-11

    The relative precision with which the initial space-charge fields should
    be calculated. More specifically, the initial space-charge fields are
    computed with an iterative Multi-Level Multi-Grid (MLMG) solver.
    For highly-relativistic beams, this solver can fail to reach the default
    precision within a reasonable time ; in that case, users can set a
    relaxed precision requirement through ``self_fields_required_precision``.

.. pp:param:: <species_name>.self_fields_absolute_tolerance
    :type: ``float``
    :default: 0.0

    The absolute tolerance with which the space-charge fields should be
    calculated in units of :math:`\mathrm{V/m}^2`. More specifically, the acceptable
    residual with which the solution can be considered converged. In general
    this should be left as the default, but in cases where the simulation state
    changes very little between steps it can occur that the initial guess for
    the MLMG solver is so close to the converged value that it fails to improve
    that solution sufficiently to reach the ``self_fields_required_precision``
    value.

.. pp:param:: <species_name>.self_fields_max_iters
    :type: ``integer``
    :default: 200

    Maximum number of iterations used for MLMG solver for initial space-charge
    fields calculation. In case if MLMG converges but fails to reach the desired
    ``self_fields_required_precision``, this parameter may be increased.

.. pp:param:: <species_name>.profile
    :type: ``string``

    Density profile for this species. The options are:

    * ``constant``: Constant density profile within the box, or between :pp:param:`<species_name>.xmin`
      and :pp:param:`<species_name>.xmax` (and same in all directions). This requires additional
      parameter ``<species_name>.density``. i.e., the plasma density in :math:`m^{-3}`.

    * ``parse_density_function``: the density is given by a function in the input file.
      It requires additional argument ``<species_name>.density_function(x,y,z)``, which is a
      mathematical expression for the density of the species, e.g.
      ``electrons.density_function(x,y,z) = "n0+n0*x**2*1.e12"`` where ``n0`` is a
      user-defined constant, see above.

    * ``read_from_file``: load the density profile from an openPMD file.
      An additional parameter, indicating the path of an openPMD data file,
      ``<species_name>.read_density_from_path`` must be specified. The openPMD
      file must contain a field with the name given by ``<species_name>.density_mesh_name``
      (default ``density``). See
      `this file <https://github.com/BLAST-WarpX/warpx/blob/development/Examples/Tests/load_density/inputs_test_3d_load_density_prepare.py>`__
      for an example of how to prepare the openPMD data file. There is
      another optional parameter,
      ``<species_name>.read_density_distributed=true``, which controls how the
      openPMD data are distributed among processes. If it is set to false, the
      openPMD data are loaded and duplicated on every process. If it is set to
      true, the openPMD data required for initializing the density profile
      are distributed among MPI processes. If particles are continuously
      injected during the simulation and
      ``<species_name>.read_density_distributed`` is true, chunks of the
      openPMD data are loaded and cached as needed.

.. pp:param:: <species_name>.flux_profile
    :type: ``string``

    Defines the expression of the flux, when using :pp:param:`<species_name>.injection_style = NFluxPerCell`

    * ``constant``: Constant flux. This requires the additional parameter ``<species_name>.flux``.
      i.e., the injection flux in :math:`m^{-2}.s^{-1}`.

    * ``parse_flux_function``: the flux is given by a function in the input file.
      It requires the additional argument ``<species_name>.flux_function(x,y,z,t)``, which is a
      mathematical expression for the flux of the species.

.. pp:param:: <species_name>.density_min
    :type: ``float``
    :default: ``0.``
    :optional:

    Minimum plasma density. No particle is injected where the density is below this value.
    This is useful because, where the density is close to zero, particles would otherwise
    still be injected between ``xmin`` and ``xmax`` etc., with a null weight. This is
    undesirable because it results in useless computing. This option applies to all density
    profiles (``constant``, ``parse_density_function`` and ``read_from_file``).

.. pp:param:: <species_name>.density_max
    :type: ``float``
    :default: ``infinity``
    :optional:

    Maximum plasma density. The density at each point is the minimum between the value given in the profile, and ``density_max``.
    This option applies to all density profiles (``constant``, ``parse_density_function`` and ``read_from_file``).

.. pp:param:: <species_name>.radial_numpercell_power
    :type: ``float``
    :default: ``0``
    :optional:

    With cylindrical and spherical geometry, specifies the radial power scaling of the number of particles per cell.
    The number of particles per cell will be proportional to :math:`r^p`, where :math:`r` is the radius, and :math:`p` is the specified power.
    The power must be greater than -1.
    When the power is 0, the default value, the number of particles per cell will be uniform.
    With a uniform density, a power of 1 for cylindrical, and a power of 2 for spherical, will give uniform particle weights.
    The total number of particles loaded along the radius will be :math:`rmax/dr*N_{percell}`, :math:`rmax` the maximum radius particles are loaded, :math:`dr` the radial grid cell size, and :math:`N_{percell}` the number of particles per cell.
    The particle weights are set accordingly depending on the power and on the specified density profile.

.. pp:param:: <species_name>.momentum_distribution_type
    :type: ``string``

    Distribution of the normalized momentum (``u=p/mc``) for this species. The options are:

    * ``at_rest``: Particles are initialized with zero momentum.

    * ``constant``: constant momentum profile. This can be controlled with the additional parameters
      ``<species_name>.ux``, ``<species_name>.uy`` and ``<species_name>.uz``, the normalized
      momenta in the x, y and z direction respectively, which are all ``0.`` by default.

    * ``uniform``: uniform probability distribution between a minimum and a maximum value.
      The x, y and z directions are sampled independently and the final momentum space is a cuboid.
      The parameters that control the minimum and maximum domain of the distribution
      are ``<species_name>.u<x,y,z>_min`` and ``<species_name>.u<x,y,z>_max`` in each
      direction respectively (e.g., ``<species_name>.uz_min = 0.2`` and ``<species_name>.uz_max = 0.4``
      to control the generation along the ``z`` direction).
      All the parameters default to ``0``.

    * ``gaussian``: gaussian momentum distribution in all 3 directions. This can be controlled with the
      additional arguments for the average momenta along each direction
      ``<species_name>.ux_m``, ``<species_name>.uy_m`` and ``<species_name>.uz_m`` as
      well as standard deviations along each direction ``<species_name>.ux_th``,
      ``<species_name>.uy_th`` and ``<species_name>.uz_th``.
      These 6 parameters are all ``0.`` by default.

    * ``gaussianflux``: Gaussian momentum flux distribution, which is Gaussian in the plane and v*Gaussian normal to the plane.
      It can only be used when ``injection_style = NFluxPerCell``.
      This can be controlled with the additional arguments to specify the plane's orientation, ``<species_name>.flux_normal_axis`` and
      ``<species_name>.flux_direction``, for the average momenta along each direction
      ``<species_name>.ux_m``, ``<species_name>.uy_m`` and ``<species_name>.uz_m``, as
      well as standard deviations along each direction ``<species_name>.ux_th``,
      ``<species_name>.uy_th`` and ``<species_name>.uz_th``.
      ``ux_m``, ``uy_m``, ``uz_m``, ``ux_th``, ``uy_th`` and ``uz_th`` are all ``0.`` by default.

    * ``maxwellian``: Maxwellian momentum distribution. The mean normalized momentum (bulk drift) and the standard deviation (thermal spread) of each
      momentum component can be specified independently. They can be given either as constants or as functions of position.
      Each normalized-momentum component is sampled independently from a Gaussian distribution.

      It requires the following arguments:

      * ``<species_name>.maxwellian_u_mean_distribution_type`` (`string`, default ``constant``):
        Specifies the distribution type for the bulk (mean) particle momentum ``u_mean``.
        Here, ``u_mean`` is a 3D vector (with components ``ux_mean``, ``uy_mean``, ``uz_mean``)
        representing the normalized momentum, defined as
        :math:`u_\mathrm{mean} = \gamma \beta`, where
        :math:`\beta = v/c` and :math:`\gamma = 1/\sqrt{1-\beta^2}`.

        * If ``constant``, the following are required: ``<species_name>.ux_mean``,
          ``<species_name>.uy_mean``, ``<species_name>.uz_mean`` (`float`, default ``0``).
          The magnitude :math:`|u_\mathrm{mean}|` must be strictly less than 1.
        * If ``parser``, the following are required:
          ``<species_name>.ux_mean_function(x,y,z)``,
          ``<species_name>.uy_mean_function(x,y,z)``,
          ``<species_name>.uz_mean_function(x,y,z)``.

      * ``<species_name>.maxwellian_u_std_distribution_type`` (`string`, default ``constant``):
        Specifies the distribution type for the thermal spread (standard deviation) of the
        particle momentum. Here, ``u_std`` is a 3D vector (with components ``ux_std``,
        ``uy_std``, ``uz_std``) representing the standard deviation of the normalized momentum
        :math:`u_\mathrm{std} = \sqrt{\theta}`, where
        :math:`\theta = \frac{k_\mathrm{B} \cdot T}{m \cdot c^2}`.

        * If ``constant``, the following are required: ``<species_name>.ux_std``,
          ``<species_name>.uy_std``, ``<species_name>.uz_std`` (`float`, default ``0``).
          These are standard deviations of :math:`u_x`, :math:`u_y`, :math:`u_z` in the drift
          frame, i.e. the thermal spread per axis.
        * If ``parser``, the following are required:
          ``<species_name>.ux_std_function(x,y,z)``,
          ``<species_name>.uy_std_function(x,y,z)``,
          ``<species_name>.uz_std_function(x,y,z)``.

        Particles may be relativistic in the lab frame, but the sampling model treats them as
        non-relativistic in the drift frame. For a relativistic thermal spread, use ``maxwell_juttner`` instead.

    * ``maxwell_juttner``: Maxwell-Juttner distribution for relativistic plasma.
      More specifically, the plasma is initialized with a Maxwell-Juttner distribution

      .. math::

        p(\mathbf{u}) \propto \exp(-\gamma(\mathbf{u})mc^2/k_B T) = \exp(-\gamma (\mathbf{u})/\theta)

      (with :math:`\gamma(\mathbf{u}) = \sqrt{1+\mathbf{u}^2}` and :math:`\theta = k_B T/m c^2`) in a
      **drifting Lorentz frame** that is moving with a bulk velocity specified by the normalized momentum
      :math:`\boldsymbol{u}_{\rm mean} = (u_{x,{\rm mean}}, u_{y,{\rm mean}}, u_{z,{\rm mean}}) = \gamma \boldsymbol{v}/c` (see below).
      Thus, particles can potentially be relativistic in two ways: by having relativistic bulk drift :math:`\beta`
      in the lab frame, and/or by having high temperature :math:`\theta` in the drift frame.

      It requires the following arguments:

      * ``<species_name>.maxwell_juttner_u_mean_distribution_type`` (`string`, default ``constant``):
        Specifies the distribution type for the bulk (mean) particle momentum ``u_mean``.
        Here, ``u_mean`` is a 3D vector (with components ``ux_mean``, ``uy_mean``, ``uz_mean``)
        representing the normalized momentum, defined as
        :math:`\boldsymbol{u}_\mathrm{mean} = \gamma \boldsymbol{\beta}`, where
        :math:`\boldsymbol{\beta} = \boldsymbol{v}/c` and :math:`\gamma = 1/\sqrt{1-|\boldsymbol{\beta}|^2}`.
        The distribution is boosted from the drift frame to the simulation frame along the
        direction of :math:`\boldsymbol{u}_\mathrm{mean}`; the bulk velocity :math:`|\boldsymbol{\beta}|` derived from
        :math:`\boldsymbol{u}_\mathrm{mean}` is therefore always in the physical range :math:`|\boldsymbol{\beta}| < 1`.

        * If ``constant``, the following are required: ``<species_name>.ux_mean``,
          ``<species_name>.uy_mean``, ``<species_name>.uz_mean`` (`float`, default ``0``).
        * If ``parser``, the following are required:
          ``<species_name>.ux_mean_function(x,y,z)``,
          ``<species_name>.uy_mean_function(x,y,z)``,
          ``<species_name>.uz_mean_function(x,y,z)``.

      * ``<species_name>.theta_distribution_type`` (`string`, default ``constant``):
        Specifies the distribution type for the temperature :math:`\theta`.
        Values less than zero are not allowed.

        * If ``constant``, the following is required: ``<species_name>.theta`` (`float`).
        * If ``parser``, the following is required: ``<species_name>.theta_function(x,y,z)``.

      Sampling uses the Sobol and flipping methods described in :cite:t:`param-ZenitaniPOP2015`.
      For :math:`\theta \lesssim 0.1`, the Sobol method becomes inefficient (its acceptance
      efficiency tends to zero as :math:`\theta \rightarrow 0`) and, at the same time, the Maxwell-Juttner
      distribution becomes almost indistinguishable from a non-relativistic Maxwellian.
      Thus, for :math:`\theta < 0.1`, the code instead samples an isotropic Maxwellian with thermal spread
      :math:`\sqrt{\theta}` per component in the drift frame, then applies the same flipping
      method and Lorentz transform as for the Sobol-sampled momenta.

    * ``parse_momentum_function``: the momentum :math:`u = (u_{x},u_{y},u_{z})=(\gamma v_{x}/c,\gamma v_{y}/c,\gamma v_{z}/c)` is given by a function in the input
      file. It requires additional arguments ``<species_name>.momentum_function_ux(x,y,z)``,
      ``<species_name>.momentum_function_uy(x,y,z)`` and ``<species_name>.momentum_function_uz(x,y,z)``,
      which give the distribution of each component of the momentum as a function of space.

.. pp:param:: <species_name>.zinject_plane
    :type: ``float``

    Only read if  ``<species_name>`` is in :pp:param:`particles.rigid_injected_species`.
    Injection plane when using the rigid injection method.
    See :pp:param:`particles.rigid_injected_species` above.

.. pp:param:: <species_name>.rigid_advance
    :type: ``string`` or ``bool``
    :default: ``vzbar``

    Only read if ``<species_name>`` is in :pp:param:`particles.rigid_injected_species`.
    Until reaching ``zinject_plane``, each particle is rigidly advanced according to
    a specified velocity,

    * ``vz`` or ``false``: each particle's longitudinal velocity :math:`v_z`

    * ``vzbar`` or ``true``: the species' average longitudinal velocity :math:`\overline{v_z}`

    * ``v``: each particle's velocity :math:`{\bf v}`, including transverse components

.. pp:param:: <species_name>.do_backward_propagation
    :type: ``bool``

    Inject a backward-propagating beam to reduce the effect of charge-separation
    fields when running in the boosted frame. See examples.

.. pp:param:: <species_name>.split_type
    :type: ``int``
    :default: ``0``
    :optional:

    Splitting technique. When ``0``, particles are split along the simulation
    axes (4 particles in 2D, 6 particles in 3D). When ``1``, particles are split
    along the diagonals (4 particles in 2D, 8 particles in 3D).

.. pp:param:: <species_name>.do_not_deposit
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    If ``1`` is given, both charge deposition and current deposition will
    not be done, thus that species does not contribute to the fields.

.. pp:param:: <species_name>.do_not_gather
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    If ``1`` is given, field gather from grids will not be done,
    thus that species will not be affected by the field on grids.

.. pp:param:: <species_name>.do_not_push
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    If ``1`` is given, this species will not be pushed
    by any pusher during the simulation.

.. pp:param:: <species_name>.addIntegerAttributes
    :type: list of ``string``

    User-defined integer particle attribute for species, ``species_name``.
    These integer attributes will be initialized with user-defined functions
    when the particles are generated.
    If the user-defined integer attribute is ``<int_attrib_name>`` then the
    following required parameter must be specified to initialize the attribute.

    * ``<species_name>.attribute.<int_attrib_name>(x,y,z,ux,uy,uz,t)`` (``string``)
        ``t`` represents the physical time in seconds during the simulation.
        ``x``, ``y``, ``z`` represent particle positions in the unit of meter.
        ``ux``, ``uy``, ``uz`` represent the particle momenta in the unit of
        :math:`\gamma v/c`, where
        :math:`\gamma` is the Lorentz factor,
        :math:`v/c` is the particle velocity normalized by the speed of light.
        E.g. If ``electrons.addIntegerAttributes = upstream``
        and ``electrons.upstream(x,y,z,ux,uy,uz,t) = (x>0.0)*1`` is provided
        then, an integer attribute ``upstream`` is added to all electron particles
        and when these particles are generated, the particles with position less than ``0``
        are assigned a value of ``1``.

.. pp:param:: <species_name>.addRealAttributes
    :type: list of ``string``

    User-defined real particle attribute for species, ``species_name``.
    These real attributes will be initialized with user-defined functions
    when the particles are generated.
    If the user-defined real attribute is ``<real_attrib_name>`` then the
    following required parameter must be specified to initialize the attribute.

    * ``<species_name>.attribute.<real_attrib_name>(x,y,z,ux,uy,uz,t)`` (``string``)
        ``t`` represents the physical time in seconds during the simulation.
        ``x``, ``y``, ``z`` represent particle positions in the unit of meter.
        ``ux``, ``uy``, ``uz`` represent the particle momenta in the unit of
        :math:`\gamma v/c`, where
        :math:`\gamma` is the Lorentz factor,
        :math:`v/c` is the particle velocity normalized by the speed of light.

.. pp:param:: <species_name>.save_particles_at_xlo/ylo/zlo/xhi/yhi/zhi/eb
    :link_aliases:
        <species_name>.save_particles_at_xlo/ylo/zlo
        <species_name>.save_particles_at_xhi/yhi/zhi
        <species_name>.save_particles_at_xlo
        <species_name>.save_particles_at_ylo
        <species_name>.save_particles_at_zlo
        <species_name>.save_particles_at_xhi
        <species_name>.save_particles_at_yhi
        <species_name>.save_particles_at_zhi
        <species_name>.save_particles_at_eb
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    If ``1`` particles of this species will be copied to the scraped particle
    buffer for the specified boundary if they leave the simulation domain in
    the specified direction. **If USE_EB=TRUE** the ``save_particles_at_eb``
    flag can be set to ``1`` to also save particle data for the particles of this
    species that are absorbed at the embedded boundary.
    The scraped particle buffer can be used to track particle fluxes out of the
    simulation.
    The particle data can be written out by setting up a ``BoundaryScrapingDiagnostic``.
    It is also accessible via the Python interface. The
    function ``get_particle_boundary_buffer``, found in the
    ``picmi.Simulation`` class as
    ``sim.extension.get_particle_boundary_buffer()``, can be
    used to access the scraped particle buffer. An entry is included for every
    particle in the buffer of the timestep at which the particle was scraped.
    This can be accessed by passing the argument ``comp_name="stepScraped"`` to
    the above mentioned function.

    .. note::

       When accessing the data via Python, the scraped particle buffer relies on the user
       to clear the buffer after processing the data. The
       buffer will grow unbounded as particles are scraped and therefore could
       lead to memory issues if not periodically cleared. To clear the buffer
       call ``clear_buffer()``.

.. pp:param:: <species_name>.do_field_ionization
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Do field ionization for this species (using the ADK theory).

.. pp:param:: <species_name>.do_adk_correction
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Whether to apply the correction to the ADK theory proposed by Zhang, Lan and Lu in `Q. Zhang et al. (Phys. Rev. A 90, 043410, 2014) <https://doi.org/10.1103/PhysRevA.90.043410>`__.
    If so, the probability of ionization is modified using an empirical model that should be more accurate in the regime of high electric fields.
    Currently, this is only implemented for Hydrogen, although Argon is also available in the same reference.

.. pp:param:: <species_name>.physical_element
    :type: ``string``

    Only read if ``do_field_ionization = 1``. Symbol of chemical element for
    this species. Example: for Helium, use ``physical_element = He``.
    All the elements up to atomic number Z=100 (Fermium) are supported.

.. pp:param:: <species_name>.ionization_product_species
    :type: ``string``

    Only read if ``do_field_ionization = 1``. Name of species in which ionized
    electrons are stored. This species must be created as a regular species
    in the input file (in particular, it must be in :pp:param:`particles.species_names`).

.. pp:param:: <species_name>.ionization_initial_level
    :type: ``int``
    :default: ``0``
    :optional:

    Only read if ``do_field_ionization = 1``. Initial ionization level of the
    species (must be smaller than the atomic number of chemical element given
    in ``physical_element``).

.. pp:param:: <species_name>.do_resampling
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    If ``1`` resampling is performed for this species. This means that the number of macroparticles
    will be reduced at specific timesteps while preserving the distribution function as much as
    possible (details depend on the chosen resampling algorithm).
    This can be useful in situations with continuous creation of particles (e.g. with ionization
    or with QED effects). At least one resampling trigger (see below) must be specified to actually
    perform resampling.

.. pp:param:: <species_name>.resampling_algorithm
    :type: ``string``
    :default: ``leveling_thinning``
    :optional:

    The algorithm used for resampling:

    * ``leveling_thinning`` This algorithm is defined in :cite:t:`param-MuravievCPC2021`.
      It has one parameter:

        * ``<species_name>.resampling_algorithm_target_ratio`` (``float``) optional (default ``1.5``)
            This **roughly** corresponds to the ratio between the number of particles before and
            after resampling.

    * ``velocity_coincidence_thinning``` The particles are sorted into phase space
      cells and merged, similar to the approach described in :cite:t:`param-Vranic2015`.
      It has three parameters:

        * ``<species_name>.resampling_algorithm_delta_ur`` (``float``)
            The width of momentum cells used in clustering particles, in m/s.

        * ``<species_name>.resampling_algorithm_n_theta`` (``int``)
            The number of cell divisions to use in the :math:`\theta` direction
            when clustering the particle velocities.

        * ``<species_name>.resampling_algorithm_n_phi`` (``int``)
            The number of cell divisions to use in the :math:`\phi` direction
            when clustering the particle velocities.

.. pp:param:: <species_name>.resampling_min_ppc
    :type: ``int``
    :default: ``1``
    :optional:

    Resampling is not performed in cells with a number of macroparticles strictly smaller
    than this parameter.

.. pp:param:: <species_name>.resampling_trigger_intervals
    :type: ``string``
    :default: ``0``
    :optional:

    Using the `Time intervals`_ syntax, this string defines timesteps at which resampling is
    performed.

.. pp:param:: <species_name>.resampling_trigger_max_avg_ppc
    :type: ``float``
    :default: ``infinity``
    :optional:

    Resampling is performed every time the number of macroparticles per cell of the species
    averaged over the whole simulation domain exceeds this parameter.

.. pp:param:: <species_name>.do_temperature_deposition
    :type: ``boolean``
    :default: ``false``
    :optional:

    When running with Ohm's Law Hybrid Solver, this will enable temperature deposition
    in each dimension with a matched shape function and filtering used for current deposition.
    This is required when using the electron energy solver with electron-ion temperature relaxation.

.. pp:param:: <species>.do_qed_virtual_photons
    :type: ``boolean``
    :default: ``false``
    :optional:

    Create a population of virtual photons associated with ``<species>``.
    It only works if ``<species>`` is an electron or a positron species.
    The virtual photon species has to be created as a regular photon species in the input file.
    Virtual photons are created from scratch at each timestep at the same position as the parent particle.
    This implies that different primary species must have different virtual photon species.
    The energy of the virtual photons is sampled from their spectrum (see :cite:t:`param-LandauVol4` section 99 for more details).
    The momentum of the virtual photons is parallel to that of the parent particle.
    This feature also requires the following input parameters:

      * ``<species>.qed_virtual_photon_species_name`` (``string``) name of the virtual photon species associated with the current lepton species.

      * ``<virtual_photon_species>.qed_virtual_photons_min_energy`` (``float``, in Joules) minimum energy of the virtual photons

      * ``<virtual_photon_species>.qed_virtual_photons_multiplier`` (``int``), sampling factor for the virtual photons.
        A sampling factor of ``f`` means that the number of virtual photons is multiplied by ``f``, while their weights are divided by ``f``.

    The virtual photons can undergo collisions via the linear Breit-Wheeler or linear Compton processes.
    This is useful to model incoherent beam-beam effects in colliders (e.g. pair generation, radiative Bhabha scattering).
    This QED feature is separated from the strong-field QED modules (quantum synchrotron and non-linear Breit-Wheeler).
    It requires WarpX to be compiled with ``WarpX_QED=ON`` (CMake) or ``QED=TRUE`` (GNU Make).

.. pp:param:: <species>.qed_virtual_photons_do_beam_size_effect
    :type: ``boolean``
    :default: ``false``
    :optional:

    Applies the beam size effect on the virtual photons.
    This effect reduces the radiative Bhabha scattering cross section by approximately half, by smearing the impact parameter of the virtual photons on a disc around the equivalent primary. This accounts for the finite transverse size of the colliding bunches. Otherwise all virtual photons are assumed at the same impact parameter. The (transverse) virtual photon coordinates will be randomized around the coordinate of the corresponding primary and distributed on a disc perpendicular to the primary's propagation direction. The radius of the disc is :math:`\rho=\frac{\hbar}{\sqrt{Q^2(1-x)}}`, where :math:`Q` is the photon virtuality and :math:`x` is the fractional photon energy.
    See :cite:t:`param-Kicsiny2024` for more details.


.. _running-cpp-parameters-fluids:

Cold Relativistic Fluid initialization
--------------------------------------

.. pp:param:: fluids.species_names
    :type: ``strings``, separated by spaces

    Defines the names of each fluid species. It is a required input to create and evolve fluid species using the cold relativistic fluid equations.
    Most of the parameters described in the section "Particle initialization" can also be used to initialize fluid properties (e.g. initial density distribution).
    For fluid-specific inputs we use ``<fluid_species_name>`` as a placeholder. Also see external fields
    for how to specify these for fluids as the function names differ.

.. _running-cpp-parameters-laser:

Laser initialization
--------------------

.. pp:param:: lasers.names
    :type: list of ``string``

    Name of each laser. This is then used in the rest of the input deck ;
    in this documentation we use ``<laser_name>`` as a placeholder. The parameters below
    must be provided for each laser pulse.

.. pp:param:: <laser_name>.position
    :type: ``3 floats in 3D and 2D``
    :unit: meters

    The coordinates of one of the point of the antenna that will emit the laser.
    The plane of the antenna is entirely defined by :pp:param:`<laser_name>.position`
    and :pp:param:`<laser_name>.direction`.

    :pp:param:`<laser_name>.position` also corresponds to the origin of the coordinates system
    for the laser transverse profile. For instance, for a Gaussian laser profile,
    the peak of intensity will be at the position given by :pp:param:`<laser_name>.position`.
    This variable can thus be used to shift the position of the laser pulse
    transversally.

    .. note::
        In 2D, :pp:param:`<laser_name>.position` is still given by 3 numbers,
        but the second number is ignored.

    When running a **boosted-frame simulation**, provide the value of
    :pp:param:`<laser_name>.position` in the laboratory frame, and use :pp:param:`warpx.gamma_boost`
    to automatically perform the conversion to the boosted frame. Note that,
    in this case, the laser antenna will be moving, in the boosted frame.

.. pp:param:: <laser_name>.polarization
    :type: ``3 floats in 3D and 2D``

    The coordinates of a vector that points in the direction of polarization of
    the laser. The norm of this vector is unimportant, only its direction matters.

    .. note::
        Even in 2D, all the 3 components of this vectors are important (i.e.
        the polarization can be orthogonal to the plane of the simulation).

.. pp:param:: <laser_name>.direction
    :type: ``3 floats in 3D``

    The coordinates of a vector that points in the propagation direction of
    the laser. The norm of this vector is unimportant, only its direction matters.

    The plane of the antenna that will emit the laser is orthogonal to this vector.

    .. warning::

        When running **boosted-frame simulations**, :pp:param:`<laser_name>.direction` should
        be parallel to :pp:param:`warpx.boost_direction`, for now.

.. pp:param:: <laser_name>.e_max
    :type: ``float``
    :unit: V/m

    Peak amplitude of the laser field, in the focal plane.

    For a laser with a wavelength :math:`\lambda = 0.8\,\mu m`, the peak amplitude
    is related to :math:`a_0` by:

    .. math::

        E_{max} = a_0 \frac{2 \pi m_e c^2}{e\lambda} = a_0 \times (4.0 \cdot 10^{12} \;V.m^{-1})

    When running a **boosted-frame simulation**, provide the value of :pp:param:`<laser_name>.e_max`
    in the laboratory frame, and use :pp:param:`warpx.gamma_boost` to automatically
    perform the conversion to the boosted frame.

.. pp:param:: <laser_name>.a0
    :type: ``float``
    :comment: dimensionless

    Peak normalized amplitude of the laser field, in the focal plane (given in the lab frame, just as ``e_max`` above).
    See the description of :pp:param:`<laser_name>.e_max` for the conversion between ``a0`` and ``e_max``.
    Either ``a0`` or ``e_max`` must be specified.

.. pp:param:: <laser_name>.wavelength
    :type: ``float``
    :unit: meters

    The wavelength of the laser in vacuum.

    When running a **boosted-frame simulation**, provide the value of
    :pp:param:`<laser_name>.wavelength` in the laboratory frame, and use :pp:param:`warpx.gamma_boost`
    to automatically perform the conversion to the boosted frame.

.. pp:param:: <laser_name>.profile
    :type: ``string``

    The spatio-temporal shape of the laser. The options that are currently
    implemented are:

    - ``"Gaussian"``: The transverse and longitudinal profiles are Gaussian.
    - ``"parse_field_function"``: the laser electric field is given by a function in the
      input file. It requires additional argument ``<laser_name>.field_function(X,Y,t)``, which
      is a mathematical expression , e.g.
      ``<laser_name>.field_function(X,Y,t) = "a0*X**2 * (X>0) * cos(omega0*t)"`` where
      ``a0`` and ``omega0`` are a user-defined constant, see above. The profile passed
      here is the full profile, not only the laser envelope. ``t`` is time and ``X``
      and ``Y`` are coordinates orthogonal to :pp:param:`<laser_name>.direction` (not necessarily the
      x and y coordinates of the simulation). All parameters above are required, but
      none of the parameters below are used when ``<laser_name>.parse_field_function=1``. Even
      though :pp:param:`<laser_name>.wavelength` and :pp:param:`<laser_name>.e_max` should be included in the laser
      function, they still have to be specified as they are used for numerical purposes.
    - ``"from_file"``: the electric field of the laser is read from an external file. Currently both
      the `lasy <https://lasydoc.readthedocs.io/en/latest/>`_ format as well as a custom binary format are supported. It requires to provide
      the name of the file to load setting the additional parameter ``<laser_name>.binary_file_name`` or ``<laser_name>.lasy_file_name`` (``string``).
      It accepts an optional parameter ``<laser_name>.time_chunk_size`` (``int``), supported for both lasy and binary files;
      this allows to read only time_chunk_size timesteps from the file. New timesteps are read as soon as they are needed.

      The default value is automatically set to the number of timesteps contained in the file
      (i.e. only one read is performed at the beginning of the simulation).
      It also accepts the optional parameter ``<laser_name>.delay`` (``float``; in seconds), which allows
      delaying (``delay > 0``) or anticipating (``delay < 0``) the laser by the specified amount of time.

      Details about the usage of the lasy format: lasy can produce either 3D Cartesian files or RZ files.
      WarpX can read both types of files independently of the geometry in which it was compiled (e.g. WarpX
      compiled with ``WarpX_DIMS=RZ`` can read 3D Cartesian lasy files). In the case where WarpX is compiled
      in 2D (or 1D) Cartesian, the laser antenna will emit the field values that correspond to the slice ``y=0``
      in the lasy file (and ``x=0`` in the 1D case). One can generate a lasy file from Python, see an example
      at ``Examples/Tests/laser_injection_from_file``.

      Details about the usage of the binary format: The external binary file should provide E(x,y,t) on a rectangular (necessarily uniform)
      grid. The code performs a bi-linear (in 2D) or tri-linear (in 3D) interpolation to set the field
      values. x,y,t are meant to be in S.I. units, while the field value is meant to be multiplied by
      :pp:param:`<laser_name>.e_max` (i.e. in most cases the maximum of abs(E(x,y,t)) should be 1,
      so that the maximum field intensity can be set straightforwardly with :pp:param:`<laser_name>.e_max`).
      The binary file has to respect the following format:

      * ``flag`` to indicate the grid is uniform (1 byte, 0 means non-uniform, !=0 means uniform) - only uniform is supported
      * ``nt``, number of timesteps (``uint32_t``, must be >=2)
      * ``nx``, number of points along x (``uint32_t``, must be >=2)
      * ``ny``, number of points along y (``uint32_t``, must be 1 for 2D simulations and >=2 for 3D simulations)
      * ``timesteps`` (``double[2]=[t_min,t_max]``)
      * ``x_coords`` (``double[2]=[x_min,x_max]``)
      * ``y_coords`` (``double[1]`` in 2D, ``double[2]=[y_min,y_max]`` in 3D)
      * ``field_data`` (``double[nt x nx * ny]``, with ``nt`` being the slowest coordinate).

      A binary file can be generated from Python, see an example at ``Examples/Tests/laser_injection_from_file``

.. pp:param:: <laser_name>.profile_t_peak
    :type: ``float``
    :unit: seconds

    The time at which the laser reaches its peak intensity, at the position
    given by :pp:param:`<laser_name>.position` (only used for the ``"gaussian"`` profile)

    When running a **boosted-frame simulation**, provide the value of
    :pp:param:`<laser_name>.profile_t_peak` in the laboratory frame, and use :pp:param:`warpx.gamma_boost`
    to automatically perform the conversion to the boosted frame.

.. pp:param:: <laser_name>.profile_duration
    :type: ``float``
    :unit: seconds

    The duration of the laser pulse for the ``"gaussian"`` profile, defined as :math:`\tau` below:

    .. math::

        E(\boldsymbol{x},t) \propto \exp\left( -\frac{(t-t_{peak})^2}{\tau^2} \right)

    Note that :math:`\tau` relates to the full width at half maximum (FWHM) of *intensity*, which is closer to pulse length measurements in experiments, as :math:`\tau = \mathrm{FWHM}_I / \sqrt{2\ln(2)}` :math:`\approx \mathrm{FWHM}_I / 1.1774`.

    For a chirped laser pulse (i.e. with a non-zero :pp:param:`<laser_name>.phi2`), ``profile_duration`` is the Fourier-limited duration of the pulse, not the actual duration of the pulse. See the documentation for :pp:param:`<laser_name>.phi2` for more detail.

    When running a **boosted-frame simulation**, provide the value of
    :pp:param:`<laser_name>.profile_duration` in the laboratory frame, and use :pp:param:`warpx.gamma_boost`
    to automatically perform the conversion to the boosted frame.

.. pp:param:: <laser_name>.profile_waist
    :type: ``float``
    :unit: meters

    The waist of the transverse Gaussian :math:`w_0`, i.e. defined such that the electric field of the
    laser pulse in the focal plane is of the form:

    .. math::

        E(\boldsymbol{x},t) \propto \exp\left( -\frac{\boldsymbol{x}_\perp^2}{w_0^2} \right)

.. pp:param:: <laser_name>.profile_focal_distance
    :type: ``float``
    :unit: meters

    The distance from ``laser_position`` to the focal plane.
    (where the distance is defined along the direction given by :pp:param:`<laser_name>.direction`.)

    Use a negative number for a defocusing laser instead of a focusing laser.

    When running a **boosted-frame simulation**, provide the value of
    :pp:param:`<laser_name>.profile_focal_distance` in the laboratory frame, and use :pp:param:`warpx.gamma_boost`
    to automatically perform the conversion to the boosted frame.

.. pp:param:: <laser_name>.phi0
    :type: ``float``
    :unit: radians
    :default: ``0.``
    :optional:

    The Carrier Envelope Phase, i.e. the phase of the laser oscillation, at the
    position where the laser envelope is maximum (only used for the ``"gaussian"`` profile)

.. pp:param:: <laser_name>.stc_direction
    :type: ``3 floats``
    :default: ``1. 0. 0.``
    :optional:

    Direction of laser spatio-temporal couplings.
    See definition in :cite:t:`param-AkturkOE2004`.

.. pp:param:: <laser_name>.zeta
    :type: ``float``
    :unit: meters.seconds
    :default: ``0.``
    :optional:

    Spatial chirp at focus in direction :pp:param:`<laser_name>.stc_direction`. See definition in
    :cite:t:`param-AkturkOE2004`.

.. pp:param:: <laser_name>.beta
    :type: ``float``
    :unit: seconds
    :default: ``0.``
    :optional:

    Angular dispersion (or angular chirp) at focus in direction :pp:param:`<laser_name>.stc_direction`.
    See definition in :cite:t:`param-AkturkOE2004`.

.. pp:param:: <laser_name>.phi2
    :type: ``float``
    :unit: seconds**2
    :default: ``0.``
    :optional:

    The amount of temporal chirp :math:`\phi^{(2)}` at focus (in the lab frame). Namely, a wave packet
    centered on the frequency :math:`(\omega_0 + \delta \omega)` will reach its peak intensity
    at :math:`z(\delta \omega) = z_0 - c \phi^{(2)} \, \delta \omega`. Thus, a positive
    :math:`\phi^{(2)}` corresponds to positive chirp, i.e. red part of the spectrum in the
    front of the pulse and blue part of the spectrum in the back. More specifically, the electric
    field in the focal plane is of the form:

    .. math::

        E(\boldsymbol{x},t) \propto Re\left[ \exp\left(  -\frac{(t-t_{peak})^2}{\tau^2 + 2i\phi^{(2)}} + i\omega_0 (t-t_{peak}) + i\phi_0 \right) \right]

    where :math:`\tau` is given by :pp:param:`<laser_name>.profile_duration` and represents the
    Fourier-limited duration of the laser pulse. Thus, the actual duration of the chirped laser pulse is:

    .. math::

        \tau' = \sqrt{ \tau^2 + 4 (\phi^{(2)})^2/\tau^2 }

    See also the definition in :cite:t:`param-AkturkOE2004`.

.. pp:param:: <laser_name>.do_continuous_injection
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Whether or not to use continuous injection.
    If the antenna starts outside of the simulation domain but enters it
    at some point (due to moving window or moving antenna in the boosted
    frame), use this so that the laser antenna is injected when it reaches
    the box boundary. If running in a boosted frame, this requires the
    boost direction, moving window direction and laser propagation direction
    to be along ``z``. If not running in a boosted frame, this requires the
    moving window and laser propagation directions to be the same (``x``, ``y``
    or ``z``)

.. pp:param:: <laser_name>.min_particles_per_mode
    :type: ``int``
    :default: ``4``
    :optional:

    When using the RZ version, this specifies the minimum number of particles
    per angular mode. The laser particles are loaded into radial spokes, with
    the number of spokes given by min_particles_per_mode*(warpx.n_rz_azimuthal_modes-1).

.. pp:param:: lasers.deposit_on_main_grid
    :type: ``int``
    :default: ``0``
    :optional:

    When using mesh refinement, whether the antenna that emits the laser
    deposits charge/current only on the main grid (i.e. level 0), or also
    on the higher mesh-refinement levels.

.. pp:param:: warpx.num_mirrors
    :type: ``int``
    :default: ``0``
    :optional:

    Users can input perfect mirror condition inside the simulation domain.
    The number of mirrors is given by :pp:param:`warpx.num_mirrors`. The mirrors are
    orthogonal to the ``z`` direction. The following parameters are required
    when :pp:param:`warpx.num_mirrors` is >0.

.. pp:param:: warpx.mirror_z
    :type: list of ``float``
    :comment: required if :pp:param:`warpx.num_mirrors > 0`

    ``z`` location of the front of the mirrors.

.. pp:param:: warpx.mirror_z_width
    :type: list of ``float``
    :comment: required if :pp:param:`warpx.num_mirrors > 0`

    ``z`` width of the mirrors.

.. pp:param:: warpx.mirror_z_npoints
    :type: list of ``int``
    :comment: required if :pp:param:`warpx.num_mirrors > 0`

    In the boosted frame, depending on ``gamma_boost``, :pp:param:`warpx.mirror_z_width`
    can be smaller than the cell size, so that the mirror would not work. This
    parameter is the minimum number of points for the mirror. If
    ``mirror_z_width < dz/cell_size``, the upper bound of the mirror is increased
    so that it contains at least ``mirror_z_npoints``.

External fields
---------------

Applied to the grid
^^^^^^^^^^^^^^^^^^^

The external fields defined with input parameters that start with ``warpx.B_ext_grid_init_`` or ``warpx.E_ext_grid_init_``
are applied to the grid directly. In particular, these fields can be seen in the diagnostics that output the fields on the grid.

    - When using an **electromagnetic** field solver, these fields are applied to the grid at the beginning of the simulation, and serve as initial condition for the Maxwell solver.
    - When using an **electrostatic** or **magnetostatic** field solver, these fields are added to the fields computed by the Poisson solver, at each timestep.

.. pp:param:: warpx.B_ext_grid_init_style
    :type: string
    :optional:

    This parameter determines the type of initialization for the external
    magnetic field. By default, the
    external magnetic field (Bx,By,Bz) is initialized to (0.0, 0.0, 0.0).
    The string can be set to "constant" if a constant magnetic field is
    required to be set at initialization. If set to "constant", then an
    additional parameter, namely, :pp:param:`warpx.B_external_grid` must be specified.
    If set to ``parse_B_ext_grid_function``, then a mathematical expression can
    be used to initialize the external magnetic field on the grid. It
    requires additional parameters in the input file, namely,
    ``warpx.Bx_external_grid_function(x,y,z)``,
    ``warpx.By_external_grid_function(x,y,z)``,
    ``warpx.Bz_external_grid_function(x,y,z)`` to initialize the external
    magnetic field for each of the three components on the grid.
    Constants required in the expression can be set using ``my_constants``.
    For example, if ``warpx.Bx_external_grid_function(x,y,z)=Bo*x + delta*(y + z)``
    then the constants ``Bo`` and ``delta`` required in the above equation
    can be set using ``my_constants.Bo=`` and ``my_constants.delta=`` in the
    input file. For a two-dimensional simulation, it is assumed that the first dimension
    is ``x`` and the second dimension is ``z``, and the value of ``y`` is set to zero.
    Note that the current implementation of the parser for external B-field
    does not work with RZ and the code will abort with an error message.

    If ``B_ext_grid_init_style`` is set to be ``read_from_file``, an additional parameter,
    indicating the path of an openPMD data file,
    ``warpx.read_fields_from_path`` must be specified,
    from which external B field data can be loaded into WarpX.
    One can refer to input files in ``Examples/Tests/LoadExternalField`` for more information.
    Regarding how to prepare the openPMD data file, one can refer to
    the `openPMD-example-datasets <https://github.com/openPMD/openPMD-example-datasets>`__.

.. pp:param:: warpx.E_ext_grid_init_style
    :type: string
    :optional:

    This parameter determines the type of initialization for the external
    electric field. By default, the
    external electric field (Ex,Ey,Ez) to (0.0, 0.0, 0.0).
    The string can be set to "constant" if a constant electric field is
    required to be set at initialization. If set to "constant", then an
    additional parameter, namely, :pp:param:`warpx.E_external_grid` must be specified
    in the input file.
    If set to ``parse_E_ext_grid_function``, then a mathematical expression can
    be used to initialize the external electric field on the grid. It
    required additional parameters in the input file, namely,
    ``warpx.Ex_external_grid_function(x,y,z)``,
    ``warpx.Ey_external_grid_function(x,y,z)``,
    ``warpx.Ez_external_grid_function(x,y,z)`` to initialize the external
    electric field for each of the three components on the grid.
    Constants required in the expression can be set using ``my_constants``.
    For example, if ``warpx.Ex_external_grid_function(x,y,z)=Eo*x + delta*(y + z)``
    then the constants ``Bo`` and ``delta`` required in the above equation
    can be set using ``my_constants.Eo=`` and ``my_constants.delta=`` in the
    input file. For a two-dimensional simulation, it is assumed that the first
    dimension is ``x`` and the second dimension is ``z``,
    and the value of ``y`` is set to zero.
    Note that the current implementation of the parser for external E-field
    does not work with RZ and the code will abort with an error message.

    If ``E_ext_grid_init_style`` is set to be ``read_from_file``, an additional parameter,
    indicating the path of an openPMD data file,
    ``warpx.read_fields_from_path`` must be specified,
    from which external E field data can be loaded into WarpX.
    One can refer to input files in ``Examples/Tests/LoadExternalField`` for more information.
    Regarding how to prepare the openPMD data file, one can refer to
    the `openPMD-example-datasets <https://github.com/openPMD/openPMD-example-datasets>`__.
    Note that if both ``B_ext_grid_init_style`` and ``E_ext_grid_init_style`` are set to
    ``read_from_file``, the openPMD file specified by ``warpx.read_fields_from_path``
    should contain both B and E external fields data.

.. pp:param:: warpx.E/B_external_grid
    :link_aliases:
        warpx.E_external_grid
        warpx.B_external_grid
    :type: list of ``3 floats``

    required when :pp:param:`warpx.E_ext_grid_init_style = "constant"`
    and when :pp:param:`warpx.B_ext_grid_init_style = "constant"`, respectively.
    External uniform and constant electrostatic and magnetostatic field added
    to the grid at initialization. Use with caution as these fields are used for
    the field solver. In particular, do not use any other boundary condition
    than periodic.

.. pp:param:: warpx.maxlevel_extEMfield_init
    :default: maximum number of levels in the simulation

    With this parameter, the externally applied electric and magnetic fields
    will not be applied for levels greater than :pp:param:`warpx.maxlevel_extEMfield_init`.
    For some mesh-refinement simulations,
    the external fields are only applied to the parent grid and not the refined patches. In such cases,
    :pp:param:`warpx.maxlevel_extEMfield_init` can be set to 0.
    In that case, the other levels have external field values of 0.

Applied to Particles
^^^^^^^^^^^^^^^^^^^^

The external fields defined with input parameters that start with ``warpx.B_ext_particle_init_`` or ``warpx.E_ext_particle_init_``
are applied to the particles directly, at each timestep. As a results, these fields **cannot** be seen in the diagnostics that output the fields on the grid.

.. pp:param:: particles.E/B_ext_particle_init_style
    :link_aliases:
        particles.E_ext_particle_init_style
        particles.B_ext_particle_init_style
    :type: string
    :default: "none"
    :optional:

    These parameters determine the type of the external electric and
    magnetic fields respectively that are applied directly to the particles at every timestep.
    The field values are specified in the lab frame.
    With the default ``none`` style, no field is applied.
    Possible values are ``constant``, ``parse_E_ext_particle_function`` or ``parse_B_ext_particle_function``, or
    ``repeated_plasma_lens``.

    * ``constant``: a constant field is applied, given by the input parameters
      ``particles.E_external_particle`` or ``particles.B_external_particle``, which are lists of the field components.

    * ``parse_E_ext_particle_function`` or ``parse_B_ext_particle_function``: the field is specified as an analytic
      expression that is a function of space (x,y,z) and time (t), relative to the lab frame.
      The E-field is specified by the input parameters:

        * ``particles.Ex_external_particle_function(x,y,z,t)``

        * ``particles.Ey_external_particle_function(x,y,z,t)``

        * ``particles.Ez_external_particle_function(x,y,z,t)``

      The B-field is specified by the input parameters:

        * ``particles.Bx_external_particle_function(x,y,z,t)``

        * ``particles.By_external_particle_function(x,y,z,t)``

        * ``particles.Bz_external_particle_function(x,y,z,t)``

      Note that the position is defined in Cartesian coordinates, as a function of (x,y,z), even for RZ, RCYLINDER, and RSPHERE.

    * ``read_from_file``: load external fields from openPMD files.

        There are two ways to specify external field data: **single-field mode**
        and **multi-field mode**.

        **Single-field mode**

        In this mode, a single external E and/or B field is loaded from the path
        given by ``particles.read_fields_from_path``. This parameter must always
        be provided when using ``read_from_file``.

        The time dependency of the E- and B-field can be specified by the input parameters:

        * ``particles.read_fields_E_dependency(t)``

        * ``particles.read_fields_B_dependency(t)``

        The time dependency scales the corresponding field uniformly in space
        and per level by the given function of time ``t`` (in seconds).

        Example:

        .. code-block:: none

            particles.E_ext_particle_init_style = read_from_file
            particles.B_ext_particle_init_style = read_from_file
            particles.read_fields_from_path = diags/field_input
            particles.read_fields_E_dependency(t) = cos(2*pi*2e6*t)
            particles.read_fields_B_dependency(t) = cos(2*pi*2e6*t + pi/2)

        If both ``B_ext_particle_init_style`` and ``E_ext_particle_init_style`` are set to
        ``read_from_file``, the same openPMD file specified by
        ``particles.read_fields_from_path`` should contain both E and B field data.

        **Multi-field mode**

        In this mode, several field maps can be loaded independently. Each field
        is given a unique name listed in

        * ``particles.E_ext_particle_fields``  (for electric fields)
        * ``particles.B_ext_particle_fields``  (for magnetic fields)

        Each named field must define its own path and may optionally define a
        time dependency. The general key ``particles.read_fields_from_path`` is
        ignored when these lists are provided.

        Example:

        .. code-block:: none

            particles.B_ext_particle_init_style = read_from_file
            particles.B_ext_particle_fields = b1 b2
            particles.b1.read_fields_from_path = diags/Bfield_map1
            particles.b1.read_fields_B_dependency(t) = cos(omega*t + phase)
            particles.b2.read_fields_from_path = diags/Bfield_map2
            particles.b2.read_fields_B_dependency(t) = cos(2*omega*t + phase)

        Each field's scaling function is evaluated independently and may contain
        user-defined constants. The expressions are parsed on the C++ side.

        .. note::

            When using ``read_from_file``, the fields loaded from file are interpolated
            to the resolution of the grid used for the simulation. These interpolated
            fields are visible to diagnostics.

        To prepare openPMD-compatible field data files, see the
        `openPMD-example-datasets <https://github.com/openPMD/openPMD-example-datasets>`__.


    * ``repeated_plasma_lens``: apply a series of plasma lenses.
      The properties of the lenses are defined in the lab frame by the input parameters:

        * ``repeated_plasma_lens_period``, the period length of the repeat, a single float number,

        * ``repeated_plasma_lens_starts``, the start of each lens relative to the period, an array of floats,

        * ``repeated_plasma_lens_lengths``, the length of each lens, an array of floats,

        * ``repeated_plasma_lens_strengths_E``, the electric focusing strength of each lens, an array of floats, when
          :pp:param:`particles.E_ext_particle_init_style` is set to ``repeated_plasma_lens``.

        * ``repeated_plasma_lens_strengths_B``, the magnetic focusing strength of each lens, an array of floats, when
          :pp:param:`particles.B_ext_particle_init_style` is set to ``repeated_plasma_lens``.

      The repeated lenses are only defined for :math:`z > 0`.
      Once the number of lenses specified in the input are exceeded, the repeated lens stops.

      The applied field is uniform longitudinally (along z) with a hard edge,
      where residence corrections are used for more accurate field calculation. On the time step when a particle enters
      or leaves each lens, the field applied is scaled by the fraction of the time step spent within the lens.
      The fields are of the form :math:`E_x = \mathrm{strength} \cdot x`, :math:`E_y = \mathrm{strength} \cdot y`,
      and :math:`E_z = 0`, and
      :math:`B_x = \mathrm{strength} \cdot y`, :math:`B_y = -\mathrm{strength} \cdot x`, and :math:`B_z = 0`.


Applied to Cold Relativistic Fluids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The external fields defined with input parameters that start with ``warpx.B_ext_init_`` or ``warpx.E_ext_init_``
are applied to the fluids directly, at each timestep. As a results, these fields **cannot** be seen in the diagnostics that output the fields on the grid.

.. pp:param:: <fluid_species_name>.E/B_ext_init_style
    :link_aliases:
        <fluid_species_name>.E_ext_init_style
        <fluid_species_name>.B_ext_init_style
    :type: string
    :default: "none"
    :optional:

    These parameters determine the type of the external electric and
    magnetic fields respectively that are applied directly to the cold relativistic fluids at every timestep.
    The field values are specified in the lab frame.
    With the default ``none`` style, no field is applied.
    Possible values are ``parse_E_ext_function`` or ``parse_B_ext_function``.

    * ``parse_E_ext_function`` or ``parse_B_ext_function``: the field is specified as an analytic
      expression that is a function of space (x,y,z) and time (t), relative to the lab frame.
      The E-field is specified by the input parameters:

        * ``<fluid_species_name>.Ex_external_function(x,y,z,t)``

        * ``<fluid_species_name>.Ey_external_function(x,y,z,t)``

        * ``<fluid_species_name>.Ez_external_function(x,y,z,t)``

      The B-field is specified by the input parameters:

        * ``<fluid_species_name>.Bx_external_function(x,y,z,t)``

        * ``<fluid_species_name>.By_external_function(x,y,z,t)``

        * ``<fluid_species_name>.Bz_external_function(x,y,z,t)``

      Note that the position is defined in Cartesian coordinates, as a function of (x,y,z), even for RZ, RCYLINDER, and RSPHERE.

Accelerator Lattice
^^^^^^^^^^^^^^^^^^^

Several accelerator lattice elements can be defined as described below.
The elements are defined relative to the ``z`` axis and in the lab frame, starting at ``z = 0``.
They are described using a simplified MAD like syntax.
Note that elements of the same type cannot overlap each other.

.. pp:param:: lattice.elements
    :type: ``list of strings``
    :default: no elements
    :optional:

    A list of names (one name per lattice element), in the order that they
    appear in the lattice.

.. pp:param:: lattice.reverse
    :type: ``boolean``
    :default: ``false``
    :optional:

    Reverse the list of elements in the lattice.

.. pp:param:: <element_name>.type
    :type: ``string``

    Indicates the element type for this lattice element. This should be one of:

        * ``drift`` for free drift. This requires this additional parameter:

            * ``<element_name>.ds`` (``float``, in meters) the segment length

        * ``quad`` for a hard edged quadrupole.
          This applies a quadrupole field that is uniform within the ``z`` extent of the element with a sharp cut off at the ends.
          This uses residence corrections, with the field scaled by the amount of time within the element for particles entering
          or leaving it, to increase the accuracy.
          This requires these additional parameters:

            * ``<element_name>.ds`` (``float``, in meters) the segment length

            * ``<element_name>.dEdx`` (``float``, in volts/meter^2) optional (default: 0.) the electric quadrupole field gradient
              The field applied to the particles will be ``Ex = dEdx*x`` and ``Ey = -dEdx*y``.

            * ``<element_name>.dBdx`` (``float``, in Tesla/meter) optional (default: 0.) the magnetic quadrupole field gradient
              The field applied to the particles will be ``Bx = dBdx*y`` and ``By = dBdx*x``.

        * ``plasmalens`` for a field modeling a plasma lens
          This applies a radially directed plasma lens field that is uniform within the ``z`` extent of the element with
          a sharp cut off at the ends.
          This uses residence corrections, with the field scaled by the amount of time within the element for particles entering
          or leaving it, to increase the accuracy.
          This requires these additional parameters:

            * ``<element_name>.ds`` (``float``, in meters) the segment length

            * ``<element_name>.dEdx`` (``float``, in volts/meter^2) optional (default: 0.) the electric field gradient
              The field applied to the particles will be ``Ex = dEdx*x`` and ``Ey = dEdx*y``.

            * ``<element_name>.dBdx`` (``float``, in Tesla/meter) optional (default: 0.) the magnetic field gradient
              The field applied to the particles will be ``Bx = dBdx*y`` and ``By = -dBdx*x``.

        * ``line`` a sub-lattice (line) of elements to append to the lattice.

            * ``<element_name>.elements`` (``list of strings``) optional (default: no elements)
              A list of names (one name per lattice element), in the order that they appear in the lattice.

            * ``<element_name>.reverse`` (``boolean``) optional (default: ``false``)
              Reverse the list of elements in the line before appending to the lattice.


.. _running-cpp-parameters-collision:

Collision models
----------------

WarpX provides several particle collision models, using varying degrees of approximation.
Details about the collision models can be found in the :ref:`theory section <multiphysics-collisions>`.

.. pp:param:: collisions.collision_names
    :type: ``strings``, separated by spaces

    The name of each collision type.
    This is then used in the rest of the input deck;
    in this documentation we use ``<collision_name>`` as a placeholder.

.. pp:param:: <collision_name>.type
    :type: ``string``
    :optional:

    The type of collision. The types implemented are:

    - ``pairwisecoulomb`` for pair-wise Coulomb collisions, the default if unspecified.
      This provides a pair-wise relativistic elastic Monte Carlo binary Coulomb collision model,
      following the algorithm given by :cite:t:`param-PerezPOP2012`.
      When the RZ mode is used, :pp:param:`warpx.n_rz_azimuthal_modes` must be set to 1 at the moment,
      since the current implementation of the collision module assumes axisymmetry.
    - ``nuclearfusion`` for fusion reactions.
      This implements the pair-wise fusion model by :cite:t:`param-HigginsonJCP2019`.
      Currently, WarpX supports deuterium-deuterium, deuterium-tritium, deuterium-helium and proton-boron fusion.
      When initializing the reactant and product species, you need to use ``species_type`` (see the documentation
      for this parameter), so that WarpX can identify the type of reaction to use.
      (e.g. :pp:param:`<species_name>.species_type = 'deuterium'`)
    - ``dsmc`` for pair-wise, non-Coulomb collisions between kinetic species.
      This is a "direct simulation Monte Carlo" treatment of collisions between
      kinetic species. See :ref:`DSMC section <multiphysics-collisions-dsmc>`.
    - ``background_mcc`` for collisions between particles and a neutral background.
      This is a relativistic Monte Carlo treatment for particles colliding
      with a neutral background gas. See :ref:`MCC section <multiphysics-collisions-mcc>`.
    - ``pulsed_decay`` for decay of a parent species into two product species with a user-defined decay rate.
      See :ref:`Pulsed Decay section <multiphysics-collisions-pulseddecay>`.
    - ``background_stopping`` for slowing of ions due to collisions with electrons or ions.
      This implements the approximate formulae as derived in Introduction to Plasma Physics,
      from Goldston and Rutherford, section 14.2.
    - ``bremsstrahlung`` for slowing of electrons due to Bremsstrahlung collisions with ions.
      This uses the cross sections as given by `Seltzer and Berger <https://doi.org/10.1016/0092-640X(86)90014-8>`__.
    - ``inverse_bremsstrahlung`` for inverse bremstrahlung absorption of photons from the collisions of electrons and ions.
      The absorbed energy and momentum from the photons is distributed among the electrons in the cell so that the quantities are exactly conserved.
    - ``linear_breit_wheeler`` for electron-positron pair creation from the annihilation of two photons, according to the linear Breit-Wheeler mechanism
      (see for example `Gould et al. (Phys. Rev. 155, 1404, 1967) <https://doi.org/10.1103/PhysRev.155.1404>`__).
      This implements the generation of electron-positron pairs based on the analytical cross-section, e.g.
      equation (1) in Gould. The angular distribution of the emitted pairs is isotropic for now
      (instead of following the correct distribution, see e.g. `Ribeyre et al. (Plasma Phys. Control. Fusion 60 104001, 2018) <https://doi.org/10.1088/1361-6587/aad6da>`__).
      The implementation follows the same numerical algorithm as that of fusion reactions (see. :cite:t:`param-HigginsonJCP2019`).
    - ``linear_compton`` for linear Compton scattering between a lepton (electron or positron, for now) and a photon, based on the Klein-Nishina cross-section
      (see for example :cite:t:`param-LandauVol4`: equations 86.10 and 86.16 for the differential and total cross sections, respectively).
      The probability of scattering is drawn from the total cross section, while the angular distribution of the scattered lepton and photon is drawn from the differential cross section.
      The implementation follows the same numerical algorithm as that of fusion reactions (see. :cite:t:`param-HigginsonJCP2019`).
      Note the difference between the linear Compton scattering module described here and
      `the quantum synchrotron QED module <https://warpx.readthedocs.io/en/latest/usage/parameters.html#lookup-tables-and-other-settings-for-qed-modules>`__.
      The former (commonly referred to simply as Compton scattering) is the collision between a single electron and a single photon,
      the latter (also known as multi-photon/nonlinear Compton or quantum synchrotron radiation) is the scattering
      of a single electron in a strong electromagnetic field.

.. pp:param:: <collision_name>.species
    :type: ``strings``

    If using ``dsmc``, ``pairwisecoulomb``, ``nuclearfusion``, ``bremsstrahlung``, or ``inverse_bremsstrahlung`` this should be the name(s) of the species,
    between which the collision will be considered. (Provide only one name for intra-species collisions.)
    With ``bremsstrahlung``, the electron species must be given first, followed by the target species.
    Wtih ``inverse_bremsstrahlung``, this is the photon species being absorbed and the electron species they are colliding with, in that order.
    If using ``background_mcc`` or ``background_stopping`` type this should be the name of the
    species for which collisions with a background will be included.
    If using ``pulsed_decay`` type this should be the name of the parent species.
    In these three cases, only one species name should be given.
    If using ``linear_breit_wheeler`` these should be two photon species.
    If using ``linear_compton``, these should be two species: first, a photon species, and second, a lepton species, in this exact order.

.. pp:param:: <collision_name>.product_species
    :type: ``strings``

    Only for ``dsmc``, ``linear_breit_wheeler``, ``nuclearfusion``, and ``bremsstrahlung``.
    The name(s) of the species in which to add the new macroparticles created by the reaction.
    If using ``dsmc`` with ionization reactions, the first species in this list must be an electron.
    If using ``dsmc`` with ``charge_exchange`` and ``twoproduct_reaction``, the order of the ``product_species`` should match the order of the species in :pp:param:`<collision_name>.species`.
    If using ``linear_breit_wheeler`` these should be two species: one of electrons and one of positrons.
    If using ``bremsstrahlung``, the product species must be of type photon.
    If using ``linear_compton``, these should be two species: first, a photon species, and second, a lepton species, in this exact order.
    If using ``pulsed_decay``, the sum of the product species charges and mass must equal those of the parent species.

.. pp:param:: <collision_name>.ndt_supercycle
    :type: ``int``
    :optional:

    Execute collision once every ``ndt_supercycle`` PIC time steps.
    The effective collision time step is ``dt_collision = ndt_supercycle * dt_PIC``.
    Must be >= 1. Mutually exclusive with ``ndt_subcycle``. Default is 1.

.. pp:param:: <collision_name>.ndt_subcycle
    :type: ``int``
    :optional:

    Execute collision ``ndt_subcycle`` times per PIC time step.
    The effective collision time step is ``dt_collision = dt_PIC / ndt_subcycle``.
    Must be >= 1. Mutually exclusive with ``ndt_supercycle``.
    Useful when a large PIC time step is desired but collisions require finer time resolution.

.. pp:param:: <collision_name>.cumulative_scattering_angle_model
    :type: ``string``
    :default: ``bobylev``
    :optional:

    Only for ``pairwisecoulomb``. Specifies the cumulative scattering distribution used to compute the scattering angle.
    The possible values are ``bobylev`` and ``nanbu``.
    With ``bobylev``, the scattering angle is sampled from Bobylev's distribution (see :cite:t:`param-BobylevJCP2013`).
    With ``nanbu``, the scattering angle is sampled from Nanbu's distribution (see :cite:t:`param-NanbuPRE1997`).
    See :cite:t:`param-AngusJCP2025` and :cite:t:`param-AngusJCP2026` for further discussion of cumulative scattering distributions.

.. pp:param:: <collision_name>.CoulombLog
    :type: ``float``
    :optional:

    Only for ``pairwisecoulomb``. A provided fixed Coulomb logarithm of the
    collision type ``<collision_name>``.
    For example, a typical Coulomb logarithm has a form of
    :math:`\ln(\lambda_D/R)`,
    where :math:`\lambda_D` is the Debye length,
    :math:`R\approx1.4A^{1/3}` is the effective Coulombic radius of the nucleus,
    :math:`A` is the mass number.
    If this is not provided, or if a non-positive value is provided,
    a Coulomb logarithm will be computed automatically according to the algorithm in
    :cite:t:`param-PerezPOP2012`.

.. pp:param:: <collision_name>.use_global_debye_length
    :type: ``bool``
    :optional:

    Only for ``pairwisecoulomb``. When set, the Debye length used in the Coulomb log
    is calculated including all species in the simulation. The lengths are combined
    using the square root of one over the sum of one over the squares of the Debye lengths
    of each species. By default, this is turned off. Note that if :pp:param:`<collision_name>.CoulombLog`
    is specified, this Debye length is not used.

.. pp:param:: <collision_name>.event_multiplier
    :type: ``float``
    :optional:

    Only for ``nuclearfusion``, ``linear_breit_wheeler``, and ``linear_compton``.
    Increasing ``event_multiplier`` creates more macroparticles products,
    but with lower weight (in such a way that the corresponding
    total number of physical particle remains the same). This can improve
    the statistics of the simulation, in the case where the collision events are very rare.
    More specifically, in a collision between two macroparticles with weight ``w_1`` and ``w_2``,
    the weight of the product macroparticles will be ``min(w_1,w_2)/event_multiplier``.
    (And the weights of the reactant macroparticles are reduced correspondingly after the collision.)
    See :cite:t:`param-HigginsonJCP2019` for more details.
    The default value of ``event_multiplier`` is 1.

.. pp:param:: <collision_name>.probability_threshold
    :type: ``float``
    :optional:

    Only for ``nuclearfusion``, ``linear_breit_wheeler``, and ``linear_compton``.
    If the event multiplier is too high and results in a probability
    that approaches 1 (for a given collision between two macroparticles), then
    there is a risk of underestimating the total yield. In these cases,
    WarpX reduces the event multiplier used in that given collision.
    ``probability_threshold`` is the probability threshold above
    which WarpX reduces the event multiplier.

.. pp:param:: <collision_name>.probability_target_value
    :type: ``float``
    :optional:

    Only for ``nuclearfusion``, ``linear_breit_wheeler``, and ``linear_compton``.
    When the probability of fusion or linear Breit-Wheeler for a given collision exceeds
    ``probability_threshold``, WarpX reduces the event multiplier for
    that collisions such that the probability approches ``probability_target_value``.

.. pp:param:: <collision_name>.scattering_angle_model
    :type: ``string``
    :default: ``isotropic``
    :optional:

    Only for ``nuclearfusion``. The scattering angle for the products of the fusion reaction.
    The possible values are ``isotropic``, ``forward`` and ``backward``.
    With ``isotropic``, the scattering angle is drawn from an isotropic distribution.
    With ``forward``, the scattering angle is set to zero, i.e. the products are emitted in the same direction as the reactant (in the center of mass frame).
    With ``backward``, the scattering angle is set to :math:`\pi`, i.e. the products are emitted in the opposite direction of the reactant (in the center of mass frame).

.. pp:param:: <collision_name>.background_density
    :type: ``float``

    Only for ``background_mcc`` and ``background_stopping``. The density of the background in :math:`m^{-3}`.
    Can also provide ``<collision_name>.background_density(x,y,z,t)`` using the parser
    initialization style for spatially and temporally varying density. With ``background_mcc``, if a function
    is used for the background density, the input parameter ``<collision_name>.max_background_density``
    must also be provided to calculate the maximum collision probability.

.. pp:param:: <collision_name>.background_temperature
    :type: ``float``

    Only for ``background_mcc`` and ``background_stopping``. The temperature of the background in Kelvin.
    Can also provide ``<collision_name>.background_temperature(x,y,z,t)`` using the parser
    initialization style for spatially and temporally varying temperature.

.. pp:param:: <collision_name>.background_mass
    :type: ``float``
    :optional:

    Only for ``background_mcc`` and ``background_stopping``. The mass of the background gas in kg.
    With ``background_mcc``, if not given the mass of the colliding species will be used unless ionization is
    included in which case the mass of the product species will be used.
    With ``background_stopping``, and ``background_type`` set to ``electrons``, if not given defaults to the electron mass. With
    ``background_type`` set to ``ions``, the mass must be given.

.. pp:param:: <collision_name>.background_charge_state
    :type: ``float``

    Only for ``background_stopping``, where it is required when ``background_type`` is set to ``ions``.
    This specifies the charge state of the background ions.

.. pp:param:: <collision_name>.background_type
    :type: ``string``

    Only for ``background_stopping``, where it is required, the type of the background.
    The possible values are ``electrons`` and ``ions``. When ``electrons``, equation 14.12 from Goldston and Rutherford is used.
    This formula is based on Coulomb collisions with the approximations that :math:`M_b >> m_e` and :math:`V << v_{thermal\_e}`,
    and the assumption that the electrons have a Maxwellian distribution with temperature :math:`T_e`.

    .. math::
        \frac{dV}{dt} = - \frac{2^{1/2}n_eZ_b^2e^4m_e^{1/2}\log\Lambda}{12\pi^{3/2}\epsilon_0M_bT_e^{3/2}}V

    where :math:`V` is each velocity component, :math:`n_e` is the background density, :math:`Z_b` is the ion charge state,
    :math:`e` is the electron charge, :math:`m_e` is the background mass, :math:`\log\Lambda=\log((12\pi/Z_b)(n_e\lambda_{de}^3))`,
    :math:`\lambda_{de}` is the DeBye length, and :math:`M_b` is the ion mass.
    The equation is integrated over a time step, giving :math:`V(t+dt) = V(t)*\exp(-\alpha*{dt})`
    where :math:`\alpha` is the factor multiplying :math:`V`.

    When ``ions``, equation 14.20 is used.
    This formula is based on Coulomb collisions with the approximations that :math:`M_b >> M` and :math:`V >> v_{thermal\_i}`.
    The background ion temperature only appears in the :math:`\log\Lambda` term.

    .. math::
        \frac{dW_b}{dt} = - \frac{2^{1/2}n_iZ^2Z_b^2e^4M_b^{1/2}\log\Lambda}{8\pi\epsilon_0MW_b^{1/2}}

    where :math:`W_b` is the ion energy, :math:`n_i` is the background density,
    :math:`Z` is the charge state of the background ions, :math:`Z_b` is the ion charge state,
    :math:`e` is the electron charge, :math:`M_b` is the ion mass, :math:`\log\Lambda=\log((12\pi/Z_b)(n_i\lambda_{di}^3))`,
    :math:`\lambda_{di}` is the DeBye length, and :math:`M` is the background ion mass.
    The equation is integrated over a time step, giving :math:`W_b(t+dt) = ((W_b(t)^{3/2}) - 3/2\beta{dt})^{2/3}`
    where :math:`\beta` is the term on the r.h.s except :math:`W_b`.

.. pp:param:: <collision_name>.scattering_processes
    :type: ``strings`` separated by spaces

    Only for ``dsmc`` and ``background_mcc``. The scattering processes that should be
    included. Available options are ``elasticX``, ``excitationX``, ``twoproduct_reaction`` and ``charge_exchange``
    for ions and ``elasticX``, ``excitationX`` and ``ionization`` for electrons.
    Multiple elastic and excitation events can be included, corresponding e.g. to
    excitation to different levels or to several elastic channels (with different
    cross-sections and/or scattering angle models); the ``X`` above can be changed
    to a unique identifier for each such process. For each scattering process specified
    a path to a cross-section data file must also be given. We use
    ``<scattering_process>`` as a placeholder going forward.

    For ``elasticX``, ``excitationX``, ``charge_exchange`` and ``twoproduct_reaction``, the
    angular distribution is controlled by the per-process
    :pp:param:`<collision_name>.<scattering_process>_scattering_angle_model` argument.

.. pp:param:: <collision_name>.<scattering_process>_cross_section
    :type: ``string``

    Only for ``dsmc`` and ``background_mcc``. Path to the file containing cross-section data
    for the given scattering processes. The cross-section file must have exactly
    2 columns of data, the first containing energies in eV and the
    second the corresponding cross-section in :math:`m^2`. The energy column should
    represent the kinetic energy of the center-of-mass frame. The energy values in this column
    must be in strictly increasing order.

.. pp:param:: <collision_name>.<scattering_process>_energy
    :type: ``float``

    Only for ``dsmc`` and ``background_mcc``. The energy cost of the process, in eV. It is
    required for ``excitationX`` and ``ionization``, optional for ``charge_exchange`` and
    ``twoproduct_reaction`` (which may impose a fixed energy loss, defaulting to 0), and
    ignored for ``elasticX`` processes (which have no energy cost).

.. pp:param:: <collision_name>.<scattering_process>_scattering_angle_model
    :type: ``string``
    :optional:

    Only for ``dsmc`` and ``background_mcc``, and only for ``elasticX``, ``excitationX``,
    ``charge_exchange`` and ``twoproduct_reaction``.
    The model used to determine the scattering angle of the products
    in the center-of-mass frame. The possible values are ``isotropic``, ``forward`` and ``backward``.
    The default is ``isotropic`` for ``elasticX`` and ``excitationX``, and ``forward`` for
    ``charge_exchange`` and ``twoproduct_reaction``.
    With ``isotropic``, the scattering angle is drawn from an isotropic distribution.
    With ``forward``, the scattering angle is set to zero, i.e. the products keep the same direction
    as the incident particle (in the center of mass frame).
    With ``backward``, the scattering angle is set to :math:`\pi`, i.e. the products are emitted in
    the opposite direction of the incident particle (in the center of mass frame).

.. pp:param:: <collision_name>.ionization_species
    :type: ``float``

    Only for ``background_mcc``. If the scattering process is ``ionization`` the
    produced species must also be given. For example if argon properties is used
    for the background gas, a species of argon ions should be specified here.

.. pp:param:: <collision_name>.ionization_target_species
    :type: ``string``

    Only for ``dsmc`` with impact ionization. This specifies which one of the
    colliding particles is ionized.

.. pp:param:: <collision_name>.decay_rate(x,y,z,t)
    :type: `string`

    The parent species decay rate (only for ``pulsed_decay``).

.. pp:param:: <collision_name>.fixed_product_weight
    :type: `float`

    Fixed particle weight of product species (only for ``pulsed_decay``).
    Can be estimated as :math:`n_{\text{target}}dV/N_{ppc}`, where :math:`n_{\text{target}}` is the target density of the product species, :math:`dV` is the cell volume, and :math:`N_{ppc}` is the target number of particle per cell for each product species.

.. pp:param:: <collision_name>.productA_temperature_eV
    :type: `float array, size 3`

    Direction-dependent temperature used to assign a random thermal velocity to product species A (only for ``pulsed_decay``).
    Example: ``<collision_name>.productA_temperature_eV = 0.1 0.2 0.3``.

.. pp:param:: <collision_name>.productB_temperature_eV
    :type: `float array, size 3`

    Direction-dependent temperature used to assign a random thermal velocity to product species B (only for ``pulsed_decay``).
    Example: ``<collision_name>.productB_temperature_eV = 0.1 0.2 0.3``.

.. pp:param:: <collision_name>.Z
    :type: ``integer``

    Only for ``bremsstrahlung``. The atomic number of the target ion species.
    Currently, only the values 1, 2, 5, 6 are supported.

.. pp:param:: <collision_name>.multiplier
    :type: ``float``

    Only for ``bremsstrahlung``. Multiplier for the collision probability.
    Any resulting photons will have the electron weight divided the multiplier.
    The default is 1. This must be greater than or equal to 1.

.. pp:param:: <collision_name>.create_photons
    :type: ``integer``

    Only for ``bremsstrahlung``. Whether photons will be created, defaults to 1 (true).

.. pp:param:: <collision_name>.koT1_cut
    :type: ``float``

    Only for ``bremsstrahlung``. Minimum energy of the photons created.
    This is relative to the electron energy, defaulting to 1.e-4.

.. pp:param:: collisions.correct_energy_momentum
    :type: ``bool``
    :default: 0
    :optional:

    Only for ``pairwisecoulomb`` collisions, whether to correct the energy and momentum after the collisions so that they are conserved.
    This can be set for each collision using :pp:param:`<collision_name>.correct_energy_momentum`.
    In binary collisions, if the weights of the colliding particles are not the same, the collision does not
    exactly conserve energy and momentum. When this option is on, after the collisions, small modifications are made to the
    particle momentum so that the energy and momentum are exactly conserved in each cell.
    This uses the algorithm described in https://doi.org/10.1016/j.jcp.2025.113927.

.. pp:param:: collisions.np_warning_threshold
    :type: ``int``
    :default: 20.
    :optional:

    Only for ``pairwisecoulomb`` collisions, with :pp:param:`collisions.correct_energy_momentum` set, this parameter controls the minimum number of particles per cell for producing warning messages when the moment-correction method fails.

.. pp:param:: collisions.energy_fraction
    :type: ``float``
    :default: 0.05
    :optional:

    Only for ``pairwisecoulomb`` collisions with :pp:param:`collisions.correct_energy_momentum` set, and for ``inverse_bremsstrahlung``.
    In both cases, the energy difference is applied to pairs of particles in their center of momentum frame in such a way that the momentum is conserved.
    This parameter limits the change in the energy of the electrons to the specified fraction of the energy in the COM frame of the pair of particles.
    With ``pairwisecoulomb`` collisions, energy can be added or removed, and if residual energy error remains after 10 passes over all particle pairs in a cell, the correction is deemed to have failed and particle velocities in the cell are restored to their pre-collision values.
    With ``inverse_bremsstrahlung``, energy is always added, and it there if residual energy remaining after 10 passes, that remaining energy is distributed evenly among the particles without conservation of momentum.
    This can be set for each collision using :pp:param:`<collision_name>.energy_fraction`.

.. pp:param:: collisions.beta_weight_exponent
    :type: ``float``
    :default: 1.
    :optional:

    Only for ``pairwisecoulomb`` collisions, with :pp:param:`collisions.correct_energy_momentum` set, this parameter controls the exponent used on the particle weight when distributing the momentum correction.
    This can be set for each collision using :pp:param:`<collision_name>.beta_weight_exponent`.
    With a value greater than 1, it will distribute more of the correction to particles with higher weights.

.. pp:param:: collisions.energy_correction_sort_by_weight
    :type: ``bool``
    :default: 0
    :optional:

    Only for ``pairwisecoulomb`` collisions, with :pp:param:`collisions.correct_energy_momentum` set, specifies whether the particles are sorted by weight when the energy correction is applied.
    This can be set for each collision using :pp:param:`<collision_name>.energy_correction_sort_by_weight`.
    When the particles have a range of weights, sorting improves the correction by applying more of it to the heavier weighted particles, which has a proportionately smaller effect on their momenta, and typically reduces the number of particles that the correction is applied to.

.. pp:param:: collisions.split_momentum_push
    :type: ``bool``
    :default: 1
    :optional:

    If true, collisions are performed in the middle of the momentum push, which is split into two substeps.
    This improves energy conservation, as demonstrated in (`Vay et al., Phys. Rev. E 111, 2025 <https://doi.org/10.1103/PhysRevE.111.025306>`__).
    This is only implemented for the explicit evolve scheme and is not available for the implicit evolve schemes, because the implicit
    formulation is intrinsically energy-conserving when combined with MCC collisions, as shown in `Angus et al., J. Comput. Phys. 456, 2022 <https://doi.org/10.1016/j.jcp.2022.111030>`__.

.. _running-cpp-parameters-numerics:

Numerics and algorithms
-----------------------

This section describes the input parameters used to select numerical methods and algorithms for your simulation setup.

Time step
^^^^^^^^^

.. pp:param:: warpx.cfl
    :type: ``float``
    :default: ``0.999``
    :optional:

    The ratio between the actual timestep that is used in the simulation
    and the Courant-Friedrichs-Lewy (CFL) limit. (e.g. for ``warpx.cfl=1``,
    the timestep will be exactly equal to the CFL limit.)
    For some speed v and grid spacing dx, this limits the timestep to ``warpx.cfl * dx / v``.
    When used with the electromagnetic solver, ``v`` is the speed of light.
    For the electrostatic solver, ``v`` is the maximum speed among all particles in the domain.

.. pp:param:: warpx.const_dt
    :type: ``float``

    Allows direct specification of the time step size, in units of seconds.
    When the electrostatic solver is being used, this must be supplied if not using adaptive timestepping.
    This can be used with the electromagnetic solver, overriding :pp:param:`warpx.cfl`, but
    it is up to the user to ensure that the CFL condition is met.

.. pp:param:: warpx.dt_update_interval
    :type: ``string``
    :optional:

    This controls adaptive timestepping, where the time step size is updated based on the conditions of the simulation, and only applies when using the explicit electrostatic or theta-implicit solvers.
    This specifies time step intervals when the time step size is updated.
    The value must be greater than ``0``.
    When specified, :pp:param:`warpx.const_dt` must not also be specified.
    The time step size is updated using the limits specified by :pp:param:`warpx.cfl`, :pp:param:`warpx.max_omegap_dt`, and :pp:param:`warpx.max_omegac_dt`.

.. pp:param:: warpx.dt_update_diagnostic_file
    :type: ``string``
    :optional:

    When adaptive timestepping is activated, information about the new time step and the simulation conditions are output to the file specified by this parameter.

.. pp:param:: warpx.max_omegap_dt
    :type: ``float``
    :optional:

    With adaptive timestepping, the time step size is limited to be less than or equal to the value specified divided by the global plasma frequency.
    The application of this limit is controlled by :pp:param:`warpx.dt_update_interval`, and is only applied when using the explicit electrostatic or theta-implicit solver..

.. pp:param:: warpx.max_omegac_dt
    :type: ``float``
    :optional:

    With adaptive timestepping, the time step size is limited to be less than or equal to the value specified divided by the maximum cyclotron frequency.
    Note that the maximum B-field is calculated from using only the constant applied B field (as set by :pp:param:`particles.B_external_particle`) and the B-field grid data.
    The application of this limit is controlled by :pp:param:`warpx.dt_update_interval`, and is only applied when using the explicit electrostatic or theta-implicit solver..

.. pp:param:: warpx.max_dt
    :type: ``float``
    :optional:

    The maximum timestep permitted when using adaptive timestepping.
    If supplied, also sets the initial timestep for these simulations, before the first timestep update.

Filtering
^^^^^^^^^

.. pp:param:: warpx.use_filter
    :type: ``0`` or ``1``

    Whether to use filtering in the simulation.
    With the explicit evolve scheme, the filtering is turned on by default, except for RZ FDTD.
    With the implicit evolve schemes, the filtering is turned off by default.
    The filtering smooths the charge and currents on the mesh, after depositing them from the macro-particles.
    With implicit schemes, the electric field is also filtered (to maintain consistency for energy conservation).
    This uses a bilinear filter (see the :ref:`filtering section <theory-filter>`).
    With the RZ PSATD solver, the filtering is done in :math:`k`-space.

    .. warning::

       Known bug: filter currently not working with FDTD solver in RZ geometry (see https://github.com/BLAST-WarpX/warpx/issues/1943).

.. pp:param:: warpx.filter_npass_each_dir
    :type: ``3 int``
    :default: ``1 1 1``
    :optional:

    Number of passes along each direction for the bilinear filter.
    In 2D simulations, only the first two values are read.

.. pp:param:: warpx.use_filter_compensation
    :type: ``0`` or ``1``
    :default: ``0``

    Whether to add compensation when applying filtering.
    This is only supported with the RZ spectral solver.

Particle push, charge and current deposition, field gathering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. pp:param:: algo.current_deposition
    :type: ``string``
    :optional:

    This parameter selects the algorithm for the deposition of the current density.
    Available options are: ``direct``, ``esirkepov``, ``villasenor``, and ``vay``. The default choice
    is ``esirkepov`` for FDTD maxwell solvers but ``direct`` for standard or
    Galilean PSATD solver (i.e. with :pp:param:`algo.maxwell_solver = psatd`) and
    for the hybrid-PIC solver (i.e. with :pp:param:`algo.maxwell_solver = hybrid`) and for
    diagnostics output with the electrostatic solvers (i.e., with
    :pp:param:`warpx.do_electrostatic = ...`).
    Note that ``vay`` is only available for :pp:param:`algo.maxwell_solver = psatd`.

    1. ``direct``

       The current density is deposited as described in the section :ref:`current_deposition`.
       This deposition scheme does not conserve charge.

    2. ``esirkepov``

       The current density is deposited as described in
       :cite:t:`param-Esirkepovcpc01`.
       This deposition scheme guarantees charge conservation for shape factors of arbitrary order.

    3. ``villasenor``

       This uses the Villasenor-Buneman algorithm which guarantees charge conservation.
       The algorithm is described in :cite:t:`pt-Villasenorcpc92`.

    4. ``vay``

       The current density is deposited as described in :cite:t:`param-VayJCP2013` (see section :ref:`current_deposition` for more details).
       This option guarantees charge conservation only when used in combination
       with :pp:param:`psatd.periodic_single_box_fft = 1`, that is, only for periodic single-box
       simulations with global FFTs without guard cells. The implementation for domain
       decomposition with local FFTs over guard cells is planned but not yet completed.

.. pp:param:: algo.charge_deposition
    :type: ``string``
    :optional:

    The algorithm for the charge density deposition. Available options are:

     - ``standard``: standard charge deposition algorithm, described in
       the :ref:`particle-in-cell theory section <theory-pic>`.

.. pp:param:: algo.field_gathering
    :type: ``string``
    :optional:

    The algorithm for field gathering. Available options are:

     * ``energy-conserving``: gathers directly from the grid points (either staggered
       or nodal grid points depending on :pp:param:`warpx.grid_type`).
     * ``momentum-conserving``: first average the fields from the grid points to
       the nodes, and then gather from the nodes.


    Default: :pp:param:`algo.field_gathering = energy-conserving` with collocated or staggered grids (note that ``energy-conserving`` and ``momentum-conserving`` are equivalent with collocated grids), :pp:param:`algo.field_gathering = momentum-conserving` with hybrid grids.

.. pp:param:: algo.particle_pusher
    :type: ``string``
    :optional:

    The algorithm for the particle pusher. Available options are:

     - ``boris``: Boris pusher.
     - ``vay``: Vay pusher (see :cite:t:`param-Vaypop2008`)
     - ``higuera``: Higuera-Cary pusher (see :cite:t:`param-HigueraPOP2017`)

    If :pp:param:`algo.particle_pusher` is not specified, ``boris`` is the default.

.. pp:param:: algo.particle_shape
    :type: ``integer``; ``1``, ``2``, ``3``, or ``4``

    The order of the shape factors (splines) for the macro-particles along all spatial directions: ``1`` for linear, ``2`` for quadratic, ``3`` for cubic, ``4`` for quartic.
    Low-order shape factors result in faster simulations, but may lead to more noisy results.
    High-order shape factors are computationally more expensive, but may increase the overall accuracy of the results. For production runs it is generally safer to use high-order shape factors, such as cubic order.

    Note that this input parameter is not optional and must always be set in all input files provided that there is at least one particle species (set in input as :pp:param:`particles.species_names`) or one laser species (set in input as :pp:param:`lasers.names`) in the simulation. No default value is provided automatically.

.. pp:param:: particles.max_grid_crossings
    :type: ``integer``
    :default: ``1``
    :optional:

    Maximum number of grid crossings the particles can do per time step.
    This is only used with the Strang and theta-implicit schemes since they allow the speed of light Courant limit to be violated.

Maxwell solver
^^^^^^^^^^^^^^

Two families of Maxwell solvers are implemented in WarpX, based on the Finite-Difference Time-Domain method (FDTD) or the Pseudo-Spectral Analytical Time-Domain method (PSATD), respectively.

.. pp:param:: algo.maxwell_solver
    :type: ``string``
    :optional:

    The algorithm for the Maxwell field solver.
    Available options are:

     - ``yee``: Yee FDTD solver.
     - ``ckc``: (not available in ``RZ``, ``RCYLINDER``, and ``RSPHERE`` geometries) Cole-Karkkainen solver with Cowan
       coefficients (see :cite:t:`param-CowanPRSTAB13`).
     - ``psatd``: Pseudo-spectral solver (see :ref:`theory <theory-mwsolve-psatd>`).
     - ``ect``: Enlarged cell technique (conformal finite difference solver. See :cite:t:`param-XiaoIEEE2004`).
     - ``hybrid``: The E-field will be solved using Ohm's law and a kinetic-fluid hybrid model (see :ref:`theory <theory-kinetic-fluid-hybrid-model>`).
     - ``none``: No field solve will be performed.

    If :pp:param:`algo.maxwell_solver` is not specified, ``yee`` is the default.

.. pp:param:: algo.em_solver_medium
    :type: ``string``
    :optional:

    The medium for evaluating the Maxwell solver. Available options are :

    - ``vacuum``: vacuum properties are used in the Maxwell solver.
    - ``macroscopic``: macroscopic Maxwell equation is evaluated. If this option is selected, then the corresponding properties of the medium must be provided using :pp:param:`macroscopic.sigma`, :pp:param:`macroscopic.epsilon`, and :pp:param:`macroscopic.mu` for each case where the initialization style is ``constant``.  Otherwise if the initialization style uses the parser, :pp:param:`macroscopic.sigma_function(x,y,z)`, :pp:param:`macroscopic.epsilon_function(x,y,z)` and/or :pp:param:`macroscopic.mu_function(x,y,z)` must be provided using the parser initialization style for spatially varying macroscopic properties.

    If :pp:param:`algo.em_solver_medium` is not specified, ``vacuum`` is the default.

Maxwell solver: PSATD method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. pp:param:: psatd.nox/noy/noz
    :link_aliases:
        psatd.nox
        psatd.noy
        psatd.noz
    :type: ``integer``
    :default: ``16`` for all
    :optional:

    The order of accuracy of the spatial derivatives, when using the code compiled with a PSATD solver.
    If :pp:param:`psatd.periodic_single_box_fft` is used, these can be set to ``inf`` for infinite-order PSATD.

.. pp:param:: psatd.nx/ny/nz_guard
    :link_aliases:
        psatd.nx_guard
        psatd.ny_guard
        psatd.nz_guard
    :type: ``integer``
    :optional:

    The number of guard cells to use with PSATD solver.
    If not set by users, these values are calculated automatically and determined *empirically* and
    equal the order of the solver for collocated grids and half the order of the solver for staggered grids.

.. pp:param:: psatd.periodic_single_box_fft
    :type: ``0`` or ``1``
    :default: 0

    If true, this will *not* incorporate the guard cells into the box over which FFTs are performed.
    This is only valid when WarpX is run with periodic boundaries and a single box.
    In this case, using :pp:param:`psatd.periodic_single_box_fft` is equivalent to using a global FFT over the whole domain.
    Therefore, all the approximations that are usually made when using local FFTs with guard cells
    (for problems with multiple boxes) become exact in the case of the periodic, single-box FFT without guard cells.

.. pp:param:: psatd.current_correction
    :type: ``0`` or ``1``
    :default: ``1``, with the exceptions mentioned below

    If true, a current correction scheme in Fourier space is applied in order to guarantee charge conservation.
    The default value is :pp:param:`psatd.current_correction = 1`, unless a charge-conserving current deposition scheme is used (by setting :pp:param:`algo.current_deposition = esirkepov` or :pp:param:`algo.current_deposition = vay`) or unless the ``div(E)`` cleaning scheme is used (by setting :pp:param:`warpx.do_dive_cleaning = 1`).

    If :pp:param:`psatd.v_galilean` is zero, the spectral solver used is the standard PSATD scheme described in :cite:t:`param-VayJCP2013` and the current correction reads

    .. math::
       \widehat{\boldsymbol{J}}^{\,n+1/2}_{\mathrm{correct}} = \widehat{\boldsymbol{J}}^{\,n+1/2}
       - \bigg(\boldsymbol{k}\cdot\widehat{\boldsymbol{J}}^{\,n+1/2}
       - i \frac{\widehat{\rho}^{n+1} - \widehat{\rho}^{n}}{\Delta{t}}\bigg) \frac{\boldsymbol{k}}{k^2}

    If :pp:param:`psatd.v_galilean` is non-zero, the spectral solver used is the Galilean PSATD scheme described in :cite:t:`param-LehePRE2016` and the current correction reads

    .. math::
       \widehat{\boldsymbol{J}}^{\,n+1/2}_{\mathrm{correct}} = \widehat{\boldsymbol{J}}^{\,n+1/2}
       - \bigg(\boldsymbol{k}\cdot\widehat{\boldsymbol{J}}^{\,n+1/2} - (\boldsymbol{k}\cdot\boldsymbol{v}_G)
       \,\frac{\widehat\rho^{n+1} - \widehat\rho^{n}\theta^2}{1 - \theta^2}\bigg) \frac{\boldsymbol{k}}{k^2}

    where :math:`\theta=\exp(i\,\boldsymbol{k}\cdot\boldsymbol{v}_G\,\Delta{t}/2)`.

    This option is currently implemented only for the standard PSATD, Galilean PSATD, and averaged Galilean PSATD schemes, while it is not yet available for the PSATD JRhom algorithm.

.. pp:param:: psatd.update_with_rho
    :type: ``0`` or ``1``

    If true, the update equation for the electric field is expressed in terms of both the current density and the charge density, namely :math:`\widehat{\boldsymbol{J}}^{\,n+1/2}`, :math:`\widehat\rho^{n}`, and :math:`\widehat\rho^{n+1}`.
    If false, instead, the update equation for the electric field is expressed in terms of the current density :math:`\widehat{\boldsymbol{J}}^{\,n+1/2}` only.
    If charge is expected to be conserved (by setting, for example, :pp:param:`psatd.current_correction = 1`), then the two formulations are expected to be equivalent.

    If :pp:param:`psatd.v_galilean` is zero, the spectral solver used is the standard PSATD scheme described in :cite:t:`param-VayJCP2013`:

    1. if :pp:param:`psatd.update_with_rho = 0`, the update equation for the electric field reads

    .. math::
       \begin{split}
       \widehat{\boldsymbol{E}}^{\,n+1}= & \:
       C \widehat{\boldsymbol{E}}^{\,n} + i \, \frac{S c}{k} \boldsymbol{k}\times\widehat{\boldsymbol{B}}^{\,n}
       - \frac{S}{\epsilon_0 c \, k} \widehat{\boldsymbol{J}}^{\,n+1/2} \\[0.2cm]
       & +\frac{1-C}{k^2} (\boldsymbol{k}\cdot\widehat{\boldsymbol{E}}^{\,n}) \boldsymbol{k}
       + \frac{1}{\epsilon_0 k^2} \left(\frac{S}{c \, k}-\Delta{t}\right)
       (\boldsymbol{k}\cdot\widehat{\boldsymbol{J}}^{\,n+1/2}) \boldsymbol{k}
       \end{split}

    2. if :pp:param:`psatd.update_with_rho = 1`, the update equation for the electric field reads

    .. math::
       \begin{split}
       \widehat{\boldsymbol{E}}^{\,n+1}= & \:
       C\widehat{\boldsymbol{E}}^{\,n} + i \, \frac{S c}{k} \boldsymbol{k}\times\widehat{\boldsymbol{B}}^{\,n}
       - \frac{S}{\epsilon_0 c \, k} \widehat{\boldsymbol{J}}^{\,n+1/2} \\[0.2cm]
       & + \frac{i}{\epsilon_0 k^2} \left(C-\frac{S}{c\,k}\frac{1}{\Delta{t}}\right)
       \widehat{\rho}^{n} \boldsymbol{k} - \frac{i}{\epsilon_0 k^2} \left(1-\frac{S}{c \, k}
       \frac{1}{\Delta{t}}\right)\widehat{\rho}^{n+1} \boldsymbol{k}
       \end{split}

    The coefficients :math:`C` and :math:`S` are defined in :cite:t:`param-VayJCP2013`.

    If :pp:param:`psatd.v_galilean` is non-zero, the spectral solver used is the Galilean PSATD scheme described in :cite:t:`param-LehePRE2016`:

    1. if :pp:param:`psatd.update_with_rho = 0`, the update equation for the electric field reads

    .. math::
       \begin{split}
       \widehat{\boldsymbol{E}}^{\,n+1} = & \:
       \theta^{2} C \widehat{\boldsymbol{E}}^{\,n} + i \, \theta^{2} \frac{S c}{k}
       \boldsymbol{k}\times\widehat{\boldsymbol{B}}^{\,n}
       + \frac{i \, \nu \, \theta \, \chi_1 - \theta^{2} S}{\epsilon_0 c \, k}
       \widehat{\boldsymbol{J}}^{\,n+1/2} \\[0.2cm]
       & + \theta^{2} \frac{\chi_2-\chi_3}{k^{2}}
       (\boldsymbol{k}\cdot\widehat{\boldsymbol{E}}^{\,n}) \boldsymbol{k}
       + i \, \frac{\chi_2\left(\theta^{2}-1\right)}{\epsilon_0 c \, k^{3} \nu}
       (\boldsymbol{k}\cdot\widehat{\boldsymbol{J}}^{\,n+1/2}) \boldsymbol{k}
       \end{split}

    2. if :pp:param:`psatd.update_with_rho = 1`, the update equation for the electric field reads

    .. math::
       \begin{split}
       \widehat{\boldsymbol{E}}^{\,n+1} = & \:
       \theta^{2} C \widehat{\boldsymbol{E}}^{\,n} + i \, \theta^{2} \frac{S c}{k}
       \boldsymbol{k}\times\widehat{\boldsymbol{B}}^{\,n}
       + \frac{i \, \nu \, \theta \, \chi_1 - \theta^{2} S}{\epsilon_0 c \, k}
       \widehat{\boldsymbol{J}}^{\,n+1/2} \\[0.2cm]
       & + i \, \frac{\theta^{2} \chi_3}{\epsilon_0 k^{2}} \widehat{\rho}^{\,n} \boldsymbol{k}
       - i \, \frac{\chi_2}{\epsilon_0 k^{2}} \widehat{\rho}^{\,n+1} \boldsymbol{k}
       \end{split}

    The coefficients :math:`C`, :math:`S`, :math:`\theta`, :math:`\nu`, :math:`\chi_1`, :math:`\chi_2`, and :math:`\chi_3` are defined in :cite:t:`param-LehePRE2016`.

    The default value for :pp:param:`psatd.update_with_rho` is ``1`` if :pp:param:`psatd.v_galilean` is non-zero and ``0`` otherwise.
    The option :pp:param:`psatd.update_with_rho = 0` is not implemented with the following algorithms:
    comoving PSATD (:pp:param:`psatd.v_comoving`), time averaging (:pp:param:`psatd.do_time_averaging = 1`), div(E) cleaning (:pp:param:`warpx.do_dive_cleaning = 1`), and PSATD JRhom (:pp:param:`psatd.JRhom`).

    Note that the update with and without rho is also supported in RZ geometry.

.. pp:param:: psatd.v_galilean
    :type: ``3 floats``
    :unit: units of the speed of light
    :default: ``0. 0. 0.``

    Defines the Galilean velocity.
    A non-zero velocity activates the Galilean algorithm, which suppresses numerical Cherenkov instabilities (NCI) in boosted-frame simulations (see the section :ref:`Numerical Stability and alternate formulation in a Galilean frame <theory-boostedframe-galilean>` for more information).
    This requires the code to be compiled with the spectral solver.
    It also requires the use of the direct current deposition algorithm (by setting :pp:param:`algo.current_deposition = direct`).

.. pp:param:: psatd.use_default_v_galilean
    :type: ``0`` or ``1``
    :default: ``0``

    This can be used in boosted-frame simulations only and sets the Galilean velocity along the :math:`z` direction automatically as :math:`v_{G} = -\sqrt{1-1/\gamma^2}`, where :math:`\gamma` is the Lorentz factor of the boosted frame (set by :pp:param:`warpx.gamma_boost`).
    See the section :ref:`Numerical Stability and alternate formulation in a Galilean frame <theory-boostedframe-galilean>` for more information on the Galilean algorithm for boosted-frame simulations.

.. pp:param:: psatd.v_comoving
    :type: 3 floating-point values
    :unit: units of the speed of light
    :default: ``0. 0. 0.``

    Defines the comoving velocity in the comoving PSATD scheme.
    A non-zero comoving velocity selects the comoving PSATD algorithm, which suppresses the numerical Cherenkov instability (NCI) in boosted-frame simulations, under certain assumptions. This option requires that WarpX is compiled with ``USE_FFT = TRUE``. It also requires the use of direct current deposition (:pp:param:`algo.current_deposition = direct`) and has neither been implemented nor tested with other current deposition schemes.

.. pp:param:: psatd.do_time_averaging
    :type: ``0`` or ``1``
    :default: 0

    Whether to use an averaged Galilean PSATD algorithm or standard Galilean PSATD.

.. pp:param:: psatd.JRhom
    :type: ``string``

    This determines whether the PSATD JRhom algorithm is used, where current deposition and field update are performed multiple times within one time step, while field gathering is performed only once.
    For simulations with strong numerical Cherenkov instability (NCI), the PSATD JRhom algorithm is recommended in combination with :pp:param:`psatd.do_time_averaging = 1`.
    The input parameter is a string composed by two characters and one digit.
    The first character represents the time dependency of J within the time step over which the electromagnetic fields are evolved, e.g., "C" for constant in time, "L" for linear in time, "Q" for quadratic in time.
    The second character represents the time dependency of rho within the time step over which the electromagnetic fields are evolved, following the same naming convention as for J.
    The last digit is an integer that represents the number of subintervals used in the JRhom algorithm.
    Examples: "CL1" (equivalent to the standard PSATD PIC algorithm), "CL2", "LL4", etc.
    By default, the string is empty and the PSATD JRhom algorithm is not used.


Maxwell solver: macroscopic media
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. pp:param:: algo.macroscopic_sigma_method
    :type: ``string``
    :optional:

    The algorithm for updating electric field when :pp:param:`algo.em_solver_medium` is macroscopic. Available options are:

    - ``backwardeuler`` is a fully-implicit, first-order in time scheme for E-update (default).
    - ``laxwendroff`` is the semi-implicit, second order in time scheme for E-update.

    Comparing the two methods, Lax-Wendroff is more prone to developing oscillations and requires a smaller timestep for stability. On the other hand, Backward Euler is more robust but it is first-order accurate in time compared to the second-order Lax-Wendroff method.

.. pp:param:: macroscopic.sigma/epsilon/mu_function(x,y,z)
    :link_aliases:
        macroscopic.sigma_function(x,y,z)
        macroscopic.epsilon_function(x,y,z)
        macroscopic.mu_function(x,y,z)
    :type: ``string``

    To initialize spatially varying conductivity, permittivity, and permeability, respectively,
    using a mathematical function in the input. Constants required in the
    mathematical expression can be set using ``my_constants``. These parameters are parsed
    if :pp:param:`algo.em_solver_medium = macroscopic`.

.. pp:param:: macroscopic.sigma/epsilon/mu
    :link_aliases:
        macroscopic.sigma
        macroscopic.epsilon
        macroscopic.mu
    :type: ``double``

    To initialize a constant conductivity, permittivity, and permeability of the
    computational medium, respectively. The default values are the corresponding values
    in vacuum.

.. _running-cpp-parameters-hybrid-model:

Maxwell solver: kinetic-fluid hybrid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    **Required Parameters:**

    - :pp:param:`hybrid_pic_model.elec_temp` must be specified when using the hybrid solver.
    - :pp:param:`hybrid_pic_model.n0_ref` should be specified if :pp:param:`hybrid_pic_model.gamma != 1`.

    **Best Practices**

    - *Grid type:* Setting :pp:param:`warpx.grid_type = collocated` is recommended
    - *Particle shape:* Linear particles (:pp:param:`algo.particle_shape = 1`) are recommended based on :cite:t:`param-Stanier2020`.

.. warning::

    **Constraints and Limitations:**

    - *Mesh refinement:* Only one level is supported (no AMR). The solver will abort if ``lev > 0``.
    - *RZ geometry:* Only the m=0 azimuthal mode is supported in RZ geometry.
    - *External vector potential:* If using :pp:param:`hybrid_pic_model.add_external_fields = true`, then :pp:param:`external_vector_potential.fields` must be non-empty.
    - *Time-dependent A fields:* When using expressions for external vector potentials, time variation must be specified via ``A_time_external_function(t)``, not directly in the ``A[x,y,z]_external_grid_function(x,y,z)`` expressions.

.. pp:param:: hybrid_pic_model.elec_temp
    :type: ``float``

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the electron temperature, in eV, used to calculate
    the electron pressure (see :ref:`here <theory-hybrid-model-elec-temp>`).

.. pp:param:: hybrid_pic_model.n0_ref
    :type: ``float``

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the reference density, in :math:`m^{-3}`, used to calculate
    the electron pressure (see :ref:`here <theory-hybrid-model-elec-temp>`).

.. pp:param:: hybrid_pic_model.gamma
    :type: ``float``
    :default: ``5/3``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the exponent used to calculate
    the electron pressure (see :ref:`here <theory-hybrid-model-elec-temp>`).

.. pp:param:: hybrid_pic_model.include_hall_term
    :type: ``bool``
    :default: ``true``
    :optional:

    Controls the Hall contribution :math:`\vec{J}\times\vec{B}/(e n_e)` in
    generalized Ohm's law. Setting this to ``false`` retains
    :math:`-\vec{u}_i\times\vec{B}` and selects the non-Hall, resistive-MHD
    limit. This switch does not disable the electron-pressure or resistive
    terms.

.. pp:param:: hybrid_pic_model.include_electron_pressure_term
    :type: ``bool``
    :default: ``true``
    :optional:

    Controls the electron-pressure contribution
    :math:`-\nabla P_e/(e n_e)` in generalized Ohm's law independently of
    the Hall and resistive terms.

.. pp:param:: hybrid_pic_model.include_electron_inertia
    :type: ``bool``
    :default: ``false``
    :optional:

    Add the electron-inertia term to the generalized Ohm's law, following
    the implicit-PIC formulation of Angus et al.:

    .. math::

        \vec{E}_\mathrm{inertial} = \frac{m_{e,\mathrm{eff}}}{e\rho}
        \left[ \frac{\partial \vec{J}_e}{\partial t}
        - \frac{\vec{J}_e}{\rho}\frac{\partial \rho}{\partial t}
        - (\vec{J}_e\cdot\nabla)\frac{\vec{J}_e}{\rho} \right],

    the :math:`\vec{J}_e`-form of the material derivative
    :math:`-(m_{e,\mathrm{eff}}/e)\,D\vec{u}_e/Dt` with
    :math:`\vec{u}_e = -\vec{J}_e/\rho` (cold-electron limit:
    :math:`\vec{E} = (m_e/(e^2 n_e))\,\partial\vec{J}_e/\partial t`, the
    collisionless electron inductance). The term is REACTIVE: it rolls the
    whistler branch over at the effective electron skin depth
    :math:`d_e = c/\omega_{pe}(m_{e,\mathrm{eff}})` --
    :math:`\omega \to \omega/(1 + k^2 d_e^2)` -- so under-resolved
    grid-scale whistlers saturate instead of stiffening as :math:`k^2`.
    The time derivative is assembled at the theta stage from the
    theta-extrapolated iterate and two per-step-frozen electron-current
    history levels, seeded from the first evaluated state. Density-floored
    cells keep their inertia (the Ohm density floor bounds the
    :math:`1/n_e`); true vacuum drops it. Requires an implicit evolve
    scheme (``theta_implicit_hybrid``, or ``theta_implicit_mhd`` with
    ``implicit_mhd.fluid_flux = central`` and ``include_hall_term = true``,
    where the term also joins the ``pc_mhd_block`` preconditioner rows).
    Restarts are not supported yet (the current history is not
    checkpointed). GMRES guidance for heavy effective mass ratios: the
    inertia Jacobian block scales as :math:`(2\theta + 1)/2\,(k d_e)^2`
    and is dt-independent, so decks with large :math:`d_e/\Delta x`
    (small ``reduced_electron_mass_ratio``) want a matched restart length
    -- ``gmres.restart_length = 200`` with ``gmres.max_iterations = 400``
    measured well on the hybrid side; the defaults are not changed.

.. pp:param:: hybrid_pic_model.reduced_electron_mass_ratio
    :type: ``float``
    :default: ``0``
    :optional:

    When > 0, the effective electron mass of the inertia term is the ion
    mass divided by this ratio (the lightest charged particle species on
    the hybrid path; the fluid ion mass :math:`q_e/(q/m)` under
    ``theta_implicit_mhd``). 0 selects the physical electron mass. The
    ratio tunes where the electron-scale dispersion acts relative to the
    grid: :math:`d_e \propto 1/\sqrt{R}` grows as the ratio shrinks.

.. pp:param:: hybrid_pic_model.electron_inertia_bdf2
    :type: ``bool``
    :default: ``true``
    :optional:

    Use the second-order three-point stencil
    :math:`\{(2\theta-1)/2,\,-2\theta,\,(2\theta+1)/2\}/\Delta t` on
    :math:`\{\vec{J}_e^{n-1}, \vec{J}_e^n, \vec{J}_e^{n+1}\}`, centered at
    the theta stage where Ohm's law is imposed (classic BDF2 at
    :math:`\theta = 1`, exactly the two-point midpoint form at
    :math:`\theta = 1/2`), for the inertial time derivative. ``false``
    selects the two-point form at every theta.

.. pp:param:: hybrid_pic_model.electron_inertia_linear_below
    :type: ``float``
    :unit: :math:`\mathrm{kg\,m^{-3}}`
    :default: ``0``
    :optional:

    Dust gate for the electron-inertia assembly: below this mass density
    the term keeps ONLY its linear :math:`\partial\vec{J}_e/\partial t`
    history-stencil piece -- exactly the response the ``pc_mhd_block``
    inertia rows fold, so residual and preconditioner agree there by
    construction -- and drops the nonlinear density-convection
    :math:`-(\vec{J}_e/\rho)\,\partial\rho/\partial t` and advection
    :math:`-(\vec{J}_e\cdot\nabla)(\vec{J}_e/\rho)` pieces. Full form
    above the threshold, C-infinity blend
    :math:`w = (1 + \tanh(s(1+s^2)))/2` with
    :math:`s = (\rho - \rho_c)/(0.3\rho_c)` between (central width
    :math:`0.3\rho_c`; the cubic-sharpened tail leaves the full-form
    weight within :math:`e^{-606}` of 1 at :math:`3\rho_c`, and exactly 1
    in double precision beyond :math:`{\sim}1.77\rho_c`, so
    above-threshold physics is bit-identical to the ungated assembly).
    The gate is evaluated on the same theta-stage density the term
    already divides by, keeping the residual smooth for the matrix-free
    Jacobian probes. ``0`` disables the gate.

    Rationale: at the Ohm floor density the effective electron skin depth
    is grid-scale (:math:`d_e \sim 1.2\,\Delta x` at
    :math:`\rho/m_i \sim 3\times 10^{17}\,\mathrm{m}^{-3}` on a
    formation-class deck), so the inertia term is :math:`O(1)` across the
    entire near-floor dust halo -- where the nonlinear pieces scale like
    :math:`1/\rho^2`, are NOT representable in the preconditioner's
    frozen linear fold, and carry no physics (the dust has no meaningful
    electron fluid). Measured on such a deck the ungated nonlinear pieces
    grow the preconditioned GMRES solve cost monotonically from 0.5 s to
    a 78 s peak (5.2 s median) per solve and Newton iteration counts from
    3-4 to 14 as the drive ramp expands the dust sea; gating them
    restores PC/residual agreement below the threshold. Set the gate at
    or a few times above the Ohm-floor mass density, well below the
    plasma densities whose electron dynamics matter.

.. _running-cpp-parameters-implicit-mhd:

.. rubric:: Theta-implicit ion-fluid MHD

The parameters below are used when
:pp:param:`algo.evolve_scheme = theta_implicit_mhd`. This mode also requires
:pp:param:`algo.maxwell_solver = hybrid` and uses the Hall, electron-pressure,
resistivity, and hyper-resistivity controls from ``hybrid_pic_model``. It
currently requires ``implicit_evolve.nonlinear_solver = newton`` so the same
fluid admissibility bounds can be applied to Newton updates and matrix-free
Jacobian probes.

.. warning::

    The implementation supports one AMR level, one ion fluid, and no kinetic
    or regular WarpX fluid species. Cartesian domains must be periodic.
    Cylindrical (RZ, ``m = 0``) domains require a non-periodic radial
    direction whose upper boundary is either ``pec`` (conducting wall; the
    fluid is reflected there) or ``open`` (the Green's-function free-space
    boundary, evaluated inside every JFNK residual; the fluid gets
    zero-gradient outflow ghosts there, and curl-free applied fields must
    enter through the split external vector potential); non-periodic z ends
    get the solver's own Neumann outflow ghost fills and require
    ``boundary.field_lo/hi = none`` in z, or ``open`` per z face to give the
    FIELD z ghosts of that cap the Green's-function free-space fill instead
    (the fluid moments keep the Neumann outflow ghosts). Embedded
    boundaries, runtime load balancing, hybrid external currents,
    Holmstrom vacuum regions, particle mass matrices, and existing field-only
    preconditioners are not supported. The prototype ``pc_mhd_block``
    preconditioner is available only with staggered fields, the
    ``legacy_e_centered`` fluid flux, and zero hyper-resistivity. It supports evolving or frozen ions and
    permits the Hall and electron-pressure Ohm terms, but its first Hall block
    is a constant-coefficient spectral-magnitude approximation rather than a
    full multidimensional electron-subsystem solve. Its scalar reference
    resistivity may be constant or time dependent, but not density, current,
    or species dependent. The Rusanov option regularizes fluid discontinuities
    but is not a total-energy conservative MHD shock solver.

.. pp:param:: implicit_mhd.reference_mass_density
    :type: ``float``
    :unit: :math:`\mathrm{kg\,m^{-3}}`

    Required positive mass-density scale used to normalize the JFNK state.

.. pp:param:: implicit_mhd.reference_magnetic_field
    :type: ``float``
    :unit: T

    Required positive magnetic-field scale used to normalize the JFNK state.

.. pp:param:: implicit_mhd.reference_velocity
    :type: ``float``
    :unit: :math:`\mathrm{m\,s^{-1}}`
    :optional:

    Velocity scale used for nonlinear-state normalization. The default is the
    Alfvén speed computed from ``reference_mass_density`` and
    ``reference_magnetic_field``.

.. pp:param:: implicit_mhd.reference_ion_pressure
    :type: ``float``
    :unit: Pa
    :default: ``0``

    Ion pressure at ``reference_mass_density``. The barotropic ion closure is
    :math:`P_i=P_{i0}(\rho_i/\rho_{i0})^{\gamma_i}`. Not used by the
    ``total_energy`` closure.

.. pp:param:: implicit_mhd.ion_charge_to_mass
    :type: ``float``
    :unit: :math:`\mathrm{C\,kg^{-1}}`
    :default: proton charge-to-mass ratio

    Charge-to-mass ratio of the single ion fluid.

.. pp:param:: implicit_mhd.gamma_e
    :type: ``float``
    :default: ``5/3``

    Electron adiabatic index. The evolved electron pressure is
    :math:`P_e=(\gamma_e-1)U_e`.

.. pp:param:: implicit_mhd.gamma_i
    :type: ``float``
    :default: ``5/3``

    Ion adiabatic index: the exponent of the barotropic ion-pressure closure,
    and the ratio of specific heats of the ``total_energy`` closure (which
    requires :math:`\gamma_i>1`).

.. pp:param:: implicit_mhd.ion_closure
    :type: ``string``
    :default: ``barotropic``

    Ion thermodynamic closure. ``barotropic`` evaluates
    :math:`P_i=P_{i0}(\rho_i/\rho_{i0})^{\gamma_i}` from
    :pp:param:`implicit_mhd.reference_ion_pressure`. ``total_energy`` evolves
    the conservative ion total energy density
    :math:`E_i=U_i+\rho_i|\boldsymbol{u}_i|^2/2` as an additional JFNK
    unknown, in enthalpy form (the flux carries the same total pressure as
    the momentum equation),

    .. math::

        \partial_t E_i +
        \nabla\cdot\left[\boldsymbol{u}_i(E_i+P_i+P_e)\right] =
        \boldsymbol{u}_i\cdot(\boldsymbol{J}\times\boldsymbol{B})
        + P_e\,\nabla\cdot\boldsymbol{u}_e,

    with :math:`P_i=(\gamma_i-1)\max(E_i-\rho_i|\boldsymbol{u}_i|^2/2,
    U_{i,\mathrm{floor}})`, so shock and compressional heating are captured
    conservatively by the same face fluxes that transport mass and momentum.
    The :math:`+P_e\nabla\cdot\boldsymbol{u}_e` source is evaluated on the
    same faces as the electron equation's
    :math:`-P_e\nabla\cdot\boldsymbol{u}_e` work, so the pair cancels
    discretely and :math:`\sum(E_i+U_e)` is conserved exactly on a periodic
    mesh (away from the floor limiters). Requires
    :pp:param:`implicit_mhd.evolve_ion_fluid` ``= true``, a positive
    :pp:param:`implicit_mhd.ion_pressure_floor`, and :math:`\gamma_i>1`; the
    ``pc_mhd_block`` preconditioner supports only the barotropic closure.

    ``dual_energy`` (``fluid_flux = hlld`` or ``central`` only) is the
    the reference code's "mixmaster" (mix = -1) split-energy form for the ions: it
    evolves BOTH the conservative total :math:`E_i` (assembled exactly as
    under ``total_energy``) and an auxiliary internal ion energy
    :math:`U_i` as JFNK unknowns, and every consumer of the ion pressure
    (momentum gradient and Maxwell-stress-adjacent gas pressure, wave-fan
    signal speeds, flux functions, drain-gate scales, conduction
    temperatures) sees the kinetic-fraction blend

    .. math::

        P_i = f_k\,(\gamma_i-1)\,\mathrm{smoothmax}(E_i - K, U_{i,\mathrm{floor}})
              + (1-f_k)\,(\gamma_i-1)\,U_i,
        \qquad f_k = 1 - \frac{K}{\gamma_i E_i}

    (:math:`K = \rho_i|\boldsymbol{u}_i|^2/2`; :math:`f_k` smoothly
    rectified to :math:`[0,1]`). Kinetic-energy-dominated cells
    (:math:`f_k \to 0`: imploding sheets, exhaust flow, stagnation on
    axis) read their pressure from :math:`U_i` and never see the
    total-minus-kinetic catastrophic-cancellation heating that leaves
    spurious warm plasma there under ``total_energy``. :math:`U_i`
    advects with the contact (the fan's star channel), gains the blended
    PdV work :math:`-P_i\nabla\cdot\boldsymbol{u}_i` pointwise, and
    receives the internal-only share of every source the :math:`E_i`
    channel books (viscous heating as the positive-definite face
    dissipation, the ion conductive flux verbatim, the pedestal-band
    internal drain, the floor machinery); KE-specific terms (Lorentz
    work, the drag's kinetic drain, the electron pdV pairing) stay with
    :math:`E_i` alone. At every accepted step end the reference code's mixmaster
    re-sync rewrites :math:`E_i := K + P_i/(\gamma_i-1)` (the identity
    in thermal cells; in KE-dominated cells it drains the step's
    cancellation drift into the :math:`U_i` anchor instead of letting it
    accumulate in the conservative :math:`E_i`), and thermal cells
    (:math:`f_k >` :pp:param:`implicit_mhd.dual_energy_sync_threshold`)
    additionally overwrite :math:`U_i := E_i - K`, the standard
    Enzo-style dual-energy re-sync. Same requirements as
    ``total_energy``.

    ``cgl`` (``fluid_flux = hlld`` only) evolves the CGL bi-Maxwellian
    ion internal energies :math:`U_\parallel = p_\parallel/2` and
    :math:`U_\perp = p_\perp` as two JFNK unknowns instead of
    :math:`E_i` — the pressure recovery is then LINEAR in the state (no
    kinetic-energy subtraction), which removes the cold-ion Newton
    degeneracy of near-stagnant kinetic-dominated cells. The Riemann
    fan carries the effective isotropic pressure
    :math:`p_\mathrm{eff}=(p_\parallel+2p_\perp)/3`; the internal
    energies ride the fan as contact-advected star channels; the
    trace-free deviation stress
    :math:`(p_\parallel-p_\perp)(\hat b\hat b - I/3)` enters the
    momentum equation pointwise (with the cylindrical geometric terms
    in RZ); and the CGL work terms use the full axisymmetric covariant
    velocity-gradient contraction. Ion--ion isotropization relaxes
    :math:`p_\perp - p_\parallel` at
    ``cgl_relaxation_scale`` :math:`\times \nu_{ii}` (Braginskii, with
    ``cgl_coulomb_log``), conserving :math:`U_\parallel + U_\perp`;
    a calibrated CI test pins the implemented rate to the analytic
    :math:`\theta`-discrete decay at machine precision. The optional
    ``ion_pressure_anisotropy(x,y,z)`` parser sets the initial
    :math:`p_\perp/p_\parallel` (default 1) with
    :math:`p_\mathrm{eff}` pinned to ``ion_pressure(x,y,z)``.

.. pp:param:: implicit_mhd.dual_energy_internal_cutoff
    :type: ``float``
    :default: ``0`` (off)

    With ``ion_closure = dual_energy``: the reference code's mix = -2 low-internal
    cutoff. Cells whose STEP-OLD :math:`U_i` sits below
    ``cutoff`` :math:`\times\max_\mathrm{domain}(U_{i,\mathrm{old}})`
    get :math:`f_k = 0` exactly (pure internal pressure in the
    halo/dust). Both the mask input and the max are per-step frozen
    (step-old) constants, so the branch is state-independent for the
    matrix-free Jacobian probes.

.. pp:param:: implicit_mhd.dual_energy_sync_threshold
    :type: ``float``
    :default: ``0.99``

    With ``ion_closure = dual_energy``: :math:`f_k` threshold of the
    end-of-step Enzo-style re-sync :math:`U_i := E_i - K` (thermal cells
    only, where the blend is total-dominated and the recovery is well
    conditioned), so the two energy variables cannot drift apart in
    thermal regions. The mixmaster :math:`E_i` rewrite (see
    ``ion_closure``) runs unconditionally.

.. pp:param:: implicit_mhd.cgl_relaxation_scale
    :type: ``float``
    :default: ``1.0``

    Multiplier on the Braginskii ion-ion isotropization rate of the
    ``cgl`` closure (0 disables relaxation; values above 1 model
    anomalous/instability-driven isotropization).

.. pp:param:: implicit_mhd.cgl_coulomb_log
    :type: ``float``
    :default: ``10.0``

    Coulomb logarithm of the ``cgl`` isotropization rate.

.. pp:param:: implicit_mhd.cgl_instability_scale
    :type: ``float``
    :default: ``0.1``

    Instability-bounded relaxation of the ``cgl`` closure: past the
    firehose bound (:math:`p_\parallel - p_\perp > B^2/\mu_0`) or the
    mirror bound (:math:`\beta_\perp(p_\perp/p_\parallel - 1) > 1`),
    kinetic instabilities isotropize on cyclotron timescales that no
    collisional rate represents; the relaxation rate gains
    ``cgl_instability_scale`` :math:`\times\,\Omega_{ci}` through
    :math:`C^\infty` switches in the dimensionless threshold measures,
    holding the anisotropy near marginal stability. ``0`` disables the
    bounds (pure collisional relaxation, which lets the anisotropy grow
    to unphysical values in collisionless regions).

.. pp:param:: implicit_mhd.cgl_instability_width
    :type: ``float``
    :default: ``0.1``

    Relative :math:`C^\infty` switch width of the instability bounds.
    Also sets the relative knee width of the marginal-stability clamp on
    the *transmitted* deviation stress: the anisotropy entering the
    stress divergence and the work terms is soft-clipped (:math:`C^2`,
    exact inside the band) to the firehose marginal value
    :math:`p_\parallel - p_\perp \le B^2/\mu_0` and the mirror marginal
    value :math:`p_\perp - p_\parallel \le (B^2/\mu_0)\,
    p_\parallel/(2 p_\perp)` — kinetic instabilities cap the stress a
    plasma can sustain, and the bounded relaxation alone cannot stop a
    transient super-marginal excursion from transmitting unbounded force
    through the dissipation-free stress divergence.

.. pp:param:: implicit_mhd.cgl_null_scale
    :type: ``float``
    :default: ``1.0``

    Magnetization-weighted null blend of the ``cgl`` closure: with the
    :math:`C^\infty` weight
    :math:`w = \Omega_{ci}^2\,\Delta x^2 / (\Omega_{ci}^2\,\Delta x^2 +
    v_{th,i}^2) = 1/(1 + (r_L/\Delta x)^2)` (:math:`\Delta x` the
    smallest cell size; :math:`w = 0` exactly at field nulls), the
    deviation stress carries :math:`w\,(p_\parallel - p_\perp)`, the
    work terms blend :math:`w` CGL :math:`+\,(1-w)` isotropic — the
    total internal work stays exactly :math:`-P_\mathrm{blend} :
    \nabla u` with :math:`P_\mathrm{blend} = p_\mathrm{eff} I +
    w\,(p_\parallel - p_\perp)(\hat b \hat b - I/3)`, the same stress
    the momentum equation integrates — and the relaxation rate gains
    ``cgl_null_scale`` :math:`\times\,(v_{th,i}/\Delta x)(1-w)`, the
    cell-transit rate of unmagnetized ions. The gyrotropic closure has
    no meaning where the gyroradius is unresolved (in particular at
    field nulls, where the :math:`\Omega_{ci}`-scaled instability
    bounds vanish by construction); the blend degenerates it smoothly
    to isotropic MHD there. ``0`` disables the added relaxation rate
    (the :math:`w` weighting of stress and work always applies under
    ``cgl``).

.. pp:param:: implicit_mhd.electron_ion_equilibration
    :type: ``float``
    :default: ``0`` (off)

    Collisional electron-ion temperature equilibration — the reference code's
    ``eq_brate`` exchange (``ntb.f90`` 658–700, active whenever the reference code
    runs two-temperature mode) — as a scale on the reference code's rate
    (``1`` = the reference code's coefficient exactly; the OFF path performs no
    arithmetic at all). The exchange is the single energy-conserving
    pair term

    .. math::

        \frac{dU_e}{dt} = +\nu\,(p_i - p_e) = -\frac{dU_i}{dt},
        \qquad
        \nu = 4.75\times 10^{-15}\,
        \ln\Lambda\,\frac{m_p}{m_i}\,
        \frac{n_i}{T_e[\mathrm{eV}]^{3/2}}\ \mathrm{s}^{-1},

    with the reference code's NRL-style :math:`\ln\Lambda` branches
    (:math:`23 + 3\ln 10 - \tfrac12\ln n + \tfrac32\ln T_e` for
    :math:`T_e \le e^2` eV, else
    :math:`24 + 3\ln 10 - \tfrac12\ln n + \ln T_e`; :math:`n` in
    :math:`\mathrm{m}^{-3}`, :math:`Z=1`). The rate is FROZEN per solve
    from the step-old density and electron pressure (the
    ``conduction_coefficient_state = step_old`` idiom — the pair term
    stays Newton-linear in the live energies; the reference code's backward-Euler
    pair update with the coefficient at its P/C iterate is identical to
    first order) and capped at the :math:`\theta`-scheme monotonicity
    bound :math:`1/((\gamma_e + \gamma_i - 2)(1-\theta)\,\Delta t)`.
    Deposits: :math:`+Q` to :math:`U_e`; :math:`-Q` to the ion channel
    of the active closure (:math:`E_i` under ``total_energy``; both
    :math:`E_i` and the auxiliary :math:`U_i` under ``dual_energy``;
    the isotropic split :math:`-Q/3` to :math:`U_\parallel` and
    :math:`-2Q/3` to :math:`U_\perp` under ``cgl``), under the same
    symmetric source envelope in both rows, so the species pair sum is
    conserved to round-off by construction. Requires an ion closure
    that evolves an ion energy block.

.. pp:param:: implicit_mhd.mass_density_floor
    :type: ``float``
    :unit: :math:`\mathrm{kg\,m^{-3}}`
    :default: ``1.e-12 * reference_mass_density``

    Positive floor used as the admissibility lower bound for nonlinear
    updates and as a pure div-by-zero guard in the fluid kernels. It is
    deliberately independent of :pp:param:`hybrid_pic_model.n_floor`: the
    Ohm's-law, resistivity, and electron-temperature evaluations floor
    their density inputs at that hybrid guard instead, so a large Ohm
    guard does not pin low-density halo cells at the Newton solver's
    admissibility bound.

.. pp:param:: implicit_mhd.halo_pedestal_fraction
    :type: ``float``
    :default: ``0`` (off)

    Offset-density (pedestal) treatment of the low-density halo, for
    Riemann fluid fluxes (``rusanov``, ``hllc``, ``hlld``). A positive
    fraction :math:`f` defines a dynamic pedestal STATE -- an
    :math:`f`-scaled image of the instantaneous peak state, recomputed
    once per step and frozen for the whole nonlinear solve: the density
    pedestal :math:`\rho_\mathrm{ped} = f\,\max(\rho_\mathrm{peak},
    \rho_\mathrm{ref})` plus an energy pedestal per evolved block
    (:math:`f` times the instantaneous peak of :math:`U_e`, of the ion
    internal energy under ``total_energy``, and of
    :math:`U_\parallel`/:math:`U_\perp` under ``cgl``). Every step
    starts by raising sub-pedestal-density cells onto the pedestal state
    (tracked non-conservation of the same class as the positivity
    floors), and the donor drain gates of each block anchor at its
    pedestal value instead of its floor, making the pedestal band
    dynamically invariant. This is an equivalent reformulation of
    advecting the *deviation* above a pedestal with the pedestal
    restored arithmetically for all derived quantities: in both forms
    the halo operates at a regular interior point of the admissible set
    -- the Newton admissibility bounds stay at the
    (orders-of-magnitude lower) positivity floors and are strictly
    inactive there, so no cell rides a bound. Bounded-Newton solves on
    magnetized floor-riding halos otherwise clamp tens of thousands of
    direction components per solve and stagnate on the bound-resident
    population; a mass-only pedestal merely moves that population into
    the energy blocks (measured: ~16k electron-energy plus ~3k
    ion-energy floor-resident components per solve, line search frozen),
    which is why the pedestal is a full state. Requires
    :math:`f\,\rho_\mathrm{ref} >` :pp:param:`implicit_mhd.mass_density_floor`.
    Reactive work and relaxation sources additionally taper
    :math:`C^1`-smoothly to zero below :math:`2\rho_\mathrm{ped}` (the
    pedestal is numerical mass with no reactive response of its own).

.. pp:param:: implicit_mhd.halo_pedestal_drag_rate
    :type: ``float``
    :unit: 1/s
    :default: ``0`` (off)

    Velocity relaxation (drag) rate of the pedestal band. Requires a
    positive :pp:param:`implicit_mhd.halo_pedestal_fraction`. The drag
    engages :math:`C^1`-smoothly below twice the density pedestal --
    full rate at :math:`\rho_\mathrm{ped}`, exactly zero at and above
    :math:`2\rho_\mathrm{ped}`, keyed to the step-old density (a
    per-solve constant mask) -- adding the diagonal, smooth momentum
    sink :math:`-\nu\,\rho\mathbf{u}` of the vacuum-dust drag pattern.
    Rationale: the pedestal's donor drain gates hold the band's mass
    flux at zero, but the band's momentum rows still integrate the
    discrete-stress truncation forcing of the surrounding plasma -- a
    residual demand the gated mass cannot supply, which otherwise pins
    the Newton solve at an irreducible least-squares plateau. The drag
    bounds the band at the terminal velocity :math:`F/(\rho\nu)`
    instead. Under the ``total_energy`` closure a matched
    kinetic-energy drain (:math:`-\nu\,|\mathbf{m}|^2/\rho`) keeps the
    ion internal energy invariant (discretely exact at
    :math:`\theta = 1/2`). A sensible rate is the ion cyclotron scale
    (of order :math:`0.1/\Delta t` for typical implicit MHD steps).

.. pp:param:: implicit_mhd.halo_pedestal_energy_rate
    :type: ``float``
    :unit: 1/s
    :default: ``0`` (off)

    Ion-energy relaxation rate of the pedestal band. Requires a
    positive :pp:param:`implicit_mhd.halo_pedestal_fraction` and
    ``implicit_mhd.ion_closure = total_energy`` or ``cgl``. Engages
    with a :math:`C^1` density mask of the drag's form but an octave
    narrower (full rate at and below :math:`1.125\rho_\mathrm{ped}`,
    exactly zero at and above :math:`1.25\rho_\mathrm{ped}`, keyed to
    the step-old density): the treatable population is the
    gate-pinned band, while the wall's live hot boundary layer (which
    cycles through :math:`1.3\text{-}1.8\,\rho_\mathrm{ped}` with an
    ion-energy stock far above the pedestal image) must not be
    drained.
    Rationale (the last free band channel, after the mass gates and
    :pp:param:`implicit_mhd.halo_pedestal_drag_rate`): under a ramping
    drive the band's density sits gate-pinned at the pedestal while
    its ion-energy rows keep integrating the drive forcing; the
    accreted energy demands a mass drain the gates refuse, which
    otherwise pins the Newton solve at an irreducible ion-energy
    residual. The diagonal relaxation drains the band's ion energy
    toward its pedestal-consistent image instead. Under the
    ``total_energy`` closure it acts only on the internal part
    :math:`e = E_i - |\mathbf{m}|^2/(2\rho)`, while the momentum drag
    owns the kinetic channel through its matched drain -- the
    composition is triangular in (kinetic, internal) and the two terms
    never double-count. The drain is one-sided (rectified): a
    :math:`C^1` gate closes it where the internal part sits at or
    below the pedestal image (full exact rate at and above twice it),
    so the term can never act as an energy *source* -- a two-sided
    KE-following form pumps kinetically dominated band cells toward
    their growing kinetic target. Under the ``cgl`` closure
    :math:`U_\parallel` and :math:`U_\perp` (purely internal) drain
    toward their pedestal images directly, with the same one-sided
    gates. Targets are clamped from below at the corresponding
    positivity floors. A sensible rate matches the drag rate (ion
    cyclotron scale, of order :math:`0.1/\Delta t` for typical
    implicit MHD steps).

.. pp:param:: implicit_mhd.advection_density_offset_fraction
    :type: ``float``
    :default: ``0`` (off, bit-identical)

    Offset-density advection: the fraction :math:`f_\mathrm{off}` of the
    shared vacuum reference density defining the advective background

    .. math::

       \rho_\mathrm{off} = f_\mathrm{off}\,\rho_\mathrm{ref},

    with :math:`\rho_\mathrm{ref}` the same reference the vacuum
    resistivity and the conduction halo boost key to (the Ohm density
    guard, raised by :pp:param:`implicit_mhd.vacuum_reference_base_density`
    and the per-step frozen
    :pp:param:`implicit_mhd.vacuum_reference_peak_fraction` times the
    density peak). The **advective mass flux only** then transports

    .. math::

       D(\rho) = \mathrm{smooth\text{-}max}(\rho - \rho_\mathrm{off},\,0)

    -- the rectified *perturbation* -- in place of :math:`\rho`. This is the flux-form transliteration of
    the subtract / advect / re-add density advance
    (:math:`n \to n - f_\mathrm{off}\rho_\mathrm{ref}`, advect,
    :math:`\max(\cdot, 0)`, add back), whose stated purpose is to
    "remove the floor value to reduce noise in the low density region":
    advecting the perturbation about a non-zero background is
    numerically cleaner than advecting the small, noisy absolute
    density. Both the source scheme's advance and every mass flux
    channel of the recast fluxes here are *linear* in the cell
    densities, and :math:`\rho_\mathrm{off}` is a constant field, so
    subtracting the offset from the state and adding it back afterwards
    cancels identically -- the only surviving arithmetic is the
    non-negativity clamp on the perturbation, which is what :math:`D`
    applies. Consequences:

    * **Exactly conservative.** The offset enters only as the density
      argument of a face flux, so the divergence still telescopes and
      total mass changes solely through the domain boundary.
    * **Exact clamp tail.** At and above :math:`1.2\,\rho_\mathrm{off}`
      the smoothing is exactly inactive (a :math:`C^1` gate whose value
      and slope both vanish there): :math:`D = \rho -
      \rho_\mathrm{off}` with no smoothing residue surviving into the
      plasma.
    * **Advectively static halo.** As :math:`\rho \to 0` the
      transported density tends to zero: a sub-offset cell has no
      advective mass flux at all, so sub-offset density structure
      produces none of the dispersive ripple a centred flux would
      otherwise amplify there.
    * **Price, stated plainly.** Above the offset the flux is the
      legacy flux *minus the background's own flux*
      :math:`\rho_\mathrm{off}\mathbf{u}`, so the core mass transport
      loses the term
      :math:`\rho_\mathrm{off}\,\nabla\!\cdot\!\mathbf{u}`: the
      numerical background does not compress with the plasma. That is
      identically zero for divergence-free flow and
      :math:`O(\rho_\mathrm{off}/\rho)` otherwise, which is why the
      offset must be kept far below the plasma density. It is the
      source scheme's own behaviour, not a deviation from it.
    * **Mass channel only.** The momentum and energy channels advect at
      the velocity :math:`u = \mathbf{m}/\rho` recovered from the full
      density, exactly as in the source scheme, and carry no explicit
      density factor to offset.

    The source scheme's in-loop :math:`\max(n, 0)` clamp is carried by
    the rectifier itself, not by the drain gates: :math:`D` vanishes
    below the offset, so a sub-offset donor has no advective outflow
    left for a gate to close, and the mass donor drain gate is
    deliberately left at its floor/pedestal anchors. Note that with a
    halo's mass flux switched off but its momentum still live, the
    halo's *temperature* (advected energy over frozen mass) can drift;
    the existing halo machinery
    (:pp:param:`implicit_mhd.halo_pedestal_drag_rate`,
    :pp:param:`implicit_mhd.halo_relaxation_rate`) is what bounds that.
    Orthogonal to, and composable with,
    :pp:param:`implicit_mhd.halo_pedestal_fraction`, which raises the
    *state* onto a pedestal once per step (tracked non-conservation);
    this knob never touches the state. Requires
    ``implicit_mhd.fluid_flux = central`` or ``hlld`` and
    :math:`f_\mathrm{off}\,\rho_\mathrm{ref} >`
    :pp:param:`implicit_mhd.mass_density_floor`.

.. pp:param:: implicit_mhd.halo_relaxation_rate
    :type: ``float``
    :unit: 1/s
    :default: ``0`` (off, bit-identical)

    Windowed halo temperature-relaxation outlet: relaxes BOTH species'
    internal energies toward the prescribed cold-medium temperature
    :pp:param:`implicit_mhd.halo_relaxation_temperature` at this rate,
    in the halo density window below
    :pp:param:`implicit_mhd.halo_relaxation_n_max` -- a model of the
    halo dissipating into some medium. The window is :math:`C^1` in the
    step-old density (a per-solve constant mask, the pedestal-energy
    ramp form transplanted to the fixed upper edge
    :math:`\rho_\mathrm{max} = n_\mathrm{max} m_i`): full rate at and
    below :math:`1.125\,\rho_\mathrm{max}`, exactly zero at and above
    :math:`1.25\,\rho_\mathrm{max}`; the lower edge is the existing
    positivity/pedestal floor machinery (targets are clamped from below
    at the positivity-floor images, and the pedestal band keeps its own
    relaxation). Per-channel targets are the cold-medium energies
    :math:`n k_B T_\mathrm{med}` at the LOCAL step-old density:
    :math:`U_e \to \rho\,(q/m)\,T_\mathrm{med}/(\gamma_e - 1)`, the ion
    internal part :math:`e = E_i - |\mathbf{m}|^2/(2\rho) \to
    \rho\,(q/m)\,T_\mathrm{med}/(\gamma_i - 1)` under ``total_energy``
    (the auxiliary :math:`U_i` mirrors it under ``dual_energy``), and
    :math:`U_\parallel \to \rho\,(q/m)\,T_\mathrm{med}/2`,
    :math:`U_\perp \to \rho\,(q/m)\,T_\mathrm{med}` under ``cgl`` (the
    barotropic closure relaxes the electron channel only). The drains
    are one-sided (rectified): a :math:`C^1` gate closes each one where
    its energy sits at or below the target (full exact rate at and
    above twice it), so the outlet only cools toward the medium and
    never heats a sub-target cell. Part of the JFNK residual
    (theta-weighted, per-solve frozen coefficients). The removed energy
    is booked per step (see
    :pp:param:`implicit_mhd.halo_relaxation_ledger_file`). Keep
    :math:`\nu\,\Delta t \lesssim 5\times 10^{-3}` (non-stiff).

.. pp:param:: implicit_mhd.halo_relaxation_target
    :type: ``string``
    :default: ``temperature``

    Target mode of the halo relaxation outlet. ``temperature`` (the
    default) relaxes both species toward the fixed cold-medium
    temperature above. ``ion`` is the reference code's :math:`t_e \le t_m` one-way
    valve (``vp.f90:731``, ``te = MIN(te, tm)``, the reference-shot-flown
    Te-runaway breaker) in relaxed, JFNK-safe form: ONLY the electron
    energy drains, toward the LOCAL ion-temperature image
    :math:`n k_B T_i/(\gamma_e - 1)` with :math:`T_i` read from the
    step-old ion energy state (:math:`p_i = (\gamma_i - 1)(E_i -
    |\mathbf{m}|^2/(2\rho))` under ``total_energy``/``dual_energy``,
    :math:`p_\mathrm{eff} = (2 U_\parallel + 2 U_\perp)/3` under
    ``cgl``; a per-solve frozen coefficient), through the same window,
    rate, one-sided gate, floor clamp, and ledger; the ion channels are
    untouched. At high rate this reproduces the reference code's clip without its
    discontinuity. Requires an ion closure with an energy channel;
    :pp:param:`implicit_mhd.halo_relaxation_temperature` is unused.

.. pp:param:: implicit_mhd.halo_relaxation_temperature
    :type: ``float``
    :unit: eV
    :default: ``0``

    Cold-medium temperature the halo energies relax toward (required
    positive when :pp:param:`implicit_mhd.halo_relaxation_rate` is on
    with the ``temperature`` target). Typical bath values are of order
    1 eV.

.. pp:param:: implicit_mhd.halo_relaxation_n_max
    :type: ``float``
    :unit: m^-3
    :default: ``0``

    Upper edge of the halo relaxation density window, as a number
    density of the quasi-neutral single-ion fluid (required positive
    when :pp:param:`implicit_mhd.halo_relaxation_rate` is on). Cells
    above :math:`1.25\,n_\mathrm{max}` (step-old) are exactly
    untouched.

.. pp:param:: implicit_mhd.halo_relaxation_ledger_file
    :type: ``string``
    :default: *none*

    File for the halo relaxation ledger (requires a positive
    :pp:param:`implicit_mhd.halo_relaxation_rate`): "step energy" rows
    appended every step with the cumulative removed fluid energy [J]
    evaluated at the accepted theta state -- the conservation
    instrument of the outlet, following the wall-ledger contract (a
    converged solve books the removal to the nonlinear solver
    tolerance).

.. pp:param:: implicit_mhd.floor_consistency_width_fraction
    :type: ``float``
    :default: ``0.1``

    Rectifier width of the floor-consistency source as a fraction of
    each cell's bound (range ``(0, 0.5]``). The bound-riding supply
    capacity is :math:`\nu_\text{eff}\, w/2`, so raising the width
    raises how much persistent sub-bound demand the source can meet per
    solve, at the cost of a wider smooth engagement band (the
    exact-zero guarantee moves to :math:`(1 + 2f)` times the bound).

.. pp:param:: implicit_mhd.floor_consistency_rate
    :type: ``float``
    :unit: 1/s
    :default: ``0`` (off, bit-identical)

    Floor-consistency relaxation rate: the consistency *completion* of
    the fluid admissibility floors. Every bounded fluid block (the same
    blocks the bounded Newton projection clamps: mass density, electron
    energy, ion energy under ``total_energy``, :math:`U_\parallel` and
    :math:`U_\perp` under ``cgl``) gains the one-sided per-cell residual
    source

    .. math:: S = \min(\nu, 1/(\theta\,\Delta t)) \,
        \mathrm{rect}_w(b - u),

    where :math:`u` is the theta-stage block value, :math:`b` is the
    *same* theta-stage admissibility image the projection enforces
    (:math:`(1-\theta)\,u^n + \theta\,\mathrm{floor}_\mathrm{cell}`,
    including the one-way temperature-coefficient ratchet), and
    :math:`\mathrm{rect}_w` is the smooth-max one-sided rectifier of
    width :math:`w = 0.1\,b`, closed by an exact-zero :math:`C^1` gate
    beyond :math:`b + 2w`. Guarantees: the source only *supplies* (never
    drains); it is linear in the Newton iterate through the engaged
    regime with per-solve-constant bound and rate (JFNK-exact); it is
    identically zero for any cell at or above :math:`(1+2\cdot 0.1)\,b`
    -- at :math:`\theta = 1/2` every cell at or above 1.5x its cell
    floor, so healthy cells and healthy runs are bit-identical, beyond
    the documented sub-``1e-12`` tail requirement; and the internal rate
    cap :math:`1/(\theta\,\Delta t)` bounds the per-solve supplied
    increment by the deficit plus :math:`w/2` (no overshoot past
    :math:`b + w`).

    Rationale (measured on production run rr13e_C1w, the annihilation
    endgame of a weak-anomalous-eta formation arm): a persistent active
    set of 240--241 pure ion-energy components -- the live-plasma /
    floor-sea interface ring -- was projected onto its admissibility
    bounds every solve; the free-subspace line-search rescue engaged (7
    accepted rescue steps) but the pinned-row defect *grew*
    monotonically 0.468 :math:`\rightarrow` 1.004 (count 157
    :math:`\rightarrow` 241) until no admissible step reduced even the
    free subspace and the freeze guard fired at 43.97 us. There the
    discrete equations continuously demand sub-floor ion-energy
    evolution: a consistency gap between the residual and the admissible
    set that no line-search policy can close. With the source, a
    bound-riding cell whose equations demand drain finds an equilibrium
    a fraction of a rectifier width above its bound where the reservoir
    supplies exactly the demanded deficit -- the residual closes, the
    active set empties, and the supplied mass/energy is *booked* (see
    :pp:param:`implicit_mhd.floor_ledger_file`). The per-solve supply
    capacity at a bound-riding cell is
    :math:`\min(\nu\theta\Delta t, 1) \cdot w/2 = 5\%` of the bound: set
    the rate so this covers the local drain demand per step (rates at or
    above :math:`1/(\theta\,\Delta t)` saturate the cap).

.. pp:param:: implicit_mhd.floor_ledger_file
    :type: ``string``
    :default: none (no file)

    File for the floor-consistency supply ledger (requires a positive
    :pp:param:`implicit_mhd.floor_consistency_rate`): rows of
    ``step mass energy`` appended every step, holding the *cumulative*
    supplied mass [kg] and fluid energy [J] (per unit cross-section in
    1D), evaluated at the accepted theta state exactly as the residual
    applied them -- the conservation instrument of the reservoir, the
    floor-side sibling of :pp:param:`implicit_mhd.wall_ledger_file`.
    The first write of a run truncates a stale file; the counters
    restart at zero on a simulation restart.

.. pp:param:: implicit_mhd.density_eater_rate
    :type: ``float``
    :unit: 1/step
    :default: ``0`` (off, bit-identical)

    The reference code's density eater (``ntb.f90`` ``density_eater``, trigger
    ``eaten_type > 0``): a one-sided per-step relaxation of the mass
    density toward a low target in a fixed axial band,

    .. math:: \rho \; := \; \min\!\big(\rho,\;
        (1 - r)\,\max(\rho, \rho_\mathrm{t}) + r\,\rho_\mathrm{t}\big),

    i.e. the excess above the target :math:`\rho_\mathrm{t}` decays by
    the factor :math:`(1 - r)` each step (the reference code hard-codes
    :math:`r = 0.2`; the rate is per *step*, dt-independent, like the
    source) and at-or-below-target cells are never touched. Applied to
    the committed :math:`t^{n+1}` state at the top of the end-of-step
    restorations -- outside every implicit solve, matching the reference code's
    call site in ``step_en`` -- so Newton/JFNK consistency is
    untouched; the eaten state is the next solve's frozen step-old
    state. Faithful to the source, ONLY the density state variable is
    modified. Under the reference shot's closure (``mix = -1``,
    two-temperature) that means: velocity is preserved, so the momentum
    density scales with the mass; the ion energy *densities* (the
    conservative :math:`E_i`, the dual-energy :math:`U_i`, and the CGL
    pair) are invariant -- the removed mass gives up nothing, so the
    per-particle ion temperature *rises*; and the electron
    *temperature* is preserved (``te`` is the reference code's electron state),
    so the electron energy density scales with the mass. A frozen ion
    fluid (``evolve_ion_fluid = 0``) disables the eater, the reference code's
    ``fluid = 0`` bypass. Wall-masked (rigid-conductor) cells are
    excluded (the reference code's ``x_kind < 1`` skip). The removed mass and
    electron energy are booked (see
    :pp:param:`implicit_mhd.density_eater_ledger_file`).

.. pp:param:: implicit_mhd.density_eater_target_fraction
    :type: ``float``
    :default: ``0.01``

    Eater target as a fraction of the eater reference density (the reference code
    ``en0/100``). The target must be at or above
    :pp:param:`implicit_mhd.mass_density_floor` (the reference code keeps it at 10x
    the ``f_en_mn`` floor).

.. pp:param:: implicit_mhd.density_eater_reference_density
    :type: ``float``
    :unit: kg/m^3
    :default: :pp:param:`implicit_mhd.reference_mass_density`

    Eater reference base density (the reference code's ``en0`` input, a *number*
    density there -- multiply by the ion mass).

.. pp:param:: implicit_mhd.density_eater_reference_peak_fraction
    :type: ``float``
    :default: ``0.1``

    The reference code's ``en0_upd = 1`` dynamic reference (``step.f90``): the eater
    reference is ``max(base, fraction x step-old global density
    peak)``, refreshed every step. ``0`` freezes the reference at the
    base (``en0_upd = 0``).

.. pp:param:: implicit_mhd.density_eater_band
    :type: ``string``: ``z_lo`` or ``z_center``
    :default: ``z_lo`` when the z_lo pmc mirror plane is active, else ``z_center``

    Axial band of the eater, the reference code's ``sym_bc`` dispatch: on a
    half-domain the band sits at the mirror plane, on a full domain it
    straddles the domain z-center.

.. pp:param:: implicit_mhd.density_eater_band_cells
    :type: ``int``
    :default: ``-1`` (the reference code's grid rule)

    Band size in cell planes: the plane count from the boundary
    (``z_lo``) or the half-width about the center (``z_center``). Any
    value <= 0 selects the reference code's grid rule -- ``2 + nz/500`` planes at
    ``z_lo`` (``sym_bc = 1``: ``ngrd = 1 + nz/500``, vertex planes
    ``1 .. 1 + ngrd``), half-width ``1 + nz/1000`` about the center
    otherwise.

.. pp:param:: implicit_mhd.density_eater_flux_sign
    :type: ``int``: ``-1``, ``0``, or ``1``
    :default: ``1`` in RZ, ``0`` (and required) in 1D

    Closed-flux gate, the reference code's ``psi(k) > 0`` condition: with
    :math:`s = \pm 1` the eater acts only where
    :math:`s\,\psi(r, z) > 0`, with :math:`\psi = \int_0^r B_z\,r'\,dr'`
    the poloidal-flux integral of the theta-stage *total* :math:`B_z`
    (plasma response plus the stored external field under the
    split-field drive; the reference code gates on the step-old :math:`\psi`, so an
    intra-step stage is faithful). ``0`` disables the gate. 1D has no
    poloidal flux, so only ``0`` is accepted there.

.. pp:param:: implicit_mhd.density_eater_ledger_file
    :type: ``string``
    :default: none (no file)

    File for the eater removal ledger (requires a positive
    :pp:param:`implicit_mhd.density_eater_rate`): rows of
    ``step mass energy`` appended every step, holding the *cumulative*
    removed mass [kg] and electron energy [J] (per unit cross-section
    in 1D) at the committed end-of-step state. The ion energy density
    is invariant under the eater by construction (the reference code's wio/wik
    preservation), so these two columns close the accounting. The
    first write of a run truncates a stale file; the counters restart
    at zero on a simulation restart.

.. pp:param:: implicit_mhd.electron_pressure_floor
    :type: ``float``
    :unit: Pa
    :default: ``0``

    Non-negative pressure floor used in Ohm's law and electron-energy terms.
    The corresponding internal-energy floor is also enforced during Newton
    updates, matrix-free Jacobian probes, and final theta extrapolation.

.. pp:param:: implicit_mhd.ion_pressure_floor
    :type: ``float``
    :unit: Pa
    :default: ``0``

    Ion-pressure floor of the ``total_energy`` closure. The corresponding
    internal-energy floor :math:`U_{i,\mathrm{floor}} =
    P_{i,\mathrm{floor}}/(\gamma_i-1)` clamps the pressure recovered from
    :math:`E_i`, bounds Newton updates and matrix-free Jacobian probes, and
    smoothly closes the advected :math:`E_i` flux of a donor cell near the
    floor. Must be positive when ``implicit_mhd.ion_closure = total_energy``;
    unused otherwise.

.. pp:param:: implicit_mhd.ion_temperature_floor
    :type: ``float``
    :unit: K
    :default: ``0`` (off)

    Non-negative ion temperature floor (opt-in; e.g. ``300`` = room temperature;
    disables). The absolute pressure floors alone leave the temperature
    :math:`T = p/(n k_B)` unbounded below at densities above the
    mass-density floor; with the temperature floor, the admissibility
    lower bound of the evolved ion energy block becomes the maximum of
    the :pp:param:`implicit_mhd.ion_pressure_floor` equivalent and the
    density-dependent :math:`n k_B T_{i,\mathrm{floor}}` equivalent
    (:math:`E_i \geq \max(\ldots)/(\gamma_i-1)` for ``total_energy``,
    which bounds the internal part; :math:`U_\parallel \geq
    \max(\ldots)/2` and :math:`U_\perp \geq \max(\ldots)` for ``cgl``).
    The bound is evaluated with the beginning-of-step density, frozen for
    the whole nonlinear solve, so the per-solve admissible set stays
    linear in the unknowns; the end-of-step floor restoration instead
    uses the end-of-step density, so every solve starts admissible.
    The temperature bound is a one-way ratchet, engaged per cell only
    where the beginning-of-step value already satisfied it: a state at or
    above the floor can never cool through it, while colder-than-floor
    initial data is never lifted (no heat injection) and remains governed
    by the absolute floors alone. Requires an ion closure that evolves an
    ion energy block (``total_energy`` or ``cgl``) when set explicitly;
    the default is inert under the barotropic closure.

.. pp:param:: implicit_mhd.electron_temperature_floor
    :type: ``float``
    :unit: K
    :default: ``0`` (off)

    Non-negative electron temperature floor (default: room temperature;
    ``0`` disables). Analogous to
    :pp:param:`implicit_mhd.ion_temperature_floor` for the electron
    energy block: :math:`U_e \geq \max(p_{e,\mathrm{floor}},
    n k_B T_{e,\mathrm{floor}})/(\gamma_e-1)` with the quasi-neutral
    density :math:`n` frozen at its beginning-of-step value during the
    nonlinear solve and re-evaluated at the end of the step for the
    floor restoration.

.. pp:param:: implicit_mhd.pinned_cell_report_max
    :type: ``int``
    :default: ``16``

    Maximum number of clamped (pinned) cells listed when a Newton
    line-search failure triggers the free-subspace rescue (see
    ``newton.max_frozen_steps``): up to this many cells from the last
    admissibility projection's active-set masks are printed, one line
    per cell — block name, cell index, and physical cell-center
    coordinates — rank-locally (each rank lists its own cells) and
    rate-limited so a long frozen plateau does not flood the log.
    ``0`` disables the listing; the global per-block pinned counts of
    the projection report are printed either way.

.. pp:param:: implicit_mhd.fluid_flux
    :type: ``string``
    :default: ``central``

    Fluid face flux: one of ``legacy_e_centered``, ``rusanov``,
    ``hllc``, ``hlld`` (opt-in via ``implicit_mhd.allow_hlld``, see
    below), or ``central`` (the default). The pre-recast name
    ``centered`` was RENAMED ``legacy_e_centered`` (the legacy E-based
    scheme) and is a HARD ERROR, not a silent alias: it is one letter
    from ``central`` (the production recast flux) and the two are
    entirely different schemes. The default ``central`` requires a
    positive :pp:param:`implicit_mhd.viscosity` (the central flux
    carries no Riemann dissipation by design), so a deck that sets
    neither ``fluid_flux`` nor ``viscosity`` aborts at setup with a
    migration message: set the viscosity, or select another flux
    explicitly.

    ``legacy_e_centered`` gives the second-order, low-dissipation
    E-based operator used by smooth-wave verification. ``rusanov`` uses a
    piecewise-constant local Lax--Friedrichs flux for conservative,
    nonoscillatory fluid transport. ``hllc`` uses a contact-preserving
    HLLC flux: stationary contacts and held equilibria are not diffused
    (Rusanov diffuses every variable at the acoustic speed there), and a
    smooth floor-outflow limiter closes the advected mass and
    electron-energy fluxes (and the expansion pdV sink) as a donor cell
    approaches its positivity floor. ``hllc`` requires
    ``hybrid_pic_model.include_hall_term = false`` because the electron
    energy is advected with the ion contact wave. None of these three
    constitute a full MHD Riemann solver because magnetic induction
    remains in the staggered Ohm/Faraday update.

    ``hlld`` selects the conservative-form recast (1D Cartesian and
    cylindrical RZ). It is **not a production flux** (``central`` with
    ``implicit_mhd.viscosity`` is) and is kept only as kernel regression
    coverage: selecting it requires the explicit opt-in
    ``implicit_mhd.allow_hlld = true`` or the solver aborts at setup.
    In this form :math:`B^{n+\theta}` replaces :math:`E` as the JFNK
    field unknown, and one smoothed HLLD (Miyoshi--Kusano) Riemann
    solution per cell face supplies the fluid fluxes, the Maxwell-stress
    momentum coupling (replacing the pointwise :math:`J \times B` force),
    the ion-energy magnetic work (from the same discrete stress-flux
    difference as the momentum equation), and the ideal EMF, whose
    theta-implicit Yee curl advances :math:`B` with :math:`\nabla \cdot
    B = 0` preserved to round-off. :math:`E` becomes a derived Ohm's-law
    quantity, :math:`E = -u \times B + \eta J - \eta_H \nabla^2 J`,
    assembled by the solver (the hyper-resistive term follows
    ``hybrid_pic_model.plasma_hyper_resistivity``, discretized with the
    same cylindrical vector Laplacian as the hybrid solver).
    In RZ, :math:`E_r` and :math:`E_z` are direct face Riemann induction
    fluxes at their native Yee staggering, :math:`E_\theta` is assembled
    on cell corners with a smoothed UCT-HLLD (Londrillo--Del Zanna
    upwind-CT) form: the four-state upwind average is weighted by the
    stored fast Davis bounds, while the dissipation coefficients come
    from the fan's rotational wave speeds :math:`S_M \mp c_{An}` — the
    scale at which HLLD itself dissipates tangential-field jumps — so a
    static pressure-balanced equilibrium suffers no corner diffusion
    instead of being destroyed at the fast-speed scale
    (:math:`E_\theta` is zero on axis by :math:`m = 0` parity). The
    cylindrical geometric source terms carry the Maxwell-stress parts
    consistently in both the momentum and ion-energy equations.
    ``hlld`` requires ``include_hall_term = false`` (like ``hllc``, its
    electron energy is advected with the ion contact wave of the star
    construction, and its corner-EMF dissipation is scaled by the ion
    fan's rotational speeds; use ``central`` for Hall MHD in the
    recast) and ``include_electron_pressure_term = false``. The wave
    fan is assembled
    :math:`C^\infty`-smooth for matrix-free Jacobian probes (smoothed
    signal bounds, telescoped region form without if-chains, blended
    :math:`B_n \to 0` rotational layer); the widths are the
    ``implicit_mhd.hlld_kappa_*`` parameters below.

    ``central`` selects the same conservative-form recast with a
    Chacón-style central conservative flux (central co-located finite
    volumes with explicit dissipation, JCP 526 (2025) 113789) in place
    of the HLLD wave fan: every face channel — mass, total momentum
    including the Maxwell stress, gas-enthalpy ion energy, electron
    energy, and the ideal induction/EMF channel — is the arithmetic
    mean of the two sides' physical fluxes (the exact zero-dissipation
    limit of the ``hlld`` fan), a few flops per face with no wave-fan
    algebra on the JFNK matvec hot path and a smoother residual for
    GMRES. The central flux carries no Riemann dissipation, so a
    positive :pp:param:`implicit_mhd.viscosity` is required for
    nonlinear stability (field dissipation comes from resistivity; the
    RZ corner-EMF dissipation coefficients vanish identically under
    ``central``). ``central`` shares all recast plumbing and
    constraints with ``hlld`` (1D/RZ only, no electron-pressure Ohm
    term), is not supported with ``ion_closure = cgl``, and keeps the
    donor-gated positivity guards of the recast face fluxes.

    ``central`` additionally supports Hall MHD
    (``hybrid_pic_model.include_hall_term = true``): the solver-assembled
    Ohm's law becomes :math:`E = -u \times B + (J \times B)/\rho_q +
    \eta J - \eta_H \nabla^2 J`, with the Hall EMF added at the same
    edge/staggered :math:`E` locations as the :math:`\eta J` terms (edge
    currents interpolated with the same stencils as the :math:`|J|`
    resistivity argument, the cell-centered total :math:`B` averaged with
    the :math:`\eta_H` stencils, and the charge-density division guarded
    by a :math:`C^\infty` smooth floor at the Ohm guard; the :math:`m=0`
    axis parities are preserved exactly, :math:`E_\theta` and the Hall
    part of :math:`E_z` vanishing on axis). The face induction fluxes
    keep the ION velocity, so the edge term converts the ideal EMF to the
    electron frame, :math:`E = -u_e \times B`, without double counting;
    the electron thermal channels (:math:`U_e` advection and the
    :math:`\nabla\cdot u_e` pressure work) use the electron velocity. The
    Hall term is reactive and keeps the global ``implicit_evolve.theta``
    staging of the ideal EMF -- it always reads the theta-stage plasma
    current, never the shifted ``implicit_mhd.resistive_theta`` stage of
    the dissipative terms (an over-implicit stage would numerically damp
    whistlers first-order in :math:`\Delta t`).

.. pp:param:: implicit_mhd.viscosity
    :type: ``float``
    :default: ``0``

    Explicit ion kinematic viscosity :math:`\nu_i` in m^2/s of
    the conservative-form recast face fluxes (``fluid_flux = hlld`` or
    ``central``; required positive for ``central``, where it provides
    the nonlinear stabilization the central flux itself omits). Adds
    the normal-gradient viscous stress
    :math:`-\rho_f \nu_i (u_{c,R} - u_{c,L})/\Delta n` (with
    :math:`\rho_f` the arithmetic face density) to each momentum flux
    component and the exactly paired stress work
    :math:`-\rho_f \nu_i \sum_c \bar{u}_c (u_{c,R} - u_{c,L})/\Delta n`
    to the ion total-energy flux, evaluated at the theta-stage states
    like the rest of the residual, so total energy exchange is
    discretely conservative. Not supported with ``ion_closure = cgl``
    (its internal-energy blocks track no kinetic energy to pair the
    stress work against). The zero-flux reflecting wall passes no
    viscous flux, matching the advective wall policy.

.. pp:param:: implicit_mhd.wall_viscosity_mask
    :type: ``bool``
    :default: ``false``

    Reference-code-style wall viscosity BAND (requires an active
    :pp:param:`implicit_mhd.wall_model`; default off is bit-identical).
    Substitutes a single viscous face COEFFICIENT — for the momentum
    stress AND its velocity-weighted heating work, which share one
    coefficient assembly and must never split (an exactly conservative
    pair) — at every face either of whose adjacent cells lies within
    :pp:param:`implicit_mhd.wall_viscosity_mask_width` cells (Chebyshev
    distance over the stair-step first-masked tables) of the masked
    wall contour. This is the reference code's wall-viscosity band (``step.f90``
    ``disip``: viscosity SET to ``small_vis`` on the ``'skin'`` /
    ``'bndy'`` / ``'subr'`` / ``'subz'`` rows), the reduced-viscosity
    band that sits INSIDE their hard no-slip pin (see
    :pp:param:`implicit_mhd.wall_no_slip`): with a uniform viscosity,
    every coil-footprint ripple of the shaped wall otherwise deposits a
    viscous-heating bead row that the reference code does not have. The
    substituted coefficient is
    :pp:param:`implicit_mhd.wall_viscosity_band_value`; its default 0
    keeps the legacy behavior of skipping the viscous block entirely on
    band faces. The band is static geometry after initialization, so
    JFNK probes see constant structure.

.. pp:param:: implicit_mhd.wall_viscosity_mask_width
    :type: ``int``
    :default: ``2``

    Width of the wall viscosity band in fluid cells measured from the
    masked contour (at least 1).

.. pp:param:: implicit_mhd.wall_viscosity_band_value
    :type: ``float`` (Pa s)
    :default: ``0`` (the legacy exact-zero band)

    Viscous face coefficient substituted on the band faces selected by
    :pp:param:`implicit_mhd.wall_viscosity_mask` (which is required when
    this is positive). ``0`` skips the viscous block entirely on those
    faces — stress and heating exactly zero, bit-identical to the
    pre-band code. A positive value is the reference code's actual contract: a hard
    SET to a fixed PEDESTAL, ``small_vis = 1e-4`` Pa s in the reference shot
    (``step.f90:90``, ``184-195``).

    UNITS. :pp:param:`implicit_mhd.viscosity` is the kinematic-style
    knob :math:`\nu` [m²/s] that the face assembly multiplies by the
    face mass density (:math:`\rho_f \nu`, a dynamic viscosity), whereas
    this band value is the DYNAMIC viscosity itself [Pa s = kg/(m s)],
    deliberately NOT scaled by density — that is the whole point of a
    pedestal: it CAPS the coefficient where compressed plasma touches
    the wall (measured 3.3–40x below :math:`\rho\nu` there in the reference shot)
    and FLOORS it in the near-vacuum halo (45x above). To match a
    kinematic :math:`\nu` at a reference density, pass
    :math:`\rho_{\rm ref}\nu`.

.. pp:param:: implicit_mhd.wall_no_slip
    :type: ``bool``
    :default: ``false``

    Reference-parity NO-SLIP PIN of the shaped wall (requires an active
    :pp:param:`implicit_mhd.wall_model`; default off is bit-identical,
    and this is INDEPENDENT of
    :pp:param:`implicit_mhd.wall_thermal_bc` — an electromagnetic-only
    wall pins too). The reference code builds its implicit momentum matrices with
    ``bnd = 'i'`` for :math:`v_r` and :math:`v_z` (``vp.f90:221/233``,
    ``sten.f:115-121``), which OMITS the rows of the wall-contour
    vertices AND of the adjacent cut-cell "skin" layer; an omitted row
    is an exact zero increment (``solv.f:88-96``), and with velocities
    allocated at exactly zero (``init_cond.f90:82``) those two
    vertex layers are pinned at :math:`u = 0` for the entire run —
    measured bit-exact zero on all 762 wall and 762 skin vertices of
    the reference shot's earlier dump while the interior ran 1.7e5–6.2e5 m/s.

    Our realization is the structural twin: the first
    :pp:param:`implicit_mhd.wall_no_slip_width` LIVE (unmasked) cell rows
    adjacent to the contour get their three MOMENTUM increments zeroed
    in the residual, making them exact identity rows
    :math:`F = m - m^n` (density and the energy channels keep evolving,
    exactly as the reference code keeps advecting ``en`` and the energies on the
    skin), and the state's band momentum is zeroed ONCE at
    load-sanitize time — the discrete analog of the reference code's zero velocity
    IC, and idempotent, so the identity rows then hold :math:`u = 0`
    bit-exactly forever. Zeroing the band momentum preserves the
    INTERNAL energy under ``total_energy``/``dual_energy`` (the removed
    kinetic part is subtracted from :math:`E_i` and floored), so the pin
    thermalizes nothing at boot.

    Without the pin our stair-face image copies the tangential momentum
    verbatim — a perfect free-slip wall with no tangential momentum sink
    at all. Pair the pin with
    :pp:param:`implicit_mhd.wall_viscosity_band_value`: the reference code
    substitutes a small fixed dynamic viscosity INSIDE the pin so the
    pinned layer does not transmit large stress into the bulk, and
    landing only one of the two halves is what a free-slip near-wall jet
    against a frozen wall image (with the viscous coefficient zeroed)
    exploits. The band is static geometry, so JFNK probes see constant
    structure, and ``pc_mhd_block`` emits the matching momentum identity
    rows.

.. pp:param:: implicit_mhd.wall_no_slip_width
    :type: ``int``
    :default: ``1``

    Number of LIVE cell rows adjacent to the masked contour whose
    momentum the no-slip pin holds at zero (at least 1). ``2``
    reproduces the reference code's wall-node + skin depth.

.. pp:param:: implicit_mhd.thermal_diffusivity_ion
    :type: ``float``
    :default: ``0``

    Ion thermal diffusivity :math:`\chi_i` in m^2/s of the
    conservative-form recast face fluxes (``fluid_flux = hlld`` or
    ``central``; 0 = off). Adds the conductive internal-energy flux
    :math:`-\chi_i \rho_f \, \partial e_i / \partial n` (the
    :math:`q = -\kappa \nabla T` heat flux of Chacón, JCP 526 (2025),
    with :math:`\kappa = \chi_i \rho c_v`) to the ion total-energy face
    channel, with :math:`\rho_f` the arithmetic face density and
    :math:`e_i = p_i / ((\gamma_i - 1)\rho)` the specific internal
    energy recovered from the same (smooth-floored) cell pressures the
    physical fluxes use. Pure energy diffusion — there is no momentum
    counterpart — added after the donor gates and inside the zero-flux
    wall mask, exactly like the viscous terms. Requires
    ``ion_closure = total_energy`` (the flux enters the total ion-energy
    channel).

.. pp:param:: implicit_mhd.thermal_diffusivity_electron
    :type: ``float``
    :default: ``0``

    Electron thermal diffusivity :math:`\chi_e` in m^2/s of the
    conservative-form recast face fluxes (0 = off): the electron
    counterpart of :pp:param:`implicit_mhd.thermal_diffusivity_ion`,
    diffusing :math:`e_e = p_e / ((\gamma_e - 1)\rho)` on the electron
    energy face channel. Available under every ion closure (the electron
    energy block is always evolved).

.. pp:param:: implicit_mhd.thermal_diffusivity_ion(rho,Te,Ti,J,t)
    :type: ``str``
    :optional:

    Parser form of :pp:param:`implicit_mhd.thermal_diffusivity_ion`
    (supply one or the other, not both; an electron twin
    ``thermal_diffusivity_electron(rho,Te,Ti,J,t)`` exists with the same
    conventions). Evaluated at the conduction faces (the
    :math:`e`-gradient faces: z-faces in 1D, r- and z-faces in RZ) from
    the donor-averaged face state, with the same symbol conventions as
    :pp:param:`hybrid_pic_model.plasma_resistivity(rho,Te,J,t)`:
    ``rho`` the Ohm-floored face charge density (:math:`C/m^3`), ``Te``
    and ``Ti`` in Kelvin — the temperature-primary face ratios
    :math:`p_\mathrm{face}/(n_f k_B)` built from the same
    recovered/floored cell pressures the physical fluxes use (``Ti`` is
    0 outside ``ion_closure = total_energy``) — ``J`` the magnitude of
    the face-averaged cell-centered total current (:math:`A/m^2`), and
    ``t`` the residual (theta-stage) time (:math:`s`). Expressions must
    be :math:`C^\infty` in the state symbols (smooth floors such as
    ``(Te^2 + Tf^2)^(1.25)``, never ``max()``) for the matrix-free JFNK
    probes.

.. pp:param:: implicit_mhd.conduction_flux_limit_factor
    :type: ``float``
    :default: ``0`` (off)

    Free-streaming limiter of BOTH conductive face fluxes:
    :math:`\chi_\mathrm{eff} = \chi / (1 + |q| / (f\, q_\mathrm{fs}))`
    with :math:`q` the unlimited conductive face flux and
    :math:`q_\mathrm{fs} = n k_B T v_\mathrm{th}` the free-streaming
    heat flux of the channel's species
    (:math:`v_\mathrm{th} = \sqrt{k_B T/m}`, electron :math:`T_e` for
    :math:`\chi_e`, ion :math:`T_i` for :math:`\chi_i`, from the
    donor-averaged face state). ``0`` disables the limiter exactly
    (bit-identical); a positive factor applies the smooth harmonic cap
    with no branches. The limiter never runs silent: its factor is
    printed in the solver banner. Under
    ``thermal_conduction_model = braginskii`` the cap applies to the
    total (normal + tangential) tensor flux.

.. pp:param:: implicit_mhd.thermal_conduction_model
    :type: ``string``
    :default: ``isotropic``

    Closure of the conductive face fluxes. ``isotropic`` (default) keeps
    the scalar :pp:param:`implicit_mhd.thermal_diffusivity_ion`/electron
    path bit-identically. ``braginskii`` replaces it with the
    anisotropic tensor flux

    .. math::

        q_n = -\rho_f \left[\chi_\perp \frac{\partial e}{\partial n}
        + (\chi_\parallel - \chi_\perp)\, \hat{b}_n\,
        (\hat{b} \cdot \nabla e)\right]

    per channel (electron always; ion under
    ``ion_closure = total_energy``), with :math:`e` the same specific
    internal energies the isotropic path diffuses,
    :math:`\hat{b}` the face unit field (single-valued staggered
    :math:`B_n` plus cell-averaged tangential total :math:`B`,
    regularized smoothly at :math:`B \to 0`), and
    :math:`\chi_\parallel/\chi_\perp` the Braginskii (1965) Z = 1
    thermal diffusivities
    :math:`\chi_{\parallel,e} = 3.16\, k_B T_e \tau_e / m_e`,
    :math:`\chi_{\parallel,i} = 3.9\, k_B T_i \tau_i / m_i` with the
    standard collision times, and the Braginskii magnetization fits for
    :math:`\chi_\perp` in :math:`x = (\Omega\tau)^2` (exactly
    :math:`\chi_\parallel` at :math:`x = 0`, so the unmagnetized limit
    reproduces the isotropic flux). The tangential gradient of the
    cross term uses the corner stencil selected by
    :pp:param:`implicit_mhd.braginskii_tangential_limiter`
    (minmod-limited by default). Mutually exclusive with the
    constant/parser diffusivities; requires ``fluid_flux = hlld`` or
    ``central`` and a positive ``electron_pressure_floor``. At a shaped
    thermal wall the interface faces keep the one-sided isotropic drain
    (with the tensor's nn projection as its scalar :math:`\chi`); the
    tensor is never extended across the wall interface.

.. pp:param:: implicit_mhd.braginskii_tangential_limiter
    :type: ``string``
    :default: ``minmod``

    Slope treatment of the tangential-gradient corner stencil in the
    Braginskii cross term :math:`(\chi_\parallel -
    \chi_\perp)\,\hat{b}_n\,(\hat{b}\cdot\nabla e)`
    (``thermal_conduction_model = braginskii`` only). ``minmod`` (the
    default) builds the tangential slope from the monotone
    Sharma--Hammett minmod pairs (Sharma & Hammett, JCP 227 (2007)
    123): the cross-term flux vanishes at tangential extrema, so the
    tensor flux cannot create new extrema — the centered corner
    stencil is non-monotone and, at clamp-forced anisotropy, carried a
    measured wall-ledger heating runaway. ``none`` selects the legacy
    centered corner average.

.. pp:param:: implicit_mhd.conduction_coulomb_log
    :type: ``float``
    :default: ``10``

    Coulomb logarithm of the Braginskii collision times
    :math:`\tau_e/\tau_i`
    (``thermal_conduction_model = braginskii``).

.. pp:param:: implicit_mhd.conduction_chi_min
    :type: ``float``
    :default: ``0`` (off)

    Optional absolute floor in m^2/s on the Braginskii
    :math:`\chi_\parallel` and :math:`\chi_\perp` (smooth-max, applied
    per channel before the free-streaming limiter). This shared bound
    is in the solver's OPERATOR convention (it is applied to
    :math:`\chi` as the internal-energy diffusion coefficient,
    unconverted); see
    :pp:param:`implicit_mhd.conduction_chi_par_min` for the
    per-component, physical-convention bounds that override it.

.. pp:param:: implicit_mhd.conduction_chi_max
    :type: ``float``
    :default: ``0`` (off)

    Optional absolute cap in m^2/s on the Braginskii
    :math:`\chi_\parallel` and :math:`\chi_\perp`: a :math:`C^2` soft
    clip with knee width ``chi_max/10`` (exact pass-through below
    ``0.9 chi_max``), applied per channel before the free-streaming
    limiter. Operator convention, like ``conduction_chi_min``.

.. pp:param:: implicit_mhd.conduction_chi_par_min
    :type: ``float``
    :default: ``-1`` (inherit)

    Per-component clamp bounds in m^2/s on the Braginskii
    diffusivities: ``conduction_chi_par_min``/``_max`` bound
    :math:`\chi_\parallel` and ``conduction_chi_perp_min``/``_max``
    bound :math:`\chi_\perp`, with the same smooth-max floor and
    :math:`C^2` soft cap as the shared clamps above. Unlike the legacy
    shared ``conduction_chi_min``/``conduction_chi_max`` (which keep
    their historical operator-convention meaning), the per-component
    bounds are DEFINED in the physical :math:`\kappa/(n k_B)`
    convention of Braginskii's tables and converted internally by the
    per-species :math:`(\gamma - 1)` factor of the internal-energy
    flux operator, so the reference code's ``xil*_mn/mx``/``xip*_mn/mx`` values drop
    in unchanged. ``-1`` (the default) inherits the corresponding
    shared ``conduction_chi_min``/``conduction_chi_max`` value,
    unconverted, so decks that set none of the four keep the
    shared-clamp bounds. Independently of these knobs, after the
    clamps (and the perp-only boosts below) the ordering guard
    :math:`\chi_\parallel \geq \chi_\perp` is ALWAYS re-applied per
    channel (the reference code's ``ntb.f90`` ``xil = MAX(xil, xip)``), so
    separately clamped or perp-boosted components cannot invert the
    tensor anisotropy — note the guard can raise
    :math:`\chi_\parallel` in decks whose ``conduction_qs_chi`` or
    ``conduction_halo_boost`` pushes the boosted :math:`\chi_\perp`
    above it, even with all four knobs at their defaults.

.. pp:param:: implicit_mhd.conduction_chi_par_max
    :type: ``float``
    :default: ``-1`` (inherit)

    Physical-convention cap on the Braginskii :math:`\chi_\parallel`;
    see :pp:param:`implicit_mhd.conduction_chi_par_min`.

.. pp:param:: implicit_mhd.conduction_chi_perp_min
    :type: ``float``
    :default: ``-1`` (inherit)

    Physical-convention floor on the Braginskii :math:`\chi_\perp`;
    see :pp:param:`implicit_mhd.conduction_chi_par_min`.

.. pp:param:: implicit_mhd.conduction_chi_perp_max
    :type: ``float``
    :default: ``-1`` (inherit)

    Physical-convention cap on the Braginskii :math:`\chi_\perp`;
    see :pp:param:`implicit_mhd.conduction_chi_par_min`.

.. pp:param:: implicit_mhd.conduction_qs_chi
    :type: ``float``
    :default: ``0`` (off)

    Quasi-shorting cross-field boost amplitude in m^2/s of the
    Braginskii :math:`\chi_\perp` (requires
    ``thermal_conduction_model = braginskii``; with the parser
    diffusivities, carry the equivalent physics inside the deck-level
    ``thermal_diffusivity_*`` expressions instead): subgrid turbulence
    on broken flux surfaces shorts the channel across surfaces, modeled
    as an ADDITIVE cross-field diffusivity keyed to the pseudo-entropy
    excess :math:`s = (T/T_0)\,(\rho_0/\rho)^{2/3}` above the load
    envelope (:math:`T_0` =
    :pp:param:`implicit_mhd.conduction_qs_reference_temperature`,
    :math:`\rho_0` = :pp:param:`implicit_mhd.reference_mass_density`,
    :math:`\rho` smoothly guarded at the Ohm density scale), ramped by
    the :math:`C^\infty` smooth-max :math:`\chi_\mathrm{add} =
    \chi_\mathrm{qs}\, \tfrac{1}{2}\big[(s - s_\mathrm{on}) + \sqrt{(s
    - s_\mathrm{on})^2 + w^2}\big]` with :math:`w = 0.3\,(s_\mathrm{on}
    - 1)`. The ramp is centered ABOVE the envelope
    (:math:`s_\mathrm{on} > 1` enforced): a ramp centered on the
    envelope leaks :math:`w/2` of the amplitude onto every on-adiabat
    cell. Electron channel always; ion channel under ``ion_closure =
    total_energy``, keyed on the ion temperature with the same
    :math:`T_0`. The :math:`s` inputs follow
    :pp:param:`implicit_mhd.conduction_coefficient_state` like every
    other Braginskii coefficient input; the ``conduction_chi_min/max``
    clamp applies AFTER the addition. Mutually exclusive with
    :pp:param:`implicit_mhd.conduction_halo_boost` (see there: against
    a reference-parity perp cap the additive term saturates every face
    onto the cap and silently erases the multiplicative boost).

.. pp:param:: implicit_mhd.conduction_qs_onset
    :type: ``float``
    :default: ``1.5``

    Pseudo-entropy onset :math:`s_\mathrm{on}` of the quasi-shorting
    ramp; must exceed 1.

.. pp:param:: implicit_mhd.conduction_qs_reference_temperature
    :type: ``float``
    :unit: eV

    Load-envelope temperature :math:`T_0` of the quasi-shorting
    pseudo-entropy. Required (positive) when ``conduction_qs_chi > 0``.

.. pp:param:: implicit_mhd.conduction_halo_boost
    :type: ``float`` (dimensionless)
    :default: ``0`` (off)

    Density-keyed halo boost of the Braginskii ION :math:`\chi_\perp`
    (requires ``thermal_conduction_model = braginskii``): the
    dimensionless ``dp_mn`` of the reference code's exact low-density
    perpendicular-conduction boost (``ntb.f90`` ``t_cond``:
    ``xip = MAX(xip, xip*(en0/en)**2*dp_mn)``; the reference code hardwires
    ``dp_mn = 1``), draining halo heat to the wall bath while the bulk
    keeps physical Braginskii. The raw ion perp value is MULTIPLIED by

    .. math::

        \max\!\big(1,\; d_p \, (\rho_\mathrm{ref}/\rho)^2\big)

    — a no-op above the reference density, growing quadratically below
    it — with :math:`\rho_\mathrm{ref}` the shared reference density
    (the Ohm guard, raised by
    :pp:param:`implicit_mhd.vacuum_reference_base_density` and
    :pp:param:`implicit_mhd.vacuum_reference_peak_fraction` — the
    the reference code's ``en0``) and the density division guarded at the positivity
    floor. ION channel ONLY (the reference code's ``te_cond`` carries no boost) and
    :math:`\chi_\perp` only (the parallel channel keeps physical
    Braginskii), applied BEFORE the ``conduction_chi_min/max`` clamps —
    the reference code's order (the boost precedes the ``xipi_mn/mx`` clamps in
    ``t_cond``), so the perp cap bounds the boosted halo diffusivity:
    at reference-shot parameters the boosted halo value rides the
    ``conduction_chi_perp_max`` cap (the thermally shorted halo). The
    boost inputs follow
    :pp:param:`implicit_mhd.conduction_coefficient_state` like every
    other Braginskii coefficient input (``step_old`` keeps
    Newton-linear diffusion), and the coefficients are frozen per solve,
    so the ``MAX`` kink is never differentiated.

    Mutually exclusive with
    :pp:param:`implicit_mhd.conduction_qs_chi`: the quasi-shorting
    pseudo-entropy is maximal on exactly the low-density halo band the
    ``dp`` boost targets, and at production amplitudes
    (``conduction_qs_chi`` :math:`\gtrsim 10^4` against a reference-parity
    perp cap :math:`\sim 10^2`) even its zero-entropy smooth-max tail
    exceeds the cap, so every face rides ``conduction_chi_perp_max``
    and the multiplicative boost is exactly invisible after the clamp.
    Arm exactly one halo-shorting mechanism.

.. pp:param:: implicit_mhd.pressure_corner_width_fraction
    :type: ``float``
    :default: ``0``

    Corner width of the :math:`C^1` smooth-max internal-energy floor in
    the recast's ion pressure recovery
    :math:`p_i = (\gamma_i - 1)(E_i - \rho u^2/2)` under
    ``ion_closure = total_energy``. The default 0 keeps the legacy width
    (the internal-energy floor itself, bit-identical behavior), which at
    production floor scales is tiny against the subtracted kinetic
    energy: a compressed cell riding the corner then presents a
    near-kink whose local Jacobian is near-singular. A positive fraction
    widens the corner to
    :math:`\max(\text{floor}, \text{fraction} \times \rho u^2/2)`,
    keying the width to the local kinetic-energy scale. The asymptotes
    stay exact (floor below, :math:`E_i - \rho u^2/2` above); the corner
    inflation is width/2, confined to near-corner cells.

.. pp:param:: implicit_mhd.r_open_fluid
    :type: ``string``
    :default: ``outflow``

    Fluid ghost treatment at an open (Green's-function) upper radial
    boundary (RZ only, with ``boundary.field_hi = open`` on ``r``).
    ``outflow`` fills zero-gradient radial ghosts at :math:`r_{\max}`,
    letting fluid leave with the field free-space-coupled. ``reflect``
    keeps the no-normal-flow conducting-wall mirror for the fluid while
    the field remains open — useful when a violent boundary relaxation
    (e.g. releasing a wall-image-supported equilibrium) would otherwise
    drain wall cells through their positivity floors. Under
    ``implicit_mhd.fluid_flux = hlld`` or ``central``, ``reflect`` is a true zero-flux
    wall: every advective fluid channel of the :math:`r_{\max}` face flux
    (mass, tangential momentum advection, electron/ion energies) is
    exactly zero and the tangential Maxwell stress and the last ring's
    magnetic work terms use the perfect-conductor image field
    (:math:`B_n = 0` at the wall face), while the normal wall pressure
    and the induction/Ohm path keep the open-field boundary values.
    ``absorb`` (``hlld``/``central`` only) models a solid conductor that absorbs
    incident plasma instead of storing it: the fluid keeps the outflow
    recipe's impedance-matched zero-gradient ghosts (so the wall-face
    Riemann fan carries wall-plasma signal speeds only), except that
    the ghost image's NORMAL momentum is :math:`C^\infty` rectified to
    its outward part — the smoothed radial twin of the z-outflow
    no-reflux clamp. Plasma incident on the wall is admitted at its
    signal-limited rate (up to the local fast/Alfvén speed, the
    worst-case sheath admission rate), while a wall band retreating
    from the wall separates from a held ghost instead of dragging a
    refilling one back in. The donor drain gates apply unchanged (the
    pedestal band still cannot drain), the reflect path's zero-flux
    wall override does not apply, and the field-side boundary is
    unchanged (Green's-function open). Because the wall is then a
    physical sink, the solver integrates the r-weighted wall-face mass
    and fluid-energy fluxes of every accepted step into two cumulative
    counters (see ``absorb_ledger_interval``).

.. pp:param:: implicit_mhd.wall_model
    :type: ``string``
    :default: ``none``

    Shaped-wall model of the theta-implicit MHD solver (RZ,
    ``implicit_mhd.fluid_flux = hlld`` or ``central``; non-periodic z).
    ``none`` (default) is bit-identical to no wall. ``pec``,
    ``pec_response`` and ``dielectric`` build the same static stair-step
    mask from the revolved poloidal polyline of ``wall_polyline_file``
    and differ only in the field-side condition the masked band imposes
    (see below). For ``pec``: every electric-field component
    located on or outside the polyline (:math:`r \ge r_\mathrm{wall}(z)`;
    by the Yee staggering an E component exactly ON a stair face is
    always tangential to it) is projected onto the conductor condition
    at the end of every Ohm's-law assembly, i.e. inside every JFNK
    residual evaluation. The condition acts on the TOTAL field: with the
    split external vector potential
    (``hybrid_pic_model.add_external_fields``) the plasma-response E is
    set to :math:`-E_\mathrm{ext}` at masked locations, so the total
    tangential E vanishes and the TOTAL magnetic flux through every
    closed contour inside the metal stays frozen at its initial value —
    the perfect-conductor eddy response, including the transient
    shielding of programmed coil ramps that pure flux programming
    misses. The projection is affine in the state (JFNK-exact), and the
    MHD block preconditioner's resistive stencil drops the same masked
    rows, so the matrix-free application, the banded factorization and
    the direct sparse assembly all see the wall consistently. The FLUID
    is deliberately untouched: the wall region is expected to lie in the
    dust/vacuum fill (frozen momentum, :math:`E \to \eta J`), which is
    the fluid-side analog of an embedded-boundary particle scraper.
    Composes with the Green's-function open boundary (the wall's surface
    eddy sheet is an interior ``curl B`` current the open-boundary source
    deposit sees like any other).

    ``pec_response`` uses the identical mask but pins the PLASMA-RESPONSE
    field only: at masked locations the plasma E is zeroed while the
    prescribed external drive passes through the wall untouched. This is
    the embedded-boundary parity contract for FITTED coil-waveform
    composites that already embed the real machine's wall response
    (one-way coupling): plasma-generated flux cannot leak through the
    wall, while the drive acts everywhere as measured. Requires the
    split external fields (``hybrid_pic_model.add_external_fields = 1``);
    aborts otherwise. The preconditioner treatment is identical in both
    modes (the masked E values are state-independent constants either
    way).

    ``dielectric`` is the formation-tube standoff: in the real machine
    the shaped wall between the plasma and the drive coils is a
    quartz-class dielectric tube — it absorbs plasma and holds a
    temperature boundary, but it is electromagnetically TRANSPARENT (no
    eddy screening, no response-field pinning). The FLUID contract is
    identical to ``pec_response`` (same mask, rigid freeze of the masked
    band, one-sided stair-face drains and no-injection absorb image,
    ``wall_thermal_bc``/``wall_temperature``/ledger, halo-pedestal
    exclusion, axis guard; requires the split external fields like
    ``pec_response`` and an ACTIVE ``wall_thermal_bc`` — the standoff's
    entire wall action is the fluid contract, so without one the mode
    would be a silent no-op), while the FIELD side imposes no constraint at
    all: the plasma-response E is never projected in the band, the
    preconditioner drops no rows, and the wall-seam guard below is
    inactive (nothing is pinned, so there is no drive-scale surface
    current for the Ohm edge stencils to ingest). The semantics follow
    WarpX's mixed PEC/insulator domain boundary
    (``Source/BoundaryConditions/PEC_Insulator``): on the insulator side
    tangential fields are left unchanged unless prescribed and the
    normal B stays evolved — nothing is prescribed here. The band's
    frozen floor-density dust plus the vacuum-keyed resistivity already
    keep the region current-free in the response solve, which is the
    physically correct insulator interior. Use this mode when the
    machine's metal structures are represented separately (e.g. as
    circuit-side eddy rings coupled through the programmed drive):
    unlike ``pec_response`` — whose pinned response makes the plasma
    flux linkage of every in-band coil contour exactly zero, killing
    two-way circuit coupling — ``dielectric`` gives the finite, physical
    response linkage at coil radii inside the (former) metal band.

    In both CONDUCTOR modes the non-ideal Ohm edge terms (the Hall EMF, the
    electron-inertia field contribution, and the hyper-resistive term)
    carry a WALL-SEAM GUARD: they are zeroed at every live E row whose
    stencil footprint reaches into the masked band. Those stencils
    would otherwise ingest the conducting wall's SURFACE current
    (:math:`\mathrm{curl}\,B/\mu_0` across the stair seam is
    drive-scale, not plasma) and masked-band field values over the
    Ohm-floored near-wall density — a spurious, drive-powered EMF pump
    at the seam, measured as the global :math:`E_\theta` maximum
    growing at a formation-section step corner. Physically,
    electron-frame boundary-layer physics at a rigid conductor is below
    grid resolution, so the ideal + resistive Ohm law is the correct
    wall-adjacent contract (the :math:`\eta J` and :math:`-u \times B`
    parts are untouched). The guard is geometry-static (built once from
    the polyline, JFNK probes see constant structure), the
    preconditioner's stencil emission drops the identical contributions
    (validated in-run by ``pc_mhd_block.resistive_validate_assembly``),
    and the setup banner reports the per-family count of seam-guarded
    rows whenever the wall composes with any of the guarded terms.

    NOTE when driving with FITTED coil waveforms: calibration factors
    fitted to composites that already embed the real machine's wall
    response will double-count the wall when combined with
    ``wall_model = pec`` (measured on the FRC formation deck: with
    33/48 coil filaments on/outside the polyline, the total-field wall
    excludes essentially the whole programmed drive) — use
    ``pec_response`` with fitted composites, and refit toward raw
    waveforms before using ``pec`` quantitatively.

.. pp:param:: implicit_mhd.wall_polyline_file
    :type: ``string``

    CSV file of the wall polyline for the active
    ``implicit_mhd.wall_model`` modes:
    rows of ``z, r`` [m] (optional header row), z non-decreasing and
    single-valued (express near-vertical wall faces with epsilon-offset
    duplicate z values). The polyline is interpolated linearly in z and
    continued constantly beyond its axial range; everything at
    :math:`r \ge r_\mathrm{wall}(z)` is conductor (the surface sits AT
    the polyline). The polyline must stay clear of the :math:`r = 0`
    axis (at least one unmasked radial cell everywhere): the axis is
    handled by the m = 0 parity reflecting boundary, and a polyline
    pinching to the axis aborts at setup. A relative path is also
    resolved against the ``AMREX_INPUTS_FILE_PREFIX`` environment
    variable, like the inputs file itself.

.. pp:param:: implicit_mhd.wall_thermal_bc
    :type: ``string``
    :default: ``none``

    Thermal boundary condition at the stair-step interface of the shaped
    wall (requires ``implicit_mhd.wall_model = pec``, ``pec_response`` or
    ``dielectric``). The conducting-wall mask is electromagnetic only:
    with the default ``none`` the conduction operator
    (``thermal_diffusivity_ion/electron``) exchanges heat blindly across
    the wall interface against conductor cells riding the density floor,
    and the cell-centered Joule heating :math:`\eta J^2` of the PEC
    surface current meets the (possibly anomalous) plasma resistivity at
    the near-floor wall-band density — an unbounded spurious heat source
    inside the metal. Any active mode upgrades the masked band to a
    RIGID conductor: every fluid increment inside masked cells is zero,
    the stair interface faces drain the interior one-sidedly (what
    crosses is gone: mass, momentum, enthalpy — the fluid analog of an
    embedded-boundary particle scraper), and faces between two masked
    cells carry no conduction. The band's state itself is the rigid
    VACUUM image: at first-step sanitize time every masked fluid cell is
    clamped to the mass-density floor with zero momentum and
    floor-consistent energies at the background temperature —
    ``wall_temperature`` itself in ``outflow_limited`` mode, so the
    interface drain starts at exactly zero gradient (whatever the IC
    loaded outside the contour is scraped; a rank-0 banner and a
    ``#``-comment header row of ``wall_ledger_file`` report the removed
    cell count and net mass), the clamp is bit-exactly idempotent
    across restarts, and the halo pedestal plus every end-of-step floor
    restoration skip the band so it stays bit-static forever.
    ``zero_flux`` additionally makes the wall adiabatic: interface faces
    carry exactly zero conductive flux. ``outflow_limited`` makes the wall a
    heat SINK instead: interface faces drain conductively toward
    ``wall_temperature`` (requires a nonzero thermal diffusivity). The
    drain is ONE-SIDED — smoothly gated to zero at/below the reachable
    anchor (the maximum of ``wall_temperature`` and the corresponding
    temperature floor) and fully open above twice it, so the reservoir
    cools
    the interior toward T_wall but never heats plasma (or the
    near-floor dust rim) above it and never fights the admissibility
    projection's temperature-floor ratchet (a reservoir-anchored gate
    with T_wall below a floor was measured to deadlock Newton: the
    projection pins the wall-adjacent energy rows at the floor bound
    while the still-open drain demands descent — iteration-0 frozen
    solves until the ``newton.max_frozen_steps`` trip) — and ALWAYS
    free-streaming limited at the wall face
    (factor = ``conduction_flux_limit_factor`` when set, else 1). A
    two-sided, un-capped Dirichlet exchange was measured fatal on the
    FRC formation ladder: the reservoir heats the machine's entire
    near-floor wall rim on the sub-step cell-diffusion time while the
    contact hot spots drain at many times the free-streaming flux —
    conduction-type Newton hostility on every wall face at once.
    ``dirichlet`` PINS the interface temperature instead (the reference code's
    ``t_bc = 'd'`` contract): a TWO-SIDED conductive exchange against
    the ``wall_temperature`` bath at the half-cell Dirichlet distance
    (the wall value sits ON the interface, not one cell in), so the
    bath restores sub-T_wall cells as well as draining hot ones. The
    exchange is free-streaming capped in BOTH directions (the same
    harmonic cap as the ``outflow_limited`` drain — the fatality above was
    the two-sided exchange WITHOUT the cap), and the pin target is
    anchored at the reachable set, the maximum of ``wall_temperature``
    and the corresponding temperature-floor image, so the pin never
    fights the admissibility projection's floor ratchet. Like
    ``outflow_limited``, it requires ``wall_temperature`` and a conduction
    channel. The mask is static geometry, so all of this is smooth in
    the state for the JFNK probes.

.. pp:param:: implicit_mhd.wall_temperature
    :type: ``float`` (eV)

    Wall reservoir temperature of
    ``implicit_mhd.wall_thermal_bc = outflow_limited`` or ``dirichlet``
    (required in both, in eV; an error in the other modes). Applied to
    both the electron and ion conduction channels. The drain's smooth
    gate closes at the maximum of ``wall_temperature`` and the
    corresponding
    :pp:param:`implicit_mhd.electron_temperature_floor` /
    :pp:param:`implicit_mhd.ion_temperature_floor` — the reachable set
    of the admissibility projection — so a ``wall_temperature`` below
    an active temperature floor cools the wall-adjacent plasma to the
    FLOOR temperature only (a rank-0 warning reports the
    mis-configuration; the reservoir value below the floor is
    unreachable by construction). Under ``dirichlet`` the same
    reachable anchor is the pin target itself.

.. pp:param:: implicit_mhd.wall_conduction_scale
    :type: ``string``
    :default: ``perp``

    Braginskii clamp CLASS of the shaped-wall interface conductance
    (requires ``implicit_mhd.thermal_conduction_model = braginskii``;
    the constant/parser diffusivities carry a single scalar chi that the
    wall drain already uses unchanged). Interface faces never use the
    anisotropic tensor's :math:`\hat b_n \hat b_n` projection — on the
    stair-step contour the face normals are grid-aligned, so that
    projection is a geometry artifact rather than a physical channel —
    they use a clamped per-component coefficient, and this knob selects
    which one. ``perp`` (default) takes the
    :pp:param:`implicit_mhd.conduction_chi_perp_max` /
    ``conduction_chi_perp_min``-clamped coefficient: the reading that a
    flux-conforming wall only ever exposes the cross-field channel
    because B is tangential to the contour. ``parallel`` takes the
    ``conduction_chi_par_min/max``-clamped coefficient instead, which
    reproduces the reference code's MEASURED wall conductance: their implicit
    temperature solve pins every wall-conforming cut-cell vertex at the
    0.5 eV anchor for BOTH species and builds the halo-to-wall
    conductance :math:`G = \sum (K\cdot 1)_i` from the PARALLEL
    conductivity clamp maximum ``xile_mx = xili_mx``, and removing that
    single term takes their halo from 0.5 eV to 5e4 eV in 73 steps.
    Under production clamp classes (perp :math:`[1, 100]` vs par
    :math:`[100, 10^6]`) the two settings differ by up to
    :math:`10^4` in wall conductance and therefore in the halo thermal
    time constant :math:`\tau = C/G`. The selection applies to BOTH
    species channels and to every reservoir
    :pp:param:`implicit_mhd.wall_thermal_bc` mode
    (``outflow_limited``, ``dirichlet``, ``dirichlet_limited``); only the
    chi that multiplies the drain changes — the drain's sign, its
    one-sided reachable-set gate and its free-streaming cap are
    untouched. Interior faces always keep the full tensor, and the z-end
    exchange (:pp:param:`implicit_mhd.z_wall_conduction`) keeps its own
    one-sided nn scalar. The selector is a per-solve constant, so the
    Newton/JFNK residual stays exactly as smooth as before.

.. pp:param:: implicit_mhd.wall_band_eta_override
    :type: ``float`` (Ohm m)
    :default: ``0`` (off)

    Electrical hygiene of the shaped-wall band (requires an active
    ``implicit_mhd.wall_model``; aimed at ``dielectric``, whose
    EM-transparent band otherwise relies on the density-keyed vacuum
    resistivity to stay current-free): when positive, the resistivity
    of the FIELD-advance Ohm's law is set to EXACTLY this constant at
    every band-INTERIOR electric-field location — one whose complete
    cell-centered interpolation neighborhood is masked. The override
    REPLACES the composed eta (user parser + current-keyed anomalous
    terms + density-keyed vacuum ramp) rather than composing with it:
    :math:`\eta_{\mathrm{band}}` is a state-independent constant with
    :math:`\partial\eta/\partial(\mathrm{state}) = 0` in the band.
    Both properties are load-bearing (measured on the FRC formation
    ladder): a density-keyed ramp that lands mid-band leaves a
    resistive shell whose :math:`\mu_0 L^2/\eta` time can sit on the
    drive timescale and screen the coil drive, and the current-keyed
    anomalous terms explode at the clamped floor density — the
    Jacobian stiffness that froze a production arm's Newton (150
    zero-progress solves, 154 pinned energy rows). Choose the value at
    or above the deck's vacuum-eta saturation so
    :math:`\mu_0 L_{\mathrm{band}}^2/\eta_{\mathrm{band}}` is far
    below the drive timescale (band :math:`L \sim 0.08` m: a 10 us
    drive needs :math:`\eta_{\mathrm{band}} \gg 10^{-3}` Ohm m). E
    rows on the stair interface itself (which bound live cells) keep
    the composed eta, so the live-side field evolution and the
    interface scraper are untouched; the preconditioner ingests the
    identically overridden eta, and Joule heating keeps the composed
    resistivity (masked cells receive no fluid increments anyway).
    Under the conductor contracts the overridden rows are subsequently
    pinned by the projection, making the override redundant there.

.. pp:param:: implicit_mhd.wall_field_freeze
    :type: ``bool``
    :default: ``0`` (off, bit-identical)

    FIELD-side freeze of the shaped-wall band (requires an active
    ``implicit_mhd.wall_model``; aimed at ``dielectric``): when on,
    every EVOLVED magnetic-field face whose complete Faraday update
    stencil (its curl-E edges) lies in the masked region becomes an
    exact identity row of the residual (:math:`F = B - B^n`, unit
    Jacobian diagonal — the field twin of the wall's fluid identity
    rows), so the plasma-response B on/outside the wall contour stays
    bit-frozen at its boot/restart value (physically ~0: the response
    starts zero and the exterior carries no fluid) and the discrete
    curl of the evolved field — the band current — is exactly zero at
    every strictly-exterior location from the first step. Faces with
    even one live E edge stay live, so the last live Faraday row still
    closes against the surface E and interior induction is untouched.
    The painted EXTERNAL fields (the split registers) are unaffected:
    the coil drive penetrates by prescription. This closes the wall's
    consistency triangle — scraped fluid outside the contour (the
    exterior clamp), frozen exterior response field, and
    :math:`J \to 0` at the boundary — and supersedes
    ``wall_band_eta_override`` for the FIELD advance (the band-interior
    E rows no longer influence any live B row; both knobs stay
    independent). Under ``pec_response`` the freeze is an exact no-op
    (the pinned E already freezes fully-masked faces); under ``pec`` it
    overrides the eddy-response closure of fully-masked faces (the
    metal then no longer shields the drive there — not recommended).
    The preconditioner mirrors the freeze exactly (identity rows in the
    resistive stencil emission plus a post-inverse restore; validated
    in-run by ``pc_mhd_block.resistive_validate_assembly``), and
    checkpoint/restart carries the frozen faces unchanged. A
    Green's-contour exterior fill (a response-consistent exterior field
    instead of a frozen one) is a documented future upgrade of this
    hook.

.. pp:param:: implicit_mhd.conduction_coefficient_state
    :type: ``string``
    :default: ``theta``

    Evaluation state of the thermal-conduction COEFFICIENT inputs (the
    face :math:`\rho_f` multiplier, charge density, and temperatures
    feeding the parser diffusivities, the Braginskii coefficients, and
    the free-streaming caps). ``theta`` evaluates them at the theta
    stage inside the residual (bit-identical default). ``step_old``
    freezes them at the step-old fields — per-solve constants, so
    Newton sees LINEAR diffusion in the energies while the flux keeps
    the live theta-state specific-energy gradient. Isolation probes on
    the formation ladder measured state-dependent diffusivities inside
    the residual as the dominant Newton-hostility class (an ~11x
    step-rate cost by themselves); freezing the coefficients removes
    it at the price of a one-step lag on the coefficient, irrelevant
    for subgrid transport models. The parser J input stays live (there
    is no step-old current register).

.. pp:param:: implicit_mhd.wall_ledger_file
    :type: ``string``

    Optional file for the shaped-wall deposition ledger (active
    whenever ``wall_thermal_bc`` is not ``none``): the cumulative mass
    [kg] and fluid energy [J] deposited through the stair-step
    interface faces into the rigid-conductor band — the advective
    capture plus the one-sided conductive drain, integrated from the
    accepted theta-state face fluxes with exact per-face annulus areas
    (both r- and z-normal interfaces). Rows of ``step mass energy``,
    written and printed on the ``absorb_ledger_interval`` cadence,
    preceded by a ``#``-comment header row reporting the first-step
    exterior clamp (scraped cell count and net mass removed); the
    accumulation always runs and the printed line appears even without
    a file. This is the wall heat-load instrument: for a converged
    solve the booked energy matches the interior's loss through the
    wall to the nonlinear tolerance.

.. pp:param:: implicit_mhd.absorb_ledger_interval
    :type: ``integer``
    :default: ``1``

    Print interval (in steps) of the absorbing-wall ledger: the
    cumulative mass [kg] and fluid energy [J] exported through the
    :math:`r_{\max}` wall face under ``r_open_fluid = absorb``,
    integrated from the accepted theta-state face fluxes each step
    (r-weighted, so a converged solve books the export to the nonlinear
    tolerance). ``0`` disables the printing; the accumulation always
    runs. The counters restart at zero on a simulation restart.

.. pp:param:: implicit_mhd.absorb_ledger_file
    :type: ``string``
    :default: *none*

    Optional file to which the absorbing-wall ledger rows
    (``step  absorbed_mass  absorbed_energy``) are appended at every
    ledger print.

.. pp:param:: implicit_mhd.z_boundary_fluid
    :type: ``string``
    :default: ``neumann``

    Fluid-moment ghost treatment at the non-periodic axial (z) ends
    (RZ, ``implicit_mhd.fluid_flux = hlld`` or ``central``; the
    field/current z-ghost handling — including the open Green's-function
    cap deferral — is unaffected). ``neumann`` keeps the passive
    zero-gradient ghosts: with identical states on both sides of the end
    face the Riemann flux of every channel vanishes, so the ends are
    zero-flux for the energies and heat generated in the device cannot
    escape through the exhaust. ``wall_temperature`` copies the end
    plane's density and momentum but sets the ghost ENERGIES to
    ``z_wall_temperature`` at the ghost density — the electron ghost
    carries the wall internal energy
    :math:`\rho (q/m) T_\mathrm{wall} / (\gamma_e - 1)` and the ion
    ghost the same-temperature internal energy plus the kinetic energy
    of the copied ghost momentum, so the condition acts on the thermal
    content only (under the CGL closure
    :math:`U_\parallel = p_\mathrm{wall}/2`,
    :math:`U_\perp = p_\mathrm{wall}`): the end faces then exchange
    advectively against a :math:`T_\mathrm{wall}` reservoir, the z
    analog of the r-wall temperature anchoring. ``outflow`` rectifies
    the ghost AXIAL momentum :math:`C^\infty`-smoothly to its OUTGOING
    part — the exact recipe of the ``r_open_fluid = absorb`` wall, the
    smoothed twin of the ``z_outflow_no_reflux`` clamp (with which it is
    mutually exclusive) — so the ends advect mass and energy out at
    interior values but never feed plasma back. Both non-default modes
    are applied identically in every JFNK residual evaluation and are
    smooth in the state.

.. pp:param:: implicit_mhd.z_wall_temperature
    :type: ``float``, in eV

    Wall temperature of the z-end fluid ghosts; required with (and only
    valid with) ``implicit_mhd.z_boundary_fluid = wall_temperature``.

.. pp:param:: implicit_mhd.z_wall_conduction
    :type: ``bool``
    :default: ``false``

    Conductive z-end exchange (requires ``implicit_mhd.z_boundary_fluid
    = wall_temperature`` and an active conduction channel). By default
    the conductive face flux at the z domain end faces goes through the
    PLAIN interior branch differencing the interior cell against the
    wall-image ghost at the one-cell Neumann-ghost distance — an
    exchange that inherits every interior guard: the free-streaming cap
    when ``conduction_flux_limit_factor`` is armed, and the pressure
    floors, which CLAMP the ghost image wherever
    :math:`n\,k_B T_\mathrm{wall}` falls below a pressure floor (at low
    exhaust densities the branch then drains toward the floor image,
    not the wall). Enabled, the z domain END boundary faces (z_hi
    always; z_lo unless it is the ``symmetry`` mirror plane) instead
    carry a HARD Dirichlet-class conductive exchange against the
    :math:`T_\mathrm{wall}` reservoir in both energy channels (electron
    always; ion under the ``total_energy``/``dual_energy`` closures,
    the only ones whose ion-energy register consumes conduction), at
    the half-cell Dirichlet distance (the wall value sits ON the face,
    flux :math:`\chi \rho_f (e - e_\mathrm{wall}) \, 2/\Delta z`) with
    NO free-streaming cap — the implicit solver converges on the
    demanded outflow, exactly like the r-wall ``wall_thermal_bc =
    dirichlet`` pin (like that pin, the bath anchors at the
    temperature-floor images, identical for wall temperatures at/above
    the floors). Under Braginskii conduction the exchange coefficient
    is the full anisotropic tensor scalar
    :math:`\chi_{nn} = \chi_\perp + (\chi_\parallel - \chi_\perp)
    b_n^2/|b|^2` evaluated at the face with the magnetic-field inputs
    taken ONE-SIDED from the interior cell plus the staggered face
    :math:`B_n` (end walls intersect an axial field at normal
    incidence, so the exchange is :math:`\chi_\parallel`-dominated —
    the physics point of the knob). The exchange REPLACES the plain
    conductive branch at those faces, is part of the JFNK residual
    (smooth in the state, conduction-stage/frozen-coefficient
    conventions identical to the interior operator), and the default
    keeps the legacy end faces bit-identical.

.. pp:param:: implicit_mhd.z_lo_boundary_fluid
    :type: ``string``
    :default: *none* (inherit ``z_boundary_fluid``)

    Fluid-moment ghost treatment at the LOWER axial end only (RZ,
    ``implicit_mhd.fluid_flux = hlld`` or ``central``). Unset, the
    z_lo face keeps ``z_boundary_fluid``. The only accepted value,
    ``symmetry``, makes :math:`z = z_{\mathrm{lo}}` a mirror-symmetry
    plane: the z_lo ghost rows are the exact linear reflection of the
    interior — even mass density and energies, even radial/azimuthal
    momentum, ODD axial momentum — so the boundary faces see symmetric
    Riemann states (zero advective mass/energy/normal-momentum flux;
    the normal momentum feels only the symmetric pressure) and the
    even-temperature mirror ghosts make the conductive heat flux
    through the plane vanish identically. It must be paired with the
    PMC field boundary at z_lo (``boundary.field_lo = ... pmc``), and a
    z_lo ``pmc`` field boundary in turn requires
    ``z_lo_boundary_fluid = symmetry`` — the PMC parities (tangential
    :math:`B_r`, :math:`B_\theta` odd, normal :math:`B_z` even;
    tangential :math:`E_r`, :math:`E_\theta` even, normal :math:`E_z`
    odd; :math:`J_r`, :math:`J_\theta` even, :math:`J_z` odd) are
    exactly the z-mirror parities of the electromagnetic subsystem, so
    field, current, and fluid see one consistent half-domain image of a
    mirror-symmetric full domain. The z_hi face keeps
    ``z_boundary_fluid`` (production: ``outflow`` at z_hi with the
    mirror at z_lo). With split external fields the deck's coil set
    must itself respect the :math:`z = 0` mirror. Applied identically
    in every JFNK residual evaluation; the MHD block preconditioner's
    resistive/whistler end rows fold their ghost columns with the same
    parities.

.. pp:param:: implicit_mhd.hllc_signal_closure
    :type: ``string``
    :default: ``consistent``

    Ion pressure used in the HLLC wave-speed estimates (Davis bounds and
    the local Lax--Friedrichs coefficient of the ion-energy channel).
    ``consistent`` uses the same pressure as the physical flux.
    ``barotropic`` evaluates the signal-speed ion pressure from the
    polytropic law :math:`P_{i,\mathrm{ref}} (\rho/\rho_{\mathrm{ref}})
    ^{\gamma_i}`, so the HLLC wave structure does not respond to the
    ion-energy unknown at all; the physical fluxes and the contact speed
    keep the consistent pressure, preserving flux continuity across the
    contact. This improves Newton robustness at near-stagnant,
    magnetically balanced interfaces where the recovered
    :math:`p_i(E_i)` otherwise couples the ion-energy Jacobian block to
    the Riemann wave fan. Requires ``fluid_flux = hllc``,
    ``ion_closure = total_energy``, and a positive
    ``reference_ion_pressure``.

.. pp:param:: implicit_mhd.hllc_contact_blend
    :type: ``float``
    :default: ``0``

    Smooth contact-side blending width :math:`\kappa` for the HLLC flux;
    ``0`` keeps the standard hard upwind-side switch at zero contact
    speed. When positive, the two side-complete star fluxes are blended
    with a :math:`C^\infty` weight over contact speeds :math:`|S_*|
    \lesssim \kappa \, (|S_L| + |S_R|)/2` (:math:`\kappa \approx 0.05`
    recommended). At :math:`S_* = 0` the two branches carry the identical
    flux, so static contacts remain machine-preserved; the blend removes
    the switch kink that matrix-free Jacobian probes otherwise straddle
    at near-stagnant balanced interfaces (e.g. an FRC separatrix), which
    can defeat the Newton line search. Requires ``fluid_flux = hllc``.

.. pp:param:: implicit_mhd.hlld_fan_closure
    :type: ``string``
    :default: ``consistent``

    Ion pressure seen by the HLLD wave-fan structure — the signal
    bounds (the Davis fast-speed estimates, which parametrize every
    channel's upwind dissipation and the RZ corner-EMF weights), the
    contact-speed estimate :math:`S_M`, and the star states built from
    it. ``consistent`` uses the recovered :math:`p_i(E_i)`.
    ``barotropic`` evaluates the fan ion pressure from the polytropic
    law (the electron and magnetic contributions stay consistent); the
    physical fluxes, and hence the jump conditions, keep the consistent
    pressure, so only the dissipation structure is re-centered. This is the robustness
    configuration for violent rotational/compound structures
    (:math:`B_t` through zero at finite :math:`B_n`), where the
    :math:`p_i(E_i)` coupling into the star algebra otherwise defeats
    the Newton line search (the Brio--Wu CI test is the reproducer);
    combine with a widened ``hlld_kappa_bn`` (:math:`\approx 0.2`).
    Requires ``fluid_flux = hlld``, ``ion_closure = total_energy``, and
    a positive ``reference_ion_pressure``.

.. pp:param:: implicit_mhd.hlld_ion_energy_flux
    :type: ``string``
    :default: ``star``

    Ion total-energy channel of the HLLD flux. ``star`` (default)
    builds fan-consistent star energies from the :math:`S_{L,R}` jump
    conditions with the gas star pressure derived from the fan's
    total star pressure: a magnetically balanced static gradient gives
    :math:`S_M = 0` and star states equal to the inputs (no energy
    pumping — the Toro gas-star pathology does not arise because
    :math:`S_M` comes from the total pressure), and contact jumps
    dissipate at :math:`|S_M|` scale. ``llf`` selects the fast-speed
    local Lax--Friedrichs enthalpy fallback, which diffuses every
    :math:`E_i` gradient at the fast-magnetosonic scale (visible as a
    steady ion-energy drain along strong-:math:`B_n` field lines in
    near-static equilibria).

.. pp:param:: implicit_mhd.hlld_all_speed
    :type: ``bool``
    :default: ``true``

    LHLLD all-speed pressure correction (Minoshima & Miyoshi, JCP 446
    (2021) 110639): the velocity-difference term of the fan's star
    total pressure — a normal-momentum diffusion that scales with the
    fast-magnetosonic speed — is scaled by :math:`\phi = \chi(2-\chi)`,
    :math:`\chi = \min(1, \max(c_{u,L}, c_{u,R})/c_{f,\max})`, with
    :math:`c_u` the fast-speed formula evaluated with :math:`|u|` in
    place of the sound speed. :math:`\phi \to 1` for super-magnetosonic
    flow (original HLLD) and :math:`\to 0` at stagnation, removing the
    spurious fast-scale response to low-Mach velocity noise. Assembled
    :math:`C^\infty`; exactly inert where :math:`u_L = u_R`.

.. pp:param:: implicit_mhd.hlld_kappa_signal
    :type: ``float``
    :default: ``0.05``

    :math:`C^\infty` smoothing widths of the HLLD wave fan, in units of
    the local fast-speed scale; ``0`` recovers the corresponding hard
    form. ``hlld_kappa_signal`` smooths the Davis signal bounds and the
    ion-energy dissipation coefficient; ``hlld_kappa_contact`` sets the
    region-blend width of the telescoped fan; ``hlld_kappa_bn`` sets the
    :math:`B_n \to 0` degeneracy scale of the rotational (Alfvén) layer
    — widen to :math:`\approx 0.2` together with
    ``hlld_fan_closure = barotropic`` for violent compound structures;
    ``hlld_kappa_denominator`` guards the star-state denominators at the
    switch-on/switch-off degeneracy.

.. pp:param:: implicit_mhd.hlld_kappa_contact
    :type: ``float``
    :default: ``0.05``

    See ``hlld_kappa_signal``.

.. pp:param:: implicit_mhd.hlld_kappa_bn
    :type: ``float``
    :default: ``0.05``

    See ``hlld_kappa_signal``.

.. pp:param:: implicit_mhd.hlld_kappa_denominator
    :type: ``float``
    :default: ``0.05``

    See ``hlld_kappa_signal``.

.. pp:param:: implicit_mhd.positivity_safety
    :type: ``float``
    :default: ``0.99``

    Safety factor in ``(0,1)`` applied when a Newton update or matrix-free
    Jacobian perturbation reaches the density or electron-energy bound.

.. pp:param:: implicit_mhd.include_joule_heating
    :type: ``bool``
    :default: ``true``

    Include :math:`\eta |\boldsymbol J|^2` in the JFNK electron-energy
    residual. This is independent of the QDSMC
    ``hybrid_pic_model.include_joule_heating`` option. Hyper-resistive
    heating is not yet included, so total-energy accounting is incomplete
    when ``plasma_hyper_resistivity`` is nonzero.

.. pp:param:: implicit_mhd.joule_ion_fraction
    :type: ``float``
    :default: ``0`` (all-electron)

    Direct ion share of the Joule power (the reference code's ``f_ohmi``,
    ``vp.f90`` ``dw0_io``): the fraction :math:`f` of the Joule
    deposit is booked into the ion energy channel, with the electron
    channel keeping :math:`1 - f`. ``0`` (the default) is the legacy
    all-electron deposit (bit-identical). A constant in ``(0, 1]``
    books that fixed share. ``-1`` selects the reference code's Te-keyed tanh
    split

    .. math::

        f = 0.1 + 0.4\,\big(1 + \tanh((T_e[\mathrm{eV}] - 60)/20)\big)

    (the ``f_ohmi = -1`` flown by the reference shot's run).
    Requires a recast flux (``fluid_flux = hlld`` or ``central``) and
    ``ion_closure = total_energy`` or ``dual_energy`` — the ion share
    needs an ion energy channel. The ion share follows the same
    wall-freeze/theta staging envelope as the electron deposit and
    the same un-boosted heating :math:`\eta` (see
    :pp:param:`implicit_mhd.joule_ohm_current`).

.. pp:param:: implicit_mhd.vacuum_resistivity_diffusivity
    :type: ``float``
    :unit: :math:`\mathrm{m^2\,s^{-1}}`
    :default: ``0`` (off)

    Density-keyed vacuum resistivity of the field advance
    (``fluid_flux = hlld``/``central`` only). The solver-assembled Ohm's law sees
    the smooth, uncapped resistivity floor

    .. math::

        \eta_\mathrm{field} = \max\!\big(\eta_\mathrm{user},\;
        \mu_0 D_\mathrm{vac}\,(\rho_\mathrm{ref,\Omega}/\rho)^2\big),

    with :math:`\rho_\mathrm{ref,\Omega}` the Ohm density guard
    (:pp:param:`hybrid_pic_model.n_floor` equivalent) and the density
    division guarded at the far-lower positivity floor. Below the Ohm
    guard the "fluid" is a numerical halo/pedestal rather than plasma,
    and the physically consistent field behavior is vacuum-like
    diffusion of the response currents: the magnetic diffusivity grows
    as :math:`\rho^{-2}` below the guard (reaching
    :math:`D_\mathrm{vac}` at the guard) and vanishes as
    :math:`\rho^{-2}` above it, so the bulk keeps the user resistivity.
    The max is assembled :math:`C^\infty`-smoothly (quadrature form) for
    matrix-free Jacobian probes. Joule heating keeps the un-boosted user
    :math:`\eta`: vacuum field diffusion never heats plasma. Pairs
    naturally with :pp:param:`implicit_mhd.resistive_theta` ``= 1``,
    which damps the stiff halo field modes this term creates. With
    :pp:param:`implicit_mhd.vacuum_reference_peak_fraction` and/or
    :pp:param:`implicit_mhd.vacuum_reference_base_density` the static
    Ohm-guard reference is raised to the dynamic tenth-of-peak /
    static ``en00`` reference (the reference code's ``en0``).

.. pp:param:: implicit_mhd.vacuum_reference_peak_fraction
    :type: ``float``
    :default: ``0`` (off)

    Reference-code-style DYNAMIC reference density of the density-keyed halo
    mechanisms (``step.f90``: ``en0 = MAX(en00, 0.1*MAXVAL(en))``,
    refreshed every step). When positive, the reference density keying
    the field-eta vacuum boost
    (:pp:param:`implicit_mhd.vacuum_resistivity_diffusivity`), the
    :pp:param:`implicit_mhd.joule_ohm_current` diffusion-dominance
    criterion (through the boosted :math:`\eta_\mathrm{field}` it
    tests), and the conduction halo boost
    (:pp:param:`implicit_mhd.conduction_halo_boost`) becomes

    .. math::

        \rho_\mathrm{ref} = \max\!\big(\rho_\mathrm{ref,\Omega},\;
        \rho_\mathrm{en00},\; f \, \rho_\mathrm{peak}\big),

    with :math:`\rho_\mathrm{peak}` the global maximum of the STEP-OLD
    mass density, recomputed once per step at step start and FROZEN for
    the whole nonlinear solve (every residual and Jacobian evaluation
    keys the boosts to the same reference: Newton-consistent),
    :math:`\rho_\mathrm{en00}` the static base
    :pp:param:`implicit_mhd.vacuum_reference_base_density`, and
    :math:`\rho_\mathrm{ref,\Omega}` the static Ohm density guard —
    the exact base ``= f = 0`` limit, so the default is bit-identical.
    Rationale: the static guard protects only near-vacuum cells,
    leaving the intermediate halo band under full Joule heating and
    physical (tiny) cross-field conduction; keying on a tenth of the
    peak tracks compression like the reference code's ``en0``. The Joule heating
    COEFFICIENT keeps the un-boosted user :math:`\eta` and the
    eta-parser density arguments keep the static Ohm-guard floor
    either way. The effective reference is printed (rank 0) on >1%
    change.

.. pp:param:: implicit_mhd.vacuum_reference_base_density
    :type: ``float``
    :unit: :math:`\mathrm{kg\,m^{-3}}`
    :default: ``0`` (off)

    Static base of the shared reference density: the reference code's ``en00``
    glob card as a mass density (the reference shot flies ``en00 = 3.3e20``
    m\ :sup:`-3`). Enters the reference as
    :math:`\max(\rho_\mathrm{ref,\Omega}, \rho_\mathrm{en00},
    f\rho_\mathrm{peak})`; ``0`` keeps the legacy Ohm-guard base.
    Calibration identity: the field-eta boost curve depends only on the
    product :math:`D_\mathrm{vac}\,\rho_\mathrm{ref}^2`, so switching
    the base from the Ohm guard :math:`n_g` to ``en00`` while rescaling
    :pp:param:`implicit_mhd.vacuum_resistivity_diffusivity` by
    :math:`(n_g/\mathrm{en00})^2` leaves the field advance unchanged
    and keys the conduction halo boost
    (:pp:param:`implicit_mhd.conduction_halo_boost`) at the reference code's
    reference.

.. pp:param:: implicit_mhd.joule_ohm_current
    :type: ``bool``
    :default: ``false``

    Reference-code-style Ohm-current Joule quench. In cells where the field
    advance is diffusion dominated,

    .. math::

        \Delta t \, \eta_\mathrm{field} / \mu_0 > \min(\Delta x)^2,

    the :math:`|\boldsymbol J|^2` of the electron-energy Joule deposit is
    evaluated from Ohm's law as :math:`|\boldsymbol
    E|^2/\eta_\mathrm{field}^2` — with :math:`\boldsymbol E` the solved
    stage electric field and :math:`\eta_\mathrm{field}` the same
    floored/boosted field-advance resistivity the field solve uses
    (including the vacuum boost of
    :pp:param:`implicit_mhd.vacuum_resistivity_diffusivity` and the
    :pp:param:`implicit_mhd.wall_band_eta_override`) — instead of the
    pointwise curl-B current. The heating coefficient keeps the
    un-boosted user :math:`\eta` (see
    :pp:param:`implicit_mhd.include_joule_heating`), so where
    :math:`\eta_\mathrm{field}` is vacuum-boosted the halo Joule deposit
    is quenched quadratically: the stiff implicit field solve relaxes
    :math:`\boldsymbol E` while the pointwise curl-B current stays at
    gradient-noise scale. Binary per-cell switch (no smooth blend),
    matching the reference code's "fancy current" rule.

.. pp:param:: implicit_mhd.resistive_theta
    :type: ``float``
    :default: :pp:param:`implicit_evolve.theta`

    Time centering of the dissipative Ohm terms (:math:`\eta
    \boldsymbol J`, including the vacuum-resistivity boost, and the
    hyper-resistive term), in ``[0.5, 1]``; the ideal EMF keeps the
    global :pp:param:`implicit_evolve.theta`. Since :math:`\boldsymbol J
    = \nabla\times\boldsymbol B/\mu_0` is linear in the unknowns, the
    dissipative terms are evaluated at the exactly extrapolated stage
    current :math:`\boldsymbol J^{n+\theta_r} = (\theta_r/\theta)
    \boldsymbol J^{n+\theta} + (1 - \theta_r/\theta) \boldsymbol J^n`,
    which Jacobian probes see exactly. Motivation: a stiff resistive
    mode with :math:`z = \eta \Delta t k^2/\mu_0 \gg 1` has trapezoidal
    amplification :math:`(1 - z/2)/(1 + z/2) \to -1` -- marginally
    damped and sign-flipping every step, measured as a never-decaying
    halo residual plateau under strong vacuum resistivity -- while
    backward Euler on the dissipative terms alone (``resistive_theta =
    1``) gives :math:`1/(1+z) \to 0`, damping those modes in one step
    with the ideal dynamics still second-order centered. Values other
    than the global theta require ``fluid_flux = hlld`` or ``central``.

.. pp:param:: implicit_mhd.resistive_direct_device_assembly
    :type: ``string``
    :default: ``auto``

    Assembly and transport mode of the direct resistive preconditioner
    block (``pc_mhd_block.resistive_solver = direct``). ``auto`` (the
    default) keeps the whole refreeze/apply cycle device-resident on
    single-rank runs: after the one-time host sparsity freeze, the CSR
    values are recomputed in kernels from the coefficient fields' device
    arrays through the frozen entry-to-slot map, and every application
    packs the right-hand side and unpacks the solution on the device --
    no per-refreeze coefficient mirrors and no per-apply host round
    trip. Multi-rank runs fall back to the host gather path
    automatically. ``on`` forces the device path (aborts on multi-rank
    runs); ``off`` forces the host gather everywhere. With
    :pp:param:`pc_mhd_block.resistive_validate_assembly` the device- and
    host-assembled value arrays are additionally gated bitwise
    (:math:`\le 4` ULP; exact equality expected, the allowance absorbs
    device FMA contraction only) at every update.

.. pp:param:: implicit_mhd.conduction_theta
    :type: ``float``
    :default: :pp:param:`implicit_evolve.theta`

    Time centering of the thermal-conduction stage, in ``[0.5, 1]``;
    the wave/advective dynamics keep the global
    :pp:param:`implicit_evolve.theta`. The conductive-flux ENERGY
    arguments -- the specific internal energies whose
    differences/gradients drive the flux, the Braginskii tangential
    stencil samples, the wall thermal-drain interior energy, and the
    free-streaming-cap temperatures -- are evaluated at the exact stage
    extrapolation :math:`e^{n+\theta_c} = (\theta_c/\theta)
    e^{n+\theta} + (1 - \theta_c/\theta) e^n`, linear in the Newton
    iterate, so matrix-free Jacobian probes see the shifted centering
    exactly (the conduction twin of
    :pp:param:`implicit_mhd.resistive_theta`). Motivation
    (L-stability): the trapezoidal rule is A-stable but not L-stable --
    at halo diffusion numbers :math:`\theta \Delta t \chi/\Delta x^2
    \gg 1` grid-Nyquist conduction modes come through with
    amplification :math:`(1 - z/2)/(1 + z/2) \to -1` (amplitude
    :math:`\sim 1`, sign flipped every step), measured in production as
    a halo temperature checkerboard preceding Newton freeze-guard
    aborts, while backward-Euler conduction (``conduction_theta = 1``)
    damps them as :math:`1/(1 + z) \to 0` in one step. The conduction
    COEFFICIENTS keep their own
    :pp:param:`implicit_mhd.conduction_coefficient_state` rule. Values
    other than the global theta require an active thermal-conduction
    channel.

.. pp:param:: implicit_mhd.evolve_ion_fluid
    :type: ``bool``
    :default: ``true``

    Advance ion mass and momentum in the JFNK residual. Setting this to
    ``false`` freezes the prescribed ion background while retaining the
    electron-energy and electromagnetic unknowns. With
    :pp:param:`hybrid_pic_model.include_hall_term = true` and zero ion
    velocity, this selects an electron-MHD limit with
    :math:`\boldsymbol u_e=-\boldsymbol J/\rho_q`.

.. pp:param:: implicit_mhd.external_field_iteration
    :type: ``bool``
    :default: ``false``

    Circuit-in-the-residual coupling. Every residual evaluation executes the
    ``externalcoiltheta`` python callback (and the end-of-step commit the
    ``externalcoilfinish`` one) and then refreshes the split external fields,
    so python can re-advance a coupled external circuit against the current
    iterate's plasma flux linkage and push updated coil scale segments
    (``warpx.set_external_vector_potential_scale``). Requires
    :pp:param:`hybrid_pic_model.add_external_fields`.

.. pp:param:: implicit_mhd.circuit_hook_scope
    :type: ``string``
    :default: ``residual``

    Scope of the ``externalcoiltheta`` hook when
    :pp:param:`implicit_mhd.external_field_iteration` is on. ``residual``
    fires it on every residual evaluation (bit-identical default).
    ``newton`` fires it only on the residual evaluations at accepted Newton
    iterates: matrix-free Jacobian probes and line-search trials skip the
    python round-trip and reuse the coil scales cached from the last iterate
    evaluation. The Jacobian then sees a per-iterate lagged circuit
    (quasi-Newton) — possibly a few extra Newton iterations, hundreds fewer
    python round-trips per step — while the converged answer is unchanged to
    solver tolerance, because convergence is still tested on the
    live-coupled iterate residual and the end-of-step ``externalcoilfinish``
    commit is unaffected.

.. pp:param:: implicit_mhd.circuit_driver
    :type: ``string``
    :default: ``python``

    Which coupler fires at the circuit hook points when
    :pp:param:`implicit_mhd.external_field_iteration` is on. ``python``
    (bit-identical default) executes the
    ``externalcoiltheta``/``externalcoilfinish`` callbacks. ``native``
    drives the C++ circuit-coupling engine in-process instead (requires
    :pp:param:`circuit.coils` and :pp:param:`circuit.engine`, typically
    ``external`` with a compiled plugin): each firing measures the
    iterate's plasma flux linkages with the engine's batched device
    probes (one device pass, one stream synchronization and one
    all-reduce per measurement, reusing the plasma current the residual
    evaluation just computed), re-advances the engine from the committed
    step entry to the theta-stage time (``accept = false``: repeatable,
    discontinuous engine transitions stay latched at committed state)
    and realizes the returned coil scales through the same external-field
    refresh as the python path. The end-of-step commit performs the
    single ``accept = true`` advance over :math:`[t^n, t^{n+1}]` on the
    accepted state, where engine switches/stage swaps latch, followed by
    the engine's ``FinishStep``. Honors both
    :pp:param:`implicit_mhd.circuit_hook_scope` values: ``residual``
    restores exact circuit-in-residual coupling (the python path's
    per-evaluation host round-trip and device-to-host flux pulls are
    eliminated entirely); ``newton`` keeps the lagged quasi-Newton
    economy. The engine passes the measured per-interval EMF unfiltered;
    any smoothing is engine-side policy.

.. pp:param:: implicit_mhd.mass_density(x,y,z)
    :type: ``string``
    :unit: :math:`\mathrm{kg\,m^{-3}}`

    Required analytic initial ion mass-density profile, evaluated at cell
    centers.

.. pp:param:: implicit_mhd.electron_pressure(x,y,z)
    :type: ``string``
    :unit: Pa

    Required analytic initial electron-pressure profile, evaluated at cell
    centers and converted to electron internal energy.

.. pp:param:: implicit_mhd.ion_pressure(x,y,z)
    :type: ``string``
    :unit: Pa

    Analytic initial ion-pressure profile, evaluated at cell centers and
    combined with the initial velocity to form the initial ion total energy.
    Required when ``implicit_mhd.ion_closure = total_energy``; ignored by the
    barotropic closure.

.. pp:param:: implicit_mhd.velocity_[x/y/z](x,y,z)
    :type: ``string``
    :unit: :math:`\mathrm{m\,s^{-1}}`
    :default: ``0``

    Analytic initial ion-fluid velocity components.

.. pp:param:: hybrid_pic_model.plasma_resistivity(rho,J,t)
    :type: ``float`` or ``str``
    :default: ``0``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the plasma resistivity in :math:`\Omega m`.
    The expression may use the charge density ``rho`` (:math:`C/m^3`), the
    current-density magnitude ``J`` (:math:`A/m^2`), and the time ``t``
    (:math:`s`).

.. pp:param:: hybrid_pic_model.plasma_resistivity(rho,Te,J,t)
    :type: ``float`` or ``str``
    :default: ``0``
    :optional:

    Extended signature of :pp:param:`hybrid_pic_model.plasma_resistivity(rho,J,t)`
    (supply one or the other, not both) adding the electron temperature
    ``Te`` in Kelvin, read from the nodal electron-temperature register at
    the same locations and staggering as the density argument (for
    ``theta_implicit_mhd`` this is the temperature-primary
    :math:`T_e = p_e/(n_f k_B)` of the evolving solver state, so
    e.g. Spitzer :math:`\eta \propto T_e^{-3/2}` sweeps respond live to
    Joule heating). All state symbols (``rho``, ``Te``, ``J``) must enter
    through smooth (:math:`C^\infty`) expressions for the implicit solvers'
    matrix-free Jacobian probes: use smooth floors such as
    ``(Te^2 + Tf^2)^(-0.75)`` instead of ``max(Te, Tf)^(-1.5)``. The
    implicit preconditioners evaluate the same parser with the state frozen
    at the preconditioner update (the standard lagged-coefficient
    treatment).

.. pp:param:: hybrid_pic_model.plasma_hyper_resistivity(rho,B)
    :type: ``float`` or ``str``
    :default: ``0``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the plasma hyper-resistivity in :math:`\Omega m^3`.

.. pp:param:: hybrid_pic_model.plasma_resistivity_<species>(rho_s,rho,Te,J,J_s,B,t)
    :type: ``float`` or ``str``
    :default: ``0``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this adds a per-species resistivity overlay in :math:`\Omega m`
    for the named charged species, on top of :pp:param:`hybrid_pic_model.plasma_resistivity(rho,J,t)`
    (see the :ref:`theory section <theory-kinetic-fluid-hybrid-model>`). The expression can depend on the species
    charge density ``rho_s`` and total charge density ``rho`` (:math:`C/m^3`), the electron temperature ``Te`` (:math:`K`),
    the current-density magnitudes ``J`` and ``J_s`` (:math:`A/m^2`), the magnetic-field magnitude ``B`` (:math:`T`)
    and the time ``t`` (:math:`s`). The same effective per-species resistivity enters the Joule-heating source of the
    electron energy equation when :pp:param:`hybrid_pic_model.include_joule_heating` is on.
    Species without their own overlay simply use the global
    :pp:param:`hybrid_pic_model.plasma_resistivity(rho,J,t)`, so existing single-resistivity input decks are unchanged.
    Note that in RZ geometry ``J_s`` is the magnitude of the raw (not volume-scaled) species current deposit;
    in Cartesian geometry it is the physical current-density magnitude.

.. pp:param:: hybrid_pic_model.solve_electron_energy_equation
    :type: ``bool``
    :default: ``false``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this evolves the electron temperature used for the
    electron pressure with the electron energy equation, solved with the QDSMC scheme
    (see the :ref:`theory section <theory-hybrid-model-electron-energy-eq>`), instead of evaluating the polytropic
    closure with the constant reference state :math:`(n_0, T_{e0})`.

.. pp:param:: hybrid_pic_model.qdsmc_n_floor
    :type: ``float``
    :default: ``1``
    :optional:

    Density floor, in :math:`m^{-3}`, below which cells are excluded from the QDSMC electron-energy-equation
    update (the electron temperature is left unchanged there).

.. pp:param:: hybrid_pic_model.include_joule_heating
    :type: ``bool``
    :default: ``false``
    :optional:

    If :pp:param:`hybrid_pic_model.solve_electron_energy_equation` is on, this adds the Joule-heating source
    consistent with the resistive friction in Ohm's law, applied per ion species with the effective resistivity
    :math:`\eta_{s,\mathrm{eff}} = \eta + \eta_s`. For a single species this reduces to
    :math:`dT_e/dt = (\gamma - 1)\,\eta J^2/(n_e k_B)`.

.. pp:param:: hybrid_pic_model.joule_redirect_Te_threshold
    :type: ``float``
    :default: ``-1`` (off)
    :optional:

    Electron temperature threshold, in eV, above which the Joule heat is redirected to the ions.
    If :pp:param:`hybrid_pic_model.include_joule_heating` is on and a threshold :math:`\geq 0` is specified,
    cells with electron temperature at or above the threshold deposit their Joule heat to the kinetic ions
    (as stochastic thermal-velocity kicks, bookkept per species) instead of the electron fluid. This caps the
    electron heating at the threshold and allows :math:`T_i > T_e` to develop, mimicking regimes where the
    electrons radiate strongly.

.. pp:param:: hybrid_pic_model.electron_ion_relaxation_rate(rho,Te,Ti,t)
    :type: ``float`` or ``str``
    :optional:

    The electron-ion relaxation rate :math:`\nu_{ei}`, in :math:`s^{-1}`. If
    :pp:param:`hybrid_pic_model.solve_electron_energy_equation` is on, specifying this rate enables the
    electron-ion thermal-equilibration exchange :math:`Q_{ei} = \sum_s 3 n_s k_B \nu_{ei} (T_e - T_{i,s})`
    as a sink on the electron fluid, paired with matching (energy-conserving) heating of the ion
    macro-particles. The required shape-aware ion temperature deposition
    (``<species>.do_temperature_deposition``) is enabled automatically on every charged species.
    The expression can depend on the total charge density ``rho`` (:math:`C/m^3`), the electron and ion
    temperatures ``Te`` and ``Ti`` (both in eV) and the time ``t`` (:math:`s`), which permits, e.g., the
    NRL-formulary Spitzer rate.

.. pp:param:: hybrid_pic_model.J[x/y/z]_external_grid_function(x,y,z,t)
    :type: ``float`` or ``str``
    :default: ``0``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the external current (on the grid) in :math:`A/m^2`.

.. pp:param:: hybrid_pic_model.n_floor
    :type: ``float``
    :default: ``1``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the plasma density floor, in :math:`m^{-3}`, which is useful since the generalized Ohm's law used to calculate the E-field includes a :math:`1/n` term.

.. pp:param:: hybrid_pic_model.substeps
    :type: ``int``
    :default: ``10``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the total number of sub-steps used to advance
    the B-field over one full timestep (split evenly between the two half-steps, so ``substeps/2`` RK4 steps are taken
    per half-step, each of duration :math:`\Delta t / \text{substeps}`). Must be divisible by 2; if not, the value is
    automatically rounded up to the next even number. When :pp:param:`hybrid_pic_model.use_rkf45` is active, this is
    instead used only as the initial substep count estimate for the adaptive solver.
    After each timestep on which :pp:param:`hybrid_pic_model.use_rkf45` is active, this value is updated based on
    ``n_attempts``, the total number of RKF45 sub-step attempts (accepted and rejected) taken in the most recent
    half-step: if the current value is less than ``2 * n_attempts``, it jumps immediately to ``2 * n_attempts``;
    otherwise it decays slowly toward that target via exponential smoothing (95% of the current value,
    5% of ``2 * n_attempts``). This warm-start guess also carries over to RK4 steps on timesteps where
    :pp:param:`hybrid_pic_model.use_rkf45` is not active.

.. pp:param:: hybrid_pic_model.use_rkf45
    :type: ``string`` or ``bool``
    :default: ``false``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this selects the B-field sub-step integrator.
    When ``false`` (default), a fixed-step classical RK4 method is used with exactly
    :pp:param:`hybrid_pic_model.substeps` total sub-steps per timestep.
    When ``true``, the adaptive Runge-Kutta-Fehlberg 4(5) (RKF45) method :cite:t:`param-Fehlberg1969`
    is used, controlling the local truncation error to stay within
    :pp:param:`hybrid_pic_model.substep_rtol` and :pp:param:`hybrid_pic_model.substep_atol`.

    This parameter also accepts the `Time intervals`_ syntax to enable the RKF45 integrator only
    on specific timesteps, e.g. ``hybrid_pic_model.use_rkf45 = 1::5``, which enables RKF45
    every 5 steps from step 1.
    When ``false`` or ``0``, the RKF45 integrator is never used.
    When ``true``, ``1``, or ``::`` the RKF45 integrator is used at every timestep.

.. pp:param:: hybrid_pic_model.substep_rtol
    :type: ``float``
    :default: ``1e-4``
    :optional:

    If :pp:param:`hybrid_pic_model.use_rkf45` is active, this sets the relative tolerance for the RKF45
    adaptive step-size control.

.. pp:param:: hybrid_pic_model.substep_atol
    :type: ``float``
    :default: ``1e-8``
    :optional:

    If :pp:param:`hybrid_pic_model.use_rkf45` is active, this sets the absolute tolerance for the RKF45
    adaptive step-size control.

.. pp:param:: hybrid_pic_model.substep_safety
    :type: ``float``
    :default: ``0.9``
    :optional:

    If :pp:param:`hybrid_pic_model.use_rkf45` is active, this sets the safety factor applied to the
    step-size adjustment formula.

.. pp:param:: hybrid_pic_model.substep_max_growth
    :type: ``float``
    :default: ``5.0``
    :optional:

    If :pp:param:`hybrid_pic_model.use_rkf45` is active, this sets the maximum factor by which the
    substep size may grow after an accepted step.

.. pp:param:: hybrid_pic_model.max_substep_attempts
    :type: ``int``
    :default: ``250``
    :optional:

    If :pp:param:`hybrid_pic_model.use_rkf45` is active, this sets the maximum number of substep attempts
    (accepted and rejected combined) per half-step before the simulation aborts.

.. pp:param:: hybrid_pic_model.holmstrom_vacuum_region
    :type: ``bool``
    :default: ``false``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the vacuum region handling of the generalized Ohm's Law to suppress vacuum fluctuations. :cite:t:`param-holmstrom2013handlingvacuumregionshybrid`.

.. pp:param:: hybrid_pic_model.add_external_fields
    :type: ``bool``
    :default: ``false``
    :optional:

    If :pp:param:`algo.maxwell_solver` is set to ``hybrid``, this sets the hybrid solver to use split external fields defined in external_vector_potential inputs.

.. pp:param:: external_vector_potential.do_diva_cleaning
    :type: ``bool``
    :default: ``true``
    :optional:

    This enables or disables the divergence cleaner application to the external A fields.

.. pp:param:: external_vector_potential.fields
    :type: list of ``str``
    :default: ``empty``
    :optional:

    If :pp:param:`hybrid_pic_model.add_external_fields` is set to ``true``, this adds a list of names for external time varying vector potentials to be added to hybrid solver.

.. pp:param:: external_vector_potential.<field_name>.read_from_file
    :type: ``bool``
    :default: ``false``
    :optional:

    If :pp:param:`hybrid_pic_model.add_external_fields` is set to ``true``, this flag determines whether to load an external field or use an implicit function to evaluate the time varying field.

.. pp:param:: external_vector_potential.<field_name>.path
    :type: ``str``
    :default: ``""``
    :optional:

    If :pp:param:`external_vector_potential.<field_name>.read_from_file` is set to ``true``, sets the path to an OpenPMD file that can be loaded externally in :math:`weber/m`.

.. pp:param:: external_vector_potential.<field_name>.A[x,y,z]_external_grid_function(x,y,z)
    :type: ``str``
    :default: ``"0"``
    :optional:

    If :pp:param:`external_vector_potential.<field_name>.read_from_file` is set to ``false``, Sets the external vector potential to be populated by an implicit function (on the grid) in :math:`weber/m`.

.. pp:param:: external_vector_potential.<field_name>.A_time_external_grid_function(t)
    :type: ``str``
    :default: ``"1"``
    :optional:

    This sets the relative strength of the external vector potential by a dimensionless implicit time function, which can compute the external B fields and E fields based on the value and first time derivative of the function.

.. pp:param:: external_vector_potential.<field_name>.python_scale
    :type: ``bool``
    :default: ``false``
    :optional:

    Drive this field's dimensionless scale through piecewise-linear segments pushed at runtime
    (``warpx.set_external_vector_potential_scale(name, s_old, s_new, t_old, t_new)`` from Python,
    e.g. from a circuit model in a callback) instead of the compiled
    :pp:param:`external_vector_potential.<field_name>.A_time_external_grid_function(t)`.
    The B field follows the linear interpolant of the segment and the E field carries its exact
    constant slope (the discrete Faraday partner of the linear B scale). The segment is
    extrapolated linearly past ``t_new``, so a fresh segment must be pushed every coupling
    interval; a refresh well past ``t_new`` records a high-priority warning.

.. pp:param:: external_vector_potential.<field_name>.initial_scale
    :type: ``float``
    :default: ``1.``
    :optional:

    With :pp:param:`external_vector_potential.<field_name>.python_scale` enabled, the constant
    scale held (e.g. a coil at its pre-ramp current) until the first runtime segment is pushed.

.. pp:param:: circuit.coils
    :type: list of ``str``
    :optional:

    Names of circular filament coils whose fields drive the hybrid solver's split external
    fields (requires :pp:param:`hybrid_pic_model.add_external_fields`). Each coil pairs with an
    entry of :pp:param:`external_vector_potential.fields` (by default the entry with the coil's
    own name) and, unless disabled, its ``<name>_Aext`` field is filled at initialization with
    the coil's field at the reference amp-turns from the axisymmetric ring kernel (a direct
    free-space evaluation; the coil may sit outside the domain). In RZ geometry the fill uses
    the discrete mesh convention (quarter-cell filament offset, Legendre-parameter clip) whose
    disk-flux self-inductance is the value a coupled circuit port must use; the discrete self-
    and mutual-inductance table of the coil set on the run mesh is printed at initialization.
    Per-coil parameters, prefixed ``circuit.<name>.``:

    * ``r``, ``z`` (required): filament radius and axial position [m]; ``r`` must be positive.
    * ``n_turns`` (default ``1``): turn count; the filled field and the inductances scale with it.
    * ``I_ref`` (default ``1``): reference current [A]; a drive scale of 1 (see
      :pp:param:`external_vector_potential.<field_name>.python_scale`) reproduces the coil at
      ``I_ref``.
    * ``field_name`` (default: the coil name): the paired external-field entry.
    * ``fill_unit_field`` (default ``1``): fill ``<field_name>_Aext`` from the ring kernel;
      disable to keep whatever the external-field initialization loaded (file or expressions).
    * ``probe`` (default ``default``): the coil's plasma flux-linkage measurement for the
      coupling engine — ``disk`` (plasma-frame :math:`B_z` through the coil circle, the same
      staircase rules as the discrete self-inductance; valid with conducting walls),
      ``reciprocity`` (:math:`\int A^\mathrm{unit} \cdot J_p\, dV`; exact in free space,
      requires the Green's-function open boundary), or ``none`` (drive-only).
      ``default`` selects reciprocity when the open boundary is active, disk otherwise.

.. pp:param:: circuit.engine
    :type: ``str``
    :default: ``none``
    :optional:

    The external circuit engine coupled to the coils per B-field substep (RZ):
    ``callbacks`` drives the coupling through the Python hooks ``circuitbeginstep`` /
    ``circuitpredict`` / ``circuitcorrect`` / ``circuitfinish`` (the handlers read the coupling
    interval and the flux-linkage registers via ``get_coupling_interval`` /
    ``get_coil_flux_linkage`` and push scale segments via
    ``set_external_vector_potential_scale``); ``external`` loads a compiled engine from
    :pp:param:`circuit.plugin_library` (a shared library exporting
    ``warpx_create_external_circuit``). Every measured coil's paired field must be declared
    with :pp:param:`external_vector_potential.<field_name>.python_scale`. Each accepted
    substep of the (possibly adaptive) B-field advance is one predictor-corrector coupling
    interval: the engine advances the circuit from the interval entry with held EMF
    estimates, the substep integrates on the refreshed circuit-driven fields, and each
    corrector pass re-advances the circuit with the measured
    :math:`\varepsilon = \Delta\lambda_p/\Delta t` and re-integrates the substep until the
    realized scales settle. The lagged variant (``corrector_iterations = 0``) is unstable
    for strong coil-plasma coupling.

.. pp:param:: circuit.plugin_library
    :type: ``str``
    :optional:

    With :pp:param:`circuit.engine` = ``external``, the path of the engine's shared library.
    The library must export ``warpx_create_external_circuit`` and
    ``warpx_external_circuit_abi_version`` (checked against the
    ``WARPX_EXTERNAL_CIRCUIT_ABI_VERSION`` of the running WarpX at load time; see
    ``Source/Circuit/ExternalCircuit.H``, whose plugin ABI is self-contained: the engine
    receives per-coil EMF estimates and returns realized coil scales, calling no WarpX
    symbols).

.. pp:param:: circuit.plugin_config
    :type: ``str``
    :optional:

    With :pp:param:`circuit.engine` = ``external``, a free-form string handed to the
    engine's ``Define`` (typically the path of the engine's own configuration file; its
    format is entirely the engine's business). An inline ``key=value,...`` string must be
    double-quoted in the inputs so the parameter parser keeps it whole.

.. pp:param:: circuit.plugin_restart_config
    :type: ``str``
    :optional:

    Optional replacement for :pp:param:`circuit.plugin_config` on restart runs. The
    restored checkpoint (``ExternalCircuit::ReadCheckpoint``) supersedes any engine-side
    boot state, so a restart configuration can skip boot work that ``Define`` would
    otherwise redo (e.g. a machine-time pre-roll). When unset, restarts hand ``Define``
    the ordinary :pp:param:`circuit.plugin_config`.

.. pp:param:: circuit.probe_crosscheck
    :type: ``bool``
    :default: ``false``
    :optional:

    Validation knob: cross-check every batched linkage measurement against the
    single-coil reference probes (``DiskFluxLinkage``/``ReciprocityLinkage``), print the
    per-coil deltas and abort when they disagree beyond
    :pp:param:`circuit.probe_crosscheck_rtol`. The batched path folds the identical
    integrand factors into per-coil weight tables and differs from the reference only in
    floating-point summation order.

.. pp:param:: circuit.probe_crosscheck_rtol
    :type: ``float``
    :default: ``1.e-12``
    :optional:

    Relative disagreement bound of :pp:param:`circuit.probe_crosscheck`.

.. pp:param:: circuit.coupling.corrector_iterations
    :type: ``int``
    :default: ``1``
    :optional:

    Corrector passes per coupling substep (``0`` = lagged predictor only).

.. pp:param:: circuit.coupling.corrector_rtol
    :type: ``float``
    :default: ``1.e-6``
    :optional:

    Early-exit tolerance of the corrector: converged when the re-advanced coil scales move
    by less than this relative amount.


Grid types (collocated, staggered, hybrid)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. pp:param:: warpx.grid_type
    :type: ``string``, ``collocated``, ``staggered`` or ``hybrid``

    Whether to use a collocated grid (all fields defined at the cell nodes),
    a staggered grid (fields defined on a Yee grid), or a hybrid grid (fields
    and currents are interpolated back and forth between a staggered grid and a
    nodal grid, must be used with momentum-conserving field gathering algorithm,
    :pp:param:`algo.field_gathering = momentum-conserving`).
    The option ``hybrid`` is currently not supported in RZ, RCYLINDER, and RSPHERE geometries.

    Default: :pp:param:`warpx.grid_type = staggered`.

.. pp:param:: interpolation.galerkin_scheme
    :type: ``0`` or ``1``

    Whether to use a Galerkin scheme when gathering fields to particles.
    When set to ``1``, the interpolation orders used for field-gathering are reduced for certain field components along certain directions.
    For example, :math:`E_z` is gathered using :pp:param:`algo.particle_shape` along :math:`(x,y)` and :pp:param:`algo.particle_shape - 1` along :math:`z`.
    See equations (21)-(23) of :cite:t:`param-Godfrey2013` and associated references for details.

    Default: :pp:param:`interpolation.galerkin_scheme = 0` with collocated grids, or momentum-conserving field gathering, or when :pp:param:`algo.current_deposition = direct` ; :pp:param:`interpolation.galerkin_scheme = 1` otherwise.

    .. warning::

        The default behavior should not normally be changed.
        At present, this parameter is intended mainly for testing and development purposes.

.. pp:param:: warpx.field_centering_nox/noy/noz
    :link_aliases:
        warpx.field_centering_nox
        warpx.field_centering_noy
        warpx.field_centering_noz
    :type: ``integer``
    :optional:

    The order of interpolation used with staggered or hybrid grids (:pp:param:`warpx.grid_type = staggered` or :pp:param:`warpx.grid_type = hybrid`) and momentum-conserving field gathering (:pp:param:`algo.field_gathering = momentum-conserving`) to interpolate the electric and magnetic fields from the cell centers to the cell nodes, before gathering the fields from the cell nodes to the particle positions.

    Default: ``warpx.field_centering_no<x,y,z> = 2`` with staggered grids, ``warpx.field_centering_no<x,y,z> = 8`` with hybrid grids (typically necessary to ensure stability in boosted-frame simulations of relativistic plasmas and beams).

.. pp:param:: warpx.current_centering_nox/noy/noz
    :link_aliases:
        warpx.current_centering_nox
        warpx.current_centering_noy
        warpx.current_centering_noz
        warpx.current_centering_no<x,y,z>
    :type: ``integer``
    :optional:

    The order of interpolation used with hybrid grids (:pp:param:`warpx.grid_type = hybrid`) to interpolate the currents from the cell nodes to the cell centers when :pp:param:`warpx.do_current_centering = 1`, before pushing the Maxwell fields on staggered grids.

    Default: :pp:param:`warpx.current_centering_no<x,y,z> = 8` with hybrid grids (typically necessary to ensure stability in boosted-frame simulations of relativistic plasmas and beams).

.. pp:param:: warpx.do_current_centering
    :type: ``bool``, ``0`` or ``1``

    If true, the current is deposited on a nodal grid and then centered to a staggered grid (Yee grid), using finite-order interpolation.

    Default: :pp:param:`warpx.do_current_centering = 0` with collocated or staggered grids, :pp:param:`warpx.do_current_centering = 1` with hybrid grids.

Additional parameters
^^^^^^^^^^^^^^^^^^^^^

.. pp:param:: warpx.do_dive_cleaning
    :type: ``0`` or ``1``
    :default: 0

    Whether to use modified Maxwell equations that progressively eliminate
    the error in :math:`div(E)-\rho`. This can be useful when using a current
    deposition algorithm which is not strictly charge-conserving, or when
    using mesh refinement. These modified Maxwell equation will cause the error
    to propagate (at the speed of light) to the boundaries of the simulation
    domain, where it can be absorbed.

.. pp:param:: warpx.do_initial_div_cleaning
    :type: ``0`` or ``1``
    :default: 0

    Whether to use a projection method to scrub A/B field divergence in externally
    loaded fields. This applies to both externally loaded grid fields
    (``warpx.B_ext_grid_init_style``) and externally applied particle fields
    (``particles.B_ext_particle_init_style = read_from_file``); when several applied
    field maps are stacked, each map is cleaned independently. It is supported for the
    electromagnetic (Yee, hybrid-PIC), electrostatic (labframe) and magnetostatic
    (labframe-electromagnetostatic, with the multigrid Poisson solver) solvers.
    This is automatically turned on if external/initial grid B or time varying A fields
    are loaded; for applied particle fields it must be enabled explicitly (opt-in).

.. pp:param:: warpx.projection_div_cleaner.rtol
    :type: ``float``
    :default: ``5e-12`` when double precision and ``5e-5`` for single precision
    :optional:

    Controls the relative tolerance when solving for the projected divergence of the field in the MLMG AMReX solver.

.. pp:param:: warpx.projection_div_cleaner.atol
    :type: ``float``
    :default: ``0``
    :optional:

    Controls the absolute tolerance when solving for the projected divergence of the field in the MLMG AMReX solver.

.. pp:param:: warpx.do_subcycling
    :type: ``0`` or ``1``
    :default: 0

    Whether or not to use sub-cycling. Different refinement levels have a
    different cell size, which results in different Courant–Friedrichs–Lewy
    (CFL) limits for the time step. By default, when using mesh refinement,
    the same time step is used for all levels. This time step is
    taken as the CFL limit of the finest level. Hence, for coarser
    levels, the timestep is only a fraction of the CFL limit for this
    level, which may lead to numerical artifacts. With sub-cycling, each level
    evolves with its own time step, set to its own CFL limit. In practice, it
    means that when level 0 performs one iteration, level 1 performs two
    iterations. Currently, this option is only supported when
    :pp:param:`amr.max_level = 1`. More information can be found at
    https://ieeexplore.ieee.org/document/8659392.

.. pp:param:: warpx.override_sync_intervals
    :type: ``string``
    :default: ``1``
    :optional:

    Using the `Time intervals`_ syntax, this string defines the timesteps at which
    synchronization of sources (``rho`` and ``J``) and fields (``E`` and ``B``) on grid nodes at box
    boundaries is performed. Since the grid nodes at the interface between two neighbor boxes are
    duplicated in both boxes, an instability can occur if they have too different values.
    This option makes sure that they are synchronized periodically.
    Note that if Perfectly Matched Layers (PML) are used, synchronization of the ``E`` and ``B`` fields
    is performed at every timestep regardless of this parameter.

.. pp:param:: warpx.do_device_synchronize
    :type: ``bool``
    :default: ``1``
    :optional:

    When running in an accelerated platform, whether to call a ``amrex::Gpu::synchronize()`` around profiling regions.
    This allows the profiler to give meaningful timers, but (hardly) slows down the simulation.

.. pp:param:: warpx.sort_intervals
    :type: ``string``
    :default: s: ``-1`` on CPU; ``4`` on GPU
    :optional:

    Using the `Time intervals`_ syntax, this string defines the timesteps at which particles are
    sorted.
    If ``<=0``, do not sort particles.
    It is turned on on GPUs for performance reasons (to improve memory locality).

.. pp:param:: warpx.sort_particles_for_deposition
    :type: ``bool``
    :default: ``true`` for the CUDA backend, otherwise ``false``
    :optional:

    This option controls the type of sorting used if particle sorting is turned on, i.e. if ``sort_intervals`` is not ``<=0``.
    If ``true``, particles will be sorted by cell to optimize deposition with many particles per cell, in the order x -> y -> z -> ppc.
    If ``false``, particles will be sorted by bin, using the ``sort_bin_size`` parameter below, in the order ppc -> x -> y -> z.
    ``true`` is recommend for best performance on NVIDIA GPUs, especially if there are many particles per cell.

.. pp:param:: warpx.sort_idx_type
    :type: list of ``int``
    :default: ``0 0 0``
    :optional:

    This controls the type of grid used to sort the particles when ``sort_particles_for_deposition`` is ``true``. Possible values are:
    ``idx_type = {0, 0, 0}``: Sort particles to a cell centered grid
    ``idx_type = {1, 1, 1}``: Sort particles to a node centered grid
    ``idx_type = {2, 2, 2}``: Compromise between a cell and node centered grid.
    In 2D (XZ and RZ), only the first two elements are read.
    In 1D, only the first element is read.

.. pp:param:: warpx.sort_bin_size
    :type: list of ``int``
    :default: ``1 1 1``
    :optional:

    If ``sort_intervals`` is activated and ``sort_particles_for_deposition`` is ``false``, particles are sorted in bins of ``sort_bin_size`` cells.
    In 2D, only the first two elements are read.

.. pp:param:: warpx.do_shared_mem_charge_deposition
    :type: ``bool``
    :default: ``false``
    :optional:

    If activated, charge deposition will allocate and use small
    temporary buffers on which to accumulate deposited charge values
    from particles. On GPUs these buffers will reside in ``__shared__``
    memory, which is faster than the usual ``__global__``
    memory. Performance impact will depend on the relative overhead
    of assigning the particles to bins small enough to fit in the
    space available for the temporary buffers.

.. pp:param:: warpx.do_shared_mem_current_deposition
    :type: ``bool``
    :default: ``false``
    :optional:

    If activated, current deposition will allocate and use small
    temporary buffers on which to accumulate deposited current values
    from particles. On GPUs these buffers will reside in ``__shared__``
    memory, which is faster than the usual ``__global__``
    memory. Performance impact will depend on the relative overhead
    of assigning the particles to bins small enough to fit in the
    space available for the temporary buffers. Performance is mostly improved
    when there is lots of contention between particles writing to the same cell
    (e.g. for high particles per cell). This feature is only available for CUDA
    and HIP, and is only recommended for 3D or 2D.

.. pp:param:: warpx.shared_tilesize
    :type: list of ``int``
    :default: ``6 6 8`` in 3D; ``14 14`` in 2D; ``1s`` otherwise
    :optional:

    Used to tune performance when ``do_shared_mem_current_deposition`` or
    ``do_shared_mem_charge_deposition`` is enabled. ``shared_tilesize`` is the
    size of the temporary buffer allocated in shared memory for a threadblock.
    A larger tilesize requires more shared memory, but gives more work to each
    threadblock, which can lead to higher occupancy, and allows for more
    buffered writes to ``__shared__`` instead of ``__global__``. The defaults
    in 2D and 3D
    are chosen from experimentation, but can be improved upon for specific
    problems. The other defaults are not optimized and should always be fine
    tuned for the problem.

.. pp:param:: warpx.shared_mem_current_tpb
    :type: ``int``
    :default: ``128``
    :optional:

    Used to tune performance when ``do_shared_mem_current_deposition`` is
    enabled. ``shared_mem_current_tpb`` controls the number of threads per
    block (tpb), i.e. the number of threads operating on a shared buffer.


.. _running-cpp-parameters-diagnostics:

Diagnostics and output
----------------------

.. _running-cpp-parameters-diagnostics-insitu:

In-situ visualization
^^^^^^^^^^^^^^^^^^^^^

WarpX has five types of diagnostics:
``Full`` diagnostics consist in dumps of fields and particles at given iterations,
``TimeAveraged`` diagnostics only allow field data, which they output after averaging over a period of time,
``BackTransformed`` diagnostics are used when running a simulation in a boosted frame, to reconstruct output data to the lab frame,
``BoundaryScraping`` diagnostics are used to collect the particles that are absorbed at the boundary, throughout the simulation, and
``ReducedDiags`` enable users to compute specific reduced quantities, such as particle temperature, energy histograms, or maximum field values, and efficiently save this in-situ analyzed data to files.
Similar to what is done for physical species, WarpX has a class Diagnostics that allows users to initialize different diagnostics, each of them with different fields, resolution and period.
This currently applies to standard diagnostics, but should be extended to back-transformed diagnostics and reduced diagnostics (and others) in a near future.

.. pp:param:: warpx.synchronize_velocity_for_diagnostics
    :type: ``0`` or ``1``
    :default: ``1``
    :optional:

    Whether to synchronize the particle velocities with the particle positions in the diagnostics.
    In its normal operation, WarpX is using the leap frog algorithm to advance the particles, and leaves the positions and velocities of the particles unsynchronized at the end of each time step, with the velocities lagging behind a half step.
    When this option is turned on, whenever any diagnostics will be calculated, the velocities will be advanced a half step to
    synchronize with the position before the diagnostics are generated.

.. _running-cpp-parameters-diagnostics-full:

Full Diagnostics
^^^^^^^^^^^^^^^^

``FullDiagnostics`` consist in dumps of fields and particles at given iterations.
Similar to what is done for physical species, WarpX has a class Diagnostics that allows users to initialize different diagnostics, each of them with different fields, resolution and period.
The user specifies the number of diagnostics and the name of each of them, and then specifies options for each of them separately.
Note that some parameter (those that do not start with a ``<diag_name>.`` prefix) apply to all diagnostics.
This should be changed in the future.
In-situ capabilities can be used by turning on Sensei or Ascent (provided they are installed) through the output format, see below.

.. pp:param:: diagnostics.enable
    :type: ``0`` or ``1``
    :default: ``1``
    :optional:

    Whether to enable or disable diagnostics. This flag overwrites all other diagnostics input parameters.

.. pp:param:: diagnostics.diags_names
    :type: list of ``string``
    :default: ``empty``
    :optional:

    Name of each diagnostics.
    example: :pp:param:`diagnostics.diags_names = diag1 my_second_diag`.

.. pp:param:: <diag_name>.intervals
    :type: ``string``

    Using the `Time intervals`_ syntax, this string defines the timesteps at which data is dumped.
    Use a negative number or 0 to disable data dumping.
    example: ``diag1.intervals = 10,20:25:1``.
    Note that by default the last timestep is dumped regardless of this parameter. This can be
    changed using the parameter :pp:param:`<diag_name>.dump_last_timestep` described below.

.. pp:param:: <diag_name>.dump_last_timestep
    :type: ``bool``
    :default: ``1``
    :optional:

    If this is ``1``, the last timestep is dumped regardless of :pp:param:`<diag_name>.intervals`.

.. pp:param:: <diag_name>.diag_type
    :type: ``string``

    Type of diagnostics. ``Full``, ``BackTransformed``, and ``BoundaryScraping``
    example: ``diag1.diag_type = Full`` or ``diag1.diag_type = BackTransformed``

.. pp:param:: <diag_name>.format
    :type: ``string``
    :default: ``plotfile``
    :optional:

    Flush format. Possible values are:

    * ``plotfile`` for native AMReX format.

    * ``checkpoint`` for a checkpoint file, only works with :pp:param:`<diag_name>.diag_type = Full`.

    * ``openpmd`` for OpenPMD format `openPMD <https://www.openPMD.org>`_.
      Requires to build WarpX with ``USE_OPENPMD=TRUE`` (see :ref:`instructions <building-openpmd>`).

    * ``ascent`` for in-situ visualization using Ascent.

    * ``sensei`` for in-situ visualization using Sensei.

    example: ``diag1.format = openpmd``.

.. pp:param:: <diag_name>.sensei_config
    :type: ``string``

    Only read if :pp:param:`<diag_name>.format = sensei`.
    Points to the SENSEI XML file which selects and configures the desired back end.

.. pp:param:: <diag_name>.sensei_pin_mesh
    :type: ``integer``
    :default: 0

    Only read if :pp:param:`<diag_name>.format = sensei`.
    When 1 lower left corner of the mesh is pinned to 0.,0.,0.

.. pp:param:: <diag_name>.openpmd_backend
    :type: ``bp5``, ``bp4``, ``h5`` or ``json``
    :optional:
    :comment: only used if :pp:param:`<diag_name>.format = openpmd`

    `I/O backend <https://openpmd-api.readthedocs.io/en/latest/backends/overview.html>`_ for `openPMD <https://www.openPMD.org>`_ data dumps.
    ``bp5``/``bp4`` is the `ADIOS I/O library <https://csmd.ornl.gov/adios>`_, ``h5`` is the `HDF5 format <https://www.hdfgroup.org/solutions/hdf5/>`_, and ``json`` is a `simple text format <https://en.wikipedia.org/wiki/JSON>`_.
    ``json`` is for debugging and only works with serial/single-rank jobs.
    When WarpX is compiled with openPMD support, the first available backend in the order given above is taken.

.. pp:param:: <diag_name>.openpmd_encoding
    :type: ``v`` (variable based), ``f`` (file based) or ``g`` (group based)
    :optional:
    :comment: only read if :pp:param:`<diag_name>.format = openpmd`.

    openPMD `file output encoding <https://openpmd-api.readthedocs.io/en/0.17.0/usage/concepts.html#iteration-and-series>`__.
    File based: one file per timestep (slower), group/variable based: one file for all steps (faster)).
    ``variable based`` is an `experimental feature with ADIOS2 BP5 <https://openpmd-api.readthedocs.io/en/0.17.0/backends/adios2.html#experimental-new-adios2-schema>`__ that will replace ``g``.
    Default: ``f`` (full diagnostics)

.. pp:param:: <diag_name>.buffer_flush_limit_btd
    :type: ``integer``
    :default: s to 5
    :optional:
    :comment: only read if :pp:param:`<diag_name>.diag_type = BackTransformed`

    This parameter is intended for ADIOS backend to group every N buffers (N is the value of this parameter) and then flush to disk.

.. pp:param:: <diag_name>.adios2_operator.type
    :type: ``zfp``, ``blosc``
    :optional:

    `ADIOS2 I/O operator type <https://openpmd-api.readthedocs.io/en/0.17.0/details/backendconfig.html#adios2>`__ for `openPMD <https://www.openPMD.org>`_ data dumps.

.. pp:param:: <diag_name>.adios2_operator.parameters.*
    :optional:

    `ADIOS2 I/O operator parameters <https://openpmd-api.readthedocs.io/en/0.17.0/details/backendconfig.html#adios2>`__ for `openPMD <https://www.openPMD.org>`_ data dumps.

    A typical example for `ADIOS2 output using lossless compression <https://openpmd-api.readthedocs.io/en/0.17.0/details/backendconfig.html#adios2>`__ with ``blosc`` using the ``zstd`` compressor and 6 CPU treads per MPI Rank (e.g. for a `GPU run with spare CPU resources <https://arxiv.org/abs/1706.00522>`__):

    .. code-block:: text

       <diag_name>.adios2_operator.type = blosc
       <diag_name>.adios2_operator.parameters.compressor = zstd
       <diag_name>.adios2_operator.parameters.clevel = 1
       <diag_name>.adios2_operator.parameters.doshuffle = BLOSC_BITSHUFFLE
       <diag_name>.adios2_operator.parameters.threshold = 2048
       <diag_name>.adios2_operator.parameters.nthreads = 6  # per MPI rank (and thus per GPU)

    or for the lossy ZFP compressor using very strong compression per scalar:

    .. code-block:: text

       <diag_name>.adios2_operator.type = zfp
       <diag_name>.adios2_operator.parameters.precision = 3

    For back-transformed diagnostics with ADIOS BP5, we are experimenting with a new option for variable-based encoding that "flattens" the output steps, aiming to increase write and read performance:

    .. code-block:: text

       <diag_name>.openpmd_backend = bp5
       <diag_name>.adios2_engine.parameters.FlattenSteps = on

.. pp:param:: <diag_name>.adios2_engine.type
    :type: ``bp5``, ``bp4``, ``sst``, ``ssc``, ``dataman``
    :optional:

    `ADIOS2 Engine type <https://openpmd-api.readthedocs.io/en/0.17.0/details/backendconfig.html#adios2>`__ for `openPMD <https://www.openPMD.org>`_ data dumps.
    See full list of engines at `ADIOS2 readthedocs <https://adios2.readthedocs.io/en/latest/engines/engines.html>`__

.. pp:param:: <diag_name>.adios2_engine.parameters.*
    :optional:

    `ADIOS2 Engine parameters <https://openpmd-api.readthedocs.io/en/0.17.0/details/backendconfig.html#adios2>`__ for `openPMD <https://www.openPMD.org>`_ data dumps.

    An example for parameters for the BP engine are setting the number of writers (``NumAggregators``), transparently redirecting data to burst buffers etc.
    A detailed list of engine-specific parameters are available at the official `ADIOS2 documentation <https://adios2.readthedocs.io/en/latest/engines/engines.html>`__

    .. code-block:: text

        <diag_name>.adios2_engine.parameters.NumAggregators = 2048
        <diag_name>.adios2_engine.parameters.BurstBufferPath="/mnt/bb/username"

.. pp:param:: <diag_name>.fields_to_plot
    :type: list of ``strings``
    :optional:

    Fields written to output.
    Possible scalar fields: ``part_per_cell`` ``rho`` ``phi`` ``F`` ``part_per_grid`` ``proc_num`` ``divE`` ``divB`` ``eb_covered`` ``rho_<species_name>`` and ``T_<species_name>``, where ``<species_name>`` must match the name of one of the available particle species.
    ``T_<species_name>`` is the temperature in eV (only valid for non-relativistic plasmas, since the code relies on the equipartition theorem to extract the temperature).
    With the hybrid-PIC solver (:pp:param:`algo.maxwell_solver` = ``hybrid``), the scalar fields ``Te`` (electron temperature in K: implied by the electron-pressure closure, or the evolved state variable when :pp:param:`hybrid_pic_model.solve_electron_energy_equation` is on) and ``Pe`` (electron pressure in Pa, as used in the Ohm's-law E-field solve) are also available.
    ``eb_covered`` is a number between 0 and 1 that indicates the fraction of the cell that is covered by the embedded boundary.
    Note that ``phi`` will only be written out when ``do_electrostatic==labframe``.
    Also, note that for :pp:param:`<diag_name>.diag_type = BackTransformed`, the only scalar field currently supported is ``rho``.
    Possible vector field components in Cartesian geometry: ``Ex`` ``Ey`` ``Ez`` ``Bx`` ``By`` ``Bz`` ``jx`` ``jy`` ``jz``.
    Possible vector field components in RZ and RCYLINDER geometry: ``Er`` ``Et`` ``Ez`` ``Br`` ``Bt`` ``Bz`` ``jr`` ``jt`` ``jz``.
    Possible vector field components in RSPHERE geometry: ``Er`` ``Et`` ``Ep`` ``Br`` ``Bt`` ``Bp`` ``jr`` ``jt`` ``jp``.
    Any MultiFab added to the internal registry can also be included in the list.
    The default :pp:param:`<diag_name>.fields_to_plot` is to write all possible field components for the geometry.
    When the special value ``none`` is specified, no fields are written out.
    Note that the fields are averaged on the cell centers before they are written to file.
    Otherwise, we reconstruct a 2D Cartesian slice of the fields for output at :math:`\theta=0`.

.. pp:param:: <diag_name>.additional_fields_to_plot
    :type: list of ``strings``
    :optional:

    Additional fields written to output, in addition to the standard default list as specified with :pp:param:`<diag_name>.fields_to_plot`.
    This allows specification of fields to plot without having to also list the default fields when they are also desired.
    Any of the same fields can be listed here.
    Any MultiFab added to the internal registry can also be included in the list.
    If :pp:param:`<diag_name>.fields_to_plot` is set to ``none``, this input is ignored.

.. pp:param:: <diag_name>.dump_rz_modes
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Whether to save all modes when in RZ.  When ``openpmd_backend = openpmd``, this parameter is ignored and all modes are saved.

.. pp:param:: <diag_name>.particle_fields_to_plot
    :type: list of ``strings``
    :optional:

    Names of per-cell diagnostics of particle properties to calculate and output as additional fields.
    Note that the deposition onto the grid does not respect the particle shape factor, but instead uses nearest-grid point interpolation.
    Default is none.
    Parser functions for these field names are specified by :pp:param:`<diag_name>.particle_fields.<field_name>(x,y,z,ux,uy,uz)`.
    Also, note that this option is only available for :pp:param:`<diag_name>.diag_type = Full`

.. pp:param:: <diag_name>.particle_fields_species
    :type: list of ``strings``
    :optional:

    Species for which to calculate ``particle_fields_to_plot``.
    Fields will be calculated separately for each specified species.
    The default is a list of all of the available particle species.

.. pp:param:: <diag_name>.particle_fields.<field_name>.do_average
    :type: ``0`` or ``1``
    :default: ``1``
    :optional:

    Whether the diagnostic is an average or a sum. With an average, the sum over the specified function is divided
    by the sum of the particle weights in each cell.

.. pp:param:: <diag_name>.particle_fields.<field_name>(x,y,z,ux,uy,uz)
   :type: parser ``string``

   Parser function to be calculated for each particle per cell. The averaged field written is

   .. math::

      \texttt{<field_name>_<species_name>} = \frac{\sum_{i=1}^N w_i \, f(x_i,y_i,z_i,u_{x,i},u_{y,i},u_{z,i})}{\sum_{i=1}^N w_i}

   where :math:`w_i` is the particle weight, :math:`f()` is the parser function, and :math:`(x_i,y_i,z_i)` are particle positions in units of a meter. The sums are over all particles of type ``<species_name>`` in a cell (ignoring the particle shape factor) that satisfy :pp:param:`<diag_name>.particle_fields.<field_name>.filter(x,y,z,ux,uy,uz)`.
   When :pp:param:`<diag_name>.particle_fields.<field_name>.do_average` is ``0``, the division by the sum over particle weights is not done.
   In 1D or 2D, the particle coordinates will follow the WarpX convention. :math:`(u_{x,i},u_{y,i},u_{z,i})` are components of the particle four-momentum. :math:`u = \gamma v/c`, :math:`\gamma` is the Lorentz factor, :math:`v` is the particle velocity and :math:`c` is the speed of light.
   For photons, we use the standardized momentum :math:`u = p/(m_{e}c)`, where :math:`p` is the momentum of the photon and :math:`m_{e}` the mass of an electron.

.. pp:param:: <diag_name>.particle_fields.<field_name>.filter(x,y,z,ux,uy,uz)
    :type: parser ``string``
    :optional:

    Parser function returning a boolean for whether to include a particle in the diagnostic.
    If not specified, all particles will be included (see above).
    The function arguments are the same as above.

.. pp:param:: <diag_name>.plot_raw_fields
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    By default, the fields written in the plot files are averaged on the cell centers.
    When :pp:param:`<diag_name>.plot_raw_fields = 1`, then the raw (i.e. non-averaged)
    fields are also saved in the output files.
    Only works with :pp:param:`<diag_name>.format = plotfile`.
    See `this section <https://yt-project.org/doc/examining/loading_data.html#viewing-raw-fields-in-warpx>`__
    in the yt documentation for more details on how to view raw fields.

.. pp:param:: <diag_name>.plot_raw_fields_guards
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Only used when :pp:param:`<diag_name>.plot_raw_fields = 1`.
    Whether to include the guard cells in the output of the raw fields.
    Only works with :pp:param:`<diag_name>.format = plotfile`.

.. pp:param:: <diag_name>.coarsening_ratio
    :type: list of ``int``
    :default: ``1 1 1``
    :optional:

    Reduce size of the selected diagnostic fields output by this ratio in each dimension.
    (For a ratio of N, this is done by averaging the fields over N or (N+1) points depending on the staggering).
    If ``blocking_factor`` and ``max_grid_size`` are used for the domain decomposition, as detailed in
    the :ref:`domain decomposition <usage_domain_decomposition>` section, ``coarsening_ratio`` should be an integer
    divisor of ``blocking_factor``. If :pp:param:`warpx.numprocs` is used instead, the total number of cells in a given
    dimension must be a multiple of the ``coarsening_ratio`` multiplied by ``numprocs`` in that dimension.

.. pp:param:: <diag_name>.file_prefix
    :type: ``string``
    :default: ``diags/<diag_name>``
    :optional:

    Root for output file names. Supports sub-directories.

.. pp:param:: <diag_name>.file_min_digits
    :type: ``int``
    :default: ``6``
    :optional:

    The minimum number of digits used for the iteration number appended to the diagnostic file names.

.. pp:param:: <diag_name>.diag_lo
    :type: list ``float``, 1 per dimension
    :default: ``-infinity -infinity -infinity``
    :optional:

    Lower corner of the output fields (if smaller than ``warpx.dom_lo``, then set to ``warpx.dom_lo``). Currently, when the ``diag_lo`` is different from ``warpx.dom_lo``, particle output is disabled.

.. pp:param:: <diag_name>.diag_hi
    :type: list ``float``, 1 per dimension
    :default: ``+infinity +infinity +infinity``
    :optional:

    Higher corner of the output fields (if larger than ``warpx.dom_hi``, then set to ``warpx.dom_hi``). Currently, when the ``diag_hi`` is different from ``warpx.dom_hi``, particle output is disabled.

.. pp:param:: <diag_name>.write_species
    :type: ``0`` or ``1``
    :default: ``1``
    :optional:

    Whether to write species output or not. For checkpoint format, always set this parameter to 1.

.. pp:param:: <diag_name>.species
    :type: list of ``string``
    :default: all physical species in the simulation

    Which species dumped in this diagnostics.

.. pp:param:: <diag_name>.<species_name>.variables
    :type: list of ``strings`` separated by spaces
    :optional:

    List of particle quantities to write to output.
    Choices are ``x``, ``y``, ``z`` for the particle positions (3D, RZ, RSPHERE), ``x`` and ``z`` in 2D, ``z`` in 1D, ``x`` and ``y`` for RCYLINDER,
    ``w`` for the particle weight and ``ux``, ``uy``, ``uz`` for the particle momenta.
    When writing to the OpenPMD format, the fields can also be obtained, ``Ex``, ``Ey``, ``Ez``, ``Bx``, ``By``, ``Bz``.
    Note that the fields gathered in the same way as during the simulation, and do not include any applied fields.
    Also, when writing to the OpenPMD format and when using the lab-frame electrostatic solver, ``phi`` (electrostatic potential, on the macroparticles) is also available.
    By default, positions, momenta, and weights are written out.
    If :pp:param:`<diag_name>.<species_name>.variables = none`, no particle data are written.

.. pp:param:: <diag_name>.<species_name>.additional_variables
    :type: list of `strings` separated by spaces
    :optional:

    List of additional particle quantities to write to output, when using the OpenPMD format.
    This allows specifying the additional particle quantities beyond the standard position, momentum, and weight.
    The options are the fields, ``Ex``, ``Ey``, ``Ez``, ``Bx``, ``By``, ``Bz``,
    and when using the lab-frame electrostatic solver, the electrostatic potential ``phi``.
    Note that the fields gathered in the same way as during the simulation, and do not include any applied fields.

.. pp:param:: <diag_name>.<species_name>.random_fraction
    :type: ``float``
    :optional:

    If provided :pp:param:`<diag_name>.<species_name>.random_fraction = a`, only ``a`` fraction of the particle data of this species will be dumped randomly in diag ``<diag_name>``, i.e. if ``rand() < a``, this particle will be dumped, where ``rand()`` denotes a random number generator.
    The value ``a`` provided should be between 0 and 1.

.. pp:param:: <diag_name>.<species_name>.uniform_stride
    :type: ``int``
    :optional:

    If provided :pp:param:`<diag_name>.<species_name>.uniform_stride = n`,
    every ``n`` particle of this species will be dumped, selected uniformly.
    The value provided should be an integer greater than or equal to 0.

.. pp:param:: <diag_name>.<species_name>.plot_filter_function(t,x,y,z,ux,uy,uz)
    :type: ``string``
    :optional:

    Users can provide an expression returning a boolean for whether a particle is dumped.
    ``t`` represents the physical time in seconds during the simulation.
    ``x, y, z`` represent particle positions in the unit of meter.
    ``ux, uy, uz`` represent particle momenta in the unit of
    :math:`\gamma v/c`, where
    :math:`\gamma` is the Lorentz factor,
    :math:`v/c` is the particle velocity normalized by the speed of light.
    E.g. If provided ``(x>0.0)*(uz<10.0)`` only those particles located at
    positions ``x`` greater than ``0``, and those having momentum ``uz`` less than 10,
    will be dumped.

.. pp:param:: amrex.async_out
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Enable asynchronous I/O for AMReX ``plotfile`` output.
    When set to ``1``, writing is handled by a background I/O
    thread so the simulation can continue while data are written to disk, which can reduce
    total time spent in I/O for large HPC runs. Actual benefits depend on the MPI
    implementation and may be negligible on a workstation.

.. pp:param:: amrex.async_out_nfiles
    :type: ``int``
    :default: ``64``
    :optional:

    Maximum number of files to use for asynchronous I/O (default: 64).
    When enabled, each MPI rank writes its own file up to this limit. If you
    run with more MPI ranks than :pp:param:`amrex.async_out_nfiles`, build WarpX with
    ``-DWarpX_MPI_THREAD_MULTIPLE=ON``.

.. pp:param:: warpx.field/particle_io_nfiles
    :link_aliases:
        warpx.field_io_nfiles
        warpx.particle_io_nfiles
    :type: ``int``
    :default: ``1024``
    :optional:

    The maximum number of files to use when writing field and particle data to plotfile directories.

.. pp:param:: warpx.mffile_nstreams
    :type: ``int``
    :default: ``4``
    :optional:

    Limit the number of concurrent readers per file.


.. _running-cpp-parameters-diagnostics-timeavg:

Time-Averaged Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^

``TimeAveraged`` diagnostics are a special type of ``Full`` diagnostics that allows for the output of time-averaged field data.
This type of diagnostics can be created using :pp:param:`<diag_name>.diag_type = TimeAveraged`.
We support only field data and related options from the list at `Full Diagnostics`_.

.. note::

    As with ``Full`` diagnostics, ``TimeAveraged`` diagnostics output the initial **instantaneous** conditions of the selected fields on step 0 (unless more specific output intervals exclude output for step 0).

In addition, ``TimeAveraged`` diagnostic options include:

.. pp:param:: <diag_name>.time_average_mode
    :type: ``string``
    :default: ``none``

    Describes the operating mode for time averaged field output.

    * ``none`` for no averaging (instantaneous fields)

    * ``fixed_start`` for a diagnostic that averages all fields between the current output step and a fixed point in time

    * ``dynamic_start`` for a constant averaging period and output at different points in time (non-overlapping)

    .. note::

        To enable time-averaged field output with intervals tightly spaced enough for overlapping averaging periods,
        please create additional instances of ``TimeAveraged`` diagnostics.

.. pp:param:: <diag_name>.average_period_steps
    :type: ``int``

    Configures the number of time steps in an averaging period.
    Set this only in the ``dynamic_start`` mode and only if ``average_period_time`` has not already been set.
    Will be ignored in the ``fixed_start`` mode (with warning).

.. pp:param:: <diag_name>.average_period_time
    :type: ``float``
    :unit: seconds

    Configures the time (SI units) in an averaging period.
    Set this only in the ``dynamic_start`` mode and only if ``average_period_steps`` has not already been set.
    Will be ignored in the ``fixed_start`` mode (with warning).

.. pp:param:: <diag_name>.average_start_step
    :type: ``int``

    Configures the time step at which time-averaging begins.
    Set this only in the ``fixed_start`` mode.
    Will be ignored in the ``dynamic_start`` mode (with warning).

.. _running-cpp-parameters-diagnostics-btd:

BackTransformed Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BackTransformed`` diag type are used when running a simulation in a boosted frame, to reconstruct output data to the lab frame. For more details on back-transformed diagnostics (BTD), see :ref:`FAQ: What about Back-transformed diagnostics (BTD)? <faq_btd>`. This option can be set using :pp:param:`<diag_name>.diag_type = BackTransformed`. We support the following list of options from `Full Diagnostics`_

    :pp:param:`<diag_name>.format`, :pp:param:`<diag_name>.openpmd_backend`, :pp:param:`<diag_name>.dump_rz_modes`, :pp:param:`<diag_name>.file_prefix`, :pp:param:`<diag_name>.diag_lo`, :pp:param:`<diag_name>.diag_hi`, :pp:param:`<diag_name>.write_species`, :pp:param:`<diag_name>.species`.

    Additional options for this diagnostic include:

.. pp:param:: <diag_name>.num_snapshots_lab
    :type: ``integer``

    Only used when :pp:param:`<diag_name>.diag_type` is ``BackTransformed``.
    The number of lab-frame snapshots that will be written.
    Only this option or ``intervals`` should be specified;
    a run-time error occurs if the user attempts to set both ``num_snapshots_lab`` and ``intervals``.

.. pp:param:: <diag_name>.intervals
    :type: ``string``
    :noindex:

    Only used when :pp:param:`<diag_name>.diag_type` is ``BackTransformed``.
    Using the `Time intervals`_ syntax, this string defines the lab frame times at which data is dumped,
    given as multiples of the step size ``dt_snapshots_lab`` or ``dz_snapshots_lab`` described below.
    Example: ``btdiag1.intervals = 10:11,20:24:2`` and ``btdiag1.dt_snapshots_lab = 1.e-12``
    indicate to dump at lab times ``1e-11``, ``1.1e-11``, ``2e-11``, ``2.2e-11``, and ``2.4e-11`` seconds.
    Note that the stop interval, the second number in the slice, must always be specified.
    Only this option or ``num_snapshots_lab`` should be specified;
    a run-time error occurs if the user attempts to set both ``num_snapshots_lab`` and ``intervals``.

.. pp:param:: <diag_name>.dt_snapshots_lab
    :type: ``float``
    :unit: seconds

    Only used when :pp:param:`<diag_name>.diag_type` is ``BackTransformed``.
    The time interval in between the lab-frame snapshots (where this
    time interval is expressed in the laboratory frame).

.. pp:param:: <diag_name>.dz_snapshots_lab
    :type: ``float``
    :unit: meters

    Only used when :pp:param:`<diag_name>.diag_type` is ``BackTransformed``.
    Distance between the lab-frame snapshots (expressed in the laboratory
    frame). ``dt_snapshots_lab`` is then computed by
    ``dt_snapshots_lab = dz_snapshots_lab/c``. Either ``dt_snapshots_lab``
    or ``dz_snapshot_lab`` is required.

.. pp:param:: <diag_name>.buffer_size
    :type: ``integer``

    Only used when :pp:param:`<diag_name>.diag_type` is ``BackTransformed``.
    The default size of the back transformed diagnostic buffers used to generate lab-frame
    data is 256. That is, when the multifab with lab-frame data has 256 z-slices,
    the data will be flushed out. However, if many lab-frame snapshots are required for
    diagnostics and visualization, the GPU may run out of memory with many large boxes with
    a size of 256 in the z-direction. This input parameter can then be used to set a
    smaller buffer-size, preferably multiples of 8, such that, a large number of
    lab-frame snapshot data can be generated without running out of gpu memory.
    The downside to using a small buffer size, is that the I/O time may increase due
    to frequent flushes of the lab-frame data. The other option is to keep the default
    value for buffer size and use slices to reduce the memory footprint and maintain
    optimum I/O performance.

.. pp:param:: <diag_name>.do_back_transformed_fields
    :type: ``0`` or ``1``
    :default: ``1``
    :optional:

    Only used when :pp:param:`<diag_name>.diag_type` is ``BackTransformed``
    Whether to back transform the fields or not.
    Note that for ``BackTransformed`` diagnostics, at least one of the options
    :pp:param:`<diag_name>.do_back_transformed_fields` or :pp:param:`<diag_name>.do_back_transformed_particles` must be 1.

.. pp:param:: <diag_name>.do_back_transformed_particles
    :type: ``0`` or ``1``
    :default: ``1``
    :optional:

    Only used when :pp:param:`<diag_name>.diag_type` is ``BackTransformed``
    Whether to back transform the particle data or not.
    Note that for ``BackTransformed`` diagnostics, at least one of the options
    :pp:param:`<diag_name>.do_back_transformed_fields` or :pp:param:`<diag_name>.do_back_transformed_particles` must be 1.
    If ``diag_name.write_species = 0``, then :pp:param:`<diag_name>.do_back_transformed_particles` will be set
    to 0 in the simulation and particles will not be backtransformed.

Boundary Scraping Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BoundaryScrapingDiagnostics`` are used to collect the particles that are absorbed at the boundaries, throughout the simulation.
This diagnostic type is specified by setting :pp:param:`<diag_name>.diag_type` = ``BoundaryScraping``.
Currently, the only supported output format is openPMD, so the user also needs to set ``<diag>.format=openpmd`` and WarpX must be compiled with openPMD turned on.
The data that is to be collected and recorded is controlled per species and per boundary by setting one or more of the flags to ``1``,
:pp:param:`<species_name>.save_particles_at_xlo/ylo/zlo`, :pp:param:`<species_name>.save_particles_at_xhi/yhi/zhi`, and :pp:param:`<species_name>.save_particles_at_eb`.
(Note that this diagnostics does not save any field ; it only saves particles.)

The data collected at each boundary is written out to a subdirectory of the diagnostics directory with the name of the boundary, for example, ``particles_at_xlo``, ``particles_at_zhi``, or ``particles_at_eb``.
By default, all of the collected particle data is written out at the end of the simulation. Optionally, the :pp:param:`<diag_name>.intervals` parameter can be given to specify writing out the data more often.
This can be important if a large number of particles are lost, avoiding filling up memory with the accumulated lost particle data.

In addition to their usual attributes, the saved particles have
   an integer attribute ``stepScraped``, which indicates the PIC iteration at which each particle was absorbed at the boundary,
   a real attribute ``deltaTimeScraped``, which indicates the time between the time associated to ``stepScraped``
   and the exact time when each particle is absorbed at the boundary,
   a real attribute ``timeScraped``, which indicates the exact time when the particle is absorbed at the boundary,
   3 real attributes ``nx``, ``ny``, ``nz``, which represents the three components of the normal to the boundary at the point where the particle is absorbed (not saved if they reach non-EB boundaries)

``BoundaryScrapingDiagnostics`` can be used with :pp:param:`<diag_name>.<species_name>.random_fraction`, :pp:param:`<diag_name>.<species_name>.uniform_stride`, and ``<diag_name>.<species_name>.plot_filter_function``, which have the same behavior as for ``FullDiagnostics``. For ``BoundaryScrapingDiagnostics``, these filters are applied at the time the data is written to file. An implication of this is that more particles may initially be accumulated in memory than are ultimately written. ``t`` in ``plot_filter_function`` refers to the time the diagnostic is written rather than the time the particle crossed the boundary.

.. _running-cpp-parameters-diagnostics-reduced:

Reduced Diagnostics
^^^^^^^^^^^^^^^^^^^

``ReducedDiags`` enable users to compute specific reduced quantities, such as particle temperature, energy histograms, or maximum field values, and efficiently save this in-situ analyzed data to files.
This shifts analysis from post-processing to runtime calculation of reduction operations (average, maximum, ...) and can greatly save disk space when "raw" particle and field outputs from ``FullDiagnostics`` can be avoided in favor of single values, 1D or 2D data at possibly even higher time resolution.

.. pp:param:: warpx.reduced_diags_names
    :type: ``strings``, separated by spaces

    A list of user-given names for reduced diagnostics.
    By default, these names are also prefixing the names of output files.
    If :pp:param:`warpx.reduced_diags_names` is not provided in the input file,
    no reduced diagnostics will be activated during the run.
    This is then used in the rest of the input deck;
    in this documentation we use ``<reduced_diags_name>`` as a placeholder.

.. pp:param:: <reduced_diags_name>.type
    :type: ``string``

    The type of reduced diagnostics associated with this ``<reduced_diags_name>``.
    For example, ``ParticleEnergy``, ``FieldEnergy``, etc.
    All available types are described below in detail.
    For all reduced diagnostics that are writing tabular data into text files,
    the first and the second columns in the output file are
    the time step and the corresponding physical time in seconds, respectively.

    * ``ParticleEnergy``
        This type computes the total and mean relativistic particle kinetic energy among all species:

        .. math::

            E_p = \sum_{i=1}^N w_i \, \left( \sqrt{|\boldsymbol{p}_i|^2 c^2 + m_0^2 c^4} - m_0 c^2 \right)

        where :math:`\boldsymbol{p}_i` is the relativistic momentum of the :math:`i`-th particle, :math:`c` is the speed of light, :math:`m_0` is the rest mass, :math:`N` is the number of particles, and :math:`w_i` is the weight of the :math:`i`-th particle.

        The output columns are the total energy of all species, the total energy per species, the total mean energy :math:`E_p / \sum_i w_i` of all species, and the total mean energy per species.

    * ``ParticleMomentum``
        This type computes the total and mean relativistic particle momentum among all species:

        .. math::

            \boldsymbol{P}_p = \sum_{i=1}^N w_i \, \boldsymbol{p}_i

        where :math:`\boldsymbol{p}_i` is the relativistic momentum of the :math:`i`-th particle, :math:`N` is the number of particles, and :math:`w_i` is the weight of the :math:`i`-th particle.

        The output columns are the components of the total momentum of all species, the total momentum per species, the total mean momentum :math:`\boldsymbol{P}_p / \sum_i w_i` of all species, and the total mean momentum per species.

    * ``FieldEnergy``
        This type computes the electromagnetic field energy

        .. math::

            E_f = \frac{1}{2} \sum_{\text{cells}} \left( \varepsilon_0 |\boldsymbol{E}|^2 + \frac{|\boldsymbol{B}|^2}{\mu_0} \right) \Delta V

        where :math:`\boldsymbol{E}` is the electric field, :math:`\boldsymbol{B}` is the magnetic field, :math:`\varepsilon_0` is the vacuum permittivity, :math:`\mu_0` is the vacuum permeability, :math:`\Delta V` is the cell volume (or cell area in 2D), and the sum is over all cells.

        The output columns are the total field energy :math:`E_f`, the :math:`\boldsymbol{E}` field energy, and the :math:`\boldsymbol{B}` field energy, at each mesh refinement level.

    * ``FieldMomentum``
        This type computes the electromagnetic field momentum

        .. math::

            \boldsymbol{P}_f = \varepsilon_0 \sum_{\text{cells}} \left( \boldsymbol{E} \times \boldsymbol{B} \right) \Delta V

        where :math:`\boldsymbol{E}` is the electric field, :math:`\boldsymbol{B}` is the magnetic field, :math:`\varepsilon_0` is the vacuum permittivity, :math:`\Delta V` is the cell volume (or cell area in 2D), and the sum is over all cells.

        The output columns are the components of the total field momentum :math:`\boldsymbol{P}_f` at each mesh refinement level.

        Note that the fields are *not* averaged on the cell centers before their energy is
        computed.

    * ``FieldMaximum``
        This type computes the maximum value of each component of the electric and magnetic fields
        and of the norm of the electric and magnetic field vectors.
        Measuring maximum fields in a plasma might be very noisy in PIC, use this instead
        for analysis of scenarios such as an electromagnetic wave propagating in vacuum.

        The output columns are
        the maximum value of the :math:`E_x` field,
        the maximum value of the :math:`E_y` field,
        the maximum value of the :math:`E_z` field,
        the maximum value of the norm :math:`|E|` of the electric field,
        the maximum value of the :math:`B_x` field,
        the maximum value of the :math:`B_y` field,
        the maximum value of the :math:`B_z` field and
        the maximum value of the norm :math:`|B|` of the magnetic field,
        at mesh refinement levels from  0 to :math:`n`.

        Note that the fields are averaged on the cell centers before their maximum values are
        computed.

    * ``FieldPoyntingFlux``
        Integrates the normal Poynting flux over each domain boundary surface and also integrates the flux over time.
        This provides the power and total energy loss into or out of the simulation domain.
        The output columns are the flux for each dimension on the lower boundaries, then the higher boundaries,
        then the integrated energy loss for each dimension on the the lower and higher boundaries.

    * ``FieldProbe``
        This type computes the value of each component of the electric and magnetic fields
        and of the Poynting vector (a measure of electromagnetic flux) at points in the domain.

        Multiple geometries for point probes can be specified via ``<reduced_diags_name>.probe_geometry = ...``:

        * ``Point`` (default): a single point
        * ``Line``: a line of points with equal spacing
        * ``Plane``: a plane of points with equal spacing

        **Point**: The point where the fields are measured is specified through the input parameters ``<reduced_diags_name>.x_probe``, ``<reduced_diags_name>.y_probe`` and ``<reduced_diags_name>.z_probe``.

        **Line**: probe a 1 dimensional line of points to create a line detector.
        Initial input parameters ``x_probe``, ``y_probe``, and ``z_probe`` designate one end of the line detector, while the far end is specified via ``<reduced_diags_name>.x1_probe``, ``<reduced_diags_name>.y1_probe``, ``<reduced_diags_name>.z1_probe``.
        Additionally, ``<reduced_diags_name>.resolution`` must be defined to give the number of detector points along the line (equally spaced) to probe.

        **Plane**: probe a 2 dimensional plane of points to create a square plane detector.
        Initial input parameters ``x_probe``, ``y_probe``, and ``z_probe`` designate the center of the detector.
        The detector plane is normal to a vector specified by ``<reduced_diags_name>.target_normal_x``, ``<reduced_diags_name>.target_normal_y``, and ``<reduced_diags_name>.target_normal_z``.
        Note that it is not necessary to specify the ``target_normal`` vector in a 2D simulation (the only supported normal is in ``y``).
        The top of the plane is perpendicular to an "up" vector denoted by ``<reduced_diags_name>.target_up_x``, ``<reduced_diags_name>.target_up_y``, and ``<reduced_diags_name>.target_up_z``.
        The detector has a square radius to be determined by ``<reduced_diags_name>.detector_radius``.
        Similarly to the line detector, the plane detector requires a resolution ``<reduced_diags_name>.resolution``, which denotes the number of detector particles along each side of the square detector.

        The output columns are
        the value of the :math:`E_x` field,
        the value of the :math:`E_y` field,
        the value of the :math:`E_z` field,
        the value of the :math:`B_x` field,
        the value of the :math:`B_y` field,
        the value of the :math:`B_z` field and
        the value of the Poynting Vector :math:`|S|` of the electromagnetic fields,
        at mesh refinement levels from  0 to :math:`n`, at point (:math:`x`, :math:`y`, :math:`z`).

        The fields are always interpolated to the measurement point.
        The interpolation order can be set by specifying ``<reduced_diags_name>.interp_order``,
        defaulting to ``1``.
        In RZ geometry, this only saves the
        0'th azimuthal mode component of the fields.
        Time integrated electric and magnetic field components can instead be obtained by specifying
        ``<reduced_diags_name>.integrate = true``.
        The integration is done every time step even when the data is written out less often.
        In a *moving window* simulation, the FieldProbe can be set to follow the moving frame by specifying ``<reduced_diags_name>.do_moving_window_FP = 1`` (default 0).

        .. warning::

           The FieldProbe reduced diagnostic does not yet add a Lorentz back transformation for boosted frame simulations.
           Thus, it records field data in the boosted frame, not (yet) in the lab frame.

    * ``RhoMaximum``
        This type computes the maximum and minimum values of the total charge density as well as
        the maximum absolute value of the charge density of each charged species.
        Please be aware that measuring maximum charge densities might be very noisy in PIC simulations.

        The output columns are
        the maximum value of the :math:`rho` field,
        the minimum value of the :math:`rho` field,
        the maximum value of the absolute :math:`|rho|` field of each charged species.

        Note that the charge densities are averaged on the cell centers before their maximum values
        are computed.

    * ``FieldReduction``
        This type computes an arbitrary reduction of the positions, the current density, and the electromagnetic fields.

        * ``<reduced_diags_name>.reduced_function(x,y,z,Ex,Ey,Ez,Bx,By,Bz,jx,jy,jz)`` (``string``)
            An analytic function to be reduced must be provided, using the math parser.

        * ``<reduced_diags_name>.reduction_type`` (``string``)
            The type of reduction to be performed. It must be either ``Maximum``, ``Minimum`` or
            ``Integral``.
            ``Integral`` computes the spatial integral of the function defined in the parser by
            summing its value on all grid points and multiplying the result by the volume of a
            cell.
            Please be also aware that measuring maximum quantities might be very noisy in PIC
            simulations.

        The only output column is the reduced value.

        Note that the fields are averaged on the cell centers before the reduction is performed.

    * ``ParticleNumber``
        This type computes the total number of macroparticles and of physical particles (i.e. the
        sum of their weights) in the whole simulation domain (for each species and summed over all
        species). It can be useful in particular for simulations with creation (ionization, QED
        processes) or removal (resampling) of particles.

        The output columns are
        total number of macroparticles summed over all species,
        total number of macroparticles of each species,
        sum of the particles' weight summed over all species,
        sum of the particles' weight of each species.

    * ``BeamRelevant``
        This type computes properties of a particle beam relevant for particle accelerators, like position, momentum, emittance, etc.

        ``<reduced_diags_name>.species`` must be provided, such that the diagnostics are done for this (beam-like) species only.

        The output columns (for 3D-XYZ) are the following, where the average is done over the whole species (typical usage: the particle beam is in a separate species):

        [0]: simulation step (iteration).

        [1]: time (s).

        [2], [3], [4]: The mean values of beam positions (m)
        :math:`\langle x \rangle`,
        :math:`\langle y \rangle`,
        :math:`\langle z \rangle`.

        [5], [6], [7]: The mean values of beam relativistic momenta (kg m/s)
        :math:`\langle p_x \rangle`,
        :math:`\langle p_y \rangle`,
        :math:`\langle p_z \rangle`.

        [8]: The mean Lorentz factor :math:`\langle \gamma \rangle`.

        [9], [10], [11]: The RMS values of beam positions (m)
        :math:`\delta_x = \sqrt{ \langle (x - \langle x \rangle)^2 \rangle }`,
        :math:`\delta_y = \sqrt{ \langle (y - \langle y \rangle)^2 \rangle }`,
        :math:`\delta_z = \sqrt{ \langle (z - \langle z \rangle)^2 \rangle }`.

        [12], [13], [14]: The RMS values of beam relativistic momenta (kg m/s)
        :math:`\delta_{px} = \sqrt{ \langle (p_x - \langle p_x \rangle)^2 \rangle }`,
        :math:`\delta_{py} = \sqrt{ \langle (p_y - \langle p_y \rangle)^2 \rangle }`,
        :math:`\delta_{pz} = \sqrt{ \langle (p_z - \langle p_z \rangle)^2 \rangle }`.

        [15]: The RMS value of the Lorentz factor
        :math:`\sqrt{ \langle (\gamma - \langle \gamma \rangle)^2 \rangle }`.

        [16], [17], [18]: beam projected transverse RMS normalized emittance (m)
        :math:`\epsilon_x = \dfrac{1}{mc} \sqrt{\delta_x^2 \delta_{px}^2 -
        \Big\langle (x-\langle x \rangle) (p_x-\langle p_x \rangle) \Big\rangle^2}`,
        :math:`\epsilon_y = \dfrac{1}{mc} \sqrt{\delta_y^2 \delta_{py}^2 -
        \Big\langle (y-\langle y \rangle) (p_y-\langle p_y \rangle) \Big\rangle^2}`,
        :math:`\epsilon_z = \dfrac{1}{mc} \sqrt{\delta_z^2 \delta_{pz}^2 -
        \Big\langle (z-\langle z \rangle) (p_z-\langle p_z \rangle) \Big\rangle^2}`.

        [19], [20]: Twiss alpha for the transverse directions
        :math:`\alpha_x = - \Big\langle (x-\langle x \rangle) (p_x-\langle p_x \rangle) \Big\rangle \Big/ \epsilon_x`,
        :math:`\alpha_y = - \Big\langle (y-\langle y \rangle) (p_y-\langle p_y \rangle) \Big\rangle \Big/ \epsilon_y`.

        [21], [22]: beta function for the transverse directions (m)
        :math:`\beta_x = \dfrac{{\delta_x}^2}{\epsilon_x}`,
        :math:`\beta_y = \dfrac{{\delta_y}^2}{\epsilon_y}`.

        [23]: The charge of the beam (C).

        For 2D-XZ,
        :math:`\langle y \rangle`,
        :math:`\delta_y`, and
        :math:`\epsilon_y` will not be outputted.

    * ``LoadBalanceCosts``
        This type computes the cost, used in load balancing, for each box on the domain.
        The cost :math:`c` is computed as

        .. math::

            c = n_{\text{particle}} \cdot w_{\text{particle}} + n_{\text{cell}} \cdot w_{\text{cell}},

        where
        :math:`n_{\text{particle}}` is the number of particles on the box,
        :math:`w_{\text{particle}}` is the particle cost weight factor (controlled by :pp:param:`algo.costs_heuristic_particles_wt`),
        :math:`n_{\text{cell}}` is the number of cells on the box, and
        :math:`w_{\text{cell}}` is the cell cost weight factor (controlled by :pp:param:`algo.costs_heuristic_cells_wt`).

    * ``LoadBalanceEfficiency``
        This type computes the load balance efficiency, given the present costs
        and distribution mapping. Load balance efficiency is computed as the
        mean cost over all ranks, divided by the maximum cost over all ranks.
        Until costs are recorded, load balance efficiency is output as ``-1``;
        at earliest, the load balance efficiency can be output starting at step
        ``2``, since costs are not recorded until step ``1``.

    * ``ParticleHistogram``
        This type computes a user defined particle histogram.

        * ``<reduced_diags_name>.species`` (``string``)
            A species name must be provided,
            such that the diagnostics are done for this species.

        * ``<reduced_diags_name>.histogram_function(t,x,y,z,ux,uy,uz)`` (``string``)
            A histogram function must be provided.
            ``t`` represents the physical time in seconds during the simulation.
            ``x, y, z`` represent particle positions in the unit of meter.
            ``ux, uy, uz`` represent the particle momenta in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            E.g.
            ``x`` produces the position (density) distribution in ``x``.
            ``ux`` produces the momentum distribution in ``x``,
            ``sqrt(ux*ux+uy*uy+uz*uz)`` produces the speed distribution.
            The default value of the histogram without normalization is
            :math:`f = \sum\limits_{i=1}^N w_i`, where
            :math:`\sum\limits_{i=1}^N` is the sum over :math:`N` particles
            in that bin,
            :math:`w_i` denotes the weight of the ith particle.

        * ``<reduced_diags_name>.bin_number`` (``int`` > 0)
            This is the number of bins used for the histogram.

        * ``<reduced_diags_name>.bin_max`` (``float``)
            This is the maximum value of the bins.

        * ``<reduced_diags_name>.bin_min`` (``float``)
            This is the minimum value of the bins.

        * ``<reduced_diags_name>.normalization`` (optional)
            This provides options to normalize the histogram:

            ``unity_particle_weight``
            uses unity particle weight to compute the histogram,
            such that the values of the histogram are
            the number of counted macroparticles in that bin,
            i.e.  :math:`f = \sum\limits_{i=1}^N 1`,
            :math:`N` is the number of particles in that bin.

            ``max_to_unity`` will normalize the histogram such that
            its maximum value is one.

            ``area_to_unity`` will normalize the histogram such that
            the area under the histogram is one,
            so the histogram is also the probability density function.

            If nothing is provided,
            the macroparticle weight will be used to compute
            the histogram, and no normalization will be done.

        * ``<reduced_diags_name>.filter_function(t,x,y,z,ux,uy,uz)`` (``string``) optional
            Users can provide an expression returning a boolean for whether a particle is taken
            into account when calculating the histogram.
            ``t`` represents the physical time in seconds during the simulation.
            ``x, y, z`` represent particle positions in the unit of meter.
            ``ux, uy, uz`` represent particle momenta in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            E.g. If provided ``(x>0.0)*(uz<10.0)`` only those particles located at
            positions ``x`` greater than ``0``, and those having momentum ``uz`` less than 10,
            will be taken into account when calculating the histogram.

        The output columns are
        values of the 1st bin, the 2nd bin, ..., the nth bin.
        An example input file and a loading python script of
        using the histogram reduced diagnostics
        are given in ``Examples/Tests/initial_distribution/``.

    * ``ParticleHistogram2D``
        This type computes a user defined, 2D particle histogram.

        * ``<reduced_diags_name>.species`` (``string``)
            A species name must be provided,
            such that the diagnostics are done for this species.

        * ``<reduced_diags_name>.file_min_digits`` (``int``) optional (default ``6``)
            The minimum number of digits used for the iteration number appended to the diagnostic file names.

        * ``<reduced_diags_name>.histogram_function_abs(t,x,y,z,ux,uy,uz,w)`` (``string``)
            A histogram function must be provided for the abscissa axis.
            ``t`` represents the physical time in seconds during the simulation.
            ``x, y, z`` represent particle positions in the unit of meter.
            ``ux, uy, uz`` represent the particle velocities in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            ``w`` represents the weight.

        * ``<reduced_diags_name>.histogram_function_ord(t,x,y,z,ux,uy,uz,w)`` (``string``)
            A histogram function must be provided for the ordinate axis.

        * ``<reduced_diags_name>.bin_number_abs`` (``int`` > 0) and ``<reduced_diags_name>.bin_number_ord`` (``int`` > 0)
            These are the number of bins used for the histogram for the abscissa and ordinate axis respectively.

        * ``<reduced_diags_name>.bin_max_abs`` (``float``) and ``<reduced_diags_name>.bin_max_ord`` (``float``)
            These are the maximum value of the bins for the abscissa and ordinate axis respectively.
            Particles with values outside of these ranges are discarded.

        * ``<reduced_diags_name>.bin_min_abs`` (``float``) and ``<reduced_diags_name>.bin_min_ord`` (``float``)
            These are the minimum value of the bins for the abscissa and ordinate axis respectively.
            Particles with values outside of these ranges are discarded.

        * ``<reduced_diags_name>.filter_function(t,x,y,z,ux,uy,uz,w)`` (``string``) optional
            Users can provide an expression returning a boolean for whether a particle is taken
            into account when calculating the histogram.
            ``t`` represents the physical time in seconds during the simulation.
            ``x, y, z`` represent particle positions in the unit of meter.
            ``ux, uy, uz`` represent particle velocities in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            ``w`` represents the weight.

        * ``<reduced_diags_name>.value_function(t,x,y,z,ux,uy,uz,w)`` (``string``) optional
            Users can provide an expression for the weight used to calculate the number of particles
            per cell associated with the selected abscissa and ordinate functions and/or the filter function.
            ``t`` represents the physical time in seconds during the simulation.
            ``x, y, z`` represent particle positions in the unit of meter.
            ``ux, uy, uz`` represent particle velocities in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            ``w`` represents the weight.

        The output is a ``<reduced_diags_name>`` folder containing a set of openPMD files.
        An example input file and a loading python script of
        using the histogram2D reduced diagnostics
        are given in ``Examples/Tests/histogram2D/``.

    * ``ParticleExtrema``
        This type computes the minimum and maximum values of
        particle position, momentum, gamma, weight,
        and the :math:`\chi` parameter for QED species.

        ``<reduced_diags_name>.species`` must be provided,
        such that the diagnostics are done for this species only.

        The output columns are
        minimum and maximum position :math:`x`, :math:`y`, :math:`z`;
        minimum and maximum momentum :math:`p_x`, :math:`p_y`, :math:`p_z`;
        minimum and maximum gamma :math:`\gamma`;
        minimum and maximum weight :math:`w`;
        minimum and maximum :math:`\chi`.

        Note that when the QED parameter :math:`\chi` is computed,
        field gather is carried out at every output,
        so the time of the diagnostic may be long
        depending on the simulation size.

    * ``ChargeOnEB``
        This type computes the total surface charge on the embedded boundary
        (in Coulombs), by using the formula

        .. math::

            Q_{tot} = \epsilon_0 \iint dS \cdot E

        where the integral is performed over the surface of the embedded boundary.

        When providing ``<reduced_diags_name>.weighting_function(x,y,z)``, the
        computed integral is weighted:

        .. math::

            Q = \epsilon_0 \iint dS \cdot E \times weighting(x, y, z)

        In particular, by choosing a weighting function which returns either
        1 or 0, it is possible to compute the charge on only some part of the
        embedded boundary.

    * ``CircuitCoupling``
        This type writes the per-step ledger of the coil / circuit coupling
        (requires :pp:param:`circuit.coils`; RZ): for every coil the realized
        drive scale :math:`s = I/I_\mathrm{ref}`, its segment slope
        :math:`ds/dt`, and the measured plasma flux linkage
        :math:`\lambda = \lambda_\mathrm{phys} I_\mathrm{ref} n` (zero for
        unmeasured coils or without a coupling engine), followed by the two
        sides of the coupling-power double entry

        .. math::

            P_\mathrm{circuit} = -\sum_k \frac{ds_k}{dt} \lambda_k,
            \qquad
            P_\mathrm{field} = \int J_p \cdot E_\mathrm{ext}\, dV,

        which agree to roundoff whenever the drive slopes are constant over
        the step (no power is created or lost in the exchange); in coupled
        runs the imbalance measures the within-step slope variation.

    * ``ColliderRelevant``
        This diagnostics computes properties of two colliding beams that are relevant for particle colliders.
        Two species must be specified. Photon species are not supported yet.
        It is assumed that the two species propagate and collide along the ``z`` direction.
        The output columns (for 3D-XYZ) are the following, where the minimum, average and maximum
        are done over the whole species:

        [0]: simulation step (iteration).

        [1]: time (s).

        [2]: time derivative of the luminosity (:math:`m^{-2}s^{-1}`) defined as:

        .. math::

            \frac{dL}{dt} = 2 c \iiint  n_1(x,y,z) n_2(x,y,z) dx dy dz

        where :math:`n_1`, :math:`n_2` are the number densities of the two colliding species.

        [3], [4], [5]: If, QED is enabled, the minimum, average and maximum values of the quantum parameter :math:`\chi` of species 1:
        :math:`\chi_{min}`,
        :math:`\langle \chi \rangle`,
        :math:`\chi_{max}`.
        If QED is not enabled, these numbers are not computed.

        [6], [7]: The average and standard deviation of the values of the transverse coordinate :math:`x` (m) of species 1:
        :math:`\langle x \rangle`,
        :math:`\sqrt{\langle x- \langle x \rangle \rangle^2}`.

        [8], [9]: The average and standard deviation of the values of the transverse coordinate :math:`y` (m) of species 1:
        :math:`\langle y \rangle`,
        :math:`\sqrt{\langle y- \langle y \rangle \rangle^2}`.

        [10], [11], [12], [13]: The minimum, average, maximum and standard deviation of the angle :math:`\theta_x = \angle (u_x, u_z)` (rad) of species 1:
        :math:`{\theta_x}_{min}`,
        :math:`\langle \theta_x \rangle`,
        :math:`{\theta_x}_{max}`,
        :math:`\sqrt{\langle \theta_x- \langle \theta_x \rangle \rangle^2}`.

        [14], [15], [16], [17]:  The minimum, average, maximum and standard deviation of the angle :math:`\theta_y = \angle (u_y, u_z)` (rad) of species 1:
        :math:`{\theta_y}_{min}`,
        :math:`\langle \theta_y \rangle`,
        :math:`{\theta_y}_{max}`,
        :math:`\sqrt{\langle \theta_y- \langle \theta_y \rangle \rangle^2}`.

        [18], ..., [32]: Analogous quantities for species 2.

        For 2D-XZ, :math:`y`-related quantities are not outputted.
        For 1D-Z, :math:`x`-related and :math:`y`-related quantities are not outputted.
        RZ, RCYLINDER, RSPHERE geometries are not supported yet.

    * ``DifferentialLuminosity``
        This type computes the differential luminosity between two species, defined as:

        .. math::

            \frac{d\mathcal{L}}{d\mathcal{E}^*}(\mathcal{E}^*, t) = \int_0^t dt'\int d\boldsymbol{x}\,d\boldsymbol{p}_1 d\boldsymbol{p}_2\;
             \sqrt{ |\boldsymbol{v}_1 - \boldsymbol{v}_2|^2 - |\boldsymbol{v}_1\times\boldsymbol{v}_2|^2/c^2} \\ f_1(\boldsymbol{x}, \boldsymbol{p}_1, t')f_2(\boldsymbol{x}, \boldsymbol{p}_2, t') \delta(\mathcal{E}^* - \mathcal{E}^*(\boldsymbol{p}_1, \boldsymbol{p}_2))

        where :math:`f_i` is the distribution function of species :math:`i` and
        :math:`\mathcal{E}^*(\boldsymbol{p}_1, \boldsymbol{p}_2) = \sqrt{m_1^2c^4 + m_2^2c^4 + 2 c^2{p_1}^\mu {p_2}_\mu}`
        is the energy in the center-of-mass frame, where :math:`p^\mu = (\sqrt{m^2 c^2 + \boldsymbol{p}^2}, \boldsymbol{p})`
        represents the 4-momentum. Note that, if :math:`\sigma^*(\mathcal{E}^*)`
        is the center-of-mass cross-section of a given collision process, then
        :math:`\int d\mathcal{E}^* \frac{d\mathcal{L}}{d\mathcal{E}^*} (\mathcal{E}^*, t)\sigma^*(\mathcal{E}^*)`
        gives the total number of collisions of that process (from the beginning of the simulation up until time :math:`t`).

        The differential luminosity is given in units of :math:`\text{m}^{-2}.\text{eV}^{-1}`. For collider-relevant WarpX simulations
        involving two crossing, high-energy beams of particles, the differential luminosity in :math:`\text{s}^{-1}.\text{m}^{-2}.\text{eV}^{-1}`
        can be obtained by multiplying the above differential luminosity by the expected repetition rate of the beams.

        In practice, the above expression of the differential luminosity is evaluated over discrete bins in energy :math:`\mathcal{E}^*`,
        and by summing over macroparticles.

        * ``<reduced_diags_name>.species`` (``list of two strings``)
            The names of the two species for which the differential luminosity is computed.

        * ``<reduced_diags_name>.bin_number`` (``int`` > 0)
            The number of bins in energy :math:`\mathcal{E}^*`

        * ``<reduced_diags_name>.bin_max`` (``float``, in eV)
            The minimum value of :math:`\mathcal{E}^*` for which the differential luminosity is computed.

        * ``<reduced_diags_name>.bin_min`` (``float``, in eV)
            The maximum value of :math:`\mathcal{E}^*` for which the differential luminosity is computed.

    * ``DifferentialLuminosity2D``
        This type computes the two-dimensional differential luminosity between two species, defined as:

        .. math::

            \frac{d^2\mathcal{L}}{dE_1 dE_2}(E_1, E_2, t) = \int_0^t dt'\int d\boldsymbol{x}\, \int d\boldsymbol{p}_1 \int d\boldsymbol{p}_2\;
             \sqrt{ |\boldsymbol{v}_1 - \boldsymbol{v}_2|^2 - |\boldsymbol{v}_1\times\boldsymbol{v}_2|^2/c^2} \\
             f_1(\boldsymbol{x}, \boldsymbol{p}_1, t')f_2(\boldsymbol{x}, \boldsymbol{p}_2, t') \delta(E_1 - E_1(\boldsymbol{p}_1)) \delta(E_2 - E_2(\boldsymbol{p}_2))

        where :math:`f_i` is the distribution function of species :math:`i`
        (normalized such that :math:`\int \int f(\boldsymbol{x} \boldsymbol{p}, t )d\boldsymbol{x} d\boldsymbol{p} = N`
        is the number of particles in species :math:`i` at time :math:`t`),
        :math:`\boldsymbol{p}_i` and :math:`E_i (\boldsymbol{p}_i) = \sqrt{m_1^2c^4 + c^2 |\boldsymbol{p}_i|^2}`
        are, respectively, the momentum and the energy of a particle of the :math:`i`-th species.
        The 2D differential luminosity is given in units of :math:`\text{m}^{-2}.\text{eV}^{-2}`.

        * ``<reduced_diags_name>.species`` (``list of two strings``)
            The names of the two species for which the differential luminosity is computed.

        * ``<reduced_diags_name>.bin_number_1`` (``int`` > 0)
            The number of bins in energy :math:`E_1`

        * ``<reduced_diags_name>.bin_max_1`` (``float``, in eV)
            The minimum value of :math:`E_1` for which the 2D differential luminosity is computed.

        * ``<reduced_diags_name>.bin_min_1`` (``float``, in eV)
            The maximum value of :math:`E_2` for which the 2D differential luminosity is compute

        * ``<reduced_diags_name>.bin_number_2`` (``int`` > 0)
            The number of bins in energy :math:`E_2`

        * ``<reduced_diags_name>.bin_max_2`` (``float``, in eV)
            The minimum value of :math:`E_2` for which the 2D differential luminosity is computed.

        * ``<reduced_diags_name>.bin_min_2`` (``float``, in eV)
            The minimum value of :math:`E_2` for which the 2D differential luminosity is computed.

        * ``<reduced_diags_name>.file_min_digits`` (``int``) optional (default ``6``)
            The minimum number of digits used for the iteration number appended to the diagnostic file names.

        The output is a ``<reduced_diags_name>`` folder containing a set of openPMD files.
        The values of the diagnostic are stored in a record labeled ``d2L_dE1_dE2``.
        An example input file and a loading python script of
        using the DifferentialLuminosity2D reduced diagnostics
        are given in ``Examples/Tests/diff_lumi_diag/``.

    * ``Timestep``
        This type outputs the simulation's physical timestep (in seconds) at each mesh refinement level.

.. pp:param:: reduced_diags.intervals
    :type: ``string``

    Using the `Time intervals`_ syntax, this string defines the timesteps at which reduced
    diagnostics are written to the file.
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.intervals``.

.. pp:param:: reduced_diags.path
    :type: ``string``
    :default: ``./diags/reducedfiles/``
    :optional:

    The path where the output file will be stored.
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.path``.

.. pp:param:: reduced_diags.extension
    :type: ``string``
    :default: ``txt``
    :optional:

    The extension of the output file (the suffix).
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.extension``.

.. pp:param:: reduced_diags.separator
    :type: ``string``
    :default: a ``whitespace``
    :optional:

    The separator between row values in the output file.
    The default separator is a whitespace.
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.separator``.

.. pp:param:: reduced_diags.precision
    :type: ``integer``
    :default: ``14``
    :optional:

    The precision used when writing out the data to the text files.
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.precision``.

.. _running-cpp-parameters-qed:

QED
---

These features require to compile with ``-DWarpX_QED=ON``, unless stated otherwise.

Nonlinear Compton scattering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This process is also known more generically as Quantum Synchrotron emission.

.. pp:param:: qed_qs.photon_creation_energy_threshold
    :type: ``float``
    :default: ``2``
    :optional:

    Energy threshold for photon particle creation in units of :math:`m_e c^2`.

.. pp:param:: <species_name>.do_qed_quantum_sync
    :type: ``int``
    :default: ``0``
    :optional:

    Enables Quantum synchrotron emission for this species.
    Quantum synchrotron lookup table should be either generated or loaded from disk to enable
    this process (see "Lookup tables for QED modules" section below).
    ``<species>`` must be either an electron or a positron species.

.. pp:param:: <species_name>.qed_quantum_sync_phot_product_species
    :type: ``string``

    If an electron or a positron species has the Quantum synchrotron process, a photon product species must be specified
    (the name of an existing photon species must be provided)

.. pp:param:: <species_name>.do_classical_radiation_reaction
    :type: ``int``
    :default: ``0``
    :optional:

    Enables Radiation Reaction (or Radiation Friction) for the species. Species
    must be either electrons or positrons. Boris pusher must be used for the
    simulation. If both ``<species>.do_classical_radiation_reaction`` and
    :pp:param:`<species_name>.do_qed_quantum_sync` are enabled, then the classical module
    will be used when the particle's chi parameter is below :pp:param:`qed_qs.chi_min`,
    the discrete quantum module otherwise. This feature does not require to compile with ``-DWarpX_QED=ON``.


Nonlinear Breit-Wheeler
^^^^^^^^^^^^^^^^^^^^^^^

.. pp:param:: <species_name>.do_qed_breit_wheeler
    :type: ``int``
    :default: ``0``
    :optional:

    Enables non-linear Breit-Wheeler process for this species.
    Breit-Wheeler lookup table should be either generated or loaded from disk to enable
    this process (see "Lookup tables for QED modules" section below).
    ``<species>`` must be a photon species (i.e., a species with :pp:param:`<species_name>.species_type` set to ``photon``)

.. pp:param:: <species_name>.qed_breit_wheeler_ele_product_species
    :type: ``string``

    If a photon species has the Breit-Wheeler process, an electron product species must be specified
    (the name of an existing electron species must be provided)

.. pp:param:: <species_name>.qed_breit_wheeler_pos_product_species
    :type: ``string``

    If a photon species has the Breit-Wheeler process, a positron product species must be specified
    (the name of an existing positron species must be provided).


Lookup tables
^^^^^^^^^^^^^

Lookup tables store pre-computed values for functions used by the nonlinear Compton Scattering and nonlinear Breit-Wheeler modules.
The lookup tables can be pre-generated using a standalone tool (see :ref:`qed tools section <generate-lookup-tables-with-tools>`).
Alternatively, one can use the low-resolution builtin tables or generate them on the fly at the beginning of the simulation.

.. pp:param:: qed_qs.lookup_table_mode
    :type: ``string``

    There are three options to prepare the lookup table required by the nonlinear Compton Scattering (or Quantum Synchrotron) module:

    * ``builtin``: a built-in table is used (Warning: the table gives reasonable results but its resolution is quite low).

    * ``generate``: a new table is generated on the fly at the beginning of the simulation. This option requires Boost math library
      (version >= 1.66) and the extra compilation flag ``-DWarpX_QED_TABLE_GEN=ON``.
      All the following parameters must be specified (table 1 is used to evolve the optical depth
      of the particles, while table 2 is used for photon emission):

        * ``qed_qs.tab_dndt_chi_min`` (``float``): minimum chi parameter for lookup table 1 (
          used for the evolution of the optical depth of electrons and positrons)

        * ``qed_qs.tab_dndt_chi_max`` (``float``): maximum chi parameter for lookup table 1

        * ``qed_qs.tab_dndt_how_many`` (``int``): number of points to be used for lookup table 1

        * ``qed_qs.tab_em_chi_min`` (``float``): minimum chi parameter for lookup table 2 (
          used for photon emission)

        * ``qed_qs.tab_em_chi_max`` (``float``): maximum chi parameter for lookup table 2

        * ``qed_qs.tab_em_chi_how_many`` (``int``): number of points to be used for chi axis in lookup table 2

        * ``qed_qs.tab_em_frac_how_many`` (``int``): number of points to be used for the second axis in lookup table 2
          (the second axis is the ratio between the quantum parameter of the photon and the
          quantum parameter of the charged particle).

        * ``qed_qs.tab_em_frac_min`` (``float``): minimum value to be considered for the second axis of lookup table 2

        * ``qed_qs.save_table_in`` (``string``): where to save the lookup table

    * ``load``: a lookup table is loaded from a pre-generated binary file. This can be a table generated by a previous run or using the standalone tool.
      The following parameter must be specified:

        * ``qed_qs.load_table_from`` (``string``): name of the lookup table file to read from.

.. pp:param:: qed_bw.lookup_table_mode
    :type: ``string``

    There are three options to prepare the lookup table required by the Breit-Wheeler module:

    * ``builtin``:  a built-in table is used (Warning: the table gives reasonable results but its resolution is quite low).

    * ``generate``: a new table is generated on the fly at the beginning of the simulation. This option requires Boost math library
      (version >= 1.66) and the extra compilation flag ``-DWarpX_QED_TABLE_GEN=ON``.
      All the following parameters must be specified (table 1 is used to evolve the optical depth
      of the photons, while table 2 is used for pair generation):

        * ``qed_bw.tab_dndt_chi_min`` (``float``): minimum chi parameter for lookup table 1 (
          used for the evolution of the optical depth of the photons)

        * ``qed_bw.tab_dndt_chi_max`` (``float``): maximum chi parameter for lookup table 1

        * ``qed_bw.tab_dndt_how_many`` (``int``): number of points to be used for lookup table 1

        * ``qed_bw.tab_pair_chi_min`` (``float``): minimum chi parameter for lookup table 2 (
          used for pair generation)

        * ``qed_bw.tab_pair_chi_max`` (``float``): maximum chi parameter for lookup table 2

        * ``qed_bw.tab_pair_chi_how_many`` (``int``): number of points to be used for chi axis in lookup table 2

        * ``qed_bw.tab_pair_frac_how_many`` (``int``): number of points to be used for the second axis in lookup table 2
          (the second axis is the ratio between the quantum parameter of the less energetic particle of the pair and the
          quantum parameter of the photon).

        * ``qed_bw.save_table_in`` (``string``): where to save the lookup table

    * ``load``: a lookup table is loaded from a pre-generated binary file. This can be a table generated by a previous run or using the standalone tool.
      The following parameter must be specified:

        * ``qed_bw.load_table_from`` (``string``): name of the lookup table file to read from.

.. pp:param:: qed_qs.chi_min
    :type: ``float``
    :comment: minimum chi parameter to be considered by the Quantum Synchrotron engine

    (suggested value : 0.001)

.. pp:param:: qed_bw.chi_min
    :type: ``float``
    :comment: minimum chi parameter to be considered by the Breit-Wheeler engine

    (suggested value : 0.01)


Schwinger process
^^^^^^^^^^^^^^^^^

.. pp:param:: warpx.do_qed_schwinger
    :type: ``bool``
    :default: ``0``
    :optional:

    If this is 1, Schwinger electron-positron pairs can be generated in vacuum in the cells where the EM field is high enough.
    If :pp:param:`warpx.do_qed_schwinger = 1`, Schwinger product species must be specified with
    :pp:param:`qed_schwinger.ele_product_species` and :pp:param:`qed_schwinger.pos_product_species`.
    Schwinger process requires either :pp:param:`warpx.grid_type = collocated` or
    :pp:param:`algo.field_gathering = momentum-conserving` (so that different field components are computed
    at the same location in the grid) and does not currently support mesh refinement, cylindrical
    coordinates or single precision.

.. pp:param:: qed_schwinger.ele_product_species
    :type: ``string``

    If Schwinger process is activated, an electron product species must be specified
    (the name of an existing electron species must be provided).

.. pp:param:: qed_schwinger.pos_product_species
    :type: ``string``

    If Schwinger process is activated, a positron product species must be specified
    (the name of an existing positron species must be provided).

.. pp:param:: qed_schwinger.y_size
    :type: ``float``
    :unit: meters

    If Schwinger process is activated with ``DIM=2D``, a transverse size must be specified.
    It is used to convert the pair production rate per unit volume into an actual number of created particles.
    This value should correspond to the typical transverse extent for which the EM field has a very high value
    (e.g. the beam waist for a focused laser beam).

.. pp:param:: qed_schwinger.xmin/ymin/zmin/xmax/ymax/zmax
    :link_aliases:
        qed_schwinger.xmin
        qed_schwinger.ymin
        qed_schwinger.zmin
        qed_schwinger.xmax
        qed_schwinger.ymax
        qed_schwinger.zmax
        qed_schwinger.xmin,ymin,zmin
        qed_schwinger.xmax,ymax,zmax
        qed_schwinger.xmin,ymin,zmin,xmax,ymax,zmax
    :type: ``float``
    :default: unlimited
    :optional:

    When :pp:param:`qed_schwinger.xmin` and :pp:param:`qed_schwinger.xmax` are set, they delimit the region within
    which Schwinger pairs can be created.
    The same is applicable in the other directions.

.. pp:param:: qed_schwinger.threshold_poisson_gaussian
    :type: ``integer``
    :default: ``25``
    :optional:

    If the expected number of physical pairs created in a cell at a given timestep is smaller than this threshold,
    a Poisson distribution is used to draw the actual number of physical pairs created.
    Otherwise a Gaussian distribution is used.
    Note that, regardless of this parameter, the number of macroparticles created is at most one per cell
    per timestep per species (with a weight corresponding to the number of physical pairs created).

.. pp:param:: warpx.use_hybrid_QED
    :type: ``bool``
    :default: 0

    Will use the Hybrid QED Maxwell solver when pushing fields: a QED correction is added to the
    field solver to solve non-linear Maxwell's equations, according to :cite:t:`param-GrismayerNJP2021`.
    Note that this option can only be used with the PSATD build. Furthermore, one must set
    :pp:param:`warpx.grid_type = collocated` (which otherwise would be ``staggered`` by default).
    This feature does not require to compile with ``-DWarpX_QED=ON``.

.. pp:param:: warpx.quantum_xi
    :type: ``float``
    :default: 1.3050122.e-52

    Overwrites the actual quantum parameter used in Maxwell's QED equations. Assigning a
    value here will make the simulation unphysical, but will allow QED effects to become more apparent.
    Note that this option will only have an effect if the ``warpx.use_Hybrid_QED`` flag is also triggered.
    This feature does not require to compile with ``-DWarpX_QED=ON``.


Checkpoints and restart
-----------------------
WarpX supports checkpoints/restart via AMReX.
The checkpoint capability can be turned with regular diagnostics: :pp:param:`<diag_name>.format = checkpoint`.

.. pp:param:: amr.restart
    :type: ``string``

    Name of the checkpoint file to restart from. Returns an error if the folder does not exist
    or if it is not properly formatted.

.. pp:param:: warpx.write_diagnostics_on_restart
    :type: ``bool``
    :default: ``false``
    :optional:

    When ``true``, write the diagnostics after restart at the time of the restart.


.. _running-cpp-parameters-test-debug:

Testing and Debugging
---------------------

When developing, testing and :ref:`debugging WarpX <debugging_warpx>`, the following options can be considered.

.. pp:param:: warpx.verbose
    :type: ``0`` or ``1``
    :default: ``1`` for true

    Controls how much information is printed to the terminal, when running WarpX.

.. pp:param:: warpx.limit_verbose_step
    :type: ``bool``
    :default: false

    If set to true, the information normally printed to the terminal at every time step
    is limited: it prints every step for the first 10 steps, every 10 steps for steps between 10 and 100,
    and once every 100 steps for steps greater than 100.

.. pp:param:: warpx.always_warn_immediately
    :type: ``0`` or ``1``
    :default: ``0`` for false

    If set to ``1``, WarpX immediately prints every warning message as soon as
    it is generated. It is mainly intended for debug purposes, in case a simulation
    crashes before a global warning report can be printed.

.. pp:param:: warpx.abort_on_warning_threshold
    :type: string
    :optional:
    :comment: ``low``, ``medium`` or ``high``

    Optional threshold to abort as soon as a warning is raised.
    If the threshold is set, warning messages with priority greater than or
    equal to the threshold trigger an immediate abort.
    It is mainly intended for debug purposes, and is best used with
    :pp:param:`warpx.always_warn_immediately = 1`.

.. pp:param:: amrex.abort_on_unused_inputs
    :type: ``0`` or ``1``
    :default: ``0`` for false

    When set to ``1``, this option causes simulation to fail *after* its completion if there were unused parameters.
    It is mainly intended for continuous integration and automated testing to check that all tests and inputs are adapted to API changes.

.. pp:param:: amrex.use_profiler_syncs
    :type: ``0`` or ``1``
    :default: ``0`` for false

    Adds a synchronization at the start of communication, so any load balance will be caught there (the timer is called ``SyncBeforeComms``), then the comm operation will run.
    This will slow down the run.

.. pp:param:: warpx.serialize_initial_conditions
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Serialize the initial conditions for reproducible testing, e.g, in our continuous integration tests.
    Mainly whether or not to use OpenMP threading for particle initialization.

.. pp:param:: warpx.safe_guard_cells
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Run in safe mode, exchanging more guard cells, and more often in the PIC loop (for debugging).

.. pp:param:: ablastr.fillboundary_always_sync
    :type: ``0`` or ``1``
    :default: ``0``
    :optional:

    Run all ``FillBoundary`` operations on ``MultiFab`` to force-synchronize shared nodal points.
    This slightly increases communication cost and can help to spot missing ``nodal_sync`` flags in these operations.

.. bibliography::
    :keyprefix: param-
