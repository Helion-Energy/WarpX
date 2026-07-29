.. _theory-kinetic-fluid-hybrid-model:

Ampere's law coupled with Ohm's law (a.k.a. "hybrid PIC")
=========================================================

Many problems in plasma physics fall in a class where both electron kinetics and electromagnetic waves do not
play a critical role in the solution. Examples of such situations include the
study of collisionless magnetic reconnection and instabilities driven by ion
temperature anisotropy, to mention only two. For these kinds of problems the
computational cost of resolving the electron dynamics can be avoided by modeling
the electrons as a neutralizing fluid rather than kinetic particles. By further
using Ohm's law to compute the electric field rather than evolving it with the
Maxwell-Faraday equation, light waves can be stepped over. The simulation resolution
can then be set by the ion time and length scales (commonly the ion cyclotron
period :math:`1/\Omega_i` and ion skin depth :math:`l_i`, respectively), which
can reduce the total simulation time drastically compared to a simulation that
has to resolve the electron Debye length and CFL-condition based on the speed
of light.

Many authors have described variations of the kinetic ion & fluid electron model,
generally referred to as particle-fluid hybrid or just hybrid-PIC models. The
implementation in WarpX is described in detail in :cite:t:`kfhm-Groenewald2023`.
The "Model derivation" section below gives a detailed description of the model
that follows mostly from the above reference, but succinctly, the model
entails the following:

The magnetic field is advanced in time using Faraday's law,

    .. math::

        \frac{\partial\vec{B}}{\partial t} = -\nabla\times\vec{E},

where the electric field is calculated from Ohm's law which involves the currents,
the magnetic field, and the electron pressure (for which an additional closure is required,
see :ref:`here <theory-hybrid-model-elec-temp>`),

    .. math::

        \vec{E} = \frac{\vec{J}_e\times\vec{B}-\nabla P_e}{en_e}
        +\eta\vec{J}-\eta_h \nabla^2\vec{J}.

The electron current is in turn obtained by subtracting the ion current (obtained from
kinetic ion macro-particles) from the total current (obtained from Ampere's law):

    .. math::

        \vec{J}_e = \vec{J} - \sum_{s\neq e}\vec{J}_s - \vec{J}_{ext}

where

    .. math::

        \mu_0\vec{J} = \vec{\nabla}\times\vec{B}.

Equivalently, for one ion fluid with
:math:`\vec{J}_i=e n_e\vec{u}_i`, the implemented ideal and Hall terms are

    .. math::

        \vec{E}
        = -\vec{u}_i\times\vec{B}
          + \chi_H\frac{\vec{J}\times\vec{B}}{e n_e}
          - \chi_P\frac{\nabla P_e}{e n_e}
          + \eta\vec{J}-\eta_h\nabla^2\vec{J}.

The runtime switches :pp:param:`hybrid_pic_model.include_hall_term` and
:pp:param:`hybrid_pic_model.include_electron_pressure_term` set
:math:`\chi_H` and :math:`\chi_P`, respectively. In particular,
:math:`\chi_H=0` removes only Hall physics and leaves the ideal
:math:`-\vec{u}_i\times\vec{B}` term intact. The standard resistive-MHD
Ohm's law is selected with :math:`\chi_H=\chi_P=0`. Keeping only one of
these two terms is supported as an exploratory extended-MHD model, but does
not retain the continuum total-energy exchange of either standard resistive
MHD or the complete Hall/electron-pressure system.

Algorithm details
-----------------

.. note::

    Various verification tests of the hybrid model implementation can be found in
    the :ref:`examples section <examples-hybrid-model>`.

.. _theory-hybrid-model-implicit-mhd:

Theta-implicit ion-fluid mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to kinetic-ion hybrid PIC, WarpX provides an initial
single-ion-fluid mode selected by
``algo.evolve_scheme = theta_implicit_mhd``. It replaces the particle push and
moment deposition inside the nonlinear loop with an ion continuity and
momentum update. This monolithic Jacobian-free Newton--Krylov treatment follows
the fully implicit MHD strategy of :cite:t:`Chacon2008`. A first
physics-based block preconditioner for the non-Hall, small-flow limit is
described below. The nonlinear unknown is

.. math::

   \mathcal{U}^{n+\theta}
   = \left(\boldsymbol E,\rho_i,\boldsymbol M_i,U_e\right)^{n+\theta},
   \qquad \boldsymbol M_i=\rho_i\boldsymbol u_i ,

and the magnetic field at the same time is obtained from Faraday's law. The
fluid part of the model is

.. math::

   \frac{\partial \rho_i}{\partial t}
   + \nabla\cdot\boldsymbol M_i = 0,

.. math::

   \frac{\partial\boldsymbol M_i}{\partial t}
   + \nabla\cdot\left(
       \frac{\boldsymbol M_i\boldsymbol M_i}{\rho_i}
       + (P_i+P_e)\mathbb I
     \right)
   = \boldsymbol J\times\boldsymbol B ,

.. math::

   \frac{\partial U_e}{\partial t}
   + \nabla\cdot(U_e\boldsymbol u_e)
   + P_e\nabla\cdot\boldsymbol u_e
   = \eta|\boldsymbol J|^2 ,

where

.. math::

   P_e=(\gamma_e-1)U_e,\qquad
   P_i=P_{i0}\left(\frac{\rho_i}{\rho_{i0}}\right)^{\gamma_i},
   \qquad
   \boldsymbol u_e
   =\boldsymbol u_i-\chi_H\frac{\boldsymbol J}{\rho_q},
   \qquad
   \rho_q=\frac{q_i}{m_i}\rho_i.

The total current is
:math:`\boldsymbol J=\nabla\times\boldsymbol B/\mu_0`, and the electric field
is constrained by the same generalized Ohm's law as the kinetic-ion model.
Thus setting ``hybrid_pic_model.include_hall_term = false`` consistently makes
:math:`\boldsymbol u_e=\boldsymbol u_i` in the energy equation and removes the
Hall term from Ohm's law. For the standard resistive-MHD limit, also set
``hybrid_pic_model.include_electron_pressure_term = false``; the evolved
electron pressure remains in the total fluid-pressure force. Leaving the
electron-pressure Ohm term enabled instead retains a Biermann-like extension
that is not part of the conservative resistive-MHD limit implemented here.
Setting ``implicit_mhd.evolve_ion_fluid = false`` instead freezes the
prescribed ion density and momentum. With Hall physics enabled and a stationary
ion background, this gives the electron-MHD limit
:math:`\boldsymbol u_e=-\boldsymbol J/\rho_q`, which is useful for whistler and
magnetic-structure studies on timescales short compared with the ion response.

For every component :math:`Q` of the fluid state, the nonlinear residual uses

.. math::

   Q^{n+\theta}-Q^n-\theta\Delta t\,
   \mathcal{R}\!\left(\mathcal{U}^{n+\theta}\right)=0.

After convergence,
:math:`Q^{n+1}=Q^n+(Q^{n+\theta}-Q^n)/\theta`. The electric and magnetic fields
use the same time centering during the coupled update; because the electric
field is algebraic, it is then recomputed from the final fluid and magnetic
state so checkpointed/output :math:`\boldsymbol E^{n+1}` satisfies Ohm's law.
Characteristic density, velocity, magnetic-field, momentum, and energy scales
normalize the composite JFNK vector so that its Euclidean norm does not mix
dimensional quantities.

Physics-based block preconditioner
""""""""""""""""""""""""""""""""""

The prototype selected with ``jacobian.pc_type = pc_mhd_block`` approximates
the inverse Jacobian with independent resistive-field and acoustic-fluid
blocks. Let :math:`h=\theta\Delta t`. In the non-Hall limit, Faraday's law and
the resistive part of Ohm's law give the field operator

.. math::

   \mathcal{P}_E\,\delta\boldsymbol E
   =
   \left[
      \mathbb I
      + \frac{h\eta_*}{\mu_0}
        \nabla\times\nabla\times
   \right]\delta\boldsymbol E ,

where :math:`\eta_*` is a scalar reference resistivity evaluated at the
theta-centered time. This staggered curl--curl system is approximately
inverted with AMReX MLMG.

The fluid block is the small-flow pressure Schur complement corresponding to
the predictor--Schur--corrector strategy in :cite:t:`Chacon2008`. This
prototype freezes domain-global/reference acoustic scalars during each Newton
linear solve rather than constructing spatially varying coefficient fields.
Specifically, it uses
:math:`a_e=\gamma_e\sum U_e/\sum\rho_i` and
:math:`c_i^2=\gamma_iP_{i,\mathrm{ref}}/\rho_{i,\mathrm{ref}}`.
Given a preconditioner right-hand side
:math:`(b_\rho,\boldsymbol b_M,b_U)`, define

.. math::

   b_p = c_i^2 b_\rho + \chi_e b_U,\qquad
   c_s^2 = c_i^2+\chi_e a_e ,

where :math:`c_i^2` is the reference-state approximation to
:math:`\partial P_i/\partial\rho_i`,
:math:`\chi_e=\partial P_e/\partial U_e`, and
:math:`a_e=(U_e+P_e)/\rho_i`. The pressure increment is obtained from

.. math::

   \left(\frac{1}{c_s^2}-h^2\nabla^2\right)\delta p
   =
   \frac{b_p}{c_s^2}-h\nabla\cdot\boldsymbol b_M ,

followed by

.. math::

   \delta\boldsymbol M
   &= \boldsymbol b_M-h\nabla\delta p,\\
   \delta\rho_i
   &= b_\rho-h\nabla\cdot\delta\boldsymbol M,\\
   \delta U_e
   &= b_U-ha_e\nabla\cdot\delta\boldsymbol M.

The scalar Helmholtz equation is approximately inverted with
``MLABecLaplacian`` and MLMG. Both multigrid applications start from zero, run
fixed cycle counts, and use a fixed smoother at the bottom. The resulting map
is stationary and linear to solver roundoff within each standard
right-preconditioned GMRES solve; AMReX MLMG can still stop early if it reaches
its internal machine-precision residual threshold.

This first version is restricted to one Cartesian, periodic AMR level, a
staggered field grid, ``implicit_mhd.fluid_flux = centered``, evolving ions,
zero Hall and electron-pressure terms in Ohm's law, and zero
hyper-resistivity. Resistivity may be constant or time dependent; density,
current, and per-species dependence is rejected because the field block uses
only the scalar value :math:`\eta_*`. Strongly nonuniform and pressure-floor
active states are also outside the robustness claim because the acoustic
coefficients are global scalars and do not linearize the pressure ``max`` at
its floor. The approximation omits background-flow advection, the
Joule-heating derivative, and ideal magnetic/Lorentz couplings, so it is not
yet the full MHD wave preconditioner of :cite:t:`Chacon2008`.

There is also a deliberate discretization mismatch in the acoustic block.
The centered nonlinear face flux composes centered cell derivatives and has
the one-dimensional pressure symbol
:math:`-\sin^2(k\Delta x)/\Delta x^2`, including its checkerboard null mode.
The compact MLMG Laplacian instead has symbol
:math:`-4\sin^2(k\Delta x/2)/\Delta x^2`. The two agree for smooth,
well-resolved modes but not near the grid scale. Thus this prototype is
expected to accelerate smooth acoustic and resistive problems; it does not
establish grid-independent robustness for discontinuities or checkerboard
modes.

The fluid divergence is evaluated from one shared face flux, so mass and
fluid momentum telescope conservatively on the periodic mesh. The default
``implicit_mhd.fluid_flux = centered`` option retains a second-order centered
operator for smooth, low-dissipation verification. The optional ``rusanov``
flux adds piecewise-constant local Lax--Friedrichs dissipation using the
ion/electron acoustic and advective speeds. Newton updates and matrix-free
Jacobian probes are restricted to positive density and electron energy, with
residual-decreasing backtracking. The bound accounts for the final
:math:`1/\theta` extrapolation as well as the intermediate theta state.

The Rusanov option is a shock-regularized *fluid* flux, not yet a conservative
MHD shock formulation: magnetic induction remains in the staggered
Ohm/Faraday operator, the ion closure is barotropic, and the evolved thermal
variable is electron internal energy rather than total energy. As discussed
by :cite:t:`Chacon2008`, a total-energy equation is preferable when exact
shock-energy conservation is required. Compatible electromagnetic work,
ion/total-energy evolution, AMR/non-periodic boundaries, and a block
preconditioner containing the complete ideal-MHD wave operator remain planned
extensions.

The kinetic-fluid hybrid extension mostly uses the same routines as the standard electromagnetic
PIC algorithm with the only exception that the E-field is calculated from Ohm's law
rather than it being updated from the full Maxwell-Ampere equation. The E-field update occurs
after particle pushing and deposition (charge and current density) has been completed. Therefore, based
on the usual time-staggering in the PIC algorithm, when the E-field is updated
at timestep :math:`t=t_n`, the quantities :math:`\rho^n`, :math:`\rho^{n+1}`, :math:`\vec{J}_i^{n-1/2}`
and  :math:`\vec{J}_i^{n+1/2}` are all known.

Field update
^^^^^^^^^^^^

The field update is done in three steps as described below.

First half step
"""""""""""""""

Firstly the E-field at :math:`t=t_n` is calculated for which the current density needs to
be interpolated to the correct time, using :math:`\vec{J}_i^n = 1/2(\vec{J}_i^{n-1/2}+ \vec{J}_i^{n+1/2})`.
The electron pressure is simply calculated using :math:`\rho^n` and the B-field is also already
known at the correct time since it was calculated for :math:`t=t_n` at the end of the last step.
Once :math:`\vec{E}^n` is calculated, it is used to push :math:`\vec{B}^n` forward in time
(using the Maxwell-Faraday equation) to :math:`\vec{B}^{n+1/2}`.

Second half step
""""""""""""""""

Next, the E-field is recalculated to get :math:`\vec{E}^{n+1/2}`. This is done
using the known fields :math:`\vec{B}^{n+1/2}`, :math:`\vec{J}_i^{n+1/2}` and
interpolated charge density :math:`\rho^{n+1/2}=1/2(\rho^n+\rho^{n+1})` (which is
also used to calculate the electron pressure). Similarly as before, the B-field
is then pushed forward to get :math:`\vec{B}^{n+1}` using the newly calculated
:math:`\vec{E}^{n+1/2}` field.

Extrapolation step
""""""""""""""""""

Obtaining the E-field at timestep :math:`t=t_{n+1}` is a well documented issue for
the hybrid model. Currently the approach in WarpX is to simply extrapolate
:math:`\vec{J}_i` forward in time, using

    .. math::

        \vec{J}_i^{n+1} = \frac{3}{2}\vec{J}_i^{n+1/2} - \frac{1}{2}\vec{J}_i^{n-1/2}.

With this extrapolation all fields required to calculate :math:`\vec{E}^{n+1}`
are known and the simulation can proceed.

Sub-stepping
^^^^^^^^^^^^

It is also well known that hybrid PIC routines require the B-field to be
updated with a smaller timestep than needed for the particles. A 4th order
Runge-Kutta scheme is used to update the B-field. The RK scheme is repeated a
number of times during each half-step outlined above. The number of sub-steps
used can be specified by the user through a runtime simulation parameter
(see :ref:`input parameters section <running-cpp-parameters-hybrid-model>`).

.. _theory-hybrid-model-elec-temp:

Electron pressure
^^^^^^^^^^^^^^^^^

The electron pressure is assumed to be a scalar quantity and calculated using the given
input parameters, :math:`T_{e0}`, :math:`n_0` and :math:`\gamma` using

    .. math::

        P_e = n_0T_{e0}\left( \frac{n_e}{n_0} \right)^\gamma.

The isothermal limit is given by :math:`\gamma = 1` while :math:`\gamma = 5/3`
(default) produces the adiabatic limit.

Alternatively, the electron temperature entering the pressure can be evolved
in space and time with the electron energy equation, as described in the next
section.

.. _theory-hybrid-model-electron-energy-eq:

Electron energy equation
^^^^^^^^^^^^^^^^^^^^^^^^

Instead of evaluating the polytropic closure with the constant reference state
:math:`(n_0, T_{e0})`, WarpX can evolve the electron temperature
:math:`T_e(\vec{x}, t)` with the electron internal-energy equation
(``hybrid_pic_model.solve_electron_energy_equation``),

    .. math::

        \frac{\partial U_e}{\partial t} + \nabla\cdot(U_e \vec{V}_e) + P_e \nabla\cdot\vec{V}_e = S_e,

where :math:`U_e = n_e k_B T_e/(\gamma - 1)` is the electron internal energy
density, :math:`\vec{V}_e = \vec{J}_e/(-e n_e)` is the electron fluid velocity
and :math:`S_e` collects the source and sink terms. The local electron
pressure :math:`P_e = n_e k_B T_e` then feeds back into Ohm's law.

The homogeneous part of the equation (the left-hand side) is solved with the
QDSMC kinetic-enslavement scheme of :cite:t:`kfhm-Belyaev2024`: the electron
entropy function :math:`K_e = T_e\, n_e^{1-\gamma}`, which the transport terms
conserve along electron-fluid characteristics, is advected by fictitious
Lagrangian markers. Each PIC step one marker is initialized at every cell
center carrying the local :math:`K_e N_e` and :math:`N_e` (with :math:`N_e`
the electron count of the cell), is pushed by one timestep with
:math:`\vec{V}_e` interpolated at its position, and both quantities are
deposited back to the grid with the standard (linear) particle shape factors.
The updated temperature is recovered from the deposited quantities and the
ion-derived density as

    .. math::

        T_e = \frac{\sum K_e N_e}{\sum N_e}\, n_e^{\gamma - 1}.

Since the scheme only advects the electron entropy, thermal conduction is
neglected (:math:`\nabla\cdot\vec{q}_e = 0`).

Two source terms can be enabled on the right-hand side. The first is the Joule
(Ohmic) heating consistent with the resistive friction in Ohm's law
(``hybrid_pic_model.include_joule_heating``), applied per ion species
:math:`s`:

    .. math::

        \frac{d T_e}{d t} = (\gamma - 1) \sum_s \frac{Z_s e^2\, \eta_{s,\mathrm{eff}}\, n_s |\Delta\vec{V}|^2}{k_B},

where :math:`\Delta\vec{V} = \vec{J}/(e n_e)` is the electron-ion relative
drift, :math:`Z_s` the charge state and
:math:`\eta_{s,\mathrm{eff}} = \eta + \eta_s` the sum of the global and
per-species resistivities (see above). For a single species this reduces
exactly to the familiar :math:`dT_e/dt = (\gamma - 1)\,\eta J^2/(n_e k_B)`.
Above a user-set electron temperature threshold the heat can optionally be
redirected to the kinetic ions instead of the electron fluid
(``hybrid_pic_model.joule_redirect_Te_threshold``), which is useful to model
regimes where the electrons radiate strongly.

The second source is the electron-ion temperature relaxation, enabled by
specifying the rate ``hybrid_pic_model.electron_ion_relaxation_rate``,

    .. math::

        Q_{ei} = \sum_s 3\, n_s k_B\, \nu_{ei}\, (T_e - T_{i,s}),

with the rate :math:`\nu_{ei}(\rho, T_e, T_i, t)` given by a user expression.
The sink on the electron fluid is paired with a matching thermal-velocity
kick on the ion macro-particles of each species so that the exchange
conserves energy exactly.

Verification tests of the transport terms (adiabatic compression), the Joule
source (force-free field decay) and the :math:`Q_{ei}` exchange are described
in the :ref:`examples section <examples-ohm-solver-electron-energy-eq>`.

Electron current
^^^^^^^^^^^^^^^^

WarpX's displacement current diagnostic can be used to output the electron current in
the kinetic-fluid hybrid model since in the absence of kinetic electrons, and under
the assumption of zero displacement current, that diagnostic simply calculates the
hybrid model's electron current.

Model derivation
----------------

The basic justification for the hybrid model is that the system to which it is
applied is dominated by ion kinetics, with ions moving much slower than electrons
and photons. In this scenario two critical approximations can be made, namely,
neutrality (:math:`n_e=n_i`) and the Maxwell-Ampere equation can be simplified by
neglecting the displacement current term :cite:p:`kfhm-Nielson1976`, giving,

    .. math::

        \mu_0\vec{J} = \vec{\nabla}\times\vec{B},

where :math:`\vec{J} = \sum_{s\neq e}\vec{J}_s + \vec{J}_e + \vec{J}_{ext}` is the total electrical current,
i.e. the sum of electron and ion currents as well as any external current (not captured through plasma
particles). Since ions are treated in the regular
PIC manner, the ion current, :math:`\sum_{s\neq e}\vec{J}_s`, is known during a simulation. Therefore,
given the magnetic field, the electron current can be calculated.

The electron momentum transport equation (obtained from multiplying the Vlasov equation by mass and
integrating over velocity), also called the generalized Ohm's law, is given by:

    .. math::

        en_e\vec{E} = \frac{m}{e}\frac{\partial \vec{J}_e}{\partial t} + \frac{m}{e}\left( \vec{U}_e\cdot\nabla \right) \vec{J}_e - \nabla\cdot {\overleftrightarrow P}_e + \vec{J}_e\times\vec{B}+\vec{R}_e

where :math:`\vec{U}_e = -\vec{J}_e/(en_e)` is the electron fluid velocity,
:math:`{\overleftrightarrow P}_e` is the electron pressure tensor and
:math:`\vec{R}_e` is the drag force due to collisions between electrons and ions.
Applying the above momentum equation to the Maxwell-Faraday equation (:math:`\frac{\partial\vec{B}}{\partial t} = -\nabla\times\vec{E}`)
and substituting in :math:`\vec{J}` calculated from the Maxwell-Ampere equation, gives,

    .. math::

        \frac{\partial\vec{J}_e}{\partial t} = -\frac{1}{\mu_0}\nabla\times\left(\nabla\times\vec{E}\right) - \frac{\partial\vec{J}_{ext}}{\partial t} - \sum_{s\neq e}\frac{\partial\vec{J}_s}{\partial t}.

Plugging this back into the generalized Ohm's law gives:

    .. math::

        \left(en_e +\frac{m}{e\mu_0}\nabla\times\nabla\times\right)\vec{E} =&
        - \frac{m}{e}\left( \frac{\partial\vec{J}_{ext}}{\partial t} + \sum_{s\neq e}\frac{\partial\vec{J}_s}{\partial t} \right) \\
        &+ \frac{m}{e}\left( \vec{U}_e\cdot\nabla \right) \vec{J}_e - \nabla\cdot {\overleftrightarrow P}_e + \vec{J}_e\times\vec{B}+\vec{R}_e.

If we now further assume electrons are inertialess (i.e. :math:`m=0`), the above equation simplifies to,

    .. math::

        en_e\vec{E} = \vec{J}_e\times\vec{B}-\nabla\cdot{\overleftrightarrow P}_e+\vec{R}_e.

Making the further simplifying assumptions that the electron pressure is isotropic and that
the electron drag term can be written using a simple resistivity (:math:`\eta`) and hyper-resistivity (:math:`\eta_h`)
i.e. :math:`\vec{R}_e = en_e(\eta-\eta_h \nabla^2)\vec{J}`, brings us to the implemented form of
Ohm's law:

    .. math::

        \vec{E} = \frac{\vec{J}_e\times\vec{B}-\nabla P_e}{en_e}
        +\eta\vec{J}-\eta_h \nabla^2\vec{J}.

Multi-species resistivity
^^^^^^^^^^^^^^^^^^^^^^^^^

The single resistivity :math:`\eta` above assumes the electron drag is the
same against every ion species. Following :cite:t:`kfhm-Belyaev2024`, the drag
can instead be resolved per species by adding to Ohm's law the overlay

    .. math::

        \vec{E} \mathrel{+}= \sum_s \eta_s\, f_s\, e n_e \left( \vec{V}_s - \vec{V}_e \right),
        \qquad f_s = \frac{\rho_s}{\sum_t \rho_t},

where :math:`f_s` is the charge-density fraction of species :math:`s`,
:math:`\vec{V}_s` its fluid velocity, and each :math:`\eta_s` is a user
expression of :math:`(\rho_s, \rho, T_e, |\vec{J}|, |\vec{J}_s|, |\vec{B}|, t)`
(``hybrid_pic_model.plasma_resistivity_<species>(rho_s,rho,Te,J,J_s,B,t)``).
This permits, for example, a temperature-dependent Spitzer drag against one
species on top of a constant background resistivity. When all ion species
drift together the overlay reduces to
:math:`\left(\sum_s f_s \eta_s\right)\vec{J}`, i.e. an effective resistivity
:math:`\eta_{\mathrm{eff}} = \eta + \sum_s f_s \eta_s`. The same
:math:`\eta_{s,\mathrm{eff}} = \eta + \eta_s` enters the per-species Joule
heating of the :ref:`electron energy equation
<theory-hybrid-model-electron-energy-eq>`.

Lastly, if an electron temperature is given from which the electron pressure can
be calculated, the model is fully constrained and can be evolved given initial
conditions.

.. bibliography::
    :keyprefix: kfhm-
