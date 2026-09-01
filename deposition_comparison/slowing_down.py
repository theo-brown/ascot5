"""Analytic steady-state slowing-down model (Stix) for NBI fast ions, in JAX.

Given the beam-ion *birth* markers (rho, energy, weight [particles/s]) produced
by a deposition calculation, this module evaluates the classical steady-state
slowing-down distribution and the collisional heating split on a shared
(rho, energy) grid, with all plasma quantities taken at the rho-bin centers
from :func:`deposition_comparison.physics.profiles`.

Physics model (references: T. H. Stix 1972 Plasma Phys. 14 367,
"Heating of toroidal plasmas by neutral injection"; NRL Plasma Formulary)
------------------------------------------------------------------------
A fast ion of mass ``m_b = mass_amu * AMU_KG`` [kg] and charge ``Z_b e`` slows
down on electrons and thermal ions according to (Stix 1972, Eq. 15)::

    dv/dt = -(v / tau_se) * (1 + v_c^3 / v^3)                            [m/s^2]

* Spitzer slowing-down time on electrons (SI units; Stix 1972, Eq. 14 /
  Wesson "Tokamaks" NBI chapter)::

      tau_se = 3 (2 pi)^{3/2} eps0^2 m_b (k Te)^{3/2}
               / (sqrt(m_e) Z_b^2 e^4 n_e lnL_e)                         [s]

  with ``k Te = Te[eV] * E_CHARGE`` [J], ``n_e`` [m^-3].  NOTE: the phase-2
  interface contract writes ``sqrt(m_e)`` in the NUMERATOR; that placement is
  dimensionally inconsistent (it yields kg*s and values ~1e-30 s).  The form
  above is the standard one — it is dimensionally a time and reproduces the
  NRL practical formula ``tau_se[s] ~= 6.27e8 A_b Te[eV]^{3/2} /
  (Z_b^2 n_e[cm^-3] lnL)`` (checked in the sanity block).  This is the one
  deliberate deviation from the contract text.

* Electron Coulomb logarithm (NRL Formulary, "thermal electron-ion
  collisions", Te > 10 Z^2 eV regime)::

      lnL_e = 24 - ln( sqrt(n_e[cm^-3]) / Te[eV] )

  clamped to ``lnL_e >= 5`` so vacuum/edge regions stay well-behaved.

* Critical velocity (Stix 1972, Eq. 16, with the electron Coulomb logarithm
  applied to ALL channels -- see "Coulomb-logarithm caveat" below)::

      v_c^3 = (3 sqrt(pi) / 4) * (m_e / n_e)
              * sum_i (n_i Z_i^2 / m_i) * v_te^3,   v_te = sqrt(2 k Te / m_e)

  and the critical energy is the beam-species kinetic energy at ``v_c``::

      E_c = (1/2) m_b v_c^2                                              [J]

  (equivalently ``E_c ~= 14.8 * A_b * Te * [sum_i n_i Z_i^2/(n_e A_i)]^{2/3}``,
  the standard NBI critical energy).

* Steady-state speed distribution of a monoenergetic source of strength
  ``S`` [1/(s m^3)] at birth speed ``v0`` (Stix 1972, Eq. 17; obtained from
  the continuity equation ``d/dv (N(v) dv/dt) = -S delta(v - v0)``)::

      N(v) = S tau_se v^2 / (v^3 + v_c^3)     for v_min < v < v0, else 0

  in [s/m^4] such that ``integral N(v) dv`` is a density.  Converted to a
  per-energy distribution with ``dE = m_b v dv``::

      N(E) = N(v) / (m_b v)                                              [1/(m^3 J)]

  ``v_min = sqrt(2 E_min / m_b)`` is the fixed thermalization boundary; below
  it markers are handed back to the bulk and contribute nothing.

* All energy-space integrals below are done in the exact closed form using
  the substitution ``u = v / v_c`` (so ``u = sqrt(E / E_c)``) and the standard
  primitive::

      F(u) = integral u / (1 + u^3) du
           = (1/6) ln( (u^2 - u + 1) / (u + 1)^2 )
             + (1/sqrt(3)) arctan( (2u - 1) / sqrt(3) )

  which is continuous for all ``u > -1`` (no branch issues on ``u >= 0``):

  - **Particle content** of a speed interval ``[v_a, v_b]`` (used for both
    the per-energy-bin f_E deposition and the total density)::

        integral_{v_a}^{v_b} N(v) dv
            = (S tau_se / 3) ln( (v_b^3 + v_c^3) / (v_a^3 + v_c^3) )

  - **Energy content**::

        integral E N(E) dE = (S tau_se m_b / 2)
                             integral v^4 / (v^3 + v_c^3) dv
            = (S tau_se m_b v_c^2 / 2) [ G(u) ]_{u_a}^{u_b},
        G(u) = u^2 / 2 - F(u)

  - **Ion heating fraction** at energy E (Stix 1972; the ion share of
    ``-dE/dt``)::

        frac_i(E) = 1 / (1 + (E / E_c)^{3/2})

    so the power a marker of weight ``w`` [1/s] delivers to thermal ions
    while slowing from ``E0`` to ``E_min`` is the exact integral::

        P_i = w * integral_{Emin}^{E0} dE / (1 + (E/E_c)^{3/2})
            = w * 2 E_c * [ F(u) ]_{u_min}^{u_0}                          [W]

    (substituting ``E = E_c u^2``), and by exact energy conservation::

        P_e = w * (E0 - E_min) - P_i

    The residual ``E_min`` per particle is returned to the bulk at
    thermalization and is NOT counted as deposited power, hence
    ``P_e + P_i == sum_markers w (E0 - E_min)`` holds to machine precision.

  - **Thermalization time** from ``E0`` down to ``E_min`` (integrating
    ``dt = -tau_se v^2 dv / (v^3 + v_c^3)`` along the slowing-down
    equation)::

        tau_th = (tau_se / 3) ln( (v0^3 + v_c^3) / (v_min^3 + v_c^3) )
               = (tau_se / 3) ln( (1 + (E0/E_c)^{3/2})
                                  / (1 + (E_min/E_c)^{3/2}) )             [s]

    i.e. the exact E_min-to-E0 form, not the ``E_min = 0`` textbook limit.

Coulomb-logarithm caveat
------------------------
Strictly, the Stix ``v_c^3`` expression carries the per-species ion-ion
Coulomb logarithms relative to the electron one:
``v_c^3 propto sum_i (n_i Z_i^2 lnL_i / m_i) / (n_e lnL_e)``.  Here the SAME
``lnL_e`` is used for every channel (the ratio ``lnL_i / lnL_e`` is set to 1),
as permitted by the module contract.  Quantitatively (physics-verifier
review, recomputing ASCOT's ``mccc_coefs_clog`` for a 100 keV D ion at the
scenario rho = 0.3 plasma): lnL_e = 17.71, lnL_D = 22.08, lnL_C = 20.76,
i.e. an effective ``lnL_i / lnL_e ~ 1.24``, so this module's ``E_c`` is
``~1.24^{2/3} ~ 15%`` LOW relative to ASCOT's effective critical energy.
The resulting shift in the heating split ``P_i/(P_e+P_i)`` for the 100 keV
component is ~0.03 absolute (0.818 here vs ~0.846 ASCOT-like) -- well inside
the 0.15 split tolerance of the analytic-vs-ASCOT comparison.

Model limitations (by design, per the contract)
-----------------------------------------------
- Pitch is ignored: no pitch-angle scattering, no orbit effects, no
  energy diffusion, no fast-ion transport -- markers slow down on the flux
  shell where they were born.
- Plasma profiles are frozen (steady state); the BBNBI weights [particles/s]
  make the accumulated distribution directly the steady-state one.

Everything is pure JAX: the public :func:`slowing_down` is ``jax.jit``-ed with
shapes taken from the input arrays, and works marker-parallel with batched
array operations (scatter-add into rho bins).
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .common import AMU_KG, E_CHARGE, Equilibrium, Plasma, shell_volumes
from .physics import profiles

# Physical constants (SI; CODATA 2018, matching common.py conventions).
EPS0 = 8.8541878128e-12   # vacuum permittivity [F/m]
M_E  = 9.1093837015e-31   # electron mass [kg]

_KEV_TO_J = 1.0e3 * E_CHARGE


class SlowingDownResult(NamedTuple):
    """Steady-state analytic slowing-down output on the shared (rho, E) grid.

    Note the trailing underscore in ``pi_`` (avoids shadowing ``math.pi``-like
    names); this field name is part of the phase-2 interface contract.
    """
    rho_edges: jnp.ndarray       # (nrho+1,) rho bin edges
    e_edges_keV: jnp.ndarray     # (nE+1,) energy bin edges [keV]
    f_E: jnp.ndarray             # (nrho, nE) steady-state fast-ion energy
                                 #   distribution [1/(m^3 keV)]
    density: jnp.ndarray         # (nrho,) fast-ion density [m^-3]
    energy_density: jnp.ndarray  # (nrho,) fast-ion energy density [J/m^3]
    pe: jnp.ndarray              # (nrho,) power to electrons [W/m^3]
    pi_: jnp.ndarray             # (nrho,) power to ions [W/m^3]
    tau_th: jnp.ndarray          # (nrho,) full-energy thermalization time [s]


def _coulomb_log_electron(ne, te_ev):
    """Electron Coulomb logarithm, NRL Plasma Formulary form.

    ``lnL_e = 24 - ln(sqrt(n_e[cm^-3]) / Te[eV])`` (thermal electron-ion
    collisions, ``Te > 10 Z^2 eV`` regime, which holds everywhere the fast
    ions live in this comparison).  Clamped to ``lnL_e >= 5`` so that tiny
    edge/vacuum densities cannot drive it to zero or negative.

    Parameters
    ----------
    ne : jnp.ndarray
        Electron density [m^-3] (should be pre-clamped positive).
    te_ev : jnp.ndarray
        Electron temperature [eV] (pre-clamped >= 1).

    Returns
    -------
    jnp.ndarray
        Coulomb logarithm, dimensionless, >= 5.
    """
    ne_cm3 = ne * 1.0e-6
    lnL = 24.0 - jnp.log(jnp.sqrt(ne_cm3) / te_ev)
    return jnp.maximum(lnL, 5.0)


def _local_sd_quantities(plasma: Plasma, rho, m_b, z_b):
    """(tau_se, v_c, E_c) of the beam species at given rho.

    Evaluates the profiles via :func:`physics.profiles` and applies the
    formulas from the module docstring:

    - ``tau_se = 3 (2 pi)^{3/2} eps0^2 m_b (k Te)^{3/2}
      / (sqrt(m_e) Z_b^2 e^4 n_e lnL_e)`` [s]  (Spitzer/Stix electron drag
      time; see module docstring for the sqrt(m_e) placement note),
    - ``v_c^3 = (3 sqrt(pi)/4) (m_e/n_e) sum_i (n_i Z_i^2 / m_i) v_te^3``
      with ``v_te = sqrt(2 k Te / m_e)`` [m/s],
    - ``E_c = m_b v_c^2 / 2`` [J].

    Inputs are clamped (``n_e >= 1e10 m^-3``, ``Te >= 1 eV``) for safety in
    vacuum regions; the species sum uses the ne-independent concentration
    ratio ``sum_i n_i Z_i^2 / (m_i n_e)`` so the clamp cannot skew it.

    Parameters
    ----------
    plasma : Plasma
        Plasma profile parameters (temperatures in eV per common.py).
    rho : jnp.ndarray
        Normalized radii, any shape ``(...)``.
    m_b : float
        Beam-ion mass [kg].
    z_b : float
        Beam-ion charge number.

    Returns
    -------
    tau_se, v_c, E_c : jnp.ndarray
        Shapes ``(...)``; units [s], [m/s], [J].
    """
    ne_raw, te_raw, ni = profiles(plasma, rho)
    ne = jnp.maximum(ne_raw, 1.0e10)
    te = jnp.maximum(te_raw, 1.0)

    lnL_e = _coulomb_log_electron(ne, te)
    kTe = te * E_CHARGE  # [J]

    tau_se = (3.0 * (2.0 * jnp.pi) ** 1.5 * EPS0**2 * m_b * kTe**1.5
              / (jnp.sqrt(M_E) * z_b**2 * E_CHARGE**4 * ne * lnL_e))

    v_te = jnp.sqrt(2.0 * kTe / M_E)
    m_i = jnp.asarray(plasma.anum, dtype=jnp.float64) * AMU_KG
    z_i = jnp.asarray(plasma.znum, dtype=jnp.float64)
    # sum_i n_i Z_i^2 / (m_i n_e): ni is proportional to ne_raw, so use the
    # matching raw density in the denominator (ratio is profile-shape free).
    ion_sum = jnp.sum(ni * z_i**2 / m_i, axis=-1) / jnp.maximum(ne_raw, 1.0e10)
    vc3 = 0.75 * jnp.sqrt(jnp.pi) * M_E * ion_sum * v_te**3
    v_c = vc3 ** (1.0 / 3.0)
    E_c = 0.5 * m_b * v_c**2
    return tau_se, v_c, E_c


def _F(u):
    """Primitive ``F(u) = integral u / (1 + u^3) du`` (module docstring).

    ``F(u) = (1/6) ln((u^2 - u + 1)/(u + 1)^2)
    + (1/sqrt(3)) arctan((2u - 1)/sqrt(3))``; continuous for ``u > -1`` so
    definite integrals over ``u >= 0`` are plain differences.
    """
    return (jnp.log((u * u - u + 1.0) / (u + 1.0) ** 2) / 6.0
            + jnp.arctan((2.0 * u - 1.0) / jnp.sqrt(3.0)) / jnp.sqrt(3.0))


def _G(u):
    """Primitive ``G(u) = integral u^4 / (1 + u^3) du = u^2/2 - F(u)``.

    (Since ``u^4/(1+u^3) = u - u/(1+u^3)``.) Used for the energy-density
    integral ``integral E N(E) dE``.
    """
    return 0.5 * u * u - _F(u)


@partial(jax.jit, static_argnames=())
def slowing_down(birth_rho, birth_energy_keV, birth_weight, eq: Equilibrium,
                 plasma: Plasma, rho_edges, e_edges_keV, *, mass_amu,
                 znum_beam=1, emin_keV=20.0) -> SlowingDownResult:
    """Analytic steady-state slowing-down of birth markers (Stix 1972).

    Each birth marker ``(rho_b, E0, w)`` is assigned to the rho bin of
    ``rho_edges`` containing ``rho_b``; the local ``(tau_se, v_c, E_c)`` are
    evaluated at that bin's CENTER via :func:`physics.profiles`, and the
    marker's exact closed-form contributions (module docstring formulas) are
    scatter-added into the bin:

    - per-energy-bin particle content -> ``f_E`` [1/(m^3 keV)] after division
      by shell volume and bin width,
    - total particle and energy content -> ``density`` [m^-3] and
      ``energy_density`` [J/m^3],
    - exact heating split -> ``pe``/``pi_`` [W/m^3] with
      ``pe + pi_ == w (E0 - Emin) / V`` identically (the residual ``Emin``
      per particle is returned to the bulk at thermalization, not counted).

    All integrals use the closed-form primitives :func:`_F` / :func:`_G`
    (no quadrature anywhere), so the only approximations are the physics
    model itself and the bin-center profile evaluation.

    Markers with ``E0 <= emin_keV`` or with ``rho_b`` outside
    ``[rho_edges[0], rho_edges[-1])`` contribute nothing (masked with
    ``jnp.where``-style zero weights; no NaNs are generated).

    .. warning::
       **Energy-grid coverage limitation.** ``f_E`` only holds the part of
       each marker's slowing-down content that lies inside
       ``[e_edges_keV[0], e_edges_keV[-1]]``; a marker born ABOVE the grid
       (``E0 > e_edges_keV[-1]``, or slowing through energies below
       ``e_edges_keV[0]`` when ``emin_keV < e_edges_keV[0]``) has that tail
       silently truncated from ``f_E``, while ``density``,
       ``energy_density``, ``pe`` and ``pi_`` always carry the FULL
       ``[emin_keV, E0]`` content — so ``sum(f_E * dE) < density`` for such
       markers (e.g. ~11% of the particles of a 120 keV birth on the shared
       20-110 keV grid).  The phase-2 comparison never hits this (max birth
       energy 100 keV < 110 keV, ``emin_keV`` equals the grid minimum), but
       choose grids with ``e_edges_keV[0] <= emin_keV`` and
       ``e_edges_keV[-1] >= max(E0)`` if the results are reused.

    ``tau_th`` is the per-bin thermalization time of a full-energy ion,
    with "full energy" taken as the maximum birth energy over all markers
    (clamped to ``>= emin_keV``), slowing from that energy down to
    ``emin_keV`` in each bin's local plasma.

    Parameters
    ----------
    birth_rho : jnp.ndarray
        (n,) marker birth normalized radii.
    birth_energy_keV : jnp.ndarray
        (n,) marker birth kinetic energies [keV] (mixed full/half/third
        components are handled per marker).
    birth_weight : jnp.ndarray
        (n,) marker weights [particles/s].
    eq : Equilibrium
        Circular equilibrium (for :func:`common.shell_volumes`).
    plasma : Plasma
        Plasma profiles (Te in eV).
    rho_edges : jnp.ndarray
        (nrho+1,) rho bin edges.
    e_edges_keV : jnp.ndarray
        (nE+1,) energy bin edges [keV].
    mass_amu : float
        Beam-ion mass [amu]; ``m_b = mass_amu * AMU_KG``.
    znum_beam : int, optional
        Beam-ion charge number ``Z_b`` (default 1).
    emin_keV : float, optional
        Thermalization boundary [keV] (default 20.0, the shared EMIN_KEV).

    Returns
    -------
    SlowingDownResult
        See the NamedTuple docstring; all arrays on the input grids.
    """
    birth_rho = jnp.asarray(birth_rho, dtype=jnp.float64)
    birth_energy_keV = jnp.asarray(birth_energy_keV, dtype=jnp.float64)
    birth_weight = jnp.asarray(birth_weight, dtype=jnp.float64)
    rho_edges = jnp.asarray(rho_edges, dtype=jnp.float64)
    e_edges_keV = jnp.asarray(e_edges_keV, dtype=jnp.float64)

    m_b = mass_amu * AMU_KG
    z_b = jnp.asarray(znum_beam, dtype=jnp.float64)
    nrho = rho_edges.shape[0] - 1
    n_e_bins = e_edges_keV.shape[0] - 1

    # --- per-bin plasma / slowing-down quantities at rho-bin centers -------
    rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    tau_se_b, v_c_b, E_c_b = _local_sd_quantities(plasma, rho_centers, m_b, z_b)
    volumes = shell_volumes(eq, rho_edges)  # (nrho,) [m^3]

    # --- map markers to rho bins and gather local quantities ---------------
    k = jnp.clip(jnp.searchsorted(rho_edges, birth_rho, side="right") - 1,
                 0, nrho - 1)
    in_grid = (birth_rho >= rho_edges[0]) & (birth_rho < rho_edges[-1])
    above_emin = birth_energy_keV > emin_keV
    w = jnp.where(in_grid & above_emin, birth_weight, 0.0)  # [1/s]

    tau_se = tau_se_b[k]
    v_c = v_c_b[k]
    E_c = E_c_b[k]

    emin_J = emin_keV * _KEV_TO_J
    # Clamp E0 to >= Emin so all interval integrals are zero (not negative /
    # NaN) for sub-threshold markers; those carry w = 0 anyway.
    E0_J = jnp.maximum(birth_energy_keV * _KEV_TO_J, emin_J)
    v0 = jnp.sqrt(2.0 * E0_J / m_b)
    v_min = jnp.sqrt(2.0 * emin_J / m_b)
    u0 = v0 / v_c
    u_min = v_min / v_c

    # --- f_E: exact particle content of each energy bin --------------------
    # Overlap of [bin_lo, bin_hi] with [Emin, E0], per marker x energy bin.
    e_lo_J = e_edges_keV[:-1] * _KEV_TO_J  # (nE,)
    e_hi_J = e_edges_keV[1:] * _KEV_TO_J
    Elo = jnp.clip(e_lo_J[None, :], emin_J, E0_J[:, None])  # (n, nE)
    Ehi = jnp.clip(e_hi_J[None, :], emin_J, E0_J[:, None])
    v_lo3 = (2.0 * Elo / m_b) ** 1.5
    v_hi3 = (2.0 * Ehi / m_b) ** 1.5
    vc3 = (v_c**3)[:, None]
    # (S tau/3) ln((v_hi^3+v_c^3)/(v_lo^3+v_c^3)); log1p form for accuracy.
    counts = ((w * tau_se / 3.0)[:, None]
              * jnp.log1p((v_hi3 - v_lo3) / (v_lo3 + vc3)))  # particles
    f_counts = jnp.zeros((nrho, n_e_bins)).at[k].add(counts)
    dE_keV = jnp.diff(e_edges_keV)
    f_E = f_counts / (volumes[:, None] * dE_keV[None, :])  # [1/(m^3 keV)]

    # --- density and energy density (closed forms over [Emin, E0]) ---------
    n_marker = (w * tau_se / 3.0) * jnp.log1p(
        (v0**3 - v_min**3) / (v_min**3 + v_c**3))  # particles
    density = jnp.zeros(nrho).at[k].add(n_marker) / volumes  # [m^-3]

    e_marker = (0.5 * w * tau_se * m_b * v_c**2) * (_G(u0) - _G(u_min))  # [J]
    energy_density = jnp.zeros(nrho).at[k].add(e_marker) / volumes  # [J/m^3]

    # --- heating split (exact) ---------------------------------------------
    pi_marker = 2.0 * E_c * w * (_F(u0) - _F(u_min))  # [W]
    ptot_marker = w * (E0_J - emin_J)  # [W]
    pi_sum = jnp.zeros(nrho).at[k].add(pi_marker)
    ptot_sum = jnp.zeros(nrho).at[k].add(ptot_marker)
    pe = (ptot_sum - pi_sum) / volumes  # [W/m^3], identity pe+pi_ = ptot/V
    pi_ = pi_sum / volumes

    # --- full-energy thermalization time per bin ---------------------------
    e_full_J = jnp.maximum(jnp.max(birth_energy_keV * _KEV_TO_J,
                                   initial=emin_J), emin_J)
    # Compute BOTH cubes with the identical expression shape (2E/m)^1.5 so
    # that the degenerate case e_full_J == emin_J cancels exactly (mixing
    # sqrt(.)^3 with (.)^1.5 rounds differently and can leave log1p with a
    # one-ulp-negative argument); the maximum(., 0) guard then makes
    # tau_th >= 0 a hard invariant.
    v_full3 = (2.0 * e_full_J / m_b) ** 1.5
    v_min3 = (2.0 * emin_J / m_b) ** 1.5
    tau_th = jnp.maximum(
        (tau_se_b / 3.0) * jnp.log1p((v_full3 - v_min3)
                                     / (v_min3 + v_c_b**3)), 0.0)  # [s]

    return SlowingDownResult(rho_edges=rho_edges, e_edges_keV=e_edges_keV,
                             f_E=f_E, density=density,
                             energy_density=energy_density, pe=pe, pi_=pi_,
                             tau_th=tau_th)


# ---------------------------------------------------------------------------
# Sanity block
# ---------------------------------------------------------------------------
def _sanity():
    """Run the contract's sanity checks and print the numbers."""
    import numpy as np

    eq = Equilibrium(R0=6.2, a=2.0)
    # Scenario-like plasma: D + 2% C, ne0=8e19 m^-3, te0=10 keV,
    # edge 1e17 m^-3 / 100 eV, alpha=1.5 (matches test_comparison scenario).
    plasma = Plasma(ne0=8.0e19, ne_edge=1.0e17, alpha_n=1.5,
                    te0=1.0e4, te_edge=100.0, alpha_t=1.5,
                    anum=jnp.array([2.0, 12.0]), znum=jnp.array([1.0, 6.0]),
                    conc=jnp.array([0.0, 0.02]))
    mass_amu = 2.0141  # deuterium
    m_b = mass_amu * AMU_KG
    emin_keV = 20.0

    rho_edges = jnp.linspace(0.0, 1.0, 26)
    e_edges_keV = jnp.linspace(20.0, 110.0, 46)

    # Uniform test source at rho = 0.05..0.15, 100 keV D, 1e19 particles/s.
    n_m = 400
    birth_rho = jnp.linspace(0.05, 0.15, n_m)
    birth_E = jnp.full(n_m, 100.0)
    birth_w = jnp.full(n_m, 1.0e19 / n_m)

    res = slowing_down(birth_rho, birth_E, birth_w, eq, plasma,
                       rho_edges, e_edges_keV, mass_amu=mass_amu,
                       znum_beam=1, emin_keV=emin_keV)
    assert isinstance(slowing_down, jax.stages.Wrapped), "not jitted"

    # -- check 1: N(E) >= 0, all outputs finite -----------------------------
    for name, arr in res._asdict().items():
        assert jnp.all(jnp.isfinite(arr)), f"{name} not finite"
    assert jnp.all(res.f_E >= 0.0), "f_E has negative entries"
    print("check 1: f_E >= 0 everywhere, all outputs finite         -- OK")

    # -- check 2: density from f_E integration vs closed form ---------------
    # Closed form per bin: n = S tau_se/3 ln((v0^3+vc^3)/(vmin^3+vc^3)),
    # S = (sum of w in bin)/V, computed here INDEPENDENTLY of the result.
    volumes = shell_volumes(eq, rho_edges)
    rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    tau_se_b, v_c_b, E_c_b = _local_sd_quantities(plasma, rho_centers, m_b, 1.0)
    kk = jnp.clip(jnp.searchsorted(rho_edges, birth_rho, side="right") - 1,
                  0, 25)
    S_bin = jnp.zeros(25).at[kk].add(birth_w) / volumes
    v0 = float(jnp.sqrt(2.0 * 100.0 * _KEV_TO_J / m_b))
    vmin = float(jnp.sqrt(2.0 * emin_keV * _KEV_TO_J / m_b))
    n_closed = (S_bin * tau_se_b / 3.0
                * jnp.log((v0**3 + v_c_b**3) / (vmin**3 + v_c_b**3)))
    n_from_fE = jnp.sum(res.f_E * jnp.diff(e_edges_keV)[None, :], axis=1)
    src = np.asarray(S_bin) > 0
    rel = np.max(np.abs(np.asarray(n_from_fE - n_closed))[src]
                 / np.asarray(n_closed)[src])
    assert rel < 1e-3, f"density from f_E vs closed form: rel {rel:.3e}"
    rel_res = np.max(np.abs(np.asarray(res.density - n_closed))[src]
                     / np.asarray(n_closed)[src])
    print(f"check 2: sum f_E dE vs closed-form density: max rel {rel:.3e} "
          f"(< 1e-3); result.density vs closed form: {rel_res:.3e}  -- OK")

    # -- check 3: exact power identity --------------------------------------
    p_dep = float(jnp.sum((res.pe + res.pi_) * volumes))
    p_expect = float(jnp.sum(birth_w * (birth_E - emin_keV) * _KEV_TO_J))
    rel_p = abs(p_dep - p_expect) / p_expect
    assert rel_p < 1e-12, f"P_e+P_i identity violated: rel {rel_p:.3e}"
    print(f"check 3: sum (pe+pi_) dV = sum w (E0-Emin) to rel {rel_p:.2e} "
          f"(< 1e-12)  -- OK  [{p_dep:.6e} W]")

    # cross-check P_i and energy density against fine trapezoid quadrature,
    # per rho bin (each bin has its own E_c)
    Eg = np.linspace(emin_keV, 100.0, 20001)
    w_bin = np.asarray(jnp.zeros(25).at[kk].add(birth_w))
    rel_q = 0.0
    for kb in np.nonzero(src)[0]:
        Ec_keV = float(E_c_b[kb]) / _KEV_TO_J
        frac_i = 1.0 / (1.0 + (Eg / Ec_keV) ** 1.5)
        pi_quad = w_bin[kb] * np.trapezoid(frac_i, Eg) * _KEV_TO_J \
            / float(volumes[kb])
        rel_q = max(rel_q, abs(float(res.pi_[kb]) - pi_quad) / pi_quad)
    assert rel_q < 1e-6, f"P_i closed form vs quadrature: rel {rel_q:.3e}"
    kb = int(kk[0])
    # energy density: integral E N(E) dE via quadrature of the exact N(E)
    tau0 = float(tau_se_b[kb])
    vg = np.sqrt(2.0 * Eg * _KEV_TO_J / m_b)
    NE = tau0 * vg / (m_b * (vg**3 + float(v_c_b[kb]) ** 3))  # per unit S
    ed_quad = w_bin[kb] * np.trapezoid(Eg * _KEV_TO_J * NE, Eg * _KEV_TO_J) \
        / float(volumes[kb])
    rel_e = abs(float(res.energy_density[kb]) - ed_quad) / ed_quad
    assert rel_e < 1e-6, f"energy density vs quadrature: rel {rel_e:.3e}"
    print(f"         P_i closed form vs 20k-pt quadrature: rel {rel_q:.2e}; "
          f"energy_density: rel {rel_e:.2e}  -- OK")

    # -- check 4: heating-split limits --------------------------------------
    # Cold plasma -> E_c << E0 -> electrons dominate; hot -> ions dominate.
    def split(te0):
        pl = plasma._replace(te0=te0)
        r = slowing_down(birth_rho, birth_E, birth_w, eq, pl, rho_edges,
                         e_edges_keV, mass_amu=mass_amu, emin_keV=emin_keV)
        p_i = float(jnp.sum(r.pi_ * volumes))
        p_e = float(jnp.sum(r.pe * volumes))
        return p_e, p_i

    pe_cold, pi_cold = split(300.0)     # E_c ~ 5 keV << 100 keV
    pe_hot, pi_hot = split(5.0e4)       # E_c ~ 900 keV >> 100 keV
    assert pe_cold > 5.0 * pi_cold, "cold-plasma limit not electron-dominated"
    assert pi_hot > 5.0 * pe_hot, "hot-plasma limit not ion-dominated"
    print(f"check 4: E0 >> Ec (Te=0.3 keV): Pe/(Pe+Pi) = "
          f"{pe_cold / (pe_cold + pi_cold):.3f} (electron-dominated);\n"
          f"         E0 << Ec (Te=50 keV):  Pi/(Pe+Pi) = "
          f"{pi_hot / (pi_hot + pe_hot):.3f} (ion-dominated)      -- OK")

    # -- scenario numbers at rho = 0.3 for a 100 keV D ion ------------------
    tau_se3, v_c3, E_c3 = _local_sd_quantities(
        plasma, jnp.asarray(0.3), m_b, 1.0)
    v0_3 = jnp.sqrt(2.0 * 100.0 * _KEV_TO_J / m_b)
    tau_th3 = (tau_se3 / 3.0) * jnp.log(
        (v0_3**3 + v_c3**3) / (vmin**3 + v_c3**3))
    ne3, te3, _ = profiles(plasma, jnp.asarray(0.3))
    lnL3 = _coulomb_log_electron(ne3, te3)
    # cross-check tau_se against the NRL practical formula
    tau_nrl = (6.27e8 * mass_amu * float(te3) ** 1.5
               / (1.0 * float(ne3) * 1e-6 * float(lnL3)))
    rel_tau = abs(float(tau_se3) - tau_nrl) / tau_nrl
    assert rel_tau < 0.02, f"tau_se vs NRL practical formula: rel {rel_tau:.3e}"
    print(f"check 5: tau_se vs NRL 6.27e8 A Te^1.5/(Z^2 ne lnL): "
          f"rel {rel_tau:.2e} (< 0.02)  -- OK")

    # -- check 6: degenerate case (no marker above Emin) -> tau_th >= 0 -----
    r_deg = slowing_down(birth_rho, jnp.full(n_m, 15.0), birth_w, eq, plasma,
                         rho_edges, e_edges_keV, mass_amu=mass_amu,
                         emin_keV=emin_keV)
    assert jnp.all(jnp.isfinite(r_deg.tau_th)), "degenerate tau_th not finite"
    assert jnp.all(r_deg.tau_th >= 0.0), \
        f"degenerate tau_th negative: min {float(jnp.min(r_deg.tau_th)):.3e}"
    assert jnp.all(r_deg.f_E == 0.0) and jnp.all(r_deg.density == 0.0) \
        and jnp.all(r_deg.pe == 0.0) and jnp.all(r_deg.pi_ == 0.0), \
        "sub-threshold markers contributed"
    print(f"check 6: all markers below Emin: tau_th >= 0 "
          f"(min {float(jnp.min(r_deg.tau_th)):.1e}), all outputs zero  "
          f"-- OK")
    print("\nscenario plasma (D + 2% C, ne0=8e19, te0=10 keV) at rho = 0.3:")
    print(f"  ne      = {float(ne3):.4e} m^-3, Te = {float(te3):.1f} eV, "
          f"lnL_e = {float(lnL3):.2f}")
    print(f"  tau_se  = {float(tau_se3):.4f} s")
    print(f"  E_c     = {float(E_c3) / _KEV_TO_J:.2f} keV "
          f"(v_c = {float(v_c3):.4e} m/s)")
    print(f"  tau_th  = {float(tau_th3):.4f} s  (100 keV -> 20 keV)")
    print(f"\nuniform-source run: density peak = "
          f"{float(jnp.max(res.density)):.4e} m^-3, "
          f"stored energy = "
          f"{float(jnp.sum(res.energy_density * volumes)):.4e} J,\n"
          f"  P_e = {float(jnp.sum(res.pe * volumes)):.4e} W, "
          f"P_i = {float(jnp.sum(res.pi_ * volumes)):.4e} W, "
          f"Pe fraction = "
          f"{float(jnp.sum(res.pe * volumes)) / p_dep:.3f}")
    print("\nall slowing_down sanity checks passed")


if __name__ == "__main__":
    _sanity()
