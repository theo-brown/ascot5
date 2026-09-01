"""Shared data structures, constants, and configuration for the BBNBI-vs-RABBIT
deposition-method comparison.

All modules in this package operate on the structures defined here. Everything
is JAX-friendly: NamedTuples of jnp arrays / floats, with static shapes so
functions can be jitted and vmapped.

Unit conventions
----------------
- Lengths in m, densities in m^-3.
- Energies in keV (beam) unless suffixed otherwise; temperatures in eV.
- Powers in W. Marker/beamlet weights in particles/s.
- Cross sections in m^2. Attenuation per length = ne * sigma [1/m].

Geometry conventions
--------------------
- Cartesian (x, y, z) with the torus axis along z. R = sqrt(x^2 + y^2).
- The equilibrium has circular, concentric flux surfaces:
      rho(R, z) = sqrt((R - R0)^2 + z^2) / a
  The plasma occupies rho < 1; outside, profiles drop to their (tiny) edge
  values so attenuation there is negligible.
- Flux-surface shell volume between rho edges [r1, r2]:
      V = 2 pi^2 R0 a^2 (r2^2 - r1^2)
"""
from typing import NamedTuple
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
E_CHARGE = 1.602176634e-19   # C (also J per eV)
AMU_KG   = 1.66053906660e-27 # kg


class Equilibrium(NamedTuple):
    """Circular concentric-flux-surface tokamak equilibrium."""
    R0: float  # major radius of magnetic axis [m]
    a: float   # minor radius [m]


class Plasma(NamedTuple):
    """Plasma profile parameters and (static-shape) species arrays.

    Profiles are parabolic-like:
        f(rho) = (f0 - fedge) * (1 - rho**2)**alpha + fedge   for rho < 1
        f(rho) = fedge                                        for rho >= 1

    Species: index 0 is the main hydrogenic ion; further entries are
    impurities. Ion densities follow from quasineutrality with fixed impurity
    concentration fractions c_imp = n_imp / n_e:
        n_main = n_e * (1 - sum_i Z_i * c_i),  n_imp_i = n_e * c_i
    """
    ne0: float      # core electron density [m^-3]
    ne_edge: float  # edge/vacuum electron density [m^-3] (tiny, e.g. 1.0)
    alpha_n: float  # density profile exponent
    te0: float      # core electron temperature [eV]
    te_edge: float  # edge electron temperature [eV]
    alpha_t: float  # temperature profile exponent
    anum: jnp.ndarray  # (nion,) species mass numbers, main ion first
    znum: jnp.ndarray  # (nion,) species charge numbers, main ion first
    conc: jnp.ndarray  # (nion,) n_i/n_e concentration; entry 0 is ignored and
                       # recomputed from quasineutrality


class Injector(NamedTuple):
    """A beamlet-based neutral beam injector (mirrors a5py's Injector)."""
    beamlet_xyz: jnp.ndarray  # (nbeamlet, 3) beamlet origins [m]
    beamlet_dir: jnp.ndarray  # (nbeamlet, 3) unit direction vectors
    energy_keV: float         # full injection energy [keV]
    efrac: jnp.ndarray        # (3,) particle fractions of full/half/third E
    power: float              # total injected power [W]
    div_h: float              # horizontal 1/e divergence half-angle [rad]
    div_v: float              # vertical 1/e divergence half-angle [rad]
    anum: int                 # beam species mass number (static)
    mass_amu: float           # beam species mass [amu]


class Markers(NamedTuple):
    """Sampled neutral markers ready for MC tracing."""
    xyz: jnp.ndarray        # (n, 3) start positions [m]
    vdir: jnp.ndarray       # (n, 3) unit velocity directions
    energy_keV: jnp.ndarray # (n,) kinetic energy [keV]
    weight: jnp.ndarray     # (n,) particles/s each marker represents


class SuzukiTables(NamedTuple):
    """Species-resolved Suzuki fit coefficients, prepared OUTSIDE jit.

    Built by physics.prepare_suzuki(anum, znum) which does all the
    species-dependent table lookups with concrete (non-traced) values, so the
    evaluation function itself is pure traced arithmetic.
    """
    # Hydrogenic species: one A-coefficient row per species, low & high E
    A_low: jnp.ndarray    # (n_h, 10)
    A_high: jnp.ndarray   # (n_h, 10)
    h_mask: jnp.ndarray   # (nion,) 1.0 where species is hydrogenic else 0.0
    # Impurity species: one B-coefficient row per species, low & high E
    B_low: jnp.ndarray    # (nion, 12), zero rows for hydrogenic entries
    B_high: jnp.ndarray   # (nion, 12), zero rows for hydrogenic entries
    znum: jnp.ndarray     # (nion,) float charge numbers
    anum: jnp.ndarray     # (nion,) float mass numbers


class DepositionResult(NamedTuple):
    """Common output of both deposition methods, on a shared rho grid."""
    rho_edges: jnp.ndarray       # (nrho+1,) bin edges in rho
    birth_rate: jnp.ndarray      # (nrho,) ion birth rate density [1/(s m^3)]
    power_density: jnp.ndarray   # (nrho,) deposited (birth) power [W/m^3]
    shinethrough_power: float    # power not absorbed in plasma [W]
    total_deposited_power: float # integral of power_density * dV [W]


# ---------------------------------------------------------------------------
# Shared helpers (implemented here so every module agrees exactly)
# ---------------------------------------------------------------------------
def shell_volumes(eq: Equilibrium, rho_edges: jnp.ndarray) -> jnp.ndarray:
    """Analytic flux-shell volumes [(nrho,) m^3] for the circular equilibrium."""
    return 2.0 * jnp.pi**2 * eq.R0 * eq.a**2 * jnp.diff(rho_edges**2)


def rho_from_xyz(eq: Equilibrium, xyz: jnp.ndarray) -> jnp.ndarray:
    """rho at Cartesian point(s); xyz has shape (..., 3)."""
    r = jnp.sqrt(xyz[..., 0]**2 + xyz[..., 1]**2)
    return jnp.sqrt((r - eq.R0)**2 + xyz[..., 2]**2) / eq.a
