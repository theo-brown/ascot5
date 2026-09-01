"""Simplified beamlet-based NBI injector and marker source.

This module builds a rectangular-grid injector (mimicking
:meth:`a5py.ascot5io.nbi.NBI.generate`) and samples neutral markers from it
for the MC (BBNBI-style) deposition method. The pencil (RABBIT-style) method
uses the beamlet origins/directions directly, without divergence.

Geometry follows the conventions in :mod:`deposition_comparison.common`:
Cartesian (x, y, z) with the torus axis along z. The injector center sits at
cylindrical (r, phi, z); the central beam direction is horizontal and chosen
so that the central ray's tangency radius (closest approach of the ray line
to the z-axis) equals ``|tanrad|``, with the sign of ``tanrad`` selecting
co- vs counter-injection. All beamlets aim at a common focal point located
``focal_length`` along the central axis.
"""
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .common import Injector, Markers, E_CHARGE

_Z_HAT = jnp.array([0.0, 0.0, 1.0])


def _central_direction(r, phi, tanrad):
    """Horizontal unit direction of the injector centerline.

    The inward unit vector ``(-cos(phi), -sin(phi), 0)`` is rotated by
    ``beta = arcsin(tanrad / r)`` about the z-axis. For a horizontal ray from
    ``P = (x0, y0)`` with unit direction ``d`` the tangency radius is
    ``|x0*dy - y0*dx|``, which evaluates to ``|r sin(beta)| = |tanrad|``.

    Parameters
    ----------
    r : float
        Injector center R-coordinate [m].
    phi : float
        Injector center toroidal angle [rad].
    tanrad : float
        Signed tangency radius of the centerline [m], ``|tanrad| <= r``.

    Returns
    -------
    d : array_like (3,)
        Horizontal unit direction vector of the central ray.
    """
    beta = jnp.arcsin(tanrad / r)
    inward = jnp.array([-jnp.cos(phi), -jnp.sin(phi), 0.0])
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    return jnp.array([cb * inward[0] - sb * inward[1],
                      sb * inward[0] + cb * inward[1],
                      0.0])


def _tangency_radius(xyz, vdir):
    """Tangency radius of ray line(s): min distance of the line to the z-axis.

    Parameters
    ----------
    xyz : array_like (..., 3)
        Point(s) on the ray [m].
    vdir : array_like (..., 3)
        Direction vector(s) (need not be horizontal or normalized).

    Returns
    -------
    rtan : array_like (...)
        Closest approach of the infinite line to the z-axis [m].
    """
    num = jnp.abs(xyz[..., 0] * vdir[..., 1] - xyz[..., 1] * vdir[..., 0])
    den = jnp.sqrt(vdir[..., 0] ** 2 + vdir[..., 1] ** 2)
    return num / den


def make_injector(*, r, phi, z, tanrad, focal_length, grid_w, grid_h,
                  nx, ny, energy_keV, efrac, power, div_h, div_v,
                  anum, mass_amu):
    """Generate a simplified rectangular-grid injector.

    Mimics :meth:`a5py.ascot5io.nbi.NBI.generate` with a deterministic
    ``nx x ny`` rectangular beamlet grid instead of random beamlet positions.
    The grid lies in the plane perpendicular to the central direction
    (horizontal axis = ``z_hat x d`` normalized, vertical axis = ``z_hat``),
    centered on the injector center, and every beamlet aims at the common
    focal point ``center + focal_length * d``.

    Parameters
    ----------
    r : float
        Injector center point R-coordinate [m].
    phi : float
        Injector center point toroidal angle [rad].
    z : float
        Injector center point z-coordinate [m].
    tanrad : float
        Signed tangency radius of the injector centerline [m]. The central
        ray's closest approach to the z-axis equals ``|tanrad|``; the sign
        selects which of the two tangent directions is used
        (co-/counter-injection).
    focal_length : float
        Distance from the grid center to the common focal point [m].
    grid_w : float
        Horizontal extent of the beamlet grid [m].
    grid_h : float
        Vertical extent of the beamlet grid [m].
    nx : int
        Number of beamlet columns (horizontal).
    ny : int
        Number of beamlet rows (vertical).
    energy_keV : float
        Full injection energy [keV].
    efrac : array_like (3,)
        Particle fractions of the full, one-half and one-third energy
        components.
    power : float
        Total injected power [W].
    div_h : float
        Horizontal 1/e divergence half-angle [rad]; the sampled Gaussian
        deflection has std div_h/sqrt(2), matching BBNBI5.
    div_v : float
        Vertical 1/e divergence half-angle [rad].
    anum : int
        Mass number of the injected species.
    mass_amu : float
        Mass of the injected species [amu].

    Returns
    -------
    inj : :class:`~deposition_comparison.common.Injector`
        Injector with ``nx * ny`` beamlets.
    """
    center = jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])
    d = _central_direction(r, phi, tanrad)

    h_axis = jnp.cross(_Z_HAT, d)
    h_axis = h_axis / jnp.linalg.norm(h_axis)
    v_axis = _Z_HAT

    xs = jnp.linspace(-0.5 * grid_w, 0.5 * grid_w, nx)
    ys = jnp.linspace(-0.5 * grid_h, 0.5 * grid_h, ny)
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
    gx = gx.ravel()
    gy = gy.ravel()

    beamlet_xyz = (center[None, :]
                   + gx[:, None] * h_axis[None, :]
                   + gy[:, None] * v_axis[None, :])

    focus = center + focal_length * d
    beamlet_dir = focus[None, :] - beamlet_xyz
    beamlet_dir = beamlet_dir / jnp.linalg.norm(
        beamlet_dir, axis=1, keepdims=True)

    return Injector(
        beamlet_xyz=beamlet_xyz,
        beamlet_dir=beamlet_dir,
        energy_keV=float(energy_keV),
        efrac=jnp.asarray(efrac, dtype=jnp.float64),
        power=float(power),
        div_h=float(div_h),
        div_v=float(div_v),
        anum=int(anum),
        mass_amu=float(mass_amu),
    )


def component_rates(injector):
    """Particle injection rates of the three energy components.

    The rates are proportional to the particle fractions ``efrac`` and
    normalized so the total injected power is reproduced exactly:

    .. math::

        P = \\sum_{k=1}^{3} \\dot N_k \\, \\frac{E_{\\mathrm{full}}}{k}
            \\times 10^3 e

    Parameters
    ----------
    injector : :class:`~deposition_comparison.common.Injector`
        The injector.

    Returns
    -------
    rates : array_like (3,)
        Particles/s injected at full, half and third energy.
    """
    k = jnp.array([1.0, 2.0, 3.0])
    e_joule = injector.energy_keV * 1e3 * E_CHARGE / k
    norm = injector.power / jnp.sum(injector.efrac * e_joule)
    return norm * injector.efrac


@partial(jax.jit, static_argnames=("n",))
def sample_markers(key, injector, n):
    """Sample ``n`` neutral markers from the injector.

    Beamlets are chosen uniformly at random; each marker starts at its
    beamlet origin (the grid supplies the spatial extent) with the beamlet
    direction perturbed by Gaussian divergence angles in the beamlet-local
    frame. Energy components use deterministic banding: the markers are split
    into three contiguous blocks with sizes proportional to ``efrac``
    (round-to-integer, last block absorbing the remainder), and each marker
    in block ``k`` carries energy ``energy_keV / k`` and weight
    ``rate_k / n_k`` so every component's total particle rate — and hence the
    total power — is exact regardless of sampling noise.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key.
    injector : :class:`~deposition_comparison.common.Injector`
        The injector to sample from.
    n : int
        Number of markers (static for jit).

    Returns
    -------
    markers : :class:`~deposition_comparison.common.Markers`
        Sampled markers with positions [m], unit directions, energies [keV]
        and weights [particles/s].
    """
    key_b, key_h, key_v = jax.random.split(key, 3)
    nbeamlet = injector.beamlet_xyz.shape[0]

    # Beamlet choice and local frame
    idx = jax.random.randint(key_b, (n,), 0, nbeamlet)
    xyz = injector.beamlet_xyz[idx]
    bdir = injector.beamlet_dir[idx]

    h_axis = jnp.cross(_Z_HAT, bdir)
    h_axis = h_axis / jnp.linalg.norm(h_axis, axis=1, keepdims=True)
    v_axis = jnp.cross(bdir, h_axis)
    v_axis = v_axis / jnp.linalg.norm(v_axis, axis=1, keepdims=True)

    # Gaussian divergence (small-angle perturbation). The divergence is the
    # 1/e half-width of the angular intensity profile, so the Gaussian
    # standard deviation is div/sqrt(2) — same convention as BBNBI5
    # (src/nbi.c, nbi_inject).
    ah = injector.div_h / jnp.sqrt(2.0) * jax.random.normal(key_h, (n,))
    av = injector.div_v / jnp.sqrt(2.0) * jax.random.normal(key_v, (n,))
    vdir = bdir + ah[:, None] * h_axis + av[:, None] * v_axis
    vdir = vdir / jnp.linalg.norm(vdir, axis=1, keepdims=True)

    # Deterministic energy banding: three contiguous blocks with sizes
    # proportional to efrac; the last block absorbs the rounding remainder.
    frac = injector.efrac / jnp.sum(injector.efrac)
    n1 = jnp.round(n * frac[0]).astype(jnp.int64)
    n2 = jnp.round(n * frac[1]).astype(jnp.int64)
    n3 = n - n1 - n2
    nk = jnp.array([n1, n2, n3])

    i = jnp.arange(n)
    block = (i >= n1).astype(jnp.int64) + (i >= n1 + n2).astype(jnp.int64)

    kfac = jnp.array([1.0, 2.0, 3.0])
    energy_keV = injector.energy_keV / kfac[block]

    rates = component_rates(injector)
    weight_per_block = rates / jnp.maximum(nk, 1)
    weight = weight_per_block[block]

    return Markers(xyz=xyz, vdir=vdir, energy_keV=energy_keV, weight=weight)


if __name__ == "__main__":
    import numpy as np

    r, phi, tanrad = 9.0, 0.3, 5.5
    energy_keV = 100.0
    efrac = [0.55, 0.30, 0.15]
    power = 1e6

    inj = make_injector(
        r=r, phi=phi, z=0.1, tanrad=tanrad, focal_length=15.0,
        grid_w=0.4, grid_h=0.8, nx=8, ny=16,
        energy_keV=energy_keV, efrac=efrac, power=power,
        div_h=0.02, div_v=0.02, anum=2, mass_amu=2.0141)

    n = 100000
    key = jax.random.PRNGKey(42)
    mk = sample_markers(key, inj, n)

    # (a) power reconstruction
    p_rec = float(jnp.sum(mk.weight * mk.energy_keV * 1e3 * E_CHARGE))
    rel_err = abs(p_rec - power) / power
    print(f"(a) power reconstruction: {p_rec:.10e} W vs {power:.10e} W, "
          f"rel err = {rel_err:.3e} ({'OK' if rel_err < 1e-10 else 'FAIL'})")

    # (b) central ray tangency radius
    center = jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), 0.1])
    d = _central_direction(r, phi, tanrad)
    rtan_central = float(_tangency_radius(center, d))
    print(f"(b) central-ray tangency radius: {rtan_central:.12f} m "
          f"(requested {abs(tanrad):.12f} m, "
          f"diff = {abs(rtan_central - abs(tanrad)):.3e})")

    # (c) sampled per-marker tangency radii
    rtan = _tangency_radius(mk.xyz, mk.vdir)
    print(f"(c) sampled tangency radii: mean = {float(jnp.mean(rtan)):.4f} m, "
          f"std = {float(jnp.std(rtan)):.4f} m")

    # (d) energy-block fractions vs efrac
    kfac = np.array([1.0, 2.0, 3.0])
    counts = [int(np.sum(np.asarray(mk.energy_keV) == energy_keV / k))
              for k in kfac]
    fracs = [c / n for c in counts]
    print(f"(d) energy-block fractions: {fracs} vs efrac {efrac}")

    # jit verification: sample_markers is a jitted wrapper, repeat call is
    # cached, and the same key reproduces identical output.
    assert isinstance(sample_markers, jax.stages.Wrapped), \
        "sample_markers is not jitted"
    mk2 = sample_markers(key, inj, n)
    identical = all(bool(jnp.all(a == b)) for a, b in zip(mk, mk2))
    print(f"jit: sample_markers is jax.stages.Wrapped; repeat call with same "
          f"key identical: {identical}")
    assert identical
