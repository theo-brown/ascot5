"""BBNBI-style Monte-Carlo NBI deposition.

Neutral markers are sampled from the beamlet grid with Gaussian divergence
(:func:`deposition_comparison.beam.sample_markers`), traced ballistically
through the plasma, and their ionization point is sampled stochastically from
the optical depth along the ray -- mirroring what ``src/bbnbi5.c`` does when
it traces neutrals until they ionize or are lost to the wall.

What this method captures that the pencil (RABBIT-style) method does not
------------------------------------------------------------------------
- **Beam divergence**: every marker's direction is individually perturbed by
  the Gaussian divergence half-angles, so the beam footprint broadens with
  distance; pencil rays follow the exact beamlet centerlines.
- **Finite grid sampling**: markers sample the beamlet grid randomly, so the
  discrete source geometry enters through actual sampled rays instead of one
  deterministic ray per beamlet.
- **Stochastic ionization**: each marker ionizes at a single random point
  drawn from ``exp(-tau)`` (optical-depth sampling), reproducing birth-point
  statistics (and their Monte-Carlo noise) instead of depositing a smooth
  attenuation profile along the ray.

The price is statistical noise ~ 1/sqrt(N) in the binned profiles.

Algorithm (per marker, fully vectorized)
----------------------------------------
1. March ``n_steps`` equal steps of length ``ds = ray_length / n_steps``
   along the straight ray; evaluate the attenuation coefficient
   ``k_j = ne_j * sigma_stop(...)`` at the segment midpoints.
2. Cumulative optical depth ``tau_j = cumsum(k_j * ds)`` (midpoint rule:
   ``tau_j`` approximates the optical depth at the *end* of segment ``j``).
3. Draw ``u ~ U(0, 1)``; the target optical depth is ``tau* = -log(1 - u)``.
   Ionization happens in the first segment with ``tau_j >= tau*`` (found by
   ``searchsorted``); if ``tau_end < tau*`` the marker is shinethrough.
4. The ionization position is interpolated linearly in tau between the
   segment boundaries, then converted to ``rho`` for binning.

Markers are processed in fixed-size chunks with :func:`jax.lax.map` (a vmapped
single-marker trace inside each chunk) so peak memory stays at
O(chunk * n_steps) instead of O(n_markers * n_steps).
"""
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .common import (DepositionResult, E_CHARGE, rho_from_xyz, shell_volumes)
from . import beam, physics

# Markers are traced in chunks of this size (padded with zero-weight markers)
# so peak memory is O(_CHUNK_SIZE * n_steps), independent of n_markers.
_CHUNK_SIZE = 2048


def _trace_one(xyz0, vdir, energy_keV, tau_star, eq, plasma, tables,
               anum, ds, n_steps):
    """Trace a single neutral marker and sample its ionization point.

    Operates on one marker only (arrays of shape ``(n_steps,)`` internally)
    so that :func:`jax.vmap` supplies the marker batching.

    Parameters
    ----------
    xyz0 : jnp.ndarray
        Marker start position [m], shape ``(3,)``.
    vdir : jnp.ndarray
        Unit direction of flight, shape ``(3,)``.
    energy_keV : float
        Marker kinetic energy [keV].
    tau_star : float
        Target optical depth ``-log(1 - u)`` with ``u ~ U(0, 1)``.
    eq, plasma, tables
        Shared scenario structures (:class:`Equilibrium`, :class:`Plasma`,
        :class:`SuzukiTables`).
    anum : int
        Beam species mass number (``injector.anum``); the divisor used to
        convert the marker energy to keV/amu for the Suzuki fit.
    ds : float
        Step length [m].
    n_steps : int
        Number of steps (static).

    Returns
    -------
    rho_ion : float
        Normalized radius of the ionization point (meaningless when the
        marker did not ionize).
    ionized : bool
        True if the marker ionized within ``ray_length``, False if it is
        shinethrough.
    """
    # Segment midpoints l_j = (j + 0.5) ds and plasma along the ray.
    l_mid = (jnp.arange(n_steps) + 0.5) * ds                # (n_steps,)
    pos = xyz0[None, :] + l_mid[:, None] * vdir[None, :]    # (n_steps, 3)
    rho = rho_from_xyz(eq, pos)                             # (n_steps,)
    ne, te, ni = physics.profiles(plasma, rho)

    # Attenuation coefficient per length k = ne * sigma [1/m].
    e_per_amu = energy_keV / anum
    sigma = physics.sigma_stop(tables, e_per_amu, ne, te, ni)
    k = ne * sigma                                          # (n_steps,)

    # Cumulative optical depth at segment ENDS (midpoint rule):
    # tau[j] ~ optical depth at l = (j + 1) ds.
    tau = jnp.cumsum(k * ds)

    # First segment j with tau[j] >= tau*; n_steps means "never" ->
    # shinethrough.
    idx = jnp.searchsorted(tau, tau_star, side="left")
    ionized = idx < n_steps
    j = jnp.minimum(idx, n_steps - 1)

    # Segment j spans l in [j ds, (j+1) ds], with optical depth running from
    # tau[j-1] (0 for j = 0) to tau[j]; interpolate linearly in tau.
    tau_lo = jnp.where(j > 0, tau[jnp.maximum(j - 1, 0)], 0.0)
    tau_hi = tau[j]
    frac = (tau_star - tau_lo) / jnp.maximum(tau_hi - tau_lo, 1e-300)
    frac = jnp.clip(frac, 0.0, 1.0)
    l_ion = (j + frac) * ds

    rho_ion = rho_from_xyz(eq, xyz0 + l_ion * vdir)
    return rho_ion, ionized


@partial(jax.jit, static_argnames=("n_markers", "n_steps"))
def deposit_mc(key, injector, eq, plasma, tables, rho_edges, *,
               n_markers, n_steps=4000, ray_length=25.0):
    """BBNBI-style Monte-Carlo NBI deposition.

    Samples ``n_markers`` neutral markers from the injector (with beamlet
    grid sampling and Gaussian divergence), traces each along a straight ray
    of length ``ray_length`` in ``n_steps`` equal steps, samples the
    ionization point from the cumulative optical depth, and histograms the
    ion births on the ``rho_edges`` grid.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key; the same key yields an identical result.
    injector : Injector
        Injector built by :func:`deposition_comparison.beam.make_injector`.
    eq : Equilibrium
        Circular equilibrium.
    plasma : Plasma
        Plasma profile parameters and species arrays.
    tables : SuzukiTables
        Suzuki coefficients from :func:`physics.prepare_suzuki` (must match
        ``plasma.anum`` / ``plasma.znum``).
    rho_edges : jnp.ndarray
        Radial bin edges, shape ``(nrho + 1,)``.
    n_markers : int
        Number of Monte-Carlo markers (static).
    n_steps : int, optional
        Steps along each ray (static). Default 4000.
    ray_length : float, optional
        Traced ray length [m]. Default 25.0.

    Returns
    -------
    DepositionResult
        Binned birth rate density [1/(s m^3)] and power density [W/m^3],
        shinethrough power [W] (markers that never ionized within
        ``ray_length``) and total deposited power [W] (histogram integral).
        Markers that ionize at ``rho`` outside ``[rho_edges[0],
        rho_edges[-1])`` are dropped from the histogram but are NOT counted
        as shinethrough, so ``injected = deposited + shinethrough`` holds
        only up to the (small) out-of-grid birth power.
    """
    key_m, key_u = jax.random.split(key)
    markers = beam.sample_markers(key_m, injector, n_markers)

    # Target optical depth per marker: u in [0, 1) so 1 - u in (0, 1] and
    # the log is finite.
    u = jax.random.uniform(key_u, (n_markers,))
    tau_star = -jnp.log1p(-u)

    ds = ray_length / n_steps

    def trace_one(xyz0, vdir, e_keV, ts):
        return _trace_one(xyz0, vdir, e_keV, ts, eq, plasma, tables,
                          injector.anum, ds, n_steps)

    # Chunked marker march: pad to a multiple of the chunk size with
    # zero-weight dummies, lax.map over chunks, vmap within a chunk.
    chunk = min(_CHUNK_SIZE, n_markers)
    n_pad = (-n_markers) % chunk
    n_tot = n_markers + n_pad

    def pad(a, fill=0.0):
        widths = ((0, n_pad),) + ((0, 0),) * (a.ndim - 1)
        return jnp.pad(a, widths, constant_values=fill)

    xyz_c = pad(markers.xyz).reshape(-1, chunk, 3)
    dir_c = pad(markers.vdir).reshape(-1, chunk, 3)
    # Dummy energies stay positive so the Suzuki logs remain finite.
    e_c = pad(markers.energy_keV,
              fill=injector.energy_keV).reshape(-1, chunk)
    ts_c = pad(tau_star).reshape(-1, chunk)

    rho_ion_c, ionized_c = jax.lax.map(
        lambda args: jax.vmap(trace_one)(*args), (xyz_c, dir_c, e_c, ts_c))
    rho_ion = rho_ion_c.reshape(n_tot)[:n_markers]
    ionized = ionized_c.reshape(n_tot)[:n_markers]

    # Histogram births on the rho grid: bins [edge_b, edge_{b+1}), markers
    # outside [rho_edges[0], rho_edges[-1]) dropped (not shinethrough).
    nrho = rho_edges.shape[0] - 1
    b = jnp.searchsorted(rho_edges, rho_ion, side="right") - 1
    in_grid = (rho_ion >= rho_edges[0]) & (rho_ion < rho_edges[-1])
    sel = ionized & in_grid
    bc = jnp.clip(b, 0, nrho - 1)

    e_J = markers.energy_keV * 1e3 * E_CHARGE
    w = jnp.where(sel, markers.weight, 0.0)
    birth_hist = jnp.zeros(nrho).at[bc].add(w)
    power_hist = jnp.zeros(nrho).at[bc].add(w * e_J)

    vols = shell_volumes(eq, rho_edges)
    birth_rate = birth_hist / vols
    power_density = power_hist / vols

    shinethrough_power = jnp.sum(
        jnp.where(ionized, 0.0, markers.weight * e_J))
    total_deposited_power = jnp.sum(power_hist)

    return DepositionResult(
        rho_edges=rho_edges,
        birth_rate=birth_rate,
        power_density=power_density,
        shinethrough_power=shinethrough_power,
        total_deposited_power=total_deposited_power,
    )


if __name__ == "__main__":
    import resource
    import time

    from .common import Equilibrium, Plasma

    eq = Equilibrium(R0=6.2, a=2.0)
    plasma = Plasma(
        ne0=8e19, ne_edge=1.0, alpha_n=1.5,
        te0=1e4, te_edge=100.0, alpha_t=1.5,
        anum=jnp.array([2.0, 12.0]),
        znum=jnp.array([1.0, 6.0]),
        conc=jnp.array([1.0, 0.02]),
    )
    tables = physics.prepare_suzuki([2, 12], [1, 6])
    inj = beam.make_injector(
        r=9.0, phi=0.0, z=0.0, tanrad=5.5, focal_length=15.0,
        grid_w=0.4, grid_h=0.8, nx=8, ny=16,
        energy_keV=100.0, efrac=[0.55, 0.3, 0.15], power=1e6,
        div_h=0.02, div_v=0.02, anum=2, mass_amu=2.0141)
    rho_edges = jnp.linspace(0.0, 1.0, 51)

    assert isinstance(deposit_mc, jax.stages.Wrapped), \
        "deposit_mc is not a jitted wrapper"
    print(f"deposit_mc is a jitted wrapper "
          f"({type(deposit_mc).__module__}.{type(deposit_mc).__name__})")

    key = jax.random.PRNGKey(1)
    n = 50000

    t0 = time.perf_counter()
    res = deposit_mc(key, inj, eq, plasma, tables, rho_edges, n_markers=n)
    jax.block_until_ready(res)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    res2 = deposit_mc(key, inj, eq, plasma, tables, rho_edges, n_markers=n)
    jax.block_until_ready(res2)
    t_second = time.perf_counter() - t0

    same = all(bool(jnp.all(a == b)) for a, b in zip(res, res2))
    assert same, "same key did not reproduce identical result"

    p_inj = inj.power
    shine = float(res.shinethrough_power)
    dep = float(res.total_deposited_power)
    resid = (p_inj - dep - shine) / p_inj
    print(f"\n50k markers x 4000 steps:")
    print(f"  shinethrough fraction        : {shine / p_inj:.5f}")
    print(f"  total deposited power        : {dep:.6e} W")
    print(f"  power balance residual       : {resid:.3e} "
          f"(= out-of-grid births)")
    print(f"  out-of-grid deposited power  : {p_inj - dep - shine:.4e} W")
    print(f"  first call (compile + run)   : {t_first:.2f} s")
    print(f"  second call (cached)         : {t_second:.3f} s")
    print(f"  same key => identical result : {same}")

    # A different key should give a statistically consistent answer.
    res3 = deposit_mc(jax.random.PRNGKey(2), inj, eq, plasma, tables,
                      rho_edges, n_markers=n)
    print(f"  key=2 shinethrough fraction  : "
          f"{float(res3.shinethrough_power) / p_inj:.5f} "
          f"(statistical agreement check)")

    # Large run: memory / wall-time check.
    n_big = 200000
    t0 = time.perf_counter()
    res_big = deposit_mc(key, inj, eq, plasma, tables, rho_edges,
                         n_markers=n_big, n_steps=4000)
    jax.block_until_ready(res_big)
    t_big = time.perf_counter() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    shine_big = float(res_big.shinethrough_power) / p_inj
    print(f"\n200k markers x 4000 steps:")
    print(f"  shinethrough fraction        : {shine_big:.5f}")
    print(f"  total deposited power        : "
          f"{float(res_big.total_deposited_power):.6e} W")
    print(f"  wall time (compile + run)    : {t_big:.2f} s")
    print(f"  peak RSS                     : {peak_gb:.2f} GB")
