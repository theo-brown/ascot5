"""RABBIT-style analytic pencil-beam NBI deposition.

Deterministic pencil rays — one per (beamlet, energy component) — are traced
from the beamlet origins along the beamlet directions with NO divergence
(deliberately: that is the methodological difference from the MC/BBNBI-style
method in :mod:`deposition_comparison.mc_deposition`). Along each ray the
optical depth is accumulated with piecewise-constant attenuation per segment,
the surviving neutral fraction is known analytically at every segment
boundary, and the fraction absorbed inside each segment is deposited into the
rho bin containing the segment midpoint. Shinethrough is the surviving
fraction at the end of the ray.

Scope note
----------
This module implements only the *deposition* stage of RABBIT (M. Weiland
et al 2018 Nucl. Fusion 58 082032). The full RABBIT code goes further:
it orbit-averages the birth profile over the fast-ion drift orbits and
evolves the fast-ion distribution with a time-dependent analytic
Fokker-Planck (slowing-down) solution. None of that is done here — the
output stops at the ionization (birth) profile and shinethrough, which is
what the comparison against the MC method needs.
"""
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .common import E_CHARGE, DepositionResult, rho_from_xyz, shell_volumes
from .physics import profiles, sigma_stop
from .beam import component_rates


def _deposit_pencil_impl(injector, eq, plasma, tables, rho_edges,
                         n_quad, ray_length):
    """Core pencil-beam deposition (traced arithmetic; see deposit_pencil).

    Returns
    -------
    result : DepositionResult
        The deposition result.
    dropped_power : jnp.ndarray (scalar)
        Power absorbed in segments whose midpoint rho falls outside
        ``[rho_edges[0], rho_edges[-1])`` — dropped from the profile and
        counted neither as deposited nor as shinethrough.
    """
    nrho = rho_edges.shape[0] - 1
    nbeamlet = injector.beamlet_xyz.shape[0]
    ds = ray_length / n_quad
    s_mid = (jnp.arange(n_quad) + 0.5) * ds

    # One pencil per (beamlet, energy component k in {1, 2, 3}), flattened
    # into a single vmapped axis of length nbeamlet * 3.
    kfac = jnp.array([1.0, 2.0, 3.0])
    e_comp_kev = injector.energy_keV / kfac                       # (3,)
    rate_comp = component_rates(injector) / nbeamlet              # (3,)

    origins = jnp.broadcast_to(
        injector.beamlet_xyz[:, None, :], (nbeamlet, 3, 3)).reshape(-1, 3)
    dirs = jnp.broadcast_to(
        injector.beamlet_dir[:, None, :], (nbeamlet, 3, 3)).reshape(-1, 3)
    e_kev = jnp.broadcast_to(
        e_comp_kev[None, :], (nbeamlet, 3)).reshape(-1)           # (np,)
    rate = jnp.broadcast_to(
        rate_comp[None, :], (nbeamlet, 3)).reshape(-1)            # (np,)
    e_joule = e_kev * 1e3 * E_CHARGE                              # (np,)

    def one_pencil(origin, direction, e_k_kev):
        """Absorbed-fraction histogram, shinethrough and dropped fraction."""
        xyz = origin[None, :] + s_mid[:, None] * direction[None, :]
        rho = rho_from_xyz(eq, xyz)                               # (n_quad,)
        ne, te, ni = profiles(plasma, rho)
        sigma = sigma_stop(tables, e_k_kev / injector.anum, ne, te, ni)
        kappa = ne * sigma                                        # [1/m]

        # Survival at segment boundaries: A_0 = 1,
        # A_j = exp(-sum_{i<j} kappa_i * ds) (piecewise-constant kappa).
        tau = jnp.concatenate(
            [jnp.zeros((1,)), jnp.cumsum(kappa * ds)])            # (n_quad+1,)
        A = jnp.exp(-tau)
        w = A[:-1] - A[1:]     # fraction absorbed in each segment (telescopes)
        shine = A[-1]

        # Bin by segment-midpoint rho; midpoints outside
        # [rho_edges[0], rho_edges[-1]) are dropped (not shinethrough).
        idx = jnp.searchsorted(rho_edges, rho, side="right") - 1
        valid = (idx >= 0) & (idx < nrho)
        hist = jnp.zeros(nrho).at[jnp.clip(idx, 0, nrho - 1)].add(
            jnp.where(valid, w, 0.0))
        dropped = jnp.sum(jnp.where(valid, 0.0, w))
        return hist, shine, dropped

    hists, shine_frac, dropped_frac = jax.vmap(one_pencil)(
        origins, dirs, e_kev)                # (np, nrho), (np,), (np,)

    birth_counts = jnp.sum(rate[:, None] * hists, axis=0)         # [1/s]
    power_counts = jnp.sum(
        (rate * e_joule)[:, None] * hists, axis=0)                # [W]
    shinethrough_power = jnp.sum(rate * e_joule * shine_frac)
    dropped_power = jnp.sum(rate * e_joule * dropped_frac)

    vols = shell_volumes(eq, rho_edges)
    result = DepositionResult(
        rho_edges=rho_edges,
        birth_rate=birth_counts / vols,
        power_density=power_counts / vols,
        shinethrough_power=shinethrough_power,
        total_deposited_power=jnp.sum(power_counts),
    )
    return result, dropped_power


@partial(jax.jit, static_argnames=("n_quad",))
def deposit_pencil(injector, eq, plasma, tables, rho_edges, *,
                   n_quad=4000, ray_length=25.0):
    """RABBIT-style pencil-beam deposition on a shared rho grid.

    One deterministic pencil ray per (beamlet, energy component): origin at
    the beamlet position, direction along the beamlet direction with no
    divergence. Each ray is split into ``n_quad`` equal segments of length
    ``ray_length / n_quad``; the attenuation coefficient
    ``kappa = ne * sigma_stop`` is evaluated at segment midpoints and treated
    as piecewise constant, giving the survival fraction
    ``A_j = exp(-sum_{i<j} kappa_i ds)`` at segment boundaries — the same
    optical-depth quadrature the MC method uses, which matters for agreement
    between the two. The fraction ``A_j - A_{j+1}`` absorbed in segment ``j``
    is deposited into the rho bin containing the segment midpoint; segments
    with midpoint rho outside ``[rho_edges[0], rho_edges[-1])`` are dropped
    (counted neither as deposited nor as shinethrough). The surviving
    fraction at the ray end is shinethrough. All pencils are handled with a
    single ``vmap`` over the flattened (beamlet, component) axis.

    With every segment inside the rho grid the absorbed fractions telescope
    exactly, so deposited + shinethrough equals injected power to floating-
    point precision.

    Parameters
    ----------
    injector : :class:`~deposition_comparison.common.Injector`
        The neutral beam injector (divergence fields are ignored).
    eq : :class:`~deposition_comparison.common.Equilibrium`
        Circular concentric-flux-surface equilibrium.
    plasma : :class:`~deposition_comparison.common.Plasma`
        Plasma profile parameters and species arrays.
    tables : :class:`~deposition_comparison.common.SuzukiTables`
        Prepared Suzuki coefficients from
        :func:`deposition_comparison.physics.prepare_suzuki`.
    rho_edges : jnp.ndarray
        Shape ``(nrho+1,)`` rho bin edges of the deposition grid.
    n_quad : int, optional
        Number of ray segments (static for jit). Default 4000.
    ray_length : float, optional
        Length of each pencil ray [m]. Default 25.0.

    Returns
    -------
    :class:`~deposition_comparison.common.DepositionResult`
        Birth-rate density, power density, shinethrough power and total
        deposited power on the rho grid.
    """
    result, _ = _deposit_pencil_impl(
        injector, eq, plasma, tables, rho_edges, n_quad, ray_length)
    return result


# Same computation, but also returning the out-of-grid dropped power (used
# by the smoke test below to close the power balance exactly).
_deposit_pencil_full = jax.jit(
    _deposit_pencil_impl, static_argnames=("n_quad",))


if __name__ == "__main__":
    import time

    from .common import Equilibrium, Plasma
    from .physics import prepare_suzuki
    from .beam import make_injector

    eq = Equilibrium(R0=6.2, a=2.0)
    plasma = Plasma(
        ne0=8e19, ne_edge=1.0, alpha_n=1.5,
        te0=1e4, te_edge=100.0, alpha_t=1.5,
        anum=jnp.array([2.0, 12.0]),
        znum=jnp.array([1.0, 6.0]),
        conc=jnp.array([1.0, 0.02]))
    tables = prepare_suzuki([2, 12], [1, 6])
    injector = make_injector(
        r=9.0, phi=0.0, z=0.0, tanrad=5.5, focal_length=15.0,
        grid_w=0.4, grid_h=0.8, nx=8, ny=16,
        energy_keV=100.0, efrac=[0.55, 0.3, 0.15], power=1e6,
        div_h=0.0, div_v=0.0, anum=2, mass_amu=2.0141)
    rho_edges = jnp.linspace(0.0, 1.0, 51)

    # First call compiles; second call measures the cached wall time.
    result, dropped = _deposit_pencil_full(
        injector, eq, plasma, tables, rho_edges, 4000, 25.0)
    jax.block_until_ready(result)
    t0 = time.perf_counter()
    result, dropped = _deposit_pencil_full(
        injector, eq, plasma, tables, rho_edges, 4000, 25.0)
    jax.block_until_ready(result)
    t_cached = time.perf_counter() - t0

    p_inj = injector.power
    p_dep = float(result.total_deposited_power)
    p_shine = float(result.shinethrough_power)
    p_drop = float(dropped)
    residual = abs(p_dep + p_shine + p_drop - p_inj) / p_inj

    vols = shell_volumes(eq, rho_edges)
    rho_mid = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    ipk = int(jnp.argmax(result.birth_rate))

    print(f"injected power           : {p_inj:.6e} W")
    print(f"shinethrough fraction    : {p_shine / p_inj:.6f} "
          f"({p_shine:.6e} W)")
    print(f"total deposited power    : {p_dep:.6e} W")
    print(f"out-of-grid dropped power: {p_drop:.6e} W")
    print(f"power-balance residual   : "
          f"|dep + shine + dropped - inj| / inj = {residual:.3e}")
    assert residual < 1e-12, "power balance violated"
    print(f"wall time (cached call)  : {t_cached * 1e3:.2f} ms")
    assert isinstance(deposit_pencil, jax.stages.Wrapped), \
        "deposit_pencil is not jitted"
    print("entry point jitted       : deposit_pencil is jax.stages.Wrapped")
    print(f"birth_rate peak          : "
          f"{float(result.birth_rate[ipk]):.6e} 1/(s m^3) "
          f"at rho = {float(rho_mid[ipk]):.3f} "
          f"(bin [{float(rho_edges[ipk]):.2f}, "
          f"{float(rho_edges[ipk + 1]):.2f}))")

    # deposit_pencil (public entry point) agrees with the full variant.
    res2 = deposit_pencil(injector, eq, plasma, tables, rho_edges,
                          n_quad=4000, ray_length=25.0)
    same = all(bool(jnp.all(a == b)) for a, b in zip(result, res2))
    print(f"deposit_pencil matches _deposit_pencil_full: {same}")
    assert same
