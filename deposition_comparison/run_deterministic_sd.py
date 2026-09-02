"""Fully deterministic NBI chain: pencil deposition -> orbit averaging ->
analytic slowing-down. No Monte-Carlo markers anywhere.

The RABBIT-style pencil deposition already resolves the source
deterministically: each (beamlet, energy component) pencil deposits
``w_j = (A_j - A_{j+1}) * rate`` in segment j, at a known position, with a
known flight direction. Those segments ARE the analytic deposition profile
from the BBNBI comparison, resolved in (R, z, direction, E) — exactly the
information the first-orbit averaging needs (pitch = direction . b_hat).

Pipeline:
1. attenuate each pencil on a fine quadrature grid (as in
   pencil_deposition; ~10 cm segments),
2. aggregate the segment deposits into ``n_cells`` coarse birth cells per
   pencil (deposit-weighted mean position; attenuation accuracy is kept,
   orbit launches are coarsened),
3. integrate one guiding-center orbit per birth cell
   (:func:`orbit_average.orbit_average_matrix`) and redistribute the cell's
   rate over the rho bins its orbit visits,
4. run the analytic Stix slowing-down on the redistributed source.

Everything is JAX (jit/vmap, autodiff field gradients) and deterministic:
same inputs, bitwise-same profiles — and differentiable end to end.

Run: python -m deposition_comparison.run_deterministic_sd
"""
import os
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from . import beam, physics
from .common import AMU_KG, E_CHARGE, rho_from_xyz, shell_volumes
from .orbit_average import bfield_cyl, orbit_average_matrix, orbit_averaged_sd
from .slowing_down import slowing_down
from .test_comparison import make_scenario, rel_l1

HERE = os.path.dirname(os.path.abspath(__file__))
EMIN_KEV = 20.0


def pencil_birth_cells(injector, eq, plasma, tables, *, n_quad=1000,
                       rho_edges=None, ray_length=25.0):
    """Deterministic birth cells from pencil-beam attenuation.

    Fine segments are aggregated per (pencil, ray leg, rho bin): rho along a
    pencil decreases to the tangency point and rises again, so each of the
    two monotone legs crosses each rho bin once. Members of a group share
    the rho bin by construction (no aliasing of the radial source profile)
    and have near-identical pitch (fixed flight direction, slowly varying
    b_hat), so one guiding-center orbit represents the group faithfully.
    Cell position = deposit-weighted mean segment position within the group.

    Returns
    -------
    r, z, vpar, mu : (npencil * 2 * nbin,) guiding-center birth coordinates
    e_keV, rate : same shape; energy and particles/s per cell (zero for
        bins a pencil never crosses).
    """
    nb = injector.beamlet_xyz.shape[0]
    rates_k = beam.component_rates(injector) / nb          # (3,)
    e_k = injector.energy_keV / jnp.array([1.0, 2.0, 3.0])  # (3,)
    nbin = rho_edges.shape[0] - 1
    ncell = 2 * nbin

    # Flatten (beamlet, component) -> pencil axis
    xyz0 = jnp.repeat(injector.beamlet_xyz, 3, axis=0)      # (np, 3)
    dirs = jnp.repeat(injector.beamlet_dir, 3, axis=0)
    e_p = jnp.tile(e_k, nb)                                 # (np,)
    rate_p = jnp.tile(rates_k, nb)

    ds = ray_length / n_quad

    def one(x0, d, e_keV, rate):
        l_mid = (jnp.arange(n_quad) + 0.5) * ds
        pos = x0[None, :] + l_mid[:, None] * d[None, :]
        rho = rho_from_xyz(eq, pos)
        ne, te, ni = physics.profiles(plasma, rho)
        k = ne * physics.sigma_stop(tables, e_keV / injector.anum, ne, te, ni)
        A = jnp.exp(-jnp.concatenate([jnp.zeros(1), jnp.cumsum(k * ds)]))
        w = (A[:-1] - A[1:]) * rate                          # (n_quad,)

        # Group index: leg (0 = inbound, rho falling; 1 = outbound) x rho bin
        leg = jnp.concatenate([jnp.zeros(1),
                               (jnp.diff(rho) >= 0).astype(jnp.float64)])
        rbin = jnp.searchsorted(rho_edges, rho, "right") - 1
        in_grid = (rbin >= 0) & (rbin < nbin)
        g = jnp.clip(leg.astype(jnp.int64) * nbin + jnp.clip(rbin, 0, nbin-1),
                     0, ncell - 1)
        wm = jnp.where(in_grid, w, 0.0)

        wsum = jnp.zeros(ncell).at[g].add(wm)                # (ncell,)
        psum = jnp.zeros((ncell, 3)).at[g].add(wm[:, None] * pos)
        cell_pos = psum / jnp.maximum(wsum, 1e-300)[:, None]
        # Zero-deposit groups: park on axis (weight 0, orbit harmless)
        cell_pos = jnp.where(wsum[:, None] > 1e-30, cell_pos,
                             jnp.array([eq.R0, 0.0, 0.0]))
        return cell_pos, wsum

    cell_pos, wsum = jax.jit(jax.vmap(one))(xyz0, dirs, e_p, rate_p)
    cell_pos = cell_pos.reshape(-1, 3)
    rate = wsum.reshape(-1)
    e_cells = jnp.repeat(e_p, ncell)
    dir_cells = jnp.repeat(dirs, ncell, axis=0)

    # GC coordinates: pitch from the (constant) pencil direction
    R = jnp.sqrt(cell_pos[:, 0] ** 2 + cell_pos[:, 1] ** 2)
    zc = cell_pos[:, 2]
    cosp, sinp = cell_pos[:, 0] / R, cell_pos[:, 1] / R
    B = jax.vmap(lambda rz: bfield_cyl(eq, rz))(jnp.stack([R, zc], axis=1))
    bx = B[:, 0] * cosp - B[:, 1] * sinp
    by = B[:, 0] * sinp + B[:, 1] * cosp
    bvec = jnp.stack([bx, by, B[:, 2]], axis=1)
    Bmag = jnp.linalg.norm(bvec, axis=1)
    xi = jnp.sum(dir_cells * bvec, axis=1) / Bmag

    m_b = injector.mass_amu * AMU_KG
    v = jnp.sqrt(2.0 * e_cells * 1e3 * E_CHARGE / m_b)
    vpar = xi * v
    mu = m_b * v**2 * (1.0 - xi**2) / (2.0 * Bmag)
    return R, zc, vpar, mu, e_cells, rate


def main():
    eq, plasma, tables, injector, _ = make_scenario(div=0.02)
    ref = np.load(os.path.join(HERE, "sd_reference.npz"))
    rho_edges = jnp.asarray(ref["rho_edges"])
    e_edges = jnp.asarray(ref["e_edges_keV"])
    vols = shell_volumes(eq, rho_edges)
    centers = np.asarray(0.5 * (rho_edges[1:] + rho_edges[:-1]))

    t0 = time.perf_counter()
    R, z, vpar, mu, e_keV, rate = pencil_birth_cells(
        injector, eq, plasma, tables, rho_edges=rho_edges)
    src_power = float(jnp.sum(rate * e_keV) * 1e3 * E_CHARGE)
    print(f"birth cells: {R.shape[0]} ({(rate > 0).sum()} with deposit), "
          f"source power {src_power/1e6:.4f} MW")

    # Local (birth-surface) analytic on the deterministic source
    rho_b = jnp.sqrt((R - eq.R0) ** 2 + z**2) / eq.a
    local = slowing_down(rho_b, e_keV, rate, eq, plasma, rho_edges, e_edges,
                         mass_amu=float(injector.mass_amu), emin_keV=EMIN_KEV)

    # Orbit-averaged (57.6k cells: shorter averaging window than the marker
    # driver, same step size; still many poloidal transits per orbit)
    frac, err = orbit_average_matrix(eq, rho_edges, R, z, vpar, mu,
                                     mass_amu=float(injector.mass_amu),
                                     t_int=5.0e-4, n_steps=2000)
    oa = orbit_averaged_sd(frac, e_keV, rate, eq, plasma, rho_edges, e_edges,
                           mass_amu=float(injector.mass_amu),
                           emin_keV=EMIN_KEV)
    jax.block_until_ready(oa.density)
    t_all = time.perf_counter() - t0
    print(f"orbit integrator energy cons. max rel: {float(err.max()):.1e}; "
          f"full deterministic chain: {t_all:.1f} s (ASCOT5: 131 s)")

    dens_ref = np.asarray(ref["density"])
    pe_ref, pi_ref = np.asarray(ref["pe"]), np.asarray(ref["pi"])
    print(f"\n{'model':<26}{'L1 n_fast':>11}{'L1 P_i':>9}{'L1 P_e':>9}"
          f"{'W_fast [kJ]':>13}")
    rows = [("local analytic (pencil)", local),
            ("orbit-avg (pencil)", oa)]
    for nm, sd in rows:
        print(f"{nm:<26}"
              f"{rel_l1(np.asarray(sd.density), dens_ref):>11.3f}"
              f"{rel_l1(np.asarray(sd.pi_), pi_ref):>9.3f}"
              f"{rel_l1(np.asarray(sd.pe), pe_ref):>9.3f}"
              f"{float(jnp.sum(sd.energy_density * vols))/1e3:>13.1f}")
    w_ref = float(np.sum(np.asarray(ref["energy_density"])
                         * np.asarray(vols))) / 1e3
    print(f"{'ASCOT5 reference':<26}{0.0:>11.3f}{0.0:>9.3f}{0.0:>9.3f}"
          f"{w_ref:>13.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    for a, (lab, attr, refv) in zip(ax, [
            ("fast-ion density [m$^{-3}$]", "density", dens_ref),
            ("$P_i$ [W/m$^3$]", "pi_", pi_ref),
            ("$P_e$ [W/m$^3$]", "pe", pe_ref)]):
        a.step(centers, refv, where="mid", color="k", lw=2, label="ASCOT5")
        a.step(centers, np.asarray(getattr(local, attr)), where="mid",
               color="C0", label="local analytic (pencil)")
        a.step(centers, np.asarray(getattr(oa, attr)), where="mid",
               color="C3", label="orbit-avg (pencil)")
        a.set_xlabel(r"$\rho$")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False, fontsize=9)
    l1d = rel_l1(np.asarray(oa.density), dens_ref)
    fig.suptitle(
        "Fully deterministic chain (no Monte-Carlo anywhere): pencil "
        "deposition -> orbit averaging -> Stix slowing-down\n"
        f"n$_f$ rel-L1 vs ASCOT5: {rel_l1(np.asarray(local.density), dens_ref):.2f}"
        f" (local) -> {l1d:.2f} (orbit-averaged)")
    fig.tight_layout()
    out = os.path.join(HERE, "comparison_deterministic.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure saved to {out}")


if __name__ == "__main__":
    main()
