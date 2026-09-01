"""Three-way slowing-down comparison: local analytic vs orbit-averaged
analytic vs the full ASCOT5 reference.

Applies the first-orbit averaging of :mod:`orbit_average` to the exact 3086
birth markers the ASCOT5 reference simulated, then re-runs the analytic Stix
slowing-down on the redistributed source and quantifies how much of the
finite-orbit-width gap it closes.

Run: python -m deposition_comparison.run_orbit_averaged
"""
import os
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .common import shell_volumes
from .orbit_average import (load_ascot_markers, orbit_average_matrix,
                            orbit_averaged_sd)
from .slowing_down import slowing_down
from .test_comparison import make_scenario, rel_l1

HERE = os.path.dirname(os.path.abspath(__file__))
H5 = os.path.join(HERE, "bbnbi_ref", "ascot.h5")
SDRUN = "run_1940713020"


def main():
    eq, plasma, _, _, _ = make_scenario()
    ref = np.load(os.path.join(HERE, "sd_reference.npz"))
    rho_edges = jnp.asarray(ref["rho_edges"])
    e_edges = jnp.asarray(ref["e_edges_keV"])
    vols = shell_volumes(eq, rho_edges)
    centers = np.asarray(0.5 * (rho_edges[1:] + rho_edges[:-1]))

    r, z, vpar, mu, e_keV, w, mamu, bdev = load_ascot_markers(H5, SDRUN, eq)
    assert bdev < 1e-8, f"field mismatch {bdev}"
    rho_b = jnp.sqrt((r - eq.R0) ** 2 + z**2) / eq.a

    # Local (birth-surface) analytic
    t0 = time.perf_counter()
    local = slowing_down(rho_b, e_keV, w, eq, plasma, rho_edges, e_edges,
                         mass_amu=mamu)
    jax.block_until_ready(local.density)
    t_local = time.perf_counter() - t0

    # Orbit-averaged analytic
    t0 = time.perf_counter()
    frac, err = orbit_average_matrix(eq, rho_edges, r, z, vpar, mu,
                                     mass_amu=mamu)
    oa = orbit_averaged_sd(frac, e_keV, w, eq, plasma, rho_edges, e_edges,
                           mass_amu=mamu)
    jax.block_until_ready(oa.density)
    t_oa = time.perf_counter() - t0

    dens_ref = np.asarray(ref["density"])
    pe_ref, pi_ref = np.asarray(ref["pe"]), np.asarray(ref["pi"])

    rows = []
    for name, sd in [("local analytic", local), ("orbit-avg analytic", oa)]:
        rows.append((
            name,
            rel_l1(np.asarray(sd.density), dens_ref),
            rel_l1(np.asarray(sd.pi_), pi_ref),
            rel_l1(np.asarray(sd.pe), pe_ref),
            float(jnp.sum(sd.energy_density * vols)) / 1e3,
        ))
    w_ref = float(np.sum(np.asarray(ref["energy_density"]) * np.asarray(vols))) / 1e3

    print(f"orbit integrator: energy cons. max rel {float(err.max()):.1e}; "
          f"cost local {t_local:.2f} s, orbit-avg {t_oa:.2f} s "
          f"(ASCOT5: 131 s)")
    print(f"\n{'model':<22}{'L1 n_fast':>11}{'L1 P_i':>9}{'L1 P_e':>9}"
          f"{'W_fast [kJ]':>13}")
    for nm, l1n, l1i, l1e, wf in rows:
        print(f"{nm:<22}{l1n:>11.3f}{l1i:>9.3f}{l1e:>9.3f}{wf:>13.1f}")
    print(f"{'ASCOT5 reference':<22}{0.0:>11.3f}{0.0:>9.3f}{0.0:>9.3f}"
          f"{w_ref:>13.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    panels = [
        ("fast-ion density [m$^{-3}$]", "density", dens_ref),
        ("$P_i$ [W/m$^3$]", "pi_", pi_ref),
        ("$P_e$ [W/m$^3$]", "pe", pe_ref),
    ]
    for a, (lab, attr, refv) in zip(ax, panels):
        a.step(centers, refv, where="mid", color="k", lw=2, label="ASCOT5")
        a.step(centers, np.asarray(getattr(local, attr)), where="mid",
               color="C0", label="local analytic")
        a.step(centers, np.asarray(getattr(oa, attr)), where="mid",
               color="C3", label="orbit-avg analytic")
        a.set_xlabel(r"$\rho$")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "First-orbit averaging closes the finite-orbit-width gap - "
        f"n$_f$ rel-L1: {rows[0][1]:.2f} (local) -> {rows[1][1]:.2f} "
        "(orbit-averaged) vs ASCOT5, same 3086 markers")
    fig.tight_layout()
    out = os.path.join(HERE, "comparison_orbitavg.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure saved to {out}")


if __name__ == "__main__":
    main()
