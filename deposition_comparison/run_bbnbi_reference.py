"""Compare the JAX deposition methods against the actual compiled BBNBI5.

This script builds an ``ascot.h5`` whose inputs reproduce the JAX comparison
scenario exactly:

- ``B_2DS`` with ``psi = ((R - R0)^2 + z^2) / a^2``, ``psi0 = 0``, ``psi1 = 1``,
  so ASCOT's ``rho = sqrt(psinorm)`` equals the circular-equilibrium rho used
  by :mod:`deposition_comparison.common` (spline interpolation of a quadratic
  is exact away from the grid boundary).
- ``plasma_1D`` sampling the same parabolic profiles on a dense rho grid.
- The identical beamlet set exported from :func:`beam.make_injector`, with the
  same divergence convention (BBNBI samples Gaussian std = div/sqrt(2)).
- Dummy atomic data, so BBNBI's ``asigma_eval_bms`` falls back to the same
  Suzuki beam-stopping model that :mod:`deposition_comparison.physics` ports.

It then runs ``build/bbnbi5``, extracts the ionization positions, and compares
the resulting birth-rate profile and shinethrough fraction against
``deposit_mc`` and ``deposit_pencil``.

Run from the repository root (requires ``build/bbnbi5`` and a5py's deps)::

    python -m deposition_comparison.run_bbnbi_reference [--n 100000]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from . import beam, physics
from .common import E_CHARGE, shell_volumes
from .mc_deposition import deposit_mc
from .pencil_deposition import deposit_pencil
from .test_comparison import make_scenario, rel_l1

HERE = os.path.dirname(os.path.abspath(__file__))
BBNBI_BIN = os.path.join(HERE, "..", "build", "bbnbi5")
WORKDIR = os.path.join(HERE, "bbnbi_ref")


def write_inputs(eq, plasma, injector, fn):
    """Write an ascot.h5 with inputs equivalent to the JAX scenario."""
    from a5py import Ascot
    from a5py.ascot5io.nbi import Injector as A5Injector, NBI
    from a5py.ascot5io.wall import wall_3D
    from a5py.ascot5io.options import Opt

    a5 = Ascot(fn, create=True)

    # --- Magnetic field: psi reproduces the circular rho map exactly -------
    nr, nz = 240, 208
    r = np.linspace(3.0, 9.6, nr)
    z = np.linspace(-2.6, 2.6, nz)
    R, Z = np.meshgrid(r, z, indexing="ij")
    psi = ((R - eq.R0) ** 2 + Z**2) / eq.a**2
    bphi = 5.3 * eq.R0 / R
    a5.data.create_input(
        "B_2DS", rmin=r[0], rmax=r[-1], nr=nr, zmin=z[0], zmax=z[-1], nz=nz,
        axisr=eq.R0, axisz=0.0, psi=psi, psi0=0.0, psi1=1.0,
        br=np.zeros((nr, nz)), bphi=bphi, bz=np.zeros((nr, nz)),
        desc="CIRCULAR")

    # --- Plasma: same parabolic profiles on a dense rho grid ---------------
    nrho = 800
    rho = np.linspace(0, 2.5, nrho)
    ne, te, ni = physics.profiles(plasma, jnp.asarray(rho))
    pls = {
        "nrho": nrho, "nion": 2, "rho": rho, "vtor": np.zeros((nrho, 1)),
        "anum": np.array([2, 12]), "znum": np.array([1, 6]),
        "mass": np.array([2.0141, 12.011]), "charge": np.array([1, 6]),
        "edensity": np.asarray(ne), "etemperature": np.asarray(te),
        "idensity": np.asarray(ni), "itemperature": np.asarray(te)}
    a5.data.create_input("plasma_1D", **pls, desc="PARABOLIC")

    # --- Wall: rotated circular torus at minor radius 2.1 m ----------------
    pol = np.linspace(0, 2 * np.pi, 181)[:-1]
    wall = {"nelements": 180,
            "r": eq.R0 + 2.1 * np.cos(pol), "z": 2.1 * np.sin(pol)}
    wall = wall_3D.convert_wall_2D(180, **wall)
    a5.data.create_input("wall_3D", **wall, desc="TORUS")

    # --- Dummies (required to be present; not used by the physics) ---------
    a5.data.create_input("N0_1D")
    a5.data.create_input("asigma_loc")
    a5.data.create_input("E_TC", exyz=np.zeros(3))

    # --- Options: defaults, no distribution output -------------------------
    opt = Opt.get_default()
    a5.data.create_input("opt", **opt, desc="BBNBIREF")

    # --- Injector: identical beamlets, same divergence convention ----------
    xyz = np.asarray(injector.beamlet_xyz)
    d = np.asarray(injector.beamlet_dir)
    nb = xyz.shape[0]
    inj = A5Injector(
        ids=1, anum=int(injector.anum), znum=1, mass=float(injector.mass_amu),
        energy=float(injector.energy_keV) * 1e3,          # eV
        efrac=np.asarray(injector.efrac, dtype=float),
        power=float(injector.power),
        divh=float(injector.div_h), divv=float(injector.div_v),
        divhalofrac=0.0, divhaloh=1e-10, divhalov=1e-10,
        nbeamlet=nb,
        beamletx=xyz[:, 0], beamlety=xyz[:, 1], beamletz=xyz[:, 2],
        beamletdx=d[:, 0], beamletdy=d[:, 1], beamletdz=d[:, 2])
    NBI.write_hdf5(fn, 1, [inj], desc="JAXMATCH")

    # Mark everything active
    a5 = Ascot(fn)
    return a5


def bbnbi_profile(fn, eq, rho_edges, volumes):
    """Extract birth-rate/power profiles and shinethrough from a BBNBI run."""
    from a5py import Ascot

    a5 = Ascot(fn)
    run = a5.data.active

    r, zc, w, ekin = run.getstate("r", "z", "weight", "ekin", mode="prt",
                                  endcond="IONIZED")
    rho = np.sqrt((np.asarray(r) - eq.R0) ** 2 + np.asarray(zc) ** 2) / eq.a
    w = np.asarray(w)
    e_j = np.asarray(ekin.to("eV")) * E_CHARGE

    edges = np.asarray(rho_edges)
    birth, _ = np.histogram(rho, bins=edges, weights=w)
    power, _ = np.histogram(rho, bins=edges, weights=w * e_j)

    # Shinethrough: markers that ended on the wall (plus aborted, if any)
    wsh, esh = 0.0, 0.0
    try:
        rw = run.getstate("weight", "ekin", mode="prt", endcond="WALL")
        wsh = np.asarray(rw[0])
        esh = np.asarray(rw[1].to("eV")) * E_CHARGE
        shine = float(np.sum(wsh * esh))
    except Exception:
        shine = 0.0

    vols = np.asarray(volumes)
    return birth / vols, power / vols, shine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000,
                    help="markers for BBNBI and the JAX MC method")
    ap.add_argument("--div", type=float, default=0.02,
                    help="1/e divergence half-angle [rad]")
    args = ap.parse_args()

    if not os.path.exists(BBNBI_BIN):
        sys.exit("build/bbnbi5 not found - compile with 'make bbnbi5' first.")

    eq, plasma, tables, injector, rho_edges = make_scenario(div=args.div)
    volumes = shell_volumes(eq, rho_edges)

    # ------------------------------------------------------------------ BBNBI
    os.makedirs(WORKDIR, exist_ok=True)
    fn = os.path.join(WORKDIR, "ascot.h5")
    if os.path.exists(fn):
        os.remove(fn)
    write_inputs(eq, plasma, injector, fn)

    t0 = time.perf_counter()
    subprocess.run([os.path.abspath(BBNBI_BIN), "--in=ascot", f"--n={args.n}",
                    "--d=JAXREF"], cwd=WORKDIR, check=True,
                   stdout=subprocess.DEVNULL)
    t_bbnbi = time.perf_counter() - t0
    s_ref, p_ref, shine_ref = bbnbi_profile(fn, eq, rho_edges, volumes)

    # ---------------------------------------------------------------- JAX MC
    key = jax.random.PRNGKey(2024)
    t0 = time.perf_counter()
    mc = deposit_mc(key, injector, eq, plasma, tables, rho_edges,
                    n_markers=args.n, n_steps=4000)
    jax.block_until_ready(mc.birth_rate)
    t_mc = time.perf_counter() - t0

    # ------------------------------------------------------------ JAX pencil
    t0 = time.perf_counter()
    pen = deposit_pencil(injector, eq, plasma, tables, rho_edges, n_quad=4000)
    jax.block_until_ready(pen.birth_rate)
    t_pen = time.perf_counter() - t0

    # ------------------------------------------------------------- reporting
    s_mc = np.asarray(mc.birth_rate)
    s_pen = np.asarray(pen.birth_rate)
    shine_mc = float(mc.shinethrough_power) / injector.power
    shine_pen = float(pen.shinethrough_power) / injector.power
    shine_ref /= injector.power

    l1_mc = rel_l1(s_mc, s_ref)
    l1_pen = rel_l1(s_pen, s_ref)

    print(f"\nScenario: 100 keV D, 1 MW, div = {args.div*1e3:.0f} mrad (1/e), "
          f"{args.n} markers")
    print(f"{'method':<28}{'shine frac':>12}{'dep power [MW]':>16}"
          f"{'rel-L1 vs BBNBI5':>18}{'wall time [s]':>15}")
    dep_ref = float(np.sum(p_ref * np.asarray(volumes))) / 1e6
    print(f"{'BBNBI5 (reference)':<28}{shine_ref:>12.4f}{dep_ref:>16.4f}"
          f"{0.0:>18.4f}{t_bbnbi:>15.1f}")
    print(f"{'JAX MC (BBNBI-style)':<28}{shine_mc:>12.4f}"
          f"{float(mc.total_deposited_power)/1e6:>16.4f}{l1_mc:>18.4f}"
          f"{t_mc:>15.1f}")
    print(f"{'JAX pencil (RABBIT-style)':<28}{shine_pen:>12.4f}"
          f"{float(pen.total_deposited_power)/1e6:>16.4f}{l1_pen:>18.4f}"
          f"{t_pen:>15.1f}")

    summary = {
        "n_markers": args.n, "div": args.div,
        "shine": {"bbnbi5": shine_ref, "mc": shine_mc, "pencil": shine_pen},
        "rel_l1_vs_bbnbi5": {"mc": l1_mc, "pencil": l1_pen},
        "wall_time_s": {"bbnbi5": t_bbnbi, "mc": t_mc, "pencil": t_pen},
    }
    with open(os.path.join(HERE, "bbnbi_reference_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (np.asarray(rho_edges)[1:] + np.asarray(rho_edges)[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    axes[0].step(centers, s_ref, where="mid", color="k", lw=2,
                 label="BBNBI5 (compiled reference)")
    axes[0].step(centers, s_mc, where="mid", color="C0",
                 label="JAX MC (BBNBI-style)")
    axes[0].plot(centers, s_pen, color="C1", label="JAX pencil (RABBIT-style)")
    axes[0].set_ylabel(r"birth rate [1/(s m$^3$)]")
    axes[1].step(centers, p_ref, where="mid", color="k", lw=2)
    axes[1].step(centers, np.asarray(mc.power_density), where="mid",
                 color="C0")
    axes[1].plot(centers, np.asarray(pen.power_density), color="C1")
    axes[1].set_ylabel(r"deposited power density [W/m$^3$]")
    for ax in axes:
        ax.set_xlabel(r"$\rho$")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"NBI deposition vs compiled BBNBI5 - 100 keV D, 1 MW, "
        f"{args.div*1e3:.0f} mrad divergence\n"
        f"rel-L1 vs BBNBI5: MC {l1_mc:.3f}, pencil {l1_pen:.3f}; "
        f"shinethrough: BBNBI5 {shine_ref:.4f}, MC {shine_mc:.4f}, "
        f"pencil {shine_pen:.4f}", fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "comparison_bbnbi.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
