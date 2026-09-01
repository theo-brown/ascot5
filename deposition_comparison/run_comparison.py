"""Driver script comparing the MC (BBNBI-style) and pencil (RABBIT-style)
NBI deposition methods on the shared ITER-ish scenario.

Run from the repository root:

    python -m deposition_comparison.run_comparison

Produces ``deposition_comparison/comparison.png`` (birth-rate and
power-density profiles for the pencil method and for MC with div=0 and
div=25 mrad) and prints a summary table to stdout.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from .mc_deposition import deposit_mc
from .pencil_deposition import deposit_pencil
from .test_comparison import make_scenario, rel_l1

N_MARKERS = 100_000
N_STEPS = 2000
DIV = 0.025

OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "comparison.png")


def summarize(name, res, injector, pencil_res=None):
    """Return a dict of summary metrics for one DepositionResult."""
    centers = 0.5 * (res.rho_edges[:-1] + res.rho_edges[1:])
    ipk = int(jnp.argmax(res.birth_rate))
    row = {
        "name": name,
        "shine_frac": float(res.shinethrough_power) / injector.power,
        "dep_power": float(res.total_deposited_power),
        "peak_birth": float(res.birth_rate[ipk]),
        "peak_rho": float(centers[ipk]),
        "l1_vs_pencil": (rel_l1(res.birth_rate, pencil_res.birth_rate)
                         if pencil_res is not None else 0.0),
    }
    return row


def main():
    eq, plasma, tables, inj0, rho_edges = make_scenario(div=0.0)
    _, _, _, inj_div, _ = make_scenario(div=DIV)
    key = jax.random.PRNGKey(1234)

    print(f"Scenario: R0={eq.R0} m, a={eq.a} m, ne0={plasma.ne0:.2e} m^-3, "
          f"E={inj0.energy_keV:.0f} keV D beam, P={inj0.power/1e6:.1f} MW")
    print(f"MC: {N_MARKERS} markers x {N_STEPS} steps; "
          f"divergence {DIV*1e3:.0f} mrad for the finite-div run\n")

    t0 = time.time()
    pencil = deposit_pencil(inj0, eq, plasma, tables, rho_edges)
    pencil.birth_rate.block_until_ready()
    print(f"pencil            done in {time.time()-t0:6.1f} s")

    t0 = time.time()
    mc0 = deposit_mc(key, inj0, eq, plasma, tables, rho_edges,
                     n_markers=N_MARKERS, n_steps=N_STEPS)
    mc0.birth_rate.block_until_ready()
    print(f"MC div=0          done in {time.time()-t0:6.1f} s")

    t0 = time.time()
    mcd = deposit_mc(key, inj_div, eq, plasma, tables, rho_edges,
                     n_markers=N_MARKERS, n_steps=N_STEPS)
    mcd.birth_rate.block_until_ready()
    print(f"MC div=25 mrad    done in {time.time()-t0:6.1f} s\n")

    rows = [
        summarize("pencil (RABBIT-style)", pencil, inj0),
        summarize("MC div=0 (BBNBI-style)", mc0, inj0, pencil),
        summarize(f"MC div={DIV*1e3:.0f} mrad (BBNBI-style)", mcd, inj_div,
                  pencil),
    ]

    # ----- summary table ---------------------------------------------------
    hdr = (f"{'method':<28} {'shine frac':>10} {'dep power [MW]':>15} "
           f"{'peak birth [1/(s m^3)]':>23} {'@ rho':>6} "
           f"{'L1 vs pencil':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<28} {r['shine_frac']:>10.4f} "
              f"{r['dep_power']/1e6:>15.6f} {r['peak_birth']:>23.4e} "
              f"{r['peak_rho']:>6.2f} {r['l1_vs_pencil']:>13.4f}")

    # ----- figure ----------------------------------------------------------
    centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    styles = [
        (pencil, "pencil (RABBIT-style)", dict(color="k", lw=2)),
        (mc0, "MC div=0 (BBNBI-style)", dict(color="C0", lw=1.2)),
        (mcd, f"MC div={DIV*1e3:.0f} mrad (BBNBI-style)",
         dict(color="C3", lw=1.2)),
    ]
    for res, label, kw in styles:
        if label.startswith("pencil"):
            ax1.plot(centers, res.birth_rate, label=label, **kw)
            ax2.plot(centers, res.power_density, label=label, **kw)
        else:
            ax1.step(centers, res.birth_rate, where="mid", label=label, **kw)
            ax2.step(centers, res.power_density, where="mid", label=label,
                     **kw)

    ax1.set_xlabel(r"$\rho$")
    ax1.set_ylabel(r"birth rate density [1/(s m$^3$)]")
    ax1.set_title("Ion birth rate")
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel(r"power density [W/m$^3$]")
    ax2.set_title("Deposited power density")
    ax1.legend(fontsize=9)

    info = "\n".join(
        f"{r['name']}: shine {100*r['shine_frac']:.2f}%, "
        f"dep {r['dep_power']/1e6:.3f} MW" for r in rows)
    ax2.text(0.03, 0.97, info, transform=ax2.transAxes, va="top",
             fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.suptitle("NBI deposition: MC (BBNBI-style) vs pencil (RABBIT-style)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
