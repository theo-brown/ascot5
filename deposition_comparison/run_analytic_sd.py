"""Analytic NBI actuator profiles: source, heating split, and current.

Drives the full analytic chain — MC deposition (marker-level, with birth
pitch from the analytic equilibrium field) followed by the Stix slowing-down
model — and plots the actuator-relevant outputs:

- particle source rate S(rho) [1/(s m^3)]
- power density to electrons and ions P_e(rho), P_i(rho) [W/m^3]
- fast-ion parallel current density j(rho) [A/m^2]
- steady-state fast-ion density n_f(rho) [m^-3]

Current-density model (not part of the reviewed slowing_down module):
each ionized marker keeps its birth pitch xi = v_hat . b_hat (no pitch-angle
scattering, consistent with the slowing-down model), so its steady-state
current contribution is

    j dV = Z_b e w xi * integral_0^{tau_th} v(t) dt
         = Z_b e w xi tau_se [ (v0 - vmin) - v_c^3 * I3(vmin, v0) ],
    I3 = int dv / (v^3 + v_c^3)   (closed form, self-checked below).

The "net" curve applies the flat electron back-current shielding factor
(1 - Z_b/Z_eff); trapped-electron corrections (neglected here) raise the
net current toward the unshielded curve.

The magnetic field for the pitch is the same analytic field the ASCOT
reference uses: psi = ((R-R0)^2 + z^2)/a^2 (B_R = -(1/R) dpsi/dz,
B_z = (1/R) dpsi/dR) with B_phi = 5.3 * R0 / R.

Run: python -m deposition_comparison.run_analytic_sd
"""
import os

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from . import beam, physics
from .common import AMU_KG, E_CHARGE, rho_from_xyz, shell_volumes
from .slowing_down import slowing_down, _local_sd_quantities
from .test_comparison import make_scenario

HERE = os.path.dirname(os.path.abspath(__file__))

RHO_EDGES = jnp.linspace(0.0, 1.0, 26)
E_EDGES_KEV = jnp.linspace(20.0, 110.0, 46)
EMIN_KEV = 20.0
B0, ZEFF_SHIELD_ZB = 5.3, 1.0


# ---------------------------------------------------------------------------
# Marker-level deposition: like mc_deposition._trace_one but returning the
# ionization point itself (position needed for the birth pitch).
# ---------------------------------------------------------------------------
def _trace_l(xyz0, vdir, energy_keV, tau_star, eq, plasma, tables, anum,
             ds, n_steps):
    l_mid = (jnp.arange(n_steps) + 0.5) * ds
    pos = xyz0[None, :] + l_mid[:, None] * vdir[None, :]
    rho = rho_from_xyz(eq, pos)
    ne, te, ni = physics.profiles(plasma, rho)
    k = ne * physics.sigma_stop(tables, energy_keV / anum, ne, te, ni)
    tau = jnp.cumsum(k * ds)
    idx = jnp.searchsorted(tau, tau_star, side="left")
    ionized = idx < n_steps
    j = jnp.minimum(idx, n_steps - 1)
    tau_lo = jnp.where(j > 0, tau[jnp.maximum(j - 1, 0)], 0.0)
    frac = jnp.clip((tau_star - tau_lo)
                    / jnp.maximum(tau[j] - tau_lo, 1e-300), 0.0, 1.0)
    return (j + frac) * ds, ionized


def bhat_xyz(eq, xyz):
    """Unit magnetic-field vector (Cartesian) of the analytic equilibrium."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    R = jnp.sqrt(x**2 + y**2)
    cosp, sinp = x / R, y / R
    # psi = ((R-R0)^2 + z^2) / a^2  [Vs/rad]
    br = -(1.0 / R) * 2.0 * z / eq.a**2
    bz = (1.0 / R) * 2.0 * (R - eq.R0) / eq.a**2
    bphi = B0 * eq.R0 / R
    bx = br * cosp - bphi * sinp
    by = br * sinp + bphi * cosp
    b = jnp.stack([bx, by, bz], axis=-1)
    return b / jnp.linalg.norm(b, axis=-1, keepdims=True)


def marker_births(key, injector, eq, plasma, tables, *, n_markers,
                  n_steps=2000, ray_length=25.0, chunk=4096):
    """Sample and trace markers; return marker-level birth data."""
    km, ku = jax.random.split(key)
    mrk = beam.sample_markers(km, injector, n_markers)
    tau_star = -jnp.log1p(-jax.random.uniform(ku, (n_markers,)))
    ds = ray_length / n_steps

    def one(args):
        xyz0, vdir, e_keV, ts = args
        return _trace_l(xyz0, vdir, e_keV, ts, eq, plasma, tables,
                        injector.anum, ds, n_steps)

    npad = (-n_markers) % chunk
    pad = lambda a: jnp.concatenate([a, jnp.zeros((npad,) + a.shape[1:])])
    args = (pad(mrk.xyz), pad(mrk.vdir),
            pad(mrk.energy_keV) + 1.0,  # keep Suzuki logs finite on padding
            pad(tau_star))
    args = tuple(a.reshape((-1, chunk) + a.shape[1:]) for a in args)
    l_ion, ionized = lax.map(jax.vmap(one, in_axes=(0,)), args)
    l_ion = l_ion.reshape(-1)[:n_markers]
    ionized = ionized.reshape(-1)[:n_markers]

    xyz_ion = mrk.xyz + l_ion[:, None] * mrk.vdir
    rho_b = rho_from_xyz(eq, xyz_ion)
    pitch = jnp.sum(mrk.vdir * bhat_xyz(eq, xyz_ion), axis=-1)
    w = jnp.where(ionized, mrk.weight, 0.0)
    return rho_b, mrk.energy_keV, w, pitch


# ---------------------------------------------------------------------------
# Fast-ion current: closed-form I3 = int dv / (v^3 + c^3)
# ---------------------------------------------------------------------------
def _I3_prim(v, c):
    """Antiderivative of 1/(v^3 + c^3)."""
    return (jnp.log((v + c) ** 2 / (v**2 - c * v + c**2)) / (6.0 * c**2)
            + jnp.arctan((2.0 * v - c) / (jnp.sqrt(3.0) * c))
            / (jnp.sqrt(3.0) * c**2))


def current_profile(rho_b, e_keV, w, pitch, eq, plasma, rho_edges,
                    mass_amu, emin_keV):
    """Unshielded fast-ion parallel current density profile [A/m^2]."""
    centers = 0.5 * (rho_edges[1:] + rho_edges[:-1])
    m_b = mass_amu * AMU_KG
    tau_se, vc, _ = _local_sd_quantities(plasma, centers, m_b, 1.0)

    idx = jnp.clip(jnp.searchsorted(rho_edges, rho_b, side="right") - 1,
                   0, centers.shape[0] - 1)
    in_grid = (rho_b >= rho_edges[0]) & (rho_b < rho_edges[-1])
    v0 = jnp.sqrt(2.0 * e_keV * 1e3 * E_CHARGE / m_b)
    vmin = jnp.sqrt(2.0 * emin_keV * 1e3 * E_CHARGE / m_b)
    vmin = jnp.minimum(vmin, v0)

    vc_m, tse_m = vc[idx], tau_se[idx]
    path = tse_m * ((v0 - vmin)
                    - vc_m**3 * (_I3_prim(v0, vc_m) - _I3_prim(vmin, vc_m)))
    contrib = jnp.where(in_grid, E_CHARGE * w * pitch * path, 0.0)
    per_bin = jnp.zeros(centers.shape[0]).at[idx].add(contrib)
    return per_bin / shell_volumes(eq, rho_edges)


def main():
    eq, plasma, tables, injector, _ = make_scenario(div=0.02)
    key = jax.random.PRNGKey(7)
    rho_b, e_keV, w, pitch = marker_births(
        key, injector, eq, plasma, tables, n_markers=100_000)

    sd = slowing_down(rho_b, e_keV, w, eq, plasma, RHO_EDGES, E_EDGES_KEV,
                      mass_amu=float(injector.mass_amu), emin_keV=EMIN_KEV)
    vols = shell_volumes(eq, RHO_EDGES)
    centers = np.asarray(0.5 * (RHO_EDGES[1:] + RHO_EDGES[:-1]))

    # Particle source rate
    idx = jnp.clip(jnp.searchsorted(RHO_EDGES, rho_b, "right") - 1, 0, 24)
    in_grid = (rho_b >= 0) & (rho_b < 1.0)
    src = (jnp.zeros(25).at[idx].add(jnp.where(in_grid, w, 0.0)) / vols)

    # Currents
    j_fast = current_profile(rho_b, e_keV, w, pitch, eq, plasma, RHO_EDGES,
                             float(injector.mass_amu), EMIN_KEV)
    # Zeff of the scenario plasma (flat by construction)
    _, _, ni_c = physics.profiles(plasma, jnp.array([0.3]))
    zeff = float(jnp.sum(ni_c * plasma.znum**2) / jnp.sum(ni_c * plasma.znum))
    j_net = j_fast * (1.0 - ZEFF_SHIELD_ZB / zeff)

    # Self-check: closed-form I3 vs numerical quadrature
    vgrid = jnp.linspace(3e5, 3.1e6, 20001)
    c = 3.9e6
    num = jnp.trapezoid(1.0 / (vgrid**3 + c**3), vgrid)
    ana = _I3_prim(vgrid[-1], c) - _I3_prim(vgrid[0], c)
    i3_err = abs(float(num / ana) - 1.0)
    assert i3_err < 1e-8, i3_err

    # Integrals
    pe_tot = float(jnp.sum(sd.pe * vols))
    pi_tot = float(jnp.sum(sd.pi_ * vols))
    s_tot = float(jnp.sum(src * vols))
    area = np.asarray(vols) / (2 * np.pi * eq.R0)
    i_fast = float(np.sum(np.asarray(j_fast) * area))
    i_net = float(np.sum(np.asarray(j_net) * area))
    w_tot = float(jnp.sum(sd.energy_density * vols))

    print(f"I3 closed-form self-check rel err: {i3_err:.2e}")
    print(f"particle source     : {s_tot:.3e} ions/s")
    print(f"P_e                 : {pe_tot/1e3:.1f} kW")
    print(f"P_i                 : {pi_tot/1e3:.1f} kW  "
          f"(P_i fraction {pi_tot/(pe_tot+pi_tot):.3f})")
    print(f"stored fast-ion E   : {w_tot/1e3:.1f} kJ")
    print(f"fast-ion current    : {i_fast/1e3:.1f} kA (unshielded)")
    print(f"net NBCD (flat 1-Zb/Zeff, Zeff={zeff:.2f}): {i_net/1e3:.1f} kA")
    print(f"mean birth pitch    : "
          f"{float(jnp.sum(w*pitch)/jnp.sum(w)):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    ax[0, 0].step(centers, np.asarray(src), where="mid", color="C2")
    ax[0, 0].set_ylabel(r"particle source [1/(s m$^3$)]")
    ax[0, 0].set_title(f"total {s_tot:.2e} ions/s")

    ax[0, 1].step(centers, np.asarray(sd.pe) / 1e3, where="mid", color="C0",
                  label=f"electrons ({pe_tot/1e3:.0f} kW)")
    ax[0, 1].step(centers, np.asarray(sd.pi_) / 1e3, where="mid", color="C3",
                  label=f"ions ({pi_tot/1e3:.0f} kW)")
    ax[0, 1].set_ylabel(r"power density [kW/m$^3$]")
    ax[0, 1].set_title("heating split (Stix)")
    ax[0, 1].legend(frameon=False)

    ax[1, 0].step(centers, np.asarray(j_fast) / 1e3, where="mid", color="C4",
                  label=f"fast-ion, unshielded ({i_fast/1e3:.0f} kA)")
    ax[1, 0].step(centers, np.asarray(j_net) / 1e3, where="mid", color="C5",
                  label=f"net, flat shielding ({i_net/1e3:.0f} kA)")
    ax[1, 0].set_ylabel(r"parallel current density [kA/m$^2$]")
    ax[1, 0].set_xlabel(r"$\rho$")
    ax[1, 0].legend(frameon=False, fontsize=9)
    ax[1, 0].set_title("NBCD (birth pitch retained; trapped-e$^-$ "
                       "corrections neglected)")

    ax[1, 1].step(centers, np.asarray(sd.density), where="mid", color="C1")
    ax[1, 1].set_ylabel(r"fast-ion density [m$^{-3}$]")
    ax[1, 1].set_xlabel(r"$\rho$")
    ax[1, 1].set_title(f"stored energy {w_tot/1e3:.0f} kJ")

    fig.suptitle("Analytic NBI actuator profiles - 100 keV D, 1 MW, "
                 "MC deposition + Stix slowing-down (100k markers)")
    fig.tight_layout()
    out = os.path.join(HERE, "analytic_sd_profiles.png")
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
