"""Multi-rung slowing-down ladder: energy-resolved orbit averaging.

The first-orbit average redistributes ALL of a fast ion's slowing-down
content with the orbit it has at birth. But orbits shrink as ions slow
(banana width ~ v), so the content deposited late in the slowing-down
should be redistributed with narrower orbits. The ladder splits each birth
cell's slowing history into ``n_rungs`` energy bands (linear in speed from
v0 down to v_min), computes the exact analytic content of each band, and
redistributes each band with a guiding-center orbit evaluated at that
band's representative energy.

Band analytics (all exact closed forms, telescoping over bands; reuses the
reviewed primitives of :mod:`slowing_down`): for a band ``v in [v_b, v_a]``
with ``u = v / v_c`` at the birth-surface (tau_se, v_c, E_c):

- time in band (-> density):  dt   = (tau_se/3) ln((v_a^3+v_c^3)/(v_b^3+v_c^3))
- stored energy in band:      dW   = S tau_se E_c [G(u_a) - G(u_b)]
- power to ions in band:      dP_i = S 2 E_c [F(u_a) - F(u_b)]
- power to electrons:         dP_e = S (E_a - E_b) - dP_i

Because the bands telescope, summing rungs reproduces the single-band
totals to machine precision, so overall power balance is exact regardless
of ``n_rungs``.

Orbit per rung: launch from the birth guiding-center position with speed
``v_rep`` (band midpoint in v; ``rung_rep="top"`` uses the band top, which
with ``n_rungs=1`` reproduces the plain first-orbit average for
validation). With ``pitch_scattering=False`` the birth pitch is kept down
the whole ladder (drag only); with ``pitch_scattering=True`` the pitch
distribution at each rung follows the Gaffey/Cordey collisional
relaxation, represented by three moment-matched pitch nodes (see the
inline derivation notes; independently reviewed in REVIEW_SD.md).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from .common import AMU_KG, E_CHARGE, Equilibrium, shell_volumes
from .orbit_average import bfield_cyl, orbit_average_matrix
from .slowing_down import (SlowingDownResult, _F, _G, _local_sd_quantities)


def ladder_sd(R, z, vpar, mu, e_keV, rate, eq: Equilibrium, plasma,
              rho_edges, e_edges_keV, *, mass_amu, emin_keV=20.0,
              n_rungs=5, rung_rep="mid", pitch_scattering=False,
              t_int=7.5e-4, n_steps=3000):
    """Slowing-down profiles with per-rung orbit averaging.

    Parameters mirror the deterministic chain's birth cells: guiding-center
    positions (R, z), parallel velocity and magnetic moment at birth
    (defining the pitch), energy [keV] and rate [1/s] per cell.

    Returns
    -------
    result : SlowingDownResult
    err : (ncell * n_rungs,) orbit-integrator energy-conservation error.
    """
    m_b = mass_amu * AMU_KG
    nbin = rho_edges.shape[0] - 1
    nE = e_edges_keV.shape[0] - 1
    centers = 0.5 * (rho_edges[1:] + rho_edges[:-1])
    vols = shell_volumes(eq, rho_edges)
    L = n_rungs

    # Birth pitch from (vpar, mu); drag preserves it down the ladder.
    B0mag = jax.vmap(lambda rz: jnp.linalg.norm(bfield_cyl(eq, rz)))(
        jnp.stack([R, z], axis=1))
    v0 = jnp.sqrt(vpar**2 + 2.0 * mu * B0mag / m_b)
    xi = vpar / jnp.maximum(v0, 1.0)
    vmin = jnp.sqrt(2.0 * emin_keV * 1e3 * E_CHARGE / m_b)
    v0 = jnp.maximum(v0, vmin)  # sub-threshold cells -> empty bands

    # Local slowing-down quantities at every destination bin center. The
    # content deposited in a bin is evaluated with THAT bin's (tau_se, v_c,
    # E_c) — the same destination-local convention as orbit_averaged_sd
    # (which feeds pseudo-markers at bin centers to slowing_down), so the
    # 1-rung ladder reproduces the plain first-orbit average exactly.
    tau_se, v_c, E_c = _local_sd_quantities(plasma, centers, m_b, 1.0)

    # Ladder edges, linear in speed: (ncell, L+1), v_edges[:,0] = v0.
    s = jnp.linspace(0.0, 1.0, L + 1)
    v_edges = v0[:, None] + (vmin - v0)[:, None] * s[None, :]
    va, vb = v_edges[:, :-1], v_edges[:, 1:]          # (ncell, L)
    Ea = 0.5 * m_b * va**2
    Eb = 0.5 * m_b * vb**2

    # Energy-grid speeds for the f_E band/bin overlap
    eE = e_edges_keV * 1e3 * E_CHARGE
    vE = jnp.sqrt(2.0 * eE / m_b)                     # (nE+1,)

    # Representative band speed per rung.
    v_rep = va if rung_rep == "top" else 0.5 * (va + vb)   # (ncell, L)

    # Pitch nodes per (cell, rung). Without pitch scattering: the birth
    # pitch, weight 1. With it: Gaffey/Cordey collisional pitch relaxation.
    # Legendre moments of the pitch distribution of ions slowing from v0
    # decay as  <P_l>(v) = P_l(xi0) X^{l(l+1) beta / 6}  with
    #   X(v) = (v^3/(v^3+v_c^3)) ((v0^3+v_c^3)/v0^3),
    #   beta = Z_eff / [Z],  [Z] = sum_i (n_i Z_i^2/n_e)(m_b/m_i)
    # (Gaffey 1976 J. Plasma Phys. 16 149; Cordey & Core 1974). beta = 1
    # exactly when m_b = m_i (this package's pure-D case gives 0.993 since
    # the beam is 2.0141 amu vs plasma 2.014). From <P_1> and <P_2> the
    # exact mean and variance follow; three Gauss-Hermite nodes
    # (mu, mu +- sqrt(3) sigma; weights 1/6, 2/3, 1/6) match both moments
    # and sum to 1, so power balance is untouched. Where a node would leave
    # |xi| < 1, sigma is shrunk to fit instead of clipping the node: the
    # mean is preserved exactly and only the (physically bounded) variance
    # is reduced. beta and v_c are evaluated at the birth-surface bin (the
    # pitch history integrates over the ion's own trajectory; the birth
    # surface is the consistent one-point choice).
    if pitch_scattering:
        from .physics import profiles as _profiles
        idx_b = jnp.clip(jnp.searchsorted(rho_edges, jnp.sqrt(
            (R - eq.R0) ** 2 + z**2) / eq.a, "right") - 1, 0, nbin - 1)
        ne_b, _, ni_b = _profiles(plasma, centers[idx_b])
        zi = jnp.asarray(plasma.znum, dtype=jnp.float64)
        mi = jnp.asarray(plasma.anum, dtype=jnp.float64) * AMU_KG
        zeff = jnp.sum(ni_b * zi**2, axis=-1) / jnp.sum(ni_b * zi, axis=-1)
        zbar = jnp.sum(ni_b * zi**2 * (m_b / mi), axis=-1) / ne_b
        beta = zeff / zbar                                  # (ncell,)

        vc_b = v_c[idx_b]
        X = ((v_rep**3 / (v_rep**3 + vc_b[:, None] ** 3))
             * ((v0**3 + vc_b**3) / v0**3)[:, None])        # (ncell, L)
        mean = xi[:, None] * X ** (beta[:, None] / 3.0)
        p2_0 = 0.5 * (3.0 * xi**2 - 1.0)
        p2 = p2_0[:, None] * X ** beta[:, None]
        xi2 = (1.0 + 2.0 * p2) / 3.0
        sig = jnp.sqrt(jnp.clip(xi2 - mean**2, 0.0))
        # Shrink sigma (never the mean) so all nodes stay inside |xi| < 1.
        sig_max = jnp.minimum(0.999 - mean, mean + 0.999) / jnp.sqrt(3.0)
        sig = jnp.minimum(sig, jnp.maximum(sig_max, 0.0))
        nodes = jnp.stack([mean - jnp.sqrt(3.0) * sig, mean,
                           mean + jnp.sqrt(3.0) * sig], axis=-1)
        node_w = jnp.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
    else:
        nodes = xi[:, None, None] * jnp.ones((1, L, 1))     # (ncell, L, 1)
        node_w = jnp.array([1.0])
    nnode = nodes.shape[-1]

    # One orbit per (cell, rung, pitch node); kernels averaged over nodes.
    vr = jnp.broadcast_to(v_rep[:, :, None],
                          (R.shape[0], L, nnode)).reshape(-1)
    xif = nodes.reshape(-1)
    Rf = jnp.repeat(R, L * nnode)
    zf = jnp.repeat(z, L * nnode)
    Bf = jnp.repeat(B0mag, L * nnode)
    frac_n, err = orbit_average_matrix(
        eq, rho_edges, Rf, zf, xif * vr,
        m_b * vr**2 * (1.0 - xif**2) / (2.0 * Bf),
        mass_amu=mass_amu, t_int=t_int, n_steps=n_steps)
    frac = jnp.einsum("n,clnb->clb", node_w,
                      frac_n.reshape(-1, L, nnode, nbin)).reshape(-1, nbin)

    # Per destination bin b: evaluate the exact band contents with that
    # bin's (tau_se, v_c, E_c), weight by the orbit-kernel column frac[:, b],
    # and sum over (cell, rung). Sequential lax.map over bins keeps the
    # (ncell, L, nE) f_E intermediate to a single bin at a time.
    va_f, vb_f = va.reshape(-1), vb.reshape(-1)        # (ncell*L,)
    rate_f = jnp.repeat(rate, L)
    dE_f = (Ea - Eb).reshape(-1)

    def per_bin(args):
        tse_b, vc_b, Ec_b, w_b = args                  # w_b: (ncell*L,)
        dt = (tse_b / 3.0) * (jnp.log(va_f**3 + vc_b**3)
                              - jnp.log(vb_f**3 + vc_b**3))
        dPi = 2.0 * Ec_b * (_F(va_f / vc_b) - _F(vb_f / vc_b)) * rate_f
        dPe = dE_f * rate_f - dPi
        dW = tse_b * Ec_b * (_G(va_f / vc_b) - _G(vb_f / vc_b)) * rate_f

        vlo = jnp.minimum(jnp.maximum(vE[None, :-1], vb_f[:, None]),
                          va_f[:, None])
        vhi = jnp.minimum(jnp.maximum(vE[None, 1:], vb_f[:, None]),
                          va_f[:, None])
        dNfE = (tse_b / 3.0) * rate_f[:, None] * (
            jnp.log(vhi**3 + vc_b**3) - jnp.log(vlo**3 + vc_b**3))

        return (jnp.sum(w_b * dt * rate_f), jnp.sum(w_b * dW),
                jnp.sum(w_b * dPe), jnp.sum(w_b * dPi),
                w_b @ dNfE)

    density, energy_density, pe, pi_, f_E = lax.map(
        per_bin, (tau_se, v_c, E_c, frac.T))
    density = density / vols
    energy_density = energy_density / vols
    pe = pe / vols
    pi_ = pi_ / vols
    f_E = f_E / vols[:, None] / jnp.diff(e_edges_keV)[None, :]

    e_full = jnp.max(e_keV) * 1e3 * E_CHARGE
    vf = jnp.sqrt(2.0 * e_full / m_b)
    tau_th = (tau_se / 3.0) * jnp.log((vf**3 + v_c**3) / (vmin**3 + v_c**3))

    return SlowingDownResult(
        rho_edges=rho_edges, e_edges_keV=e_edges_keV, f_E=f_E,
        density=density, energy_density=energy_density, pe=pe, pi_=pi_,
        tau_th=tau_th), err


if __name__ == "__main__":
    import os
    import time
    import numpy as np

    from .orbit_average import orbit_averaged_sd
    from .run_deterministic_sd import pencil_birth_cells
    from .test_comparison import make_scenario, rel_l1

    HERE = os.path.dirname(os.path.abspath(__file__))
    eq, plasma, tables, injector, _ = make_scenario(div=0.02)
    ref = np.load(os.path.join(HERE, "sd_reference.npz"))
    rho_edges = jnp.asarray(ref["rho_edges"])
    e_edges = jnp.asarray(ref["e_edges_keV"])
    vols = shell_volumes(eq, rho_edges)
    mamu = float(injector.mass_amu)

    R, z, vpar, mu, e_keV, rate = pencil_birth_cells(
        injector, eq, plasma, tables, rho_edges=rho_edges)

    # --- validation 1: n_rungs=1 with rep="top" == plain first-orbit avg
    lad1, _ = ladder_sd(R, z, vpar, mu, e_keV, rate, eq, plasma, rho_edges,
                        e_edges, mass_amu=mamu, n_rungs=1, rung_rep="top",
                        t_int=2.5e-4, n_steps=1000)
    frac0, _ = orbit_average_matrix(eq, rho_edges, R, z, vpar, mu,
                                    mass_amu=mamu, t_int=2.5e-4,
                                    n_steps=1000)
    oa0 = orbit_averaged_sd(frac0, e_keV, rate, eq, plasma, rho_edges,
                            e_edges, mass_amu=mamu)
    d1 = rel_l1(np.asarray(lad1.density), np.asarray(oa0.density))
    print(f"validation: 1-rung(top) vs first-orbit density L1 = {d1:.2e} "
          "(should be ~0)")
    assert d1 < 1e-10

    # --- validation 2: exact power balance, any rung count
    for L in (1, 5):
        lad, _ = ladder_sd(R, z, vpar, mu, e_keV, rate, eq, plasma,
                           rho_edges, e_edges, mass_amu=mamu, n_rungs=L,
                           t_int=1.0e-4, n_steps=200)
        dep = float(jnp.sum((lad.pe + lad.pi_) * vols))
        exp = float(jnp.sum(rate * jnp.clip(e_keV - 20.0, 0.0) * 1e3
                            * E_CHARGE))
        print(f"power balance, {L} rungs: rel {abs(dep/exp-1):.2e}")
        assert abs(dep / exp - 1.0) < 1e-10

    # --- the ladder runs (drag-only, and with Gaffey pitch scattering)
    t0 = time.perf_counter()
    lad, err = ladder_sd(R, z, vpar, mu, e_keV, rate, eq, plasma, rho_edges,
                         e_edges, mass_amu=mamu, n_rungs=5,
                         t_int=5.0e-4, n_steps=2000)
    jax.block_until_ready(lad.density)
    t_lad = time.perf_counter() - t0
    print(f"\n5-rung ladder: {t_lad:.1f} s, orbit energy cons. max "
          f"{float(err.max()):.1e}")

    t0 = time.perf_counter()
    ladp, errp = ladder_sd(R, z, vpar, mu, e_keV, rate, eq, plasma,
                           rho_edges, e_edges, mass_amu=mamu, n_rungs=5,
                           pitch_scattering=True, t_int=5.0e-4,
                           n_steps=2000)
    jax.block_until_ready(ladp.density)
    t_ladp = time.perf_counter() - t0
    dep = float(jnp.sum((ladp.pe + ladp.pi_) * vols))
    exp = float(jnp.sum(rate * jnp.clip(e_keV - 20.0, 0.0) * 1e3
                        * E_CHARGE))
    print(f"5-rung ladder + pitch scattering (3 nodes/rung): {t_ladp:.1f} s,"
          f" orbit energy cons. max {float(errp.max()):.1e}, "
          f"power balance rel {abs(dep/exp-1):.1e}")
    assert abs(dep / exp - 1.0) < 1e-10

    dens_ref = np.asarray(ref["density"])
    pe_ref, pi_ref = np.asarray(ref["pe"]), np.asarray(ref["pi"])
    w_ref = float(np.sum(np.asarray(ref["energy_density"])
                         * np.asarray(vols))) / 1e3
    print(f"\n{'model':<26}{'L1 n_fast':>11}{'L1 P_i':>9}{'L1 P_e':>9}"
          f"{'W_fast [kJ]':>13}")
    for nm, sd in [("first-orbit avg", oa0), ("5-rung ladder", lad),
                   ("ladder + pitch scatter", ladp)]:
        print(f"{nm:<26}"
              f"{rel_l1(np.asarray(sd.density), dens_ref):>11.3f}"
              f"{rel_l1(np.asarray(sd.pi_), pi_ref):>9.3f}"
              f"{rel_l1(np.asarray(sd.pe), pe_ref):>9.3f}"
              f"{float(jnp.sum(sd.energy_density * vols))/1e3:>13.1f}")
    print(f"{'ASCOT5 reference':<26}{0.0:>11.3f}{0.0:>9.3f}{0.0:>9.3f}"
          f"{w_ref:>13.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = np.asarray(0.5 * (rho_edges[1:] + rho_edges[:-1]))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    for a, (lab, attr, refv) in zip(ax, [
            ("fast-ion density [m$^{-3}$]", "density", dens_ref),
            ("$P_i$ [W/m$^3$]", "pi_", pi_ref),
            ("$P_e$ [W/m$^3$]", "pe", pe_ref)]):
        a.step(centers, refv, where="mid", color="k", lw=2, label="ASCOT5")
        a.step(centers, np.asarray(getattr(oa0, attr)), where="mid",
               color="C3", label="first-orbit avg")
        a.step(centers, np.asarray(getattr(lad, attr)), where="mid",
               color="C2", label="5-rung ladder (drag only)")
        a.step(centers, np.asarray(getattr(ladp, attr)), where="mid",
               color="C4", lw=1.8, label="ladder + pitch scattering")
        a.set_xlabel(r"$\rho$")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Slowing-down ladder with Gaffey pitch relaxation, deterministic "
        "pencil source - n$_f$ rel-L1 vs ASCOT5: "
        f"{rel_l1(np.asarray(oa0.density), dens_ref):.2f} (first-orbit), "
        f"{rel_l1(np.asarray(lad.density), dens_ref):.2f} (drag-only "
        "ladder), "
        f"{rel_l1(np.asarray(ladp.density), dens_ref):.2f} (+ pitch)")
    fig.tight_layout()
    out = os.path.join(HERE, "comparison_ladder.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure saved to {out}")
