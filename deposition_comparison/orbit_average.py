"""Guiding-center orbit averaging of the NBI source in JAX.

The analytic slowing-down model deposits each fast ion's density and power on
its birth flux surface. In this scenario the poloidal field is weak
(q_eff ~ 9) and 100 keV D banana orbits are radially wide
(``delta rho ~ 0.1-0.2``), which is why the local analytic profiles miss the
ASCOT5 reference by rel-L1 ~ 0.6. This module supplies the leading
correction — the same trick RABBIT uses: distribute each marker's source over
the flux surfaces its **first guiding-center orbit** actually visits, weighted
by the time spent per surface, before applying the (local) slowing-down.

Guiding-center equations (standard, low-beta):

    dX/dt     = v_par b + (1/(q_c B)) b x ( mu grad(B)/? ... ) — implemented as
                v_par b + b/(q_c B) x ( mu grad(B) + m v_par^2 (b . grad) b )
    dv_par/dt = -(mu/m) b . grad(B)

with mu = m v_perp^2 / (2 B) conserved. The field is the analytic circular
equilibrium the whole package (and the ASCOT reference input) uses:

    psi   = ((R - R0)^2 + z^2) / a^2        [Vs/rad]
    B_R   = -(1/R) dpsi/dz,  B_z = (1/R) dpsi/dR,  B_phi = B0 R0 / R

All spatial derivatives of B come from :func:`jax.jacfwd` on the analytic
field — no hand-coded gradients. Integration is fixed-step RK4 under
``lax.scan`` (vmapped over markers), accumulating a time-in-rho-bin histogram
on the fly. Axisymmetry means only (R, z) matter for rho.

Validation hooks: energy and mu are conserved by construction (mu exactly,
energy checked numerically in the sanity block); the integrator is also
checked against ASCOT's stored inistate B components.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from .common import AMU_KG, E_CHARGE, Equilibrium

B0 = 5.3  # [T] on-axis toroidal field, matching the ASCOT reference input


def bfield_cyl(eq: Equilibrium, rz):
    """Magnetic field (B_R, B_phi, B_z) [T] at (R, z)."""
    R, z = rz[0], rz[1]
    br = -(1.0 / R) * 2.0 * z / eq.a**2
    bz = (1.0 / R) * 2.0 * (R - eq.R0) / eq.a**2
    bphi = B0 * eq.R0 / R
    return jnp.array([br, bphi, bz])


def _gc_rhs(eq, m_b, q_c, y):
    """Guiding-center RHS in (R, z, v_par); axisymmetric, phi ignorable.

    Cartesian-free formulation: work in the local cylindrical triad
    (e_R, e_phi, e_z) at the marker; axisymmetry makes d/dphi = 0, and the
    curvature of the triad enters only through the phi-direction, which does
    not move the marker in (R, z, rho). The drift is evaluated with jacfwd
    of the cylindrical components plus the analytic e_phi curvature term.
    """
    rz, vpar, mu = y[:2], y[2], y[3]
    B = bfield_cyl(eq, rz)
    Bmag = jnp.linalg.norm(B)
    b = B / Bmag

    # Gradients of the cylindrical components wrt (R, z)
    J = jax.jacfwd(lambda x: bfield_cyl(eq, x))(rz)     # (3 comp, 2 coord)
    gradBmag = (J.T @ B) / Bmag                          # (2,) d|B|/d(R,z)

    # (b . grad) b in cylindrical coords with curvature of e_phi:
    # (b.grad)v |_R   = b_R db_R/dR + b_z db_R/dz - b_phi^2 / R
    # (b.grad)v |_phi = b_R db_phi/dR + b_z db_phi/dz + b_phi b_R / R
    # (b.grad)v |_z   = b_R db_z/dR + b_z db_z/dz
    R = rz[0]
    db = jax.jacfwd(lambda x: bfield_cyl(eq, x) /
                    jnp.linalg.norm(bfield_cyl(eq, x)))(rz)  # (3, 2)
    bg = jnp.array([
        b[0] * db[0, 0] + b[2] * db[0, 1] - b[1] ** 2 / R,
        b[0] * db[1, 0] + b[2] * db[1, 1] + b[1] * b[0] / R,
        b[0] * db[2, 0] + b[2] * db[2, 1],
    ])

    # Drift: b x (mu gradB + m vpar^2 kappa) / (q B); gradBmag has no phi comp.
    force = jnp.array([mu * gradBmag[0], 0.0, mu * gradBmag[1]]) \
        + m_b * vpar**2 * bg
    drift = jnp.cross(b, force) / (q_c * Bmag)

    v = vpar * b + drift
    dvpar = -(mu / m_b) * (b[0] * gradBmag[0] + b[2] * gradBmag[1])
    return jnp.array([v[0], v[2], dvpar, 0.0])


def _orbit_histogram(eq, m_b, q_c, rho_edges, y0, dt, n_steps):
    """Integrate one orbit; return (time-in-bin histogram, y_final)."""
    nbin = rho_edges.shape[0] - 1

    def rk4(y):
        k1 = _gc_rhs(eq, m_b, q_c, y)
        k2 = _gc_rhs(eq, m_b, q_c, y + 0.5 * dt * k1)
        k3 = _gc_rhs(eq, m_b, q_c, y + 0.5 * dt * k2)
        k4 = _gc_rhs(eq, m_b, q_c, y + dt * k3)
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(carry, _):
        y, hist = carry
        rho = jnp.sqrt((y[0] - eq.R0) ** 2 + y[1] ** 2) / eq.a
        idx = jnp.clip(jnp.searchsorted(rho_edges, rho, "right") - 1,
                       0, nbin - 1)
        ok = (rho >= rho_edges[0]) & (rho < rho_edges[-1])
        hist = hist.at[idx].add(jnp.where(ok, dt, 0.0))
        return (rk4(y), hist), None

    (yf, hist), _ = lax.scan(step, (y0, jnp.zeros(nbin)), None,
                             length=n_steps)
    return hist, yf


def orbit_average_matrix(eq, rho_edges, r, z, vpar, mu, *, mass_amu,
                         t_int=1.0e-3, n_steps=4000):
    """Fraction of time each marker spends in each rho bin.

    Parameters
    ----------
    r, z : (n,) guiding-center birth position [m]
    vpar : (n,) parallel velocity [m/s]
    mu : (n,) magnetic moment [J/T]
    t_int : float
        Integration time [s]; ~1 ms covers many poloidal transits/bounces of
        a 100 keV D ion here while staying << the slowing-down time (0.1 s).

    Returns
    -------
    frac : (n, nbin) row-stochastic (up to out-of-grid time) time fractions.
    efinal_rel : (n,) relative energy-conservation error of the integrator.
    """
    m_b = mass_amu * AMU_KG
    q_c = E_CHARGE
    dt = t_int / n_steps

    def one(ri, zi, vi, mui):
        y0 = jnp.array([ri, zi, vi, mui])
        hist, yf = _orbit_histogram(eq, m_b, q_c, rho_edges, y0, dt, n_steps)
        e0 = 0.5 * m_b * vi**2 + mui * jnp.linalg.norm(
            bfield_cyl(eq, y0[:2]))
        ef = 0.5 * m_b * yf[2] ** 2 + mui * jnp.linalg.norm(
            bfield_cyl(eq, yf[:2]))
        return hist / t_int, jnp.abs(ef / e0 - 1.0)

    frac, err = jax.jit(jax.vmap(one))(r, z, vpar, mu)
    return frac, err


def orbit_averaged_sd(frac, birth_energy_keV, birth_weight, eq, plasma,
                      rho_edges, e_edges_keV, *, mass_amu, emin_keV=20.0):
    """Analytic slowing-down on the orbit-averaged source.

    Expands each (marker, bin) pair into a pseudo-marker at the bin center
    carrying weight w * frac and feeds the flattened set to
    :func:`slowing_down.slowing_down`.
    """
    from .slowing_down import slowing_down

    centers = 0.5 * (rho_edges[1:] + rho_edges[:-1])
    n, nbin = frac.shape
    rho_flat = jnp.tile(centers, n)
    e_flat = jnp.repeat(birth_energy_keV, nbin)
    w_flat = (birth_weight[:, None] * frac).reshape(-1)
    return slowing_down(rho_flat, e_flat, w_flat, eq, plasma, rho_edges,
                        e_edges_keV, mass_amu=mass_amu, emin_keV=emin_keV)


def load_ascot_markers(h5path, run, eq):
    """Read (r, z, vpar, mu, energy_keV, weight) of the SDMAIN inistate.

    Also returns the max relative deviation between the analytic field and
    ASCOT's stored inistate B components, as an equilibrium cross-check.
    """
    import h5py
    import numpy as np

    with h5py.File(h5path, "r") as f:
        ini = f["results"][run]["inistate"]
        r = np.asarray(ini["r"]).ravel()
        z = np.asarray(ini["z"]).ravel()
        ppar = np.asarray(ini["ppar"]).ravel()
        mu = np.asarray(ini["mu"]).ravel() * E_CHARGE  # eV/T -> J/T
        w = np.asarray(ini["weight"]).ravel()
        mass = np.asarray(ini["mass"]).ravel() * AMU_KG
        br = np.asarray(ini["br"]).ravel()
        bphi = np.asarray(ini["bphi"]).ravel()
        bz = np.asarray(ini["bz"]).ravel()

    rz = jnp.stack([jnp.asarray(r), jnp.asarray(z)], axis=1)
    Bana = jax.vmap(lambda x: bfield_cyl(Equilibrium(eq.R0, eq.a), x))(rz)
    Basc = jnp.stack([jnp.asarray(br), jnp.asarray(bphi),
                      jnp.asarray(bz)], axis=1)
    bdev = float(jnp.max(jnp.linalg.norm(Bana - Basc, axis=1)
                         / jnp.linalg.norm(Basc, axis=1)))

    bmag = np.linalg.norm(np.stack([br, bphi, bz], axis=1), axis=1)
    vpar = ppar / mass
    e_J = 0.5 * mass * vpar**2 + mu * bmag
    e_keV = e_J / (1e3 * E_CHARGE)
    return (jnp.asarray(r), jnp.asarray(z), jnp.asarray(vpar),
            jnp.asarray(mu), jnp.asarray(e_keV), jnp.asarray(w),
            float(mass.mean() / AMU_KG), bdev)


if __name__ == "__main__":
    import numpy as np
    from .test_comparison import make_scenario

    eq, plasma, _, _, _ = make_scenario()
    rho_edges = jnp.linspace(0.0, 1.0, 26)

    r, z, vpar, mu, e_keV, w, mamu, bdev = load_ascot_markers(
        "deposition_comparison/bbnbi_ref/ascot.h5", "run_1940713020", eq)
    print(f"markers: {r.shape[0]}, mass {mamu:.4f} amu")
    print(f"analytic-vs-ASCOT inistate B field max rel dev: {bdev:.2e}")
    print(f"energy range: {float(e_keV.min()):.1f}..{float(e_keV.max()):.1f}"
          " keV (should be ~33/50/100 components)")

    frac, err = orbit_average_matrix(eq, rho_edges, r, z, vpar, mu,
                                     mass_amu=mamu)
    cov = jnp.sum(frac, axis=1)
    print(f"orbit-integrator energy conservation: max rel {float(err.max()):.2e}")
    print(f"time-in-grid coverage: min {float(cov.min()):.4f} "
          f"mean {float(cov.mean()):.4f} (lost-orbit fraction of weight: "
          f"{float(jnp.sum(w * (cov < 0.99)) / jnp.sum(w)):.4f})")
    rho_b = jnp.sqrt((r - eq.R0) ** 2 + z**2) / eq.a
    widths = jnp.sqrt(jnp.sum(frac * (0.5 * (rho_edges[1:] + rho_edges[:-1])
                                      - rho_b[:, None]) ** 2, axis=1)
                      / jnp.clip(cov, 1e-12))
    print(f"rms orbit-excursion in rho: mean {float(jnp.mean(widths)):.3f}")
