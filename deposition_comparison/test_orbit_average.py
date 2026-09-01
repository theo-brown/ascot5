"""Tests for the first-orbit-averaging correction.

Requires the local ASCOT reference h5 (deposition_comparison/bbnbi_ref/,
gitignored) for the marker initial states; skips cleanly without it.
"""
import os

import numpy as np
import pytest

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
NPZ = os.path.join(HERE, "sd_reference.npz")
SDRUN = "run_1940713020"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(H5) and os.path.exists(NPZ)),
    reason="ASCOT reference h5/npz not present (h5 is a local run artifact; "
           "regenerate with run_bbnbi_reference + run_ascot_reference)")


@pytest.fixture(scope="module")
def setup():
    eq, plasma, _, _, _ = make_scenario()
    ref = np.load(NPZ)
    rho_edges = jnp.asarray(ref["rho_edges"])
    e_edges = jnp.asarray(ref["e_edges_keV"])
    mrk = load_ascot_markers(H5, SDRUN, eq)
    r, z, vpar, mu, e_keV, w, mamu, bdev = mrk
    frac, err = orbit_average_matrix(eq, rho_edges, r, z, vpar, mu,
                                     mass_amu=mamu)
    return eq, plasma, ref, rho_edges, e_edges, mrk, frac, err


def test_field_matches_ascot_inistate(setup):
    """The analytic B field is the equilibrium ASCOT actually ran with."""
    bdev = setup[5][7]
    assert bdev < 1e-8


def test_orbit_integrator_conserves_energy(setup):
    err = setup[7]
    assert float(jnp.max(err)) < 1e-3


def test_orbits_stay_in_grid(setup):
    frac = setup[6]
    cov = jnp.sum(frac, axis=1)
    assert float(jnp.min(cov)) > 0.99


def test_orbit_averaging_closes_the_gap(setup):
    """Orbit averaging must improve every profile vs ASCOT, substantially
    for the density (the orbit-width-dominated quantity)."""
    eq, plasma, ref, rho_edges, e_edges, mrk, frac, _ = setup
    r, z, vpar, mu, e_keV, w, mamu, _ = mrk
    rho_b = jnp.sqrt((r - eq.R0) ** 2 + z**2) / eq.a

    local = slowing_down(rho_b, e_keV, w, eq, plasma, rho_edges, e_edges,
                         mass_amu=mamu)
    oa = orbit_averaged_sd(frac, e_keV, w, eq, plasma, rho_edges, e_edges,
                           mass_amu=mamu)

    for attr, refkey in [("density", "density"), ("pi_", "pi"),
                         ("pe", "pe")]:
        l1_local = rel_l1(np.asarray(getattr(local, attr)),
                          np.asarray(ref[refkey]))
        l1_oa = rel_l1(np.asarray(getattr(oa, attr)),
                       np.asarray(ref[refkey]))
        assert l1_oa < l1_local, (attr, l1_oa, l1_local)
    # Density: 0.62 -> 0.34 measured; guard with headroom. Remaining gap is
    # slowing-down-time orbit evolution (pitch scattering) + collisional
    # transport, not first-orbit width.
    l1_dens = rel_l1(np.asarray(oa.density), np.asarray(ref["density"]))
    assert l1_dens < 0.45


def test_power_identity_preserved(setup):
    """Orbit averaging redistributes the source; total deposited power must
    still satisfy the analytic identity (frac rows sum to ~1)."""
    eq, plasma, ref, rho_edges, e_edges, mrk, frac, _ = setup
    _, _, _, _, e_keV, w, mamu, _ = mrk
    oa = orbit_averaged_sd(frac, e_keV, w, eq, plasma, rho_edges, e_edges,
                           mass_amu=mamu)
    vols = shell_volumes(eq, rho_edges)
    dep = float(jnp.sum((oa.pe + oa.pi_) * vols))
    expect = float(jnp.sum(w * jnp.clip(e_keV - 20.0, 0.0, None))
                   * 1e3 * 1.602176634e-19)
    assert abs(dep / expect - 1.0) < 1e-6
