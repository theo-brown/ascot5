"""Pytest comparison suite for the two NBI deposition methods.

Compares the BBNBI-style Monte-Carlo method (:func:`mc_deposition.deposit_mc`)
against the RABBIT-style pencil method (:func:`pencil_deposition.deposit_pencil`)
on a shared ITER-ish scenario defined by :func:`make_scenario`.

Run from the repository root:

    python -m pytest deposition_comparison/test_comparison.py -v

Performance notes
-----------------
The MC tracer's runtime is dominated by ``n_markers * n_steps`` trace points
and each distinct ``(n_markers, n_steps)`` pair triggers a fresh XLA compile.
Only two combinations are used, and each expensive deposition runs exactly
once inside a module-scoped fixture that all asserts share:

- ``(100_000, 2000)`` for the low-density (shinethrough/determinism) runs;
- ``(400_000, 1000)`` for the default-density div=0 / div=0.025 pair. The
  larger marker count is required by the divergence test: at 100k markers
  the relative-L1 MC noise floor vs the pencil solution (~0.02) is as large
  as the 25 mrad smearing systematic (~0.017), and with an unlucky key the
  two partially cancel; at 400k the noise floor drops to ~0.01 and the test
  passes with a stable margin (verified over seeds 1234, 7, 99).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Every package module enables x64 at import time; importing them normally is
# all that is needed before any jax arrays are created.
from deposition_comparison import beam, physics
from deposition_comparison.common import Equilibrium, Plasma
from deposition_comparison.mc_deposition import deposit_mc
from deposition_comparison.pencil_deposition import deposit_pencil

# Two (n_markers, n_steps) combinations -> two XLA compiles total.
N_MARKERS = 100_000        # low-density runs
N_STEPS = 2000
N_MARKERS_BIG = 400_000    # default-density div=0 / div=0.025 pair
N_STEPS_BIG = 1000

# All MC runs share this key; the divergence test relies on the div=0 and
# div=0.025 runs using the same key.
KEY = jax.random.PRNGKey(1234)

# Low-density scenario for the shinethrough comparison. Chosen empirically:
# with ne0=5e18 the pencil method gives a shinethrough fraction of ~0.17,
# comfortably inside the target window [0.05, 0.6]. (At the 1.2e19 starting
# point the beam was still almost fully absorbed: shinethrough ~0.016.)
NE0_LOW = 5e18


def make_scenario(*, ne0=8e19, div=0.02, energy_keV=100.0):
    """Build the shared comparison scenario.

    Returns
    -------
    (eq, plasma, tables, injector, rho_edges)
    """
    eq = Equilibrium(R0=6.2, a=2.0)
    plasma = Plasma(
        ne0=ne0, ne_edge=1.0, alpha_n=1.5,
        te0=1e4, te_edge=100.0, alpha_t=1.5,
        anum=jnp.array([2.0, 12.0]),
        znum=jnp.array([1.0, 6.0]),
        conc=jnp.array([1.0, 0.02]),
    )
    tables = physics.prepare_suzuki(plasma.anum, plasma.znum)
    injector = beam.make_injector(
        r=9.0, phi=0.0, z=0.0, tanrad=5.5, focal_length=15.0,
        grid_w=0.4, grid_h=0.8, nx=8, ny=16,
        energy_keV=energy_keV, efrac=[0.55, 0.3, 0.15], power=1e6,
        div_h=div, div_v=div, anum=2, mass_amu=2.0141)
    rho_edges = jnp.linspace(0.0, 1.0, 51)
    return eq, plasma, tables, injector, rho_edges


def rel_l1(a, b):
    """Relative L1 difference sum|a - b| / sum b."""
    return float(jnp.sum(jnp.abs(a - b)) / jnp.sum(b))


def shine_frac(result, injector):
    return float(result.shinethrough_power) / injector.power


# ---------------------------------------------------------------------------
# Module-scoped fixtures: every expensive deposition runs exactly once.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scenario_div0():
    """Default density, zero divergence."""
    return make_scenario(div=0.0)


@pytest.fixture(scope="module")
def scenario_div():
    """Default density, finite divergence (25 mrad)."""
    return make_scenario(div=0.025)


@pytest.fixture(scope="module")
def scenario_low():
    """Low density (nonzero shinethrough), zero divergence."""
    return make_scenario(ne0=NE0_LOW, div=0.0)


@pytest.fixture(scope="module")
def pencil_default(scenario_div0):
    eq, plasma, tables, inj, edges = scenario_div0
    return deposit_pencil(inj, eq, plasma, tables, edges)


@pytest.fixture(scope="module")
def pencil_low(scenario_low):
    eq, plasma, tables, inj, edges = scenario_low
    return deposit_pencil(inj, eq, plasma, tables, edges)


@pytest.fixture(scope="module")
def mc_default_div0(scenario_div0):
    eq, plasma, tables, inj, edges = scenario_div0
    return deposit_mc(KEY, inj, eq, plasma, tables, edges,
                      n_markers=N_MARKERS_BIG, n_steps=N_STEPS_BIG)


@pytest.fixture(scope="module")
def mc_default_div(scenario_div):
    eq, plasma, tables, inj, edges = scenario_div
    return deposit_mc(KEY, inj, eq, plasma, tables, edges,
                      n_markers=N_MARKERS_BIG, n_steps=N_STEPS_BIG)


@pytest.fixture(scope="module")
def mc_low_div0(scenario_low):
    eq, plasma, tables, inj, edges = scenario_low
    return deposit_mc(KEY, inj, eq, plasma, tables, edges,
                      n_markers=N_MARKERS, n_steps=N_STEPS)


# ---------------------------------------------------------------------------
# Power balance
# ---------------------------------------------------------------------------
def test_power_balance_pencil(scenario_div0, pencil_default):
    _, _, _, inj, _ = scenario_div0
    res = pencil_default
    residual = abs(inj.power - float(res.total_deposited_power)
                   - float(res.shinethrough_power)) / inj.power
    assert residual < 1e-9


def test_power_balance_mc(scenario_div0, mc_default_div0):
    _, _, _, inj, _ = scenario_div0
    res = mc_default_div0
    residual = abs(inj.power - float(res.total_deposited_power)
                   - float(res.shinethrough_power)) / inj.power
    assert residual < 1e-6


# ---------------------------------------------------------------------------
# Shinethrough agreement (low density, zero divergence)
# ---------------------------------------------------------------------------
def test_shinethrough_agreement(scenario_low, pencil_low, mc_low_div0):
    _, _, _, inj, _ = scenario_low
    s_pencil = shine_frac(pencil_low, inj)
    s_mc = shine_frac(mc_low_div0, inj)
    # Validate the hard-coded NE0_LOW choice: meaningfully nonzero.
    assert 0.05 < s_pencil < 0.6
    # Zero divergence -> the methods trace the same rays; they must agree.
    assert abs(s_mc - s_pencil) < 0.01


# ---------------------------------------------------------------------------
# Profile agreement at zero divergence (default density)
# ---------------------------------------------------------------------------
def test_profiles_agree_zero_divergence(pencil_default, mc_default_div0):
    # MC samples beamlet origins from the same discrete beamlet set the
    # pencil method integrates, so with div=0 the only differences are MC
    # noise and ionization-point-vs-segment-midpoint binning.
    l1_birth = rel_l1(mc_default_div0.birth_rate, pencil_default.birth_rate)
    assert l1_birth < 0.08
    l1_power = rel_l1(mc_default_div0.power_density,
                      pencil_default.power_density)
    assert l1_power < 0.08


# ---------------------------------------------------------------------------
# Divergence matters: finite-divergence MC differs more from the pencil
# solution than the div=0 MC run does (same PRNG key for both MC runs).
# ---------------------------------------------------------------------------
def test_divergence_smears(pencil_default, mc_default_div0, mc_default_div):
    l1_div0 = rel_l1(mc_default_div0.birth_rate, pencil_default.birth_rate)
    l1_div = rel_l1(mc_default_div.birth_rate, pencil_default.birth_rate)
    assert l1_div > l1_div0


# ---------------------------------------------------------------------------
# Implementation hygiene
# ---------------------------------------------------------------------------
def test_jit_wrappers():
    # jax 0.10 PjitFunction satisfies both isinstance(f, jax.stages.Wrapped)
    # and hasattr(f, "lower"); assert the former (verified to hold).
    assert isinstance(deposit_mc, jax.stages.Wrapped)
    assert isinstance(deposit_pencil, jax.stages.Wrapped)


def test_mc_deterministic(scenario_low, mc_low_div0):
    eq, plasma, tables, inj, edges = scenario_low
    res2 = deposit_mc(KEY, inj, eq, plasma, tables, edges,
                      n_markers=N_MARKERS, n_steps=N_STEPS)
    assert np.array_equal(np.asarray(mc_low_div0.birth_rate),
                          np.asarray(res2.birth_rate))
