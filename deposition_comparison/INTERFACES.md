# Deposition comparison package — module contracts

Comparison of two NBI *deposition* methods (no slowing-down, no full ASCOT run):

- **MC method (BBNBI-style)**: sample neutral markers from beamlets with
  divergence, trace rays, sample the ionization point from the optical depth.
- **Pencil method (RABBIT-style)**: deterministic pencil rays (one per beamlet
  and energy component), analytic attenuation, deposit continuously along ray.

Both use the SAME Suzuki beam-stopping cross section, equilibrium, profiles,
and rho binning defined in `common.py`. Read `common.py` first — all shared
NamedTuples, units, and geometry conventions live there and MUST NOT be
redefined elsewhere.

Style requirements (all modules):
- JAX only (`jax`, `jax.numpy`); no loops over markers/beamlets — use `vmap`
  (or batched array ops) and wrap the public entry points in `jax.jit`
  (with `static_argnames` for shapes/step counts).
- 64-bit: every module that runs as a script or is imported first must call
  `jax.config.update("jax_enable_x64", True)` at import time (do it in each
  module; it is idempotent).
- Pure functions; PRNG via explicit `jax.random.PRNGKey` threading.

## Module ownership (one agent each; do not edit other modules)

### 1. `physics.py`
```python
def profiles(plasma: Plasma, rho):  # rho shape (...)
    """Return (ne, te, ni) with ne,te shape (...), ni shape (..., nion).

    Parabolic profiles per Plasma docstring; ni from quasineutrality:
    conc[0] is recomputed as (1 - sum_{i>0} znum[i]*conc[i]) / znum[0].
    Outside rho>=1 use edge values (ni scaled consistently)."""

def prepare_suzuki(anum, znum) -> SuzukiTables:
    """Concrete-species table selection (NOT jit-traceable; call at setup).
    Port of src/suzuki.c tables. Use Zeff-validity windows to pick the
    impurity B row (assume 1 < Zeff < 4 regime; pick the first matching row
    with Zeffmin=1). Raise ValueError for unsupported species."""

def sigma_stop(tables: SuzukiTables, e_kev_per_amu, ne, te_ev, ni):
    """Beam-stopping cross section [m^2], pure-JAX port of suzuki_sigmav
    (WITHOUT the final vnorm multiplication: return sigma, not sigma*v).
    Broadcasts over leading dims of (e_kev_per_amu, ne, te_ev) with ni
    shape (..., nion). Select low/high-E tables with jnp.where on
    e_kev_per_amu (low if 9 <= E < 100 keV/amu, else high). Clamp inputs
    (ne >= 1e10, te >= 1 eV) so logs are safe in vacuum regions."""
```
Also include `_test_sigma_sanity()` runnable via `python -m
deposition_comparison.physics`: sigma for a 50 keV/amu D beam in
ne=1e20, te=5 keV pure-D plasma should be ~O(1e-20 .. 1e-19) m^2, positive,
finite; and print a comparison table over E = [20, 50, 100, 500, 1000]
keV/amu. Verify against the C implementation's structure line by line
(`src/suzuki.c` in this repo).

### 2. `beam.py`
```python
def make_injector(*, r, phi, z, tanrad, focal_length, grid_w, grid_h,
                  nx, ny, energy_keV, efrac, power, div_h, div_v,
                  anum, mass_amu) -> Injector:
    """Simplified rectangular-grid injector, mimicking a5py NBI.generate:
    beamlets on an nx-x-ny grid of size grid_w x grid_h [m], centered at
    cylindrical (r, phi[rad], z), grid plane perpendicular to the central
    aim direction. Central direction: horizontal, tangent such that the
    beam center line has tangency radius |tanrad| (sign = direction);
    each beamlet aims at the common focal point at distance focal_length
    along the central axis (focusing). Pure numpy/jnp setup code; no jit
    needed."""

def component_rates(injector: Injector) -> jnp.ndarray:
    """(3,) particles/s injected at full, half, third energy.
    P = sum_k rate_k * (E_full/k) * 1e3 * E_CHARGE with
    rate_k ∝ efrac[k]."""

def sample_markers(key, injector: Injector, n: int) -> Markers:
    """Sample n markers: beamlet uniformly; energy component
    multinomial-free: split deterministically proportional to efrac
    (jnp.repeat-style banding is fine) with per-marker weight =
    component_rate/(n_markers_in_component); direction = beamlet dir
    perturbed by Gaussian divergence angles (div_h, div_v) in the local
    beamlet frame. jit with static n."""
```
Runnable sanity: `python -m deposition_comparison.beam` prints total power
reconstruction error (sum weight_i * E_i vs injector.power, < 1e-10 rel) and
mean tangency radius of sampled rays vs tanrad.

### 3. `mc_deposition.py` (BBNBI-style)
```python
def deposit_mc(key, injector, eq, plasma, tables, rho_edges, *,
               n_markers: int, n_steps: int = 4000,
               ray_length: float = 25.0) -> DepositionResult
```
- Sample markers via `beam.sample_markers`.
- For each marker (vmapped): march `n_steps` equal steps ds along the ray;
  evaluate attenuation coefficient k(l) = ne(l) * sigma_stop(...) at step
  midpoints; cumulative optical depth tau(l); draw u~U(0,1); ionization at
  first l with tau >= -log(u) (use searchsorted on cumsum); if never reached,
  the marker is shinethrough.
- Bin ionized markers' birth rho into rho_edges weighted by weight
  (birth_rate) and weight*E (power), divide by `shell_volumes`. Markers
  born at rho >= rho_edges[-1] count as deposited outside the grid
  (report in shinethrough? NO — add them to neither; keep power balance by
  computing total_deposited_power from the histogram and shinethrough from
  non-ionized weight; ionized-outside-grid should be negligible but assert
  nothing about it).
- Whole pipeline jitted (static n_markers, n_steps).

### 4. `pencil_deposition.py` (RABBIT-style)
```python
def deposit_pencil(injector, eq, plasma, tables, rho_edges, *,
                   n_quad: int = 4000, ray_length: float = 25.0)
                   -> DepositionResult
```
- One deterministic pencil per (beamlet, energy component): ray from beamlet
  origin along beamlet dir, NO divergence (that is the methodological
  difference vs MC).
- Attenuation coefficient k(l) as above on n_quad midpoints; survival
  A(l) = exp(-cumtrapz k); segment deposition w_seg = A_in - A_out per step
  (exact per-step absorbed fraction); deposit each segment's rate into the
  rho bin of its midpoint via `jnp.bincount`-style scatter-add
  (segment_sum / .at[].add on digitized rho). Shinethrough = A(end).
- vmap over beamlets and energy components; jit (static n_quad).
- Same convention for out-of-grid rho segments as MC (dropped from profile;
  shinethrough only counts survival past ray end).

### 5. `test_comparison.py` + `run_comparison.py`
- `test_comparison.py` (pytest, CPU, ~1 min budget): build a shared scenario:
  ITER-ish circular eq (R0=6.2, a=2.0), D plasma with C impurity
  (conc_C=0.02), ne0=8e19, te0=10 keV, edge 1e17/100eV, alpha=1.5;
  injector: r=9.0, phi=0, z=0, tanrad=5.5, focal 15 m, grid 0.4x0.8,
  nx=8, ny=16, E=100 keV D, efrac=[0.55,0.3,0.15], P=1 MW.
  - test_power_balance_mc / _pencil: deposited + shinethrough == injected
    within 1e-6 (pencil) / 2% (MC, out-of-grid births allowed slack).
  - test_methods_agree_zero_divergence: div=0 injector, 200k MC markers:
    shinethrough fractions agree within 3 sigma-ish (abs 0.01) and
    birth_rate profiles agree: relative L1 difference
    sum|S_mc-S_pencil| / sum S_pencil < 0.08.
  - test_divergence_smears: with div=25 mrad the MC profile differs more
    from the (divergence-free) pencil profile than the div=0 MC run does —
    demonstrating what the pencil approximation misses.
  - test_jit_and_vmap_used: check public entry points are jitted (e.g.
    repeated call much faster / or simply assert they are Jitted wrappers
    via type inspection `isinstance(f, jax.stages.Wrapped)` or presence of
    `.lower`).
- `run_comparison.py`: script producing `comparison.png` (matplotlib):
  panel 1 birth_rate profiles (MC vs pencil, div=0 and finite div);
  panel 2 power_density; annotate shinethrough fractions; plus a printed
  summary table.

## Scenario constants
Put the shared test scenario in `test_comparison.py` as `make_scenario()`
(agent 5 owns it); agents 3 & 4 write their own minimal smoke checks inline
under `if __name__ == "__main__":` with a scenario of their choosing.
