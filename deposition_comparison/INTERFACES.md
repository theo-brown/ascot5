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

---

# Phase 2: slowing-down distributions (analytic vs full ASCOT5)

Goal: starting from the SAME beam-ion birth markers, compare the analytic
steady-state slowing-down model (Stix) against a full `ascot5_main`
guiding-center slowing-down simulation. ASCOT is compute-heavy: the reference
run is calibrated first and hard-capped in wall time.

Shared conventions for this phase:
- SD rho grid: `rho_edges_sd = linspace(0, 1, 26)` (25 bins - coarser than
  deposition, for statistics).
- Energy grid: `e_edges_keV = linspace(20, 110, 46)` (2 keV bins).
- Thermalization boundary: fixed `EMIN_KEV = 20.0`. ASCOT uses
  `ENDCOND_MIN_ENERGY = 20e3` eV with `ENDCOND_MIN_THERMAL` set small enough
  (0.1) that the fixed threshold dominates everywhere; the analytic model
  integrates the slowing-down only from birth energy down to EMIN_KEV.
- Steady state: BBNBI weights are particles/s, so ASCOT's time-accumulated
  distribution IS the steady-state distribution; no extra normalization.
- The analytic model consumes the exact birth markers (rho, energy, weight) of
  the subset ASCOT simulates, carried in the npz below - deposition-method
  differences cancel by construction.

## `slowing_down.py` (agent A) - analytic model, JAX

```python
class SlowingDownResult(NamedTuple):
    rho_edges: jnp.ndarray       # (nrho+1,)
    e_edges_keV: jnp.ndarray     # (nE+1,)
    f_E: jnp.ndarray             # (nrho, nE) steady-state fast-ion energy
                                 # distribution [1/(m^3 keV)]
    density: jnp.ndarray         # (nrho,) fast-ion density [m^-3]
    energy_density: jnp.ndarray  # (nrho,) fast-ion energy density [J/m^3]
    pe: jnp.ndarray              # (nrho,) power to electrons [W/m^3]
    pi_: jnp.ndarray             # (nrho,) power to ions [W/m^3]
    tau_th: jnp.ndarray          # (nrho,) full-energy thermalization time [s]

def slowing_down(birth_rho, birth_energy_keV, birth_weight, eq, plasma,
                 rho_edges, e_edges_keV, *, mass_amu, znum_beam=1,
                 emin_keV=20.0) -> SlowingDownResult
```
Physics (all evaluated at bin-center rho from `physics.profiles`; document each
formula with a reference in the docstring; jit + vmap over markers/bins):
- Coulomb logarithm: NRL formulary, electron-ion form
  `lnL = 24 - ln(sqrt(ne_cm3)/Te_eV)` for fast-ion-electron collisions and
  a documented ion-ion form for fast-ion-ion collisions (state formulas;
  reviewer will verify). Small differences vs ASCOT's internal clog are
  acceptable and covered by comparison tolerances.
- Spitzer slowing-down time on electrons `tau_se` (SI):
  `tau_se = 3 (2 pi)^{3/2} eps0^2 m_b sqrt(m_e) (k Te)^{3/2}
            / (Z_b^2 e^4 n_e lnL_e)`.
- Critical velocity/energy: `v_c^3 = (3 sqrt(pi)/4) (m_e / n_e)
  sum_i(n_i Z_i^2 / m_i) v_te^3` with `v_te = sqrt(2 k Te / m_e)`; `E_c` is
  the kinetic energy of the beam species at `v_c`.
- Steady-state distribution of a source S [1/(s m^3)] at birth speed v0:
  `N(v) = S tau_se v^2 / (v^3 + v_c^3)` for v in (v_min, v0), 0 outside;
  convert to per-energy via `N(E) = N(v) / (m_b v)`.
- Fast-ion density/energy density: integrals of N(E) (analytic or accurate
  quadrature on the energy grid).
- Heating split: ion fraction at energy E is
  `1 / (1 + (E/E_c)^{3/2})`; `P_i = sum_markers w * int_{Emin}^{E0}
  frac_i(E) dE`, `P_e` likewise with `1 - frac_i`; per volume via
  `common.shell_volumes`. Total `P_e + P_i` must equal
  `sum w (E0 - Emin)` exactly (markers hand back Emin to the bulk at
  thermalization; do NOT count it as deposited).
- `tau_th = (tau_se / 3) ln(1 + (E0/E_c)^{3/2} ... )` evaluated from Emin to
  E0 (use the exact integral, not the E_min=0 form).
- Pitch is ignored (no pitch-angle scattering, no orbit effects) - document.

Sanity block (`python -m deposition_comparison.slowing_down`): uniform test
source at rho=0.05..0.15, checks: N(E) >= 0; density equals the closed-form
`S tau_se/3 ln(1+(v0/vc)^3)`-type expression evaluated between Emin and E0;
`P_e + P_i = sum w (E0 - Emin)` to 1e-12 rel; E0 >> Ec limit gives P_e
dominant, E0 << Ec limit gives P_i dominant.

## `run_ascot_reference.py` (agent B) - full ASCOT5 run

Builds on the phase-1 `bbnbi_ref/ascot.h5` (BBNBI run with 100k markers,
div=0.02 scenario). Steps:
1. `make ascot5_main -j4` if `build/ascot5_main` is missing.
2. Add dummy `Boozer` and `MHD_STAT` inputs (required by ascot5_main).
3. Marker subset: random `n_sd` ionized markers from the BBNBI run via
   `getstate_markers("gc", ids=...)`, weights scaled by
   (total ionized weight / subset weight sum) so the subset carries the full
   source rate; write as "gc" marker input.
4. Options: SIM_MODE=2, ENABLE_ADAPTIVE=1, orbit following + Coulomb
   collisions on, ENDCOND_ENERGYLIM=1 with MIN_ENERGY=20e3/MIN_THERMAL=0.1,
   SIMTIMELIM + MAX_MILEAGE=0.4 (safety net), CPUTIMELIM + MAX_CPUTIME as an
   extra guard, ENABLE_DIST_RHO5D=1 on the shared rho/E-compatible grid
   (rho 25 bins on [0,1], theta/phi/time/charge 1 bin, ppar/pperp sized for
   110 keV D: |p| <= 1.1e-19 kgm/s check numerically ~ sqrt(2 m E)),
   with enough momentum bins (>= 100 x 50) for a clean E-xi conversion.
5. CALIBRATE: run `timeout 510 build/ascot5_main --in=ascot --n?` no - marker
   count is fixed by the marker input group, so write a SMALL marker group
   first (64 markers), run, measure wall time, extrapolate linearly, then
   choose `n_sd` so the main run finishes in <= 8 minutes wall
   (`timeout 510`), floor 500 / cap 4000 markers. Then write the full marker
   group + rerun. Report both timings.
6. Extract to `deposition_comparison/sd_reference.npz`:
   - `rho_edges`, `e_edges_keV` (the shared grids)
   - `f_E (nrho, nE)` [1/(m^3 keV)]: rho5d dist -> exi conversion
     (`getdist("rho5d", exi=True, ekin_edges=<J or eV per a5py API>)`),
     integrate over theta/phi/pitch/charge/time, divide by shell volume
     (`common.shell_volumes` values are exact for this equilibrium) and by
     energy bin width.
   - `density (nrho,)`, `energy_density (nrho,)`: momentum-space integrals of
     the same dist.
   - `pe (nrho,)`, `pi (nrho,)` [W/m^3]: `getdist_moments(dist,
     "electronpowerdep", "ionpowerdep")` using libascot with inputs
     initialized (bfield+plasma); resample/verify the moment grid matches
     rho bins; if the moment machinery fails, fall back to writing whatever
     IS extractable and document loudly in the npz meta and your report.
   - `birth_rho`, `birth_energy_keV`, `birth_weight`: the subset's INISTATE
     (BBNBI birth coordinates of exactly the simulated markers, scaled
     weights) - the analytic model's source.
   - meta: n_markers, wall time, endcond summary counts, option values.
7. The h5 stays in bbnbi_ref/ (gitignored); the npz is committed.

## `test_slowing_down.py` + `run_sd_comparison.py` (agent C)

- Loads `sd_reference.npz`; runs `slowing_down()` on the npz birth arrays with
  the same scenario plasma (import `make_scenario` from test_comparison,
  div irrelevant).
- Tests (pytest, no ASCOT execution - npz only; skip cleanly with a clear
  message if npz missing so CI without the binary still passes):
  - analytic internal consistency (the sanity checks from agent A's block);
  - `P_e + P_i` volume-integrals agree between analytic and ASCOT within 20%
    each, and their SUM within 10% (orbit losses make ASCOT lower);
  - fast-ion stored energy (volume-integrated) within 25%;
  - density profile shape: rel-L1 < 0.35 (orbit width smears);
  - electron/ion split: analytic `P_e/(P_e+P_i)` within 0.15 absolute of
    ASCOT's.
  Tolerances are physics-motivated starting points: if a comparison fails,
  INVESTIGATE (plot, check units/normalization) before loosening; report
  any tolerance you change and why.
- `run_sd_comparison.py`: figure `comparison_sd.png` with 4 panels:
  (a) n_fast(rho), (b) P_e and P_i profiles (both methods),
  (c) volume-integrated f(E) spectra, (d) f_E at one core rho bin;
  annotate integrated powers and stored energy; stdout summary table.
