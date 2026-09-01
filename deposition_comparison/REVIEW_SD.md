# REVIEW_SD.md

## Review of slowing_down.py (agent D)

Independent adversarial review of
`/home/user/ascot5/deposition_comparison/slowing_down.py` (agent A).
Method: re-derived every closed form from `dv/dt = -(v/tau_se)(1 + v_c^3/v^3)`
with independent numpy/scipy, cross-checked against high-resolution quadrature,
ran the actual module through edge-case and end-to-end recomputation tests, and
compared the Coulomb-log treatment against ASCOT's `mccc_coefs_clog`
(`src/simulate/mccc/mccc_coefs.h:228-256`). Scratch scripts:
`scratchpad/verify_sd.py`, `scratchpad/verify_module.py` (session scratchpad,
not committed).

### Verification of the deliberate contract deviation (tau_se)

Agent A moved `sqrt(m_e)` from the numerator (as written in INTERFACES.md
Phase 2) to the denominator. **A is right; the contract text is a typo.**

- Dimensional analysis of the contract form
  `3 (2pi)^{3/2} eps0^2 m_b sqrt(m_e) (kTe)^{3/2} / (Z^2 e^4 n_e lnL)`:
  `[C^4 J^-2 m^-2][kg][kg^{1/2}][J^{3/2}] / [C^4 m^-3] = kg^{3/2} m J^{-1/2}
  = kg * s` — not a time (and numerically ~1e-31 x the correct value).
  With `sqrt(m_e)` in the denominator the units reduce exactly to seconds.
- Numerically, the implemented SI form reduces to the NRL practical formula
  `tau_se = C * A_b Te[eV]^{3/2} / (Z^2 n_e[cm^-3] lnL)` with exact
  coefficient `C = 3 (2pi)^{3/2} eps0^2 amu e^{3/2} / (sqrt(m_e) e^4) * 1e-6
  = 6.2722e8`, vs NRL's rounded `6.27e8`: relative difference **3.4e-4** at
  every (A, Te, ne) I tried. Confirmed independently of the module and via the
  module's own check 5.

### Independent derivation results (all closed forms confirmed)

From the continuity equation `d/dv[N v_dot] = -S delta(v - v0)` with
`v_dot = -(v^3+v_c^3)/(tau_se v^2)`:

- `N(v) = S tau_se v^2/(v^3+v_c^3)` on `(v_min, v0)` — matches (Stix Eq. 17).
- Particle content `(S tau/3) ln((v_b^3+v_c^3)/(v_a^3+v_c^3))` — trivially the
  antiderivative; quadrature agreement <= 2.4e-16 rel.
- Energy content: `int E N(E) dE = (S tau m_b/2) int v^4/(v^3+v_c^3) dv
  = (S tau m_b v_c^2/2)[G]`, `G(u) = u^2/2 - F(u)` since
  `u^4/(1+u^3) = u - u/(1+u^3)` — confirmed; quadrature rel <= 1.8e-14.
- Ion fraction: `dE/dt = -(2E/tau)(1 + v_c^3/v^3)`, ion share
  `v_c^3/(v^3+v_c^3) = 1/(1+(E/E_c)^{3/2})` — verified as an identity to
  1e-12 rel.
- `P_i/w = int_{Emin}^{E0} dE/(1+(E/E_c)^{3/2}) = 2 E_c [F(u)]` via
  `E = E_c u^2` — confirmed against scipy.quad at 6 triples spanning
  `E0 < E_c` (100/162.9), `E0 >> E_c` (100/5), `E0 << E_c` (100/900),
  `Emin -> E0` (Emin = 99.999, E0 = 100), third-energy (25/162.9), and
  (1000/30): max rel **5.4e-11** (that worst case is the vanishing
  `Emin -> E0` interval; all others <= 5e-15).
- Primitive `F(u) = (1/6)ln((u^2-u+1)/(u+1)^2) + (1/sqrt3) arctan((2u-1)/sqrt3)`:
  finite-difference derivative matches `u/(1+u^3)` to 5e-8 (FD-limited) on
  u in [0.001, 20]; definite integrals match quad to <= 2e-16. Continuous on
  u >= 0 as claimed (arctan argument never crosses a branch).
- `tau_th = (tau/3) ln((v0^3+v_c^3)/(v_min^3+v_c^3))` — same integral as the
  density; confirmed.
- `v_c^3 = (3 sqrt(pi)/4)(m_e/n_e) sum_i(n_i Z_i^2/m_i) v_te^3`,
  `v_te = sqrt(2kTe/m_e)`: reproduces the standard
  `E_c = 14.8 A_b Te [sum n_i Z_i^2/(n_e A_i)]^{2/3}` to ratio 0.998 (the 14.8
  is itself rounded); pure-D check gives the textbook `E_c ~ 18.6 Te`.

### Units audit (line-by-line)

- `slowing_down.py:240-241` tau_se: `kTe = te*E_CHARGE` [J]; result [s]. OK.
- `:243-251` v_c [m/s], `E_c = m_b v_c^2/2` [J]. The species sum uses
  `ni/ne_raw` (both from the same `profiles` call) so the 1e10 clamp cannot
  skew the concentration ratio — correct and careful.
- `:383-387` counts = `w tau/3 * log1p(...)` = [1/s][s] = particles;
  `f_E = counts/(V dE_keV)` = [1/(m^3 keV)]. OK, and the `log1p` form is a
  good accuracy choice for thin bins.
- `:394` e_marker = `[1/s][s][kg][m^2/s^2]` = J; `/V` -> J/m^3. OK.
- `:398` pi_marker = `E_c[J] * w[1/s]` = W; `/V` -> W/m^3. OK.
- `pe` computed as `ptot - pi_` per bin: the identity
  `pe + pi_ = sum w (E0-Emin)/V` holds by construction (verified 1.8e-16 on a
  5000-marker mixed run).

### End-to-end recomputation through the module

Single marker (rho = 0.51, E0 = 100 keV, w = 1e18/s), everything recomputed
from scratch in numpy (profiles at bin center rho = 0.5, lnL = 16.99,
tau_se = 0.7545 s, E_c = 122.37 keV):

| quantity | module | independent | rel |
|---|---|---|---|
| density | 6.28279790e15 m^-3 | 6.28279790e15 | 2.6e-15 |
| energy_density | 6.19486649e1 J/m^3 | quad | 9.2e-16 |
| pi_ | 4.90357551e2 W/m^3 | quad | 8.1e-16 |
| tau_th | 0.123025 s | 0.123025 | 2.0e-15 |

Module sanity block (`python -m deposition_comparison.slowing_down`): all 5
checks pass (power identity 5.7e-16; f_E-vs-closed-form density 5.5e-16;
tau_se-vs-NRL 3.4e-4; cold/hot split limits 0.962/0.983). Scenario numbers
(rho = 0.3: lnL_e = 17.13, tau_se = 0.860 s, E_c = 162.9 keV,
tau_th = 0.100 s) all reproduce independently.

### Edge cases (tested through the actual module)

- `E0 <= emin_keV` (15, 20, 5 keV) and `E0 = -5` keV: all outputs exactly zero
  and finite (the `E0_J = max(., emin_J)` clamp plus `w = 0` masking works).
- `rho >= 1.0` (1.0, 1.5): excluded, total matches an in-grid-only rerun
  bitwise. `rho` exactly at an interior edge goes to the right-hand bin
  (searchsorted side="right") — consistent.
- Empty bins and an empty (n = 0) marker array: zeros, no NaN (shell volumes
  are strictly positive so the `/volumes` divisions are safe).
- **f_E clipping**: marker with E0 = 65 keV (mid-bin [64, 66)): integral of
  f_E dE equals density to 0 ulp; all bins above 65 keV exactly zero; the
  partial bin holds exactly the closed-form [64, 65] content (rel 6.6e-15).
- Vacuum-ish edge (ne_edge = 1 m^-3, te_edge = 0.01 eV): finite (clamps + the
  lnL >= 5 floor work).
- 5000-marker mixed run: `sum(pe+pi_) dV` equals `sum w (E0-Emin)` over
  in-grid, above-threshold markers to 1.8e-16; f_E/density totals agree to
  8.3e-16.

### Findings

**BLOCKING: none found.** The physics, closed forms, units, and edge handling
are all correct within the evidence above; the one contract deviation is a
genuine contract typo, verified dimensionally and numerically.

**MINOR 1 — docstring understates the E_c systematic vs ASCOT
(slowing_down.py:117-126).** The "Coulomb-logarithm caveat" claims the
single-lnL approximation shifts E_c by "~5%". Recomputing ASCOT's per-species
`mccc_coefs_clog` (Debye length over all species, bmin = max(classical,
quantum) with `vbar = va^2 + 2Tb/mb`) for a 100 keV D ion at the scenario
rho = 0.3 plasma gives lnL_e = 17.71, lnL_D = 22.08, lnL_C = 20.76, i.e. an
effective `lnL_i/lnL_e ~ 1.24`, so ASCOT's effective E_c is
`1.24^{2/3} ~ 1.15x` A's — a **~15%** shift, not ~5%. Consequence for the
comparison: `P_i/(P_e+P_i)` for the 100 keV component moves from 0.818 (A) to
~0.846 (ASCOT-like), i.e. ~0.03 absolute — comfortably inside agent C's 0.15
split tolerance, so no code change is required, but the docstring number
should read ~15% (suggested fix: correct the sentence).

**MINOR 2 — tau_th can be a tiny negative number in the degenerate no-source
case (slowing_down.py:406-411).** With zero markers (or all markers at/below
Emin), `e_full_J` falls back to `emin_J`, but `v_full3 = (2E/m)^1.5` and
`v_min3 = (sqrt(2E/m))^3` round differently, so `log1p` receives ~-1e-16 and
`tau_th ~ -1.6e-17 s` (measured). Harmless (degenerate case, magnitude one
ulp), but computing `v_min3 = (2.0*emin_J/m_b)**1.5` (same expression shape as
`v_full3`) would make the cancellation exact and tau_th >= 0 always.

**MINOR 3 — f_E silently truncates markers born above the energy grid.** A
marker with E0 = 120 keV on the shared 20-110 keV grid contributes its full
content to `density`/`energy_density`/`pe`/`pi_` but f_E misses the
[110, 120] tail (measured: 10.8% of that marker's particles), so
`int f_E dE < density` for such markers. Unused in this comparison (max birth
energy is 100 keV < 110), and the contract fixes both grids, so this is
robustness-only; a one-line docstring warning (or debug assert) would prevent
surprise if the grids are ever reused.

**NOTE 1 — expected lnL_e systematic vs ASCOT (affects tau_se, density,
stored energy).** A's NRL `24 - ln(sqrt(ne_cm3)/Te)` is 3.3-4.0% below
ASCOT's clog across the profile (17.14 vs 17.71 at rho = 0.3; 15.5 vs 16.2 at
rho = 0.98). Since tau_se ~ 1/lnL, the analytic tau_se/density/stored-energy
will run **~3-4% above** ASCOT from this alone — small against the 25%
(stored energy) and 35% (profile L1) tolerances. Direction: analytic high.

**NOTE 2 — combined direction of systematics for the phase-2 comparison.**
lnL_e (NOTE 1) pushes analytic density/energy high by ~3-4%; the E_c
underestimate (MINOR 1) pushes the analytic P_i share LOW by ~0.03 absolute
(P_e correspondingly high); orbit losses/orbit width in ASCOT push ASCOT's
absolute powers low and smear profiles. None threatens the stated tolerances.

**NOTE 3 — bin-center profile evaluation.** Per contract, tau_se/v_c/E_c are
evaluated at rho-bin centers, while ASCOT uses the actual particle position;
with 25 bins and the alpha = 1.5 profiles this is a sub-percent effect except
in the outermost bins where profiles are steep — covered by the profile-shape
tolerance.

**NOTE 4 — "full energy" for tau_th** is taken as the max birth energy over
all markers. The contract ("full-energy thermalization time") is ambiguous;
this reading is reasonable and self-consistent (equals the injector full
energy whenever any full-energy marker is present).

### Verdict

**APPROVE-WITH-MINOR.** No blocking issues: every closed form was re-derived
and matches independent quadrature to <= 5e-11 (typically <= 1e-14); units are
consistent end-to-end; all edge cases (sub-threshold, out-of-grid, empty bins,
empty input, mid-bin E0 clipping, vacuum) behave correctly; the tau_se
contract deviation is verified correct (contract typo). The three MINOR items
are a docstring number (~5% should be ~15%), a one-ulp negative tau_th in a
degenerate case, and a robustness doc note — none changes phase-2 results.

### Response (agent A)

All three MINOR findings addressed in `slowing_down.py` (no behavior change
to any phase-2 result; sanity block re-run, all checks pass):

- **MINOR 1 (docstring E_c systematic):** the "Coulomb-logarithm caveat"
  section now carries the reviewer's recomputed ASCOT `mccc_coefs_clog`
  numbers (lnL_e = 17.71, lnL_D = 22.08, lnL_C = 20.76 at rho = 0.3;
  effective lnL_i/lnL_e ~ 1.24), states that this module's E_c is ~15% LOW
  vs ASCOT's effective critical energy (not the previous "~5%"), and gives
  the resulting P_i/(P_e+P_i) shift of ~0.03 absolute (0.818 vs ~0.846)
  against the 0.15 split tolerance.
- **MINOR 2 (one-ulp-negative tau_th):** `v_min3` in the tau_th block is now
  computed with the identical expression shape as `v_full3`
  (`(2.0*emin_J/m_b)**1.5` instead of `sqrt(...)**3`), so the degenerate
  `e_full_J == emin_J` case cancels exactly; a `jnp.maximum(., 0.0)` guard
  additionally makes `tau_th >= 0` a hard invariant. New sanity check 6
  runs the module with every marker below Emin and asserts tau_th finite
  and >= 0 (measured min exactly 0.0) and all other outputs exactly zero.
- **MINOR 3 (f_E energy-grid truncation):** added a `.. warning::` block to
  the `slowing_down` docstring stating explicitly that f_E truncates
  content outside `[e_edges_keV[0], e_edges_keV[-1]]` while
  density/energy_density/pe/pi_ keep the full [Emin, E0] content (so
  `sum(f_E dE) < density` for, e.g., a 120 keV birth on the 20-110 keV
  grid, ~11% of that marker's particles), with the grid-choice rule to
  avoid it. No behavior change, per the finding.

Sanity output after the edits: checks 1-5 identical to the reviewed run
(power identity 5.7e-16, f_E-vs-closed-form 5.5e-16, quadrature
cross-checks 1.1e-11 / 6.2e-11, split limits 0.962 / 0.983, tau_se-vs-NRL
3.4e-4); new check 6 passes; scenario numbers at rho = 0.3 unchanged
(tau_se = 0.8604 s, E_c = 162.91 keV, tau_th = 0.1005 s).
