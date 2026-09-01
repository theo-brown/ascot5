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

## Review of ASCOT reference + comparison (agent E)

Adversarial integration review of `run_ascot_reference.py` (agent B) and
`test_slowing_down.py` / `run_sd_comparison.py` (agent C). Method: every claim
re-derived numerically from the committed `sd_reference.npz`, the read-only
`bbnbi_ref/ascot.h5` (endstate, raw rho5d dist, libascot post-processing —
no ASCOT/BBNBI binary was executed), and the C/a5py sources. Scratch:
`scratchpad/check_npz.py`, `check_analytic.py`, `check_h5*.py`,
`check_moments.py` (session scratchpad, not committed).

### Verification of agent B's deliverable

**npz self-consistency (all pass):**

- `pmax` typo correction verified: `sqrt(2 m_D 110 keV) = 1.0857544e-20`
  kg m/s, bit-identical to the meta's `DIST_PMAX_SI`; the contract's
  "1.1e-19" is off by 10x and would have wasted ~90% of the momentum bins.
  B's deviation is correct.
- Weight rescale exact: `sum(birth_weight) = w_total = 8.322e19 /s` (rel 0);
  `3086 x 32.40441 = 100000.0` exactly (BBNBI weights are uniform), subset
  mix 1728/903/455 full/half/third vs 55/30/15% nominal. Birth power
  1.0072 MW: consistent with BBNBI shinethrough = 0 (phase-1 summary) plus
  +0.7% subset energy-mix noise.
- Integrals: `sum((pe+pi) V) = 777.686 kW`, `sum(energy_density V) =
  65.505 kJ`, Pe fraction 0.23684 — all equal the meta and C's baked-in
  numbers; identity `sum w (E0-20keV) = 740.538 kW`, ASCOT/identity = +5.02%.
- E-xi extraction closes EXACTLY: `sum(density V) - sum(f_E dE V) =
  7.298e16` particles = raw-dist content above 110 keV (6.919e16, the
  energy-diffusion upscatter tail) + below 20 keV (3.79e15). The <= 3.2%
  per-bin `f_E`-vs-density deviations are that windowing, not a conversion
  bug. `ekin_edges` eV interpretation confirmed numerically.
- Moments: re-running `getdist_moments` reproduces the npz `pe`/`pi`
  bit-for-bit; moment-vs-analytic shell volumes max rel dev 4.948e-4
  independently reconfirmed (B's <5e-4 claim holds). No unit slips, no bin
  off-by-ones, no charge/time double counting found anywhere in `extract()`.

**Steady-state normalization (independent route, passes):** dist fast-ion
inventory `sum(density V) = 6.7635e18` vs `sum_i w_i x mileage_i = 6.7735e18`
from the endstate — agreement to 0.15% (residual = dwell outside the dist
box: the >|pmax| corner leak and rho>1 excursions). Equivalently
inventory/source = 0.0813 s = the weighted mean thermalization mileage
0.0814 s, vs the analytic mean dwell 0.1103 s (+36%, the documented
lnL/E_c systematics — REVIEW_SD.md NOTES 1-2). The accumulated rho5d dist
IS weight x residence time; no spurious time-bin (0.5 s) or mileage-cap
factor anywhere.

**Wall-hit deviation (ENDCOND_WALLHIT=0) — mechanism verified in source:**

- `simulate_gc_adaptive.c:216` compares `fabs(p0.phi-p.phi)` directly
  against `ada_max_dphi`; `hdf5_options.c:81` reads ADAPTIVE_MAX_DPHI with
  NO deg2rad conversion (unlike the DIST phi/theta options at :302-303,
  :474-475), so the default 2.0 permits 114.6-degree toroidal steps.
- The phase-1 h5 wall is `wall_3D` (`run_bbnbi_reference.py:84-87`), and
  `wall_3d_hit_wall` tests the straight 3D Cartesian chord of the step: a
  115-degree chord at R ~ 6.9 m passes through R_mid ~ 3.7 m < inner wall
  radius 4.1 m — the spurious inner-wall hit is geometrically real.
- `WIENERSLOTS 20` (`ascot5.h:111`) and `ERR_WIENER_ARRAY` on overflow
  (`mccc_wiener.c:115,183`) confirmed; the dphi cap forces rejections via
  `hnext = -hin/dphi` (`simulate_gc_adaptive.c:220`), each parking a
  future-time Wiener entry — the overflow account is consistent.
- Physics stake of neglecting the wall: 3.7% of source weight (3.2% of
  power) is born at rho > 0.9 (max birth rho 0.988); banana HALF-widths
  (Bpol = (d/2)/R from psi = d^2/4, Bphi = 32.86/R; effective q = 8.4-9.4)
  are Delta-rho = 0.09-0.20 for 33-100 keV D, so true wall losses at a
  rho ~ 1.05 wall would be O(1%) of power — far inside every tolerance.
  All 3086 markers end EMIN, max mileage 0.360 s < the 0.4 s cap.
  The deviation is sound and defensible.

**MAX_CPUTIME=480 s:** acceptable as a coarse guard given the documented
per-lane accounting (10 s -> CPUMAX at 17 s wall); the real cap is the
external `timeout 510`, and the endcond summary confirms no truncation
(main run 131 s wall). **numpy-2 shims:** both reviewed — semantics match
the removed/refused numpy-1 behavior, no-ops otherwise, vendored a5py
untouched. Calibration logic (64 -> extrapolate -> floor/cap 500/4000,
one documented relaxation fallback) matches the contract.

### Verification of agent C's deliverable

Suite runs 7 passed / 1 skipped; every measured number in the docstrings
reproduces exactly (Pe +0.517, Pi -0.223, split +0.1405, sum -0.048,
stored +0.194, L1 0.624, analytic peak rho 0.74, birth peak 0.66, axis bins
46% of ASCOT peak).

**Channel bands (0.80 / 0.32) — derivation verified, adjustment legitimate:**
with frac_s = 0.237, split +/-0.15 abs and sum +/-10% give exactly
Pe in [-67%, +80%], Pi in [-28%, +32%] (recomputed independently). The
contract's "each within 20%" was indeed internally inconsistent with its own
split tolerance for an ion-dominated plasma (a 0.15 split shift alone is a
63% Pe-channel move). C kept the two binding physics gates and derived the
channel bands from them — this is a principled resolution, not
"measured plus margin". The real gate, the contract-original split 0.15,
passes at 0.1405.

**Density L1 0.70 — orbit-width claim verified quantitatively:** my
independent banana-width estimate gives Delta-rho 0.09-0.20 (above), C's
"0.1-0.15" is right (slightly conservative at full energy); |grad psi| =
0.6 Wb/rad/m at rho 0.6 confirmed. Decisive check that this is NOT a
normalization bug: the SHAPE-ONLY L1 (both profiles normalized to unit sum)
is 0.631 ~ the raw 0.624 — the miss is dominated by profile shape (smearing
+ inward shift; ASCOT deposition-power centroid rho 0.506 vs birth 0.636),
not by the +36% amplitude offset, and the amplitude offset itself decomposes
into the reviewed lnL/E_c systematics. Tolerance justified.

### Findings

**BLOCKING: none.** All comparison results stand; no normalization or unit
bug found on either side.

**MINOR C1 — the documented explanation of the +5% power excess is
quantitatively wrong (test_slowing_down.py module docstring items around
lines 23-30, test_total_deposited_power_sum, and the "~+0.016" clause in
test_electron_ion_split; also B's "5% slack: MC-noise" comment at
run_ascot_reference.py:468).** The claim is that markers overshoot the
20 keV endcond and the moments capture that sub-threshold deposition. The
endstate refutes the magnitude: mean end energy is 19.80 keV (min 17.3), so
`sum w (20keV - E_end)` = **2.7 kW** — under 10% of the 37.1 kW excess.
The actual cause, verified in source and closed numerically: the
`electron/ionpowerdep` moments are **gross drag** — `int f m v K` with only
the friction coefficient K (`a5py/ascot5io/dist.py:835-900`) — and do not
subtract the collisional energy-diffusion return flux `int f m Dpar`, which
I evaluated over the accumulated dist via libascot: **45.9 kW** (8.7 e /
37.2 i). Closure: gross 777.7 - return 45.9 = 731.7 kW vs the marker
bookkeeping net `sum w (E0 - E_end)` = **743.3 kW** (1.6%, the residual
being the 2.7 kW undershoot + out-of-box dwell + bin-center coefficient
discretization). Net-vs-net the codes agree to **+0.4%** (743.3 vs 740.5) —
a stronger result than the docstrings claim. No test outcome changes (the
10% band covers gross-vs-net, and the net split moves only 0.237 -> 0.240),
but the docstrings would misdirect anyone tightening tolerances later:
fix the wording in both files (C's three passages, B's one comment).

**MINOR C2 — the tau_th cross-check is dead code that need not be.**
`test_tau_th_vs_ascot_mileage` probes meta keys (`mean_mileage_s`, ...) that
`run_ascot_reference.py` never writes, so the factor-2 sub-check permanently
skips. The needed quantity is derivable from the npz alone: mean
thermalization time = inventory/source = `sum(density V)/sum(w)` = 0.0813 s
(verified equal to the h5 weighted mean mileage, 0.0814 s). Against the
core tau_th 0.138 s the ratio is 1.69 — the check would pass. Implement it
via N/S and drop the meta-key probing (or have B add the mileage to meta).

**MINOR B1 — extract() does not gate on endcond composition.** A
CPUMAX/TLIM-truncated run (a quietly non-steady-state dist, biased low)
would pass every verification check and write the npz; only the meta would
tell. Add an assertion such as EMIN count >= 95% of n_markers next to the
existing checks (this run: 3086/3086 EMIN, so no impact on the artifact).

**NOTE 1 — npz `density`/`energy_density` are not the integral of `f_E`.**
They include the >110 keV upscattered tail (1.02% of particles; energy
diffusion pushes ~100 keV markers above the birth energy) and the sub-20 keV
dwell (0.06%), while `f_E` is windowed to [20, 110] keV. Per-bin
discrepancies reach 3.2%. Consistent and harmless for the tests as written
(C's f_E-vs-density identity test is analytic-only), but undocumented in
the npz meta — one sentence there would prevent surprise.

**NOTE 2 — "traced orbits stay at rho <= 0.92" (write_options docstring)
overstates confinement for the ensemble.** Births extend to rho 0.988 and
the accumulated dist has (tiny) content in the [0.92, 1.0] bins. The correct
defense of WALLHIT=0 is the quantified smallness above (~1% power at stake),
which B's own numbers support — soften the sentence.

**NOTE 3 — small docstring slips in C's files.** (a) The module header says
the ASCOT density peak is at rho 0.58 (correct, bin 14) but item 2 says the
peak "shifts ... to 0.50" — 0.51 is the deposition-power centroid, not the
density peak. (b) In test_electron_ion_split, the clause blaming the Stix
"v >> v_ti least accurate" regime is directionally dubious: at
x = v/v_ti ~ 1.7-2.2 the Chandrasekhar G(x) is BELOW its 1/(2x^2)
asymptote, i.e. the true ion drag is weaker, which would push ASCOT's Pe
fraction UP, not down. The residual is instead carried by the
orbit-shifted deposition into the hotter core (centroid rho 0.51 vs 0.64,
local E_c x ~1.4 — listed in the same docstring) plus the gross-drag moment
definition (MINOR C1). Trim the v_ti clause.

**NOTE 4 — thin split margin.** The one contract-original gate, the Pe
split, passes at 0.1405 vs 0.15. It is genuinely systematic (single-lnL
E_c ~ +0.04, hotter-core orbit sampling ~ +0.08, moment definition ~ +0.01,
threshold undershoot small) — but any future change to the analytic lnL
convention or plasma scenario can push it over. Expected, honest, watch it.

### Verdicts

**Agent B (run_ascot_reference.py): APPROVE-WITH-MINOR.** Every extractable
number was independently reproduced (weights exact, moments bit-identical,
volumes 4.9e-4, steady-state normalization confirmed to 0.15% via the
mileage route, E-xi conversion content-exact); the pmax correction and both
numpy-2 shims are right; the WALLHIT=0 deviation is verified line-by-line
in the C sources and its physics neglect bounded at O(1%) of power. MINOR:
no endcond gate in extract() (B1) and one mislabeled "MC-noise" comment
(shared with C1).

**Agent C (test_slowing_down.py + run_sd_comparison.py): APPROVE-WITH-MINOR.**
All reported metrics reproduce exactly; the channel-band tolerances are
derived from the contract's own binding gates (not fitted to the data); the
L1 relaxation is backed by a banana-width estimate this review confirms
(shape-only L1 0.631 proves the miss is orbit-shape, not normalization);
the suite is green (7 passed / 1 skipped). MINOR: the +5% power-excess
explanation in the docstrings is quantitatively wrong (real cause:
gross-drag moments vs net energy transfer; C1) and the tau_th cross-check
is an eternally-skipping stub although the npz already contains what it
needs (C2). Neither changes any pass/fail result.

### Response (agent C)

All findings touching my files addressed in `test_slowing_down.py` and
`run_sd_comparison.py` (no tolerance value changed; suite now 8 passed /
0 skipped):

- **MINOR C1 (wrong +5% explanation):** all three passages rewritten — the
  module-docstring reference block, `test_power_identity`, and
  `test_total_deposited_power_sum` now state the verified cause: the a5py
  `electron/ionpowerdep` moments are GROSS drag (`int f m v K`, friction
  only) omitting the energy-diffusion return flux `int f m Dpar` = 45.9 kW;
  threshold overshoot is only 2.7 kW (mean end energy 19.80 keV); net-vs-net
  agreement is +0.4% (743.3 vs 740.5 kW). The `run_sd_comparison` stdout
  footnote carries the same gross-vs-net note. The figure was regenerated;
  its panel text never contained the wrong explanation, so the plots are
  unchanged.
- **MINOR C2 (dead tau_th check):** `test_tau_th_vs_ascot_mileage` now
  computes ASCOT's mean thermalization time from the npz alone via the
  steady-state identity `<tau> = sum(density V) / sum(birth_weight)` =
  0.0813 s and asserts the analytic core-bin tau_th (0.138 s, full-energy)
  is within a factor 2 (ratio 1.69, passes). The meta-key probing and the
  skip path are gone; the mileage-cap consistency assert is kept.
- **NOTE 3a (0.50 vs 0.58):** the module docstring and
  `test_density_profile_shape` now distinguish the density peak bin (0.58,
  on a broad rho ~ 0.44-0.58 plateau) from the deposition-power centroid
  (0.51 vs birth 0.64), and cite the reviewer's shape-only L1 (0.631).
- **NOTE 3b (v/v_ti clause directionally backwards):** the clause is
  removed from `test_electron_ion_split`; the residual is now attributed
  per this review — hotter-core orbit sampling (~+0.08, local E_c x ~1.4),
  single-lnL E_c (~+0.04, agent D), gross-drag moment definition (~+0.01;
  net split 0.240), threshold undershoot small.
- **NOTE 4 (thin split margin):** `test_electron_ion_split` now carries an
  explicit "Do NOT widen this tolerance" comment acknowledging the 0.1405
  vs 0.15 margin, citing agent D's quantified E_c systematic as the cause
  and flagging that a future lnL-convention or scenario change pushing it
  over is a physics regression to investigate, not a tolerance to relax.

### Response (agent B)

All findings touching `run_ascot_reference.py` addressed (code/doc changes
only; no ASCOT rerun; `sd_reference.npz` untouched, md5
4f746e8ddbf1d01daf3a428345e2688c before and after):

- **MINOR B1 (endcond gate):** `extract()` now gates on the endcond
  composition before any npz write: zero reported errors, every marker
  accounted for by a legitimate terminal condition (EMIN + TLIM + WALL ==
  n_markers), and EMIN >= 95% of n_markers; otherwise it raises with the
  full endcond/error breakdown and refuses to write. Dry-run verified: the
  committed SDMAIN run (3086/3086 EMIN, no errors) passes; a synthetic
  CPUMAX-truncated composition fails.
- **MINOR C1 (my "5% slack: MC-noise" comment + explanation text):** the
  comment at the deposited-power check now states the reviewer's verified
  mechanism - `electron/ionpowerdep` are GROSS collisional drag
  (friction-only `int f m v K`, a5py/ascot5io/dist.py), omitting the ~6%
  diffusive energy-return flux `int f m Dpar` (45.9 kW = 8.7 e / 37.2 i),
  so gross 777.7 kW vs net bookkeeping 743.3 kW; threshold overshoot is
  only 2.7 kW, and net-vs-net agreement is ~0.4%. The same definition is
  now in the `extract()` docstring and, for future runs, in
  `meta_json["pe_pi_definition"]` (the committed npz's meta keeps its old
  text, per instructions).
- **NOTE 1 (density != integral f_E dE):** documented in the `extract()`
  docstring and in `meta_json["f_E_window_note"]` for future runs: f_E is
  windowed to [20, 110] keV while density/energy_density integrate the
  full raw dist, so the ~1.02% >110 keV collisional-upscatter tail (plus
  ~0.06% sub-20 keV dwell) makes per-bin deviations of up to 3.2%.
- **NOTE 2 ("orbits <= rho 0.92" overstated):** the `write_options`
  docstring no longer rests the WALLHIT=0 defense on the 4-orbit trace;
  it now carries the quantified bound (3.7% of source weight born at
  rho > 0.9, max birth rho 0.99, tiny dist content in [0.92, 1.0], banana
  half-widths Delta-rho ~ 0.09-0.20 => true wall losses O(1%) of power),
  plus a pointer to the new endcond gate.
- **NOTE 3 (0.58-vs-0.50 density-peak slip):** not in my file -
  `extract()` prints the density peak from argmax (0.58 for the committed
  run); the slip is in agent C's module docstring.

Validation after the edits: `py_compile` clean, module imports, `--help`
parses (all three flags present), and no other file modified
(`git status`: only `run_ascot_reference.py` + the npz remain untracked,
npz hash unchanged).
