"""Full ASCOT5 guiding-center slowing-down reference simulation.

Phase-2 companion of :mod:`run_bbnbi_reference`: starting from the beam-ion
birth markers of the completed BBNBI run in ``bbnbi_ref/ascot.h5`` (100k
markers, active run desc JAXREF), run ``ascot5_main`` with Coulomb collisions
until the fast ions slow down to ``EMIN_KEV`` and extract the steady-state
slowing-down distribution into ``sd_reference.npz`` for comparison against the
analytic model in :mod:`slowing_down`.

Pipeline (each stage is idempotent / individually skippable):

1. Build ``build/ascot5_main`` if missing (``make ascot5_main -j4``).
2. Regenerate ``bbnbi_ref/ascot.h5`` via ``run_bbnbi_reference`` if it is
   missing or holds no BBNBI run.
3. Add dummy ``Boozer`` and ``MHD_STAT`` inputs (required by ``ascot5_main``,
   not used by the physics since ENABLE_MHD stays 0).
4. Write a slowing-down options group (documented in :func:`write_options`).
5. COMPUTE-SAFETY CALIBRATION: run a small (default 64) random marker subset
   first, measure wall time, and size the main run by linear extrapolation so
   it finishes within the wall-clock budget (<= 8 min; every ``ascot5_main``
   call is wrapped in ``timeout 510``). Marker count floor/cap: 500/4000. If
   even 500 markers extrapolates over budget the mileage/tolerances are
   relaxed once (documented) and calibration is repeated.
6. Main run with ``n_sd`` markers: a random ionized subset of the BBNBI
   endstate, weights rescaled so the subset carries the FULL ionized source
   rate (the analytic model must see the 1 MW-scale source).
7. Extraction & verification into ``sd_reference.npz``.

Steady state: BBNBI weights are particles/s, so the time-accumulated rho5d
distribution (weight x residence time) IS the steady-state distribution.

Run from the repository root::

    python -m deposition_comparison.run_ascot_reference [--calibrate-only]
    python -m deposition_comparison.run_ascot_reference --extract-only
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# numpy >= 2 compatibility shim: a5py (bbnbi5.getstate ids filter) still calls
# np.in1d, which numpy 2 removed in favor of np.isin. Unlike isin, in1d
# flattened its first argument, which a5py relies on. Patching here keeps the
# vendored a5py untouched.
if not hasattr(np, "in1d"):
    np.in1d = lambda ar1, ar2, **kw: np.isin(np.asarray(ar1).ravel(), ar2,
                                             **kw)


def _patch_a5py_numpy2():
    """Second numpy >= 2 shim, applied once a5py is importable.

    ``Dist.get`` calls ``int(f["abscissa_ndim"][:])`` on a shape-(1,) h5
    dataset, which numpy 2 refuses ("only 0-dimensional arrays can be
    converted"). Replace the method with the same code reading element [0];
    the vendored a5py stays untouched.
    """
    import unyt
    from a5py.ascot5io.dist import Dist, DistData
    if getattr(Dist.get, "_np2_patched", False):
        return

    def get(self):
        with self as f:
            histogram = np.sum(f["ordinate"][:], axis=0) * unyt.particles
            abscissa_edges = {}
            ndim = int(np.asarray(f["abscissa_ndim"][:]).ravel()[0])
            for i in range(ndim):
                abscissa = f["abscissa_vec_0" + str(i + 1)]
                name = abscissa.attrs["name_0" + str(i)].decode("utf-8")
                unit = abscissa.attrs["unit_0" + str(i)].decode("utf-8")
                try:
                    abscissa_edges[name] = abscissa[:] * unyt.Unit(unit)
                except Exception:
                    unit = unit.replace(" ", "*")
                    abscissa_edges[name] = abscissa[:] * unyt.Unit(unit)
        return DistData(histogram, **abscissa_edges)

    get._np2_patched = True
    Dist.get = get

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO      = os.path.abspath(os.path.join(HERE, ".."))
ASCOT_BIN = os.path.join(REPO, "build", "ascot5_main")
WORKDIR   = os.path.join(HERE, "bbnbi_ref")
H5FN      = os.path.join(WORKDIR, "ascot.h5")
NPZ_OUT   = os.path.join(HERE, "sd_reference.npz")
META_JSON = os.path.join(WORKDIR, "sd_run_meta.json")  # sidecar for reruns

# --- Shared phase-2 grids (INTERFACES.md) ----------------------------------
RHO_EDGES_SD = np.linspace(0.0, 1.0, 26)   # 25 rho bins
E_EDGES_KEV  = np.linspace(20.0, 110.0, 46)  # 45 x 2 keV bins
EMIN_KEV     = 20.0

# --- Group descs used in the h5 (unique tags -> unambiguous lookup) --------
BBNBI_TAG  = "JAXREF"     # phase-1 BBNBI run
CAL_DESC   = "SDCAL"      # calibration run
MAIN_DESC  = "SDMAIN"     # main slowing-down run
MRK_CAL    = "SDMRKCAL"   # calibration marker input
MRK_MAIN   = "SDMRKMAIN"  # main marker input
OPT_DESC   = "SDOPT"      # slowing-down options

SEED        = 20260901
N_CAL       = 64          # calibration marker count
BUDGET_S    = 480.0       # main-run wall-clock budget (8 min)
SAFETY      = 0.85        # use only this fraction of the budget when sizing
TIMEOUT_S   = 510         # hard cap on every ascot5_main invocation
N_FLOOR     = 500
N_CAP       = 4000

# Beam/plasma constants of the scenario (D beam, 110 keV dist headroom)
E_DIST_MAX_KEV = 110.0    # dist momentum grid sized for this energy
M_D_KG         = 2.0141 * 1.66053906660e-27


def _pmax_si():
    """|p| grid limit: sqrt(2 m_D E) for E = 110 keV, non-relativistic."""
    e_j = E_DIST_MAX_KEV * 1e3 * 1.602176634e-19
    return float(np.sqrt(2.0 * M_D_KG * e_j))  # ~1.086e-20 kg m/s


# ---------------------------------------------------------------------------
# Stage 1-2: prerequisites
# ---------------------------------------------------------------------------
def ensure_ascot5_main():
    if os.path.exists(ASCOT_BIN):
        return
    print("build/ascot5_main missing - building with 'make ascot5_main -j4'")
    subprocess.run(["make", "ascot5_main", "-j4"], cwd=REPO, check=True)


def ensure_bbnbi_run():
    """Make sure bbnbi_ref/ascot.h5 exists and holds the BBNBI run."""
    if os.path.exists(H5FN):
        try:
            from a5py import Ascot
            run = Ascot(H5FN).data[BBNBI_TAG]
            run.getstate("ids", endcond="IONIZED")
            return
        except Exception as err:
            print(f"bbnbi_ref/ascot.h5 unusable ({err}); regenerating.")
    print("Running phase-1 BBNBI reference (~4 min) ...")
    subprocess.run(
        [sys.executable, "-m", "deposition_comparison.run_bbnbi_reference",
         "--n", "100000"], cwd=REPO, check=True)


# ---------------------------------------------------------------------------
# Stage 3-4: inputs
# ---------------------------------------------------------------------------
def _destroy_by_desc(node, desc):
    """Remove all child groups of a tree node whose desc matches ``desc``.

    ``node`` may be None (parent group not present in the h5 yet).
    """
    if node is None:
        return
    for qid in list(node._qids):
        grp = node["q" + qid]
        if grp.get_desc().split(" ")[0] == desc:
            grp.destroy(repack=False)


def ensure_dummy_inputs(a5):
    """Add dummy Boozer + MHD_STAT inputs required by ascot5_main.

    ENABLE_MHD = 0, so these are never evaluated; they only need to exist.
    """
    created = False
    for name, inp in (("boozer", "Boozer"), ("mhd", "MHD_STAT")):
        node = getattr(a5.data, name, None)
        if node is None or node.active is None:
            a5.data.create_input(inp, desc="DUMMY", activate=True)
            created = True
    return created


def select_subset(bbnbi_run, n, rng):
    """Random ionized-marker subset with weights rescaled to the full source.

    The subset total weight is scaled to the TOTAL ionized weight so the
    marker population carries the full ~1 MW source rate; energy, pitch and
    position stay exactly as BBNBI produced them (guiding-center endstate =
    the ionization point). Marker time is reset to 0 (BBNBI endstate time is
    the neutral flight time, irrelevant for the steady-state slowing-down).
    """
    ids_ion, w_ion = bbnbi_run.getstate("ids", "weight", endcond="IONIZED")
    ids_ion = np.asarray(ids_ion, dtype=int)
    w_total = float(np.sum(np.asarray(w_ion)))

    pick = rng.choice(ids_ion.size, size=n, replace=False)
    sub_ids = np.sort(ids_ion[pick])

    mrk = bbnbi_run.getstate_markers("gc", ids=sub_ids)
    scale = w_total / float(np.sum(np.asarray(mrk["weight"])))
    mrk["weight"] = mrk["weight"] * scale
    mrk["time"]   = 0.0 * mrk["time"]
    return mrk, {"n_ionized": int(ids_ion.size), "w_total": w_total,
                 "scale": scale}


def write_markers(a5, mrk, desc):
    _destroy_by_desc(getattr(a5.data, "marker", None), desc)
    a5.data.create_input("gc", **mrk, desc=desc, activate=True)


def write_options(a5, max_mileage=0.4, tol_ccol=1e-1, tol_orbit=1e-8):
    """Write the slowing-down options group (activated).

    Option choices (INTERFACES.md phase-2 contract):

    - SIM_MODE=2 + ENABLE_ADAPTIVE=1: adaptive guiding-center integration.
    - Physics: orbit following + Coulomb collisions only (no MHD/atomic).
    - ENDCOND_ENERGYLIM with MIN_ENERGY = 20 keV = EMIN_KEV; MIN_THERMAL=0.1
      so the fixed 20 keV threshold dominates everywhere (0.1*Te <= 1 keV).
    - ENDCOND_SIMTIMELIM, MAX_MILEAGE=0.4 s: safety net well above the core
      slowing-down time (~0.1 s from 100 to 20 keV here).
    - ENDCOND_CPUTIMELIM, MAX_CPUTIME=480 s: coarse runaway guard only. In
      this OpenMP/SIMD build the per-marker "cputime" counter accrues close
      to the WALL time of the whole simulation (each lane adds the full
      loop time), so a small value acts as a run-wide guillotine that
      truncates markers unfinished (observed: 10 s produced CPUMAX/NONE
      endconds at 17 s wall). The real hard cap is the external
      `timeout 510` around every ascot5_main call.
    - SIMTIMELIM uses mileage; LIM_SIMTIME=1.0 is unreachable since marker
      time starts at 0 and mileage caps at 0.4 s.
    - ENDCOND_WALLHIT stays 0 (the default, per the contract's "everything
      else default/off"). This was investigated, not just assumed:
      * With WALLHIT=1 and the default ADAPTIVE_MAX_DPHI=2.0 every confined
        marker spuriously ended WALL at rho ~ 0.6 within ~5e-5 s. Cause:
        the C code compares MAX_DPHI against the per-step phi change in
        RADIANS (simulate_gc_adaptive.c), so 2.0 permits ~115 degree
        toroidal steps, and the wall check tests the straight 3D chord of
        each accepted step - such a chord cuts through the torus hole and
        "hits" the inner wall even though the orbit never leaves rho < 1.
      * Capping MAX_DPHI at 0.2 rad fixes the geometry but makes the dphi
        step-rejection fire on most steps (adaptive dt otherwise grows to
        ~1 rad/step); each rejection leaves a future-time entry in the
        collision integrator's fixed-size Wiener array (WIENERSLOTS=20 in
        ascot5.h) which then overflows: most markers abort with "Wiener
        array is full" (also with ADAPTIVE_TOL_ORBIT down to 1e-10).
      With WALLHIT=0 and default MAX_DPHI the same markers run cleanly
      (all EMIN, no errors). Physics impact is negligible here: orbits in
      this circular equilibrium are well confined (traced orbits stay at
      rho <= 0.92; the wall sits at rho ~ 1.05), so wall losses are ~0
      and the endcond summary is checked to confirm no aborts.
    - Distribution: ONLY rho5d, on the shared grid: 25 rho bins on [0,1],
      theta/phi/time/charge single bins, ppar in [-pmax, pmax] x 100 and
      pperp in [0, pmax] x 50 with pmax = sqrt(2 m_D 110 keV) ~ 1.09e-20
      kg m/s (>= 100 x 50 momentum bins for a clean E-xi conversion).
    - Everything else default/off (no orbit output, no 5D R-z dist).
    """
    from a5py.ascot5io.options import Opt

    pmax = _pmax_si()
    opt = Opt.get_default()
    opt.update({
        "SIM_MODE": 2, "ENABLE_ADAPTIVE": 1,
        "ADAPTIVE_TOL_ORBIT": tol_orbit, "ADAPTIVE_TOL_CCOL": tol_ccol,
        "ENABLE_ORBIT_FOLLOWING": 1, "ENABLE_COULOMB_COLLISIONS": 1,
        "ENDCOND_ENERGYLIM": 1,
        "ENDCOND_MIN_ENERGY": EMIN_KEV * 1e3,   # eV
        "ENDCOND_MIN_THERMAL": 0.1,
        "ENDCOND_SIMTIMELIM": 1, "ENDCOND_MAX_MILEAGE": max_mileage,
        "ENDCOND_LIM_SIMTIME": 1.0,
        "ENDCOND_CPUTIMELIM": 1, "ENDCOND_MAX_CPUTIME": 480.0,
        "ENABLE_DIST_RHO5D": 1,
        "DIST_MIN_RHO": 0.0,    "DIST_MAX_RHO": 1.0,   "DIST_NBIN_RHO": 25,
        "DIST_MIN_THETA": 0.0,  "DIST_MAX_THETA": 360, "DIST_NBIN_THETA": 1,
        "DIST_MIN_PHI": 0.0,    "DIST_MAX_PHI": 360,   "DIST_NBIN_PHI": 1,
        "DIST_MIN_PPA": -pmax,  "DIST_MAX_PPA": pmax,  "DIST_NBIN_PPA": 100,
        "DIST_MIN_PPE": 0.0,    "DIST_MAX_PPE": pmax,  "DIST_NBIN_PPE": 50,
        "DIST_MIN_TIME": 0.0,   "DIST_MAX_TIME": 0.5,  "DIST_NBIN_TIME": 1,
        "DIST_MIN_CHARGE": 0.0, "DIST_MAX_CHARGE": 2.0,
        "DIST_NBIN_CHARGE": 1,
    })
    _destroy_by_desc(a5.data.options, OPT_DESC)
    a5.data.create_input("opt", **opt, desc=OPT_DESC, activate=True)
    return {"ENDCOND_MAX_MILEAGE": max_mileage,
            "ADAPTIVE_TOL_CCOL": tol_ccol, "ADAPTIVE_TOL_ORBIT": tol_orbit,
            "ENDCOND_WALLHIT": 0,
            "ENDCOND_MIN_ENERGY_eV": EMIN_KEV * 1e3,
            "ENDCOND_MIN_THERMAL": 0.1, "ENDCOND_MAX_CPUTIME": 480.0,
            "DIST_PMAX_SI": pmax}


# ---------------------------------------------------------------------------
# Stage 5-6: running ascot5_main
# ---------------------------------------------------------------------------
def run_ascot(desc):
    """Run ascot5_main on the active inputs; return wall time in seconds.

    Every invocation is wrapped in ``timeout 510`` (compute-safety hard cap).
    """
    from a5py import Ascot
    _destroy_by_desc(Ascot(H5FN).data, desc)  # idempotent reruns

    cmd = ["timeout", str(TIMEOUT_S), ASCOT_BIN, "--in=ascot", f"--d={desc}"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=WORKDIR, stdout=subprocess.DEVNULL)
    wall = time.perf_counter() - t0
    if proc.returncode == 124:
        raise RuntimeError(
            f"ascot5_main ({desc}) hit the {TIMEOUT_S}s timeout - the run "
            "was NOT calibrated tightly enough; rerun calibration.")
    if proc.returncode != 0:
        raise RuntimeError(
            f"ascot5_main ({desc}) failed with code {proc.returncode}")
    return wall


def endcond_summary(run):
    econds, emsg = run.getstate_markersummary()
    return {name: int(cnt) for cnt, name in econds}, emsg


def calibrate(a5cls, bbnbi_tag, rng, n_cal):
    """Run the small calibration and choose n_sd by linear extrapolation.

    Returns (n_sd, t_cal, t_est, options_used). If even N_FLOOR markers
    extrapolate over budget with the contract options, the mileage cap is
    halved and the tolerances loosened (documented in ``settings``), and the
    calibration is repeated once with those relaxed options.
    """
    settings = [
        {"max_mileage": 0.4, "tol_ccol": 1e-1, "tol_orbit": 1e-8},
        # Fallback (only if N_FLOOR markers would bust the budget): shorter
        # mileage safety net + looser tolerances. Physics impact is small:
        # core thermalization takes ~0.1 s << 0.2 s.
        {"max_mileage": 0.2, "tol_ccol": 2e-1, "tol_orbit": 1e-7},
    ]
    for i, s in enumerate(settings):
        a5 = a5cls(H5FN)
        # Stale runs must go before their referenced inputs can be replaced.
        _destroy_by_desc(a5.data, MAIN_DESC)
        _destroy_by_desc(a5.data, CAL_DESC)
        optmeta = write_options(a5, **s)
        mrk, _ = select_subset(a5.data[bbnbi_tag], n_cal, rng)
        write_markers(a5, mrk, MRK_CAL)

        t_cal = run_ascot(CAL_DESC)
        ec, _ = endcond_summary(a5cls(H5FN).data[CAL_DESC])
        print(f"Calibration ({n_cal} markers, options set {i}): "
              f"{t_cal:.1f} s wall, endconds {ec}")

        # Linear extrapolation, conservatively counting all overhead as
        # per-marker cost.
        per_marker = t_cal / n_cal
        n_sd = int(BUDGET_S * SAFETY / per_marker)
        n_sd = max(N_FLOOR, min(N_CAP, n_sd))
        t_est = per_marker * n_sd
        print(f"  -> per-marker {per_marker:.3f} s, n_sd = {n_sd}, "
              f"extrapolated main run {t_est:.0f} s")
        if t_est <= BUDGET_S:
            return n_sd, t_cal, t_est, optmeta
        print("  Even the floor marker count extrapolates over budget - "
              "relaxing mileage/tolerances and recalibrating.")
    raise RuntimeError("Calibration failed to fit the wall-clock budget.")


# ---------------------------------------------------------------------------
# Stage 7: extraction
# ---------------------------------------------------------------------------
def extract(eq, meta):
    """Extract sd_reference.npz from the SDMAIN run and verify it."""
    import unyt
    from a5py import Ascot
    import a5py.physlib as physlib
    from .common import shell_volumes

    _patch_a5py_numpy2()
    a5 = Ascot(H5FN)
    run = a5.data[MAIN_DESC]
    vols = np.asarray(shell_volumes(eq, RHO_EDGES_SD))          # (25,) m^3
    mass = np.mean(run.getstate("mass", state="ini"))           # unyt amu

    # --- density & energy density: momentum integrals of the raw dist ------
    raw = run.getdist("rho5d")  # abscissae: rho,theta,phi,ppar,pperp,time,charge
    d = raw.integrate(copy=True, theta=np.s_[:], phi=np.s_[:],
                      time=np.s_[:], charge=np.s_[:])
    hist = d.histogram()                                        # (25,100,50)
    ppa, ppe = np.meshgrid(d.abscissa("ppar"), d.abscissa("pperp"),
                           indexing="ij")
    pnorm = np.sqrt(ppa**2 + ppe**2)
    ekin_j = ((physlib.gamma_momentum(mass, pnorm) - 1)
              * mass * unyt.c**2).to("J").v
    density = hist.v.sum(axis=(1, 2)) / vols                    # 1/m^3
    energy_density = np.einsum("ijk,jk->i", hist.v, ekin_j) / vols  # J/m^3

    # --- f_E on the shared rho x E grid via the E-xi conversion ------------
    # a5py's ekin_edges: plain array is interpreted in eV.
    exi = run.getdist("rho5d", exi=True, ekin_edges=E_EDGES_KEV * 1e3,
                      pitch_edges=2)  # 2 edges -> single [-1,1] pitch bin
    exi.integrate(theta=np.s_[:], phi=np.s_[:], pitch=np.s_[:],
                  time=np.s_[:], charge=np.s_[:])
    de_kev = np.diff(E_EDGES_KEV)
    f_E = exi.histogram().v / vols[:, None] / de_kev[None, :]   # 1/(m^3 keV)

    # --- pe/pi from collision-operator moments (needs libascot + inputs) ---
    pe = np.full(25, np.nan)
    pi_ = np.full(25, np.nan)
    moments_failed = True
    volratio = np.nan
    try:
        a5.input_init(bfield=True, plasma=True)
        try:
            mom = run.getdist_moments(
                run.getdist("rho5d"), "electronpowerdep", "ionpowerdep",
                volmethod="prism")
        finally:
            a5.input_free()
        # Single theta/phi bin -> toravg+polavg just squeezes to (25,)
        pe = mom.ordinate("electronpowerdep", toravg=True,
                          polavg=True).to("W/m**3").v
        pi_ = mom.ordinate("ionpowerdep", toravg=True,
                           polavg=True).to("W/m**3").v
        # Sanity: the moment machinery's own volumes vs the analytic shells.
        momvol = mom.volume.to("m**3").v.sum(axis=(1, 2))
        volratio = float(np.max(np.abs(momvol / vols - 1.0)))
        moments_failed = False
    except Exception as err:  # documented fallback, do not fake values
        print(f"WARNING: getdist_moments failed ({err!r}); writing NaN pe/pi")

    # --- birth arrays: the INISTATE of exactly the simulated subset --------
    r, z, ekin0, w0 = run.getstate("r", "z", "ekin", "weight",
                                   state="ini", mode="gc")
    birth_rho = np.sqrt((r.to("m").v - eq.R0) ** 2 + z.to("m").v ** 2) / eq.a
    birth_energy_keV = ekin0.to("keV").v
    birth_weight = np.asarray(w0)                                # particles/s

    ec, emsg = endcond_summary(run)
    if emsg:
        print(f"WARNING: run reported errors: {emsg}")

    # --- verification before writing ---------------------------------------
    e_j = 1.602176634e-19
    p_birth = float(np.sum(birth_weight * birth_energy_keV * 1e3 * e_j))
    p_dep = float(np.sum((pe + pi_) * vols)) if not moments_failed else np.nan
    n_pk = int(np.argmax(density))
    rho_pk = 0.5 * (RHO_EDGES_SD[n_pk] + RHO_EDGES_SD[n_pk + 1])
    w_stored = float(np.sum(energy_density * vols))

    print("\nVerification:")
    print(f"  birth power sum(w*E0)        = {p_birth/1e6:.4f} MW "
          "(expect ~1 MW: shinethrough ~ 0)")
    if not moments_failed:
        print(f"  deposited power sum((pe+pi)V)= {p_dep/1e6:.4f} MW "
              f"(expect 0.3 MW < P <= birth; pe {np.sum(pe*vols)/1e6:.4f} / "
              f"pi {np.sum(pi_*vols)/1e6:.4f} MW)")
        print(f"  moment-vs-analytic volumes   : max rel dev {volratio:.3f}")
    print(f"  fast-ion stored energy       = {w_stored/1e3:.1f} kJ")
    print(f"  density peak at rho          = {rho_pk:.2f}")
    print(f"  endconds                     = {ec}")

    checks = [
        ("birth power ~ 1 MW", 0.85e6 < p_birth < 1.15e6),
        ("density non-negative", bool(np.all(density >= 0.0))),
        ("density peaked inside rho<0.8", rho_pk < 0.8),
    ]
    if not moments_failed:
        checks += [
            ("deposited power > 0.3 MW", p_dep > 0.3e6),
            # 5% slack: MC-noise in the collision-coefficient moments.
            ("deposited power <= birth power", p_dep <= 1.05 * p_birth),
            ("pe, pi non-negative", bool(np.all(pe >= 0) and
                                         np.all(pi_ >= 0))),
        ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise RuntimeError(f"Verification failed: {failed} - fix before "
                           "writing the npz (check units/normalization).")

    meta.update({
        "n_markers": int(birth_rho.size), "endcond_counts": ec,
        "moments_failed": moments_failed,
        "birth_power_W": p_birth, "deposited_power_W": p_dep,
        "stored_energy_J": w_stored,
        "moment_volume_max_rel_dev": volratio,
        "emin_keV": EMIN_KEV, "seed": SEED,
    })
    np.savez(
        NPZ_OUT,
        rho_edges=RHO_EDGES_SD, e_edges_keV=E_EDGES_KEV,
        f_E=f_E, density=density, energy_density=energy_density,
        pe=pe, pi=pi_,
        birth_rho=birth_rho, birth_energy_keV=birth_energy_keV,
        birth_weight=birth_weight,
        n_markers=np.int64(birth_rho.size),
        wall_s=np.float64(meta.get("main_wall_s", np.nan)),
        meta_json=np.bytes_(json.dumps(meta).encode()),
    )
    print(f"\nWrote {NPZ_OUT}")
    print(f"  f_E {f_E.shape}, density {density.shape}, pe/pi "
          f"{'NaN' if moments_failed else 'OK'}, "
          f"birth arrays ({birth_rho.size},)")
    return meta


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibrate-only", action="store_true",
                    help="stop after the calibration run")
    ap.add_argument("--extract-only", action="store_true",
                    help="redo extraction from the existing SDMAIN run")
    ap.add_argument("--n-cal", type=int, default=N_CAL,
                    help="calibration marker count")
    args = ap.parse_args()

    from .test_comparison import make_scenario
    eq = make_scenario()[0]  # shared scenario equilibrium (R0=6.2, a=2.0)

    if args.extract_only:
        meta = {"note": "extract-only rerun"}
        if os.path.exists(META_JSON):  # timings/options of the SDMAIN run
            with open(META_JSON) as f:
                meta = json.load(f)
        extract(eq, meta)
        return

    ensure_ascot5_main()
    ensure_bbnbi_run()

    from a5py import Ascot
    a5 = Ascot(H5FN)
    ensure_dummy_inputs(a5)

    rng = np.random.default_rng(SEED)
    n_sd, t_cal, t_est, optmeta = calibrate(Ascot, BBNBI_TAG, rng, args.n_cal)
    if args.calibrate_only:
        print("--calibrate-only: stopping before the main run.")
        return

    a5 = Ascot(H5FN)
    mrk, subinfo = select_subset(a5.data[BBNBI_TAG], n_sd, rng)
    write_markers(a5, mrk, MRK_MAIN)
    print(f"\nMain run: {n_sd} markers (subset of {subinfo['n_ionized']} "
          f"ionized; weight scale {subinfo['scale']:.2f}), "
          f"extrapolated {t_est:.0f} s ...")
    t_main = run_ascot(MAIN_DESC)
    print(f"Main run finished in {t_main:.1f} s wall.")

    meta = {
        "options": optmeta, "n_cal": args.n_cal, "cal_wall_s": t_cal,
        "extrapolated_main_s": t_est, "main_wall_s": t_main,
        "weight_scale": subinfo["scale"],
        "n_ionized_total": subinfo["n_ionized"],
        "w_total_per_s": subinfo["w_total"],
    }
    with open(META_JSON, "w") as f:  # so --extract-only reruns keep the meta
        json.dump(meta, f, indent=2)
    extract(eq, meta)


if __name__ == "__main__":
    main()
