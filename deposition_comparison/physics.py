"""Plasma profiles and the Suzuki beam-stopping cross section in JAX.

This module owns three functions of the deposition-comparison package:

- :func:`profiles` evaluates the parabolic-like plasma profiles defined in
  :class:`~deposition_comparison.common.Plasma`.
- :func:`prepare_suzuki` performs all species-dependent (concrete, non-traced)
  coefficient-table lookups and returns a
  :class:`~deposition_comparison.common.SuzukiTables` instance.
- :func:`sigma_stop` evaluates the beam-stopping cross section [m^2] as pure
  traced JAX arithmetic (jit- and vmap-safe).

The Suzuki model is a faithful port of ``src/suzuki.c`` (S. Suzuki et al 1998
Plasma Phys. Control. Fusion 40 2097), with one intentional difference: the C
function returns sigma*v while :func:`sigma_stop` returns sigma alone (in m^2,
i.e. the cm^2 -> m^2 conversion is included but the final multiplication by
the beam velocity is NOT).

Table-layout note
-----------------
``SuzukiTables`` documents ``A_low``/``A_high`` as having shape ``(n_h, 10)``
(one row per hydrogenic species). For a fully jit-safe evaluation this module
instead stores them **zero-row padded to shape (nion, 10)**: row ``i`` holds
the A coefficients of species ``i`` if that species is hydrogenic and zeros
otherwise, mirroring the layout of ``B_low``/``B_high``. Hydrogenic terms are
selected with ``h_mask`` at evaluation time. Only :func:`prepare_suzuki` and
:func:`sigma_stop` touch these arrays, so the deviation is internal to this
module; it is documented here per the package interface contract.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .common import Plasma, SuzukiTables

# ---------------------------------------------------------------------------
# Suzuki (1998) fit-coefficient tables, ported verbatim from src/suzuki.c
# ---------------------------------------------------------------------------
# Table 2a: Aijk for H, D, T; valid for 100 < E [keV/amu] < 10000.
_A_HIGHE = np.array([
    [12.7, 1.25, 0.452, 0.0105, 0.547, -0.102,   0.360, -0.0298, -0.0959,
     4.21e-3],
    [14.1, 1.11, 0.408, 0.0105, 0.547, -0.0403,  0.345, -0.0288, -0.0971,
     4.74e-3],
    [12.7, 1.26, 0.449, 0.0105, 0.547, -0.00577, 0.336, -0.0282, -0.0974,
     4.87e-3],
])

# Table 3a: Aijk for H, D, T; valid for 10 < E [keV/amu] < 100.
_A_LOWE = np.array([
    [-52.9, -1.36, 0.0719, 0.0137, 0.454, 0.403, -0.220, 0.0666, -0.0677,
     -1.48e-3],
    [-67.9, -1.22, 0.0814, 0.0139, 0.454, 0.465, -0.273, 0.0751, -0.0630,
     -5.08e-4],
    [-74.2, -1.18, 0.0843, 0.0139, 0.453, 0.491, -0.294, 0.0788, -0.0612,
     -1.85e-4],
])

# Charge number and Zeff validity window for each tabulated Bijk row. The
# Li and Be rows present in the paper are commented out in suzuki.c (their
# published values are identical, presumed erroneous) and are skipped here
# exactly the same way, leaving 7 rows.
_Z_IMP       = np.array([2,   6,   6,   8,   7,   5,   26])
_ZEFFMIN_IMP = np.array([1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0])
_ZEFFMAX_IMP = np.array([2.1, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0])

# Table 2bc (partially): Bijk; valid for 100 < E [keV/amu] < 10000.
_B_HIGHE = np.array([
    [ 0.231,     0.343,    -0.185,    -0.162e-1,  0.105,    -0.703e-1,
      0.531e-1,  0.342e-2, -0.838e-2,  0.415e-2, -0.335e-2, -0.221e-3],
    [-0.101e1,  -0.865e-2, -0.124,    -0.145e-1,  0.391,     0.161e-1,
      0.298e-1,  0.332e-2, -0.248e-1, -0.104e-2, -0.152e-2, -0.189e-3],
    [-0.100e1,  -0.255e-1, -0.125,    -0.142e-1,  0.388,     0.206e-1,
      0.297e-1,  0.326e-2, -0.246e-1, -0.131e-2, -0.148e-2, -0.180e-3],
    [-0.102e1,  -0.148e-1, -0.674e-1, -0.917e-2,  0.359,     0.143e-1,
      0.139e-1,  0.184e-2, -0.209e-1, -0.732e-3, -0.502e-3, -0.949e-4],
    [-0.102e1,  -0.139e-1, -0.979e-1, -0.117e-1,  0.375,     0.156e-1,
      0.224e-1,  0.254e-2, -0.226e-1, -0.889e-3, -0.104e-2, -0.139e-3],
    [-0.732,     0.183e-1, -0.155,    -0.172e-1,  0.321,     0.946e-2,
      0.397e-1,  0.420e-2, -0.204e-1, -0.619e-3, -0.224e-2, -0.254e-3],
    [-0.820,    -0.636e-2,  0.542e-1,  0.395e-2,  0.202,     0.806e-3,
     -0.200e-2, -0.178e-2, -0.610e-2,  0.651e-3,  0.175e-2,  0.146e-3],
])

# Table 3bc (partially): Bijk; valid for 10 < E [keV/amu] < 100.
_B_LOWE = np.array([
    [-0.792,     0.420e-1,  0.530e-1,  -0.139e-1,  0.301,    -0.264e-1,
     -0.299e-1,  0.607e-2,  0.272e-3,   0.611e-2,  0.347e-2, -0.919e-3],
    [ 0.161,     0.598e-1, -0.336e-2,  -0.426e-2, -0.157,    -0.396e-1,
      0.460e-2,  0.219e-2,  0.391e-1,   0.711e-2, -0.144e-2, -0.385e-3],
    [ 0.158,     0.554e-1, -0.431e-2,  -0.335e-2, -0.155,    -0.374e-1,
      0.537e-2,  0.174e-2,  0.388e-1,   0.683e-2, -0.160e-2, -0.322e-3],
    [ 0.111,     0.541e-1, -0.346e-3,  -0.368e-2, -0.108,    -0.347e-1,
      0.193e-2,  0.181e-2,  0.280e-1,   0.604e-2, -0.841e-3, -0.317e-3],
    [ 0.139,     0.606e-1, -0.306e-2,  -0.455e-2, -0.133,    -0.394e-1,
      0.399e-2,  0.236e-2,  0.335e-1,   0.690e-2, -0.124e-2, -0.405e-3],
    [ 0.122,     0.527e-1, -0.430e-3,  -0.318e-2, -0.151,    -0.364e-1,
      0.343e-2,  0.151e-2,  0.420e-1,   0.692e-2, -0.141e-2, -0.290e-3],
    [-0.110e-1,  0.202e-1,  0.946e-3,  -0.409e-2, -0.666e-2, -0.117e-1,
     -0.236e-3,  0.202e-2,  0.408e-2,   0.185e-2, -0.648e-4, -0.313e-3],
])


def profiles(plasma: Plasma, rho):
    """Evaluate plasma profiles at given normalized radii.

    Profiles are parabolic-like per the :class:`Plasma` docstring:
    ``f(rho) = (f0 - fedge) * (1 - rho**2)**alpha + fedge`` for ``rho < 1``
    and ``f(rho) = fedge`` outside. Ion densities follow from the fixed
    impurity concentrations and quasineutrality, with ``conc[0]`` recomputed
    as ``(1 - sum_{i>0} znum[i]*conc[i]) / znum[0]``; since ``ni`` is
    proportional to ``ne`` everywhere, the edge values scale consistently.

    Parameters
    ----------
    plasma : Plasma
        Plasma profile parameters and species arrays.
    rho : array_like
        Normalized radius, any shape ``(...)``.

    Returns
    -------
    ne : jnp.ndarray
        Electron density [m^-3], shape ``(...)``.
    te : jnp.ndarray
        Electron temperature [eV], shape ``(...)``.
    ni : jnp.ndarray
        Ion densities [m^-3], shape ``(..., nion)``.
    """
    rho = jnp.asarray(rho)
    # Clamp the parabola base so fractional exponents never see a negative
    # argument; the jnp.where then selects the edge value outside rho >= 1.
    base = jnp.maximum(1.0 - rho**2, 0.0)
    inside = rho < 1.0
    ne = jnp.where(
        inside, (plasma.ne0 - plasma.ne_edge) * base**plasma.alpha_n
        + plasma.ne_edge, plasma.ne_edge)
    te = jnp.where(
        inside, (plasma.te0 - plasma.te_edge) * base**plasma.alpha_t
        + plasma.te_edge, plasma.te_edge)

    conc = jnp.asarray(plasma.conc, dtype=jnp.float64)
    znum = jnp.asarray(plasma.znum, dtype=jnp.float64)
    c0 = (1.0 - jnp.sum(znum[1:] * conc[1:])) / znum[0]
    conc = conc.at[0].set(c0)
    ni = ne[..., None] * conc

    return ne, te, ni


def prepare_suzuki(anum, znum) -> SuzukiTables:
    """Select Suzuki fit-coefficient rows for a concrete species list.

    Runs at setup time with concrete (non-traced) species numbers so that
    :func:`sigma_stop` itself is pure traced arithmetic. Hydrogenic species
    (``znum == 1``) get an A row selected by mass number (1 -> H, 2 -> D,
    3 -> T). Impurities get the B row whose charge number matches and whose
    Zeff validity window starts at 1 (``Zeffmin == 1.0``); carbon appears
    twice in the tables — with windows (1, 5) and (5, 6) — and the (1, 5)
    row is chosen, i.e. the ``1 < Zeff < 4``-ish regime is assumed.

    Note: as documented in the module docstring, ``A_low``/``A_high`` are
    stored zero-row padded to shape ``(nion, 10)`` rather than the packed
    ``(n_h, 10)`` shape mentioned in ``common.py``.

    Parameters
    ----------
    anum : array_like of int
        Species mass numbers, shape ``(nion,)``.
    znum : array_like of int
        Species charge numbers, shape ``(nion,)``.

    Returns
    -------
    SuzukiTables
        Prepared per-species coefficient arrays.

    Raises
    ------
    ValueError
        If a hydrogenic species has ``anum`` not in {1, 2, 3}, or an
        impurity's charge number is not tabulated (supported: He, B, C, N,
        O, Fe), or no hydrogenic species is present.
    """
    anum = np.asarray(anum, dtype=int).ravel()
    znum = np.asarray(znum, dtype=int).ravel()
    if anum.shape != znum.shape:
        raise ValueError("anum and znum must have the same shape")
    nion = anum.size

    A_low  = np.zeros((nion, 10))
    A_high = np.zeros((nion, 10))
    B_low  = np.zeros((nion, 12))
    B_high = np.zeros((nion, 12))
    h_mask = np.zeros(nion)

    for i in range(nion):
        if znum[i] == 1:
            if anum[i] not in (1, 2, 3):
                raise ValueError(
                    f"Hydrogenic species {i} has unsupported mass number "
                    f"{anum[i]} (must be 1, 2, or 3)")
            h_mask[i] = 1.0
            A_low[i]  = _A_LOWE[anum[i] - 1]
            A_high[i] = _A_HIGHE[anum[i] - 1]
        else:
            rows = np.nonzero(
                (_Z_IMP == znum[i]) & (_ZEFFMIN_IMP == 1.0))[0]
            if rows.size == 0:
                raise ValueError(
                    f"Impurity species {i} with charge number {znum[i]} is "
                    f"not tabulated (supported Z: 2, 5, 6, 7, 8, 26)")
            j = int(rows[0])
            B_low[i]  = _B_LOWE[j]
            B_high[i] = _B_HIGHE[j]

    if not np.any(h_mask):
        raise ValueError("At least one hydrogenic species is required")

    return SuzukiTables(
        A_low=jnp.asarray(A_low),
        A_high=jnp.asarray(A_high),
        h_mask=jnp.asarray(h_mask),
        B_low=jnp.asarray(B_low),
        B_high=jnp.asarray(B_high),
        znum=jnp.asarray(znum, dtype=jnp.float64),
        anum=jnp.asarray(anum, dtype=jnp.float64),
    )


def sigma_stop(tables: SuzukiTables, e_kev_per_amu, ne, te_ev, ni):
    """Suzuki beam-stopping cross section [m^2] (pure traced JAX).

    Port of ``suzuki_sigmav`` in ``src/suzuki.c`` WITHOUT the final
    multiplication by the beam velocity: this returns sigma [m^2], not
    sigma*v. Low-energy tables are used for ``9 <= E < 100`` keV/amu and the
    high-energy tables otherwise (selected with ``jnp.where`` so the
    function is jit- and vmap-safe). Inputs are clamped
    (``ne >= 1e10 m^-3``, ``te >= 1 eV``) so logarithms are safe in vacuum
    regions.

    Parameters
    ----------
    tables : SuzukiTables
        Coefficients from :func:`prepare_suzuki` (zero-row-padded layout,
        see module docstring).
    e_kev_per_amu : array_like
        Beam energy per mass number [keV/amu], shape ``(...)``.
    ne : array_like
        Electron density [m^-3], shape broadcastable to ``(...)``.
    te_ev : array_like
        Electron temperature [eV], shape broadcastable to ``(...)``.
    ni : array_like
        Ion densities [m^-3], shape ``(..., nion)``.

    Returns
    -------
    jnp.ndarray
        Beam-stopping cross section [m^2], shape ``(...)``.
    """
    E  = jnp.asarray(e_kev_per_amu, dtype=jnp.float64)
    ne = jnp.maximum(jnp.asarray(ne, dtype=jnp.float64), 1e10)
    te = jnp.maximum(jnp.asarray(te_ev, dtype=jnp.float64), 1.0)
    ni = jnp.asarray(ni, dtype=jnp.float64)

    logE = jnp.log(E)
    N    = ne * 1.0e-19
    logN = jnp.log(N)
    U    = jnp.log(te * 1.0e-3)

    # Zeff over all species (traced).
    z = tables.znum
    zeff_num = jnp.sum(ni * z * z, axis=-1)
    zeff_den = jnp.sum(ni * z, axis=-1)
    Zeff = zeff_num / jnp.maximum(zeff_den, 1e-300)

    # Low- vs high-energy table selection, elementwise in E.
    low = (E >= 9.0) & (E < 100.0)
    sel = low[..., None, None]
    A = jnp.where(sel, tables.A_low, tables.A_high)    # (..., nion, 10)
    B = jnp.where(sel, tables.B_low, tables.B_high)    # (..., nion, 12)

    # Broadcast scalar-per-point quantities against the species axis.
    E_    = E[..., None]
    logE_ = logE[..., None]
    N_    = N[..., None]
    logN_ = logN[..., None]
    U_    = U[..., None]

    # Equation 28: sigma_H, density-weighted over hydrogenic species.
    # Guard pow(1 - exp(-A3*N), A4) as N -> 0 by clamping the base.
    pow_base = jnp.maximum(1.0 - jnp.exp(-A[..., 3] * N_), 1e-30)
    term_H = (
        (A[..., 0] * 1.0e-16 / E_)
        * (1.0 + A[..., 1] * logE_ + A[..., 2] * logE_**2)
        * (1.0 + pow_base**A[..., 4]
           * (A[..., 5] + A[..., 6] * logE_ + A[..., 7] * logE_**2))
        * (1.0 + A[..., 8] * U_ + A[..., 9] * U_**2)
    )
    dens_H  = jnp.sum(ni * tables.h_mask, axis=-1)
    sigma_H = jnp.sum(ni * tables.h_mask * term_H, axis=-1) \
        / jnp.maximum(dens_H, 1e-300)

    # Equations 26 & 27: sigma_Z with Rui's correction. The weight
    # z*(z-1)*ni vanishes for hydrogenic species (z == 1) automatically,
    # and their B rows are zero anyway.
    poly_B = (
        B[..., 0]
        + B[..., 1]  * U_
        + B[..., 2]  * logN_
        + B[..., 3]  * logN_ * U_
        + B[..., 4]  * logE_
        + B[..., 5]  * logE_ * U_
        + B[..., 6]  * logE_ * logN_
        + B[..., 7]  * logE_ * logN_ * U_
        + B[..., 8]  * logE_**2
        + B[..., 9]  * logE_**2 * U_
        + B[..., 10] * logE_**2 * logN_
        + B[..., 11] * logE_**2 * logN_ * U_
    )
    w = z * (z - 1.0) * ni * (1.0 - tables.h_mask)
    denominator = jnp.sum(w, axis=-1)
    numerator   = jnp.sum(w * poly_B, axis=-1)
    sigma_Z = jnp.where(
        denominator > 0.0,
        numerator / jnp.maximum(denominator, 1e-300), 0.0)

    # Equation 24, converting cm^2 to m^2.
    return sigma_H * (1.0 + (Zeff - 1.0) * sigma_Z) * 1e-4


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def _test_sigma_sanity():
    """Run basic sanity checks of :func:`sigma_stop` and print a table."""
    # Pure D plus 2% carbon.
    tables = prepare_suzuki([2, 12], [1, 6])
    ne = 1e20
    te = 5000.0
    c_C = 0.02
    ni = jnp.array([ne * (1.0 - 6.0 * c_C), ne * c_C])

    sigma = sigma_stop(tables, 50.0, ne, te, ni)
    assert jnp.isfinite(sigma), "sigma is not finite"
    assert sigma > 0.0, "sigma is not positive"
    assert 1e-21 < float(sigma) < 1e-18, \
        f"sigma = {float(sigma):.3e} m^2 outside (1e-21, 1e-18)"
    print(f"50 keV/amu, ne=1e20 m^-3, te=5 keV, D + 2% C: "
          f"sigma = {float(sigma):.6e} m^2  -- OK")

    energies = jnp.array([20.0, 50.0, 100.0, 500.0, 1000.0])
    print("\n  E [keV/amu]    sigma [m^2]")
    for e in energies:
        s = sigma_stop(tables, e, ne, te, ni)
        print(f"  {float(e):11.1f}    {float(s):.6e}")

    # jit / vmap consistency. vmap of the eager function is bitwise
    # identical; XLA-compiled (jit) code may differ by 1-2 ULP from eager
    # evaluation due to instruction fusion, so allow rtol = 1e-14 there.
    f = lambda e: sigma_stop(tables, e, ne, te, ni)
    ref = f(energies)
    jitted = jax.jit(f)(energies)
    vmapped = jax.vmap(f)(energies)
    assert jnp.array_equal(ref, vmapped), "vmap result differs"
    rel_jit = float(jnp.max(jnp.abs(jitted - ref) / ref))
    assert rel_jit < 1e-14, f"jit result differs (rel {rel_jit:.3e})"
    print(f"\nvmap identical to eager; jit within {rel_jit:.2e} rel "
          f"(ULP-level) -- OK")


if __name__ == "__main__":
    _test_sigma_sanity()
