"""
QED-MP2 correction routines.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def eri_ao_to_mo_chemist(eri_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    eri_ao: (nbf,nbf,nbf,nbf) in chemist notation (pq|rs)
    returns eri_mo: (nmo,nmo,nmo,nmo) in chemist notation (pq|rs)
    """
    return np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True)


def mp2_corr_rhf_from_eri_mo(eps: np.ndarray, eri_mo: np.ndarray, nocc: int) -> float:
    """
    Standard RHF MP2 correlation from MO ERIs in chemist form (pq|rs),
    using spatial orbitals (closed-shell).
    """
    eps = np.asarray(eps)
    nmo = eps.shape[0]
    o = slice(0, nocc)
    v = slice(nocc, nmo)

    # (ia|jb)
    g_iajb = eri_mo[o, v, o, v]  # i a j b
    # (ib|ja)
    g_ibja = eri_mo[o, v, o, v].transpose(0, 3, 2, 1)  # i b j a

    e_occ = eps[o]
    e_vir = eps[v]
    denom = (
        e_occ[:, None, None, None] + e_occ[None, None, :, None]
        - e_vir[None, :, None, None] - e_vir[None, None, None, :]
    )
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

    emp2 = np.sum(g_iajb * (2.0 * g_iajb - g_ibja) / denom)
    return float(emp2)


def qed_mp2_correction(
    *,
    eps: np.ndarray,
    C: np.ndarray,
    nocc: int,
    eri_qed: np.ndarray,     # AO ERIs that ALREADY include DSE 2e term
    hc_ao: np.ndarray,      # h_c = lambda · mu in AO (your hc_ao)
    omega: float,
    spin_factor: str = "closed_shell",  # "closed_shell" or "ghf_equivalent"
    verbose: bool = True,
) -> Dict[str, float]:
    """
    QED-MP2 correction on top of a QED-HF reference.
    """

    # --- electronic MP2 on ERIs that already include DSE ---
    eri_mo = eri_ao_to_mo_chemist(eri_qed, C)
    e_el = mp2_corr_rhf_from_eri_mo(eps, eri_mo, nocc)

    # --- photon term from hc_ao (lambda·mu) ---
    hc_mo = C.conj().T @ hc_ao @ C
    h_ov = hc_mo[:nocc, nocc:]  # i,a (spatial)

    eps = np.asarray(eps)
    dE = (eps[nocc:][None, :] - eps[:nocc][:, None])  # Δ = eps_a - eps_i
    denom = dE + omega
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

    hov2 = np.abs(h_ov) ** 2

    if spin_factor == "closed_shell":
        e_ph = -omega * np.sum(hov2 / denom)
    elif spin_factor == "ghf_equivalent":
        e_ph = -(0.5 * omega) * np.sum(hov2 / denom)
    else:
        raise ValueError("spin_factor must be 'closed_shell' or 'ghf_equivalent'")

    e_qedmp2 = e_el + e_ph

    if verbose:
        print("\n=== QED-MP2 ===")
        print(f"electronic MP2: {e_el: .12f} Ha")
        print(f"photon term:    {e_ph: .12f} Ha   (spin_factor={spin_factor})")
        print(f"QED-MP2 corr:   {e_qedmp2: .12f} Ha")

    return {"emp2_el": float(e_el), "emp2_ph": float(e_ph), "emp2_total": float(e_qedmp2)}


def eri_ao_to_mo_uhf(eri_ao: np.ndarray, Ca: np.ndarray, Cb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    AO → MO ERIs for UHF (chemist notation).

    Returns three spin blocks: ``g_aa``, ``g_bb``, and ``g_ab``.
    """

    g_aa = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, Ca, Ca, Ca, Ca, optimize=True)
    g_bb = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, Cb, Cb, Cb, Cb, optimize=True)
    g_ab = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, Ca, Ca, Cb, Cb, optimize=True)
    return g_aa, g_bb, g_ab


def mp2_corr_uhf(
    eps_a: np.ndarray,
    eps_b: np.ndarray,
    g_aa: np.ndarray,
    g_bb: np.ndarray,
    g_ab: np.ndarray,
    nocc_a: int,
    nocc_b: int,
) -> float:
    """Standard UMP2 correlation energy."""

    oa = slice(0, nocc_a)
    va = slice(nocc_a, g_aa.shape[0])
    ob = slice(0, nocc_b)
    vb = slice(nocc_b, g_bb.shape[0])

    # αα
    denom_aa = (
        eps_a[oa, None, None, None] + eps_a[None, None, oa, None]
        - eps_a[None, va, None, None] - eps_a[None, None, None, va]
    )
    denom_aa = np.where(np.abs(denom_aa) < 1e-14, 1e-14, denom_aa)
    E_aa = np.sum(
        g_aa[oa, va, oa, va]
        * (g_aa[oa, va, oa, va] - g_aa[oa, va, oa, va].transpose(0, 3, 2, 1))
        / denom_aa
    )

    # ββ
    denom_bb = (
        eps_b[ob, None, None, None] + eps_b[None, None, ob, None]
        - eps_b[None, vb, None, None] - eps_b[None, None, None, vb]
    )
    denom_bb = np.where(np.abs(denom_bb) < 1e-14, 1e-14, denom_bb)
    E_bb = np.sum(
        g_bb[ob, vb, ob, vb]
        * (g_bb[ob, vb, ob, vb] - g_bb[ob, vb, ob, vb].transpose(0, 3, 2, 1))
        / denom_bb
    )

    # αβ
    denom_ab = (
        eps_a[oa, None, None, None] + eps_b[None, None, ob, None]
        - eps_a[None, va, None, None] - eps_b[None, None, None, vb]
    )
    denom_ab = np.where(np.abs(denom_ab) < 1e-14, 1e-14, denom_ab)
    E_ab = np.sum(g_ab[oa, va, ob, vb] ** 2 / denom_ab)

    return float(E_aa + E_bb + E_ab)


def qed_mp2_correction_uhf(
    *,
    eps_a: np.ndarray,
    eps_b: np.ndarray,
    C_a: np.ndarray,
    C_b: np.ndarray,
    nocc_a: int,
    nocc_b: int,
    eri_qed: np.ndarray,
    hc_ao: np.ndarray,
    omega: float,
    verbose: bool = True,
) -> Dict[str, float]:
    """QED-MP2 correction for an open-shell (UHF) reference."""

    g_aa, g_bb, g_ab = eri_ao_to_mo_uhf(eri_qed, C_a, C_b)
    e_el = mp2_corr_uhf(eps_a, eps_b, g_aa, g_bb, g_ab, nocc_a, nocc_b)

    hc_a = C_a.conj().T @ hc_ao @ C_a
    hc_b = C_b.conj().T @ hc_ao @ C_b

    def photon_term(eps: np.ndarray, h_mo: np.ndarray, nocc: int) -> float:
        h_ov = h_mo[:nocc, nocc:]
        dE = eps[nocc:][None, :] - eps[:nocc][:, None]
        denom = np.where(np.abs(dE + omega) < 1e-14, 1e-14, dE + omega)
        return -omega * np.sum(np.abs(h_ov) ** 2 / denom)

    e_ph = photon_term(eps_a, hc_a, nocc_a) + photon_term(eps_b, hc_b, nocc_b)
    e_tot = e_el + e_ph

    if verbose:
        print("\n=== QED-MP2 (open-shell) ===")
        print(f"electronic MP2: {e_el: .12f} Ha")
        print(f"photon term:    {e_ph: .12f} Ha")
        print(f"QED-MP2 corr:   {e_tot: .12f} Ha")

    return {"emp2_el": float(e_el), "emp2_ph": float(e_ph), "emp2_total": float(e_tot)}
