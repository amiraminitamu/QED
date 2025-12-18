"""
QED-MP2 correction routines.
"""

from __future__ import annotations

from typing import Dict

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
