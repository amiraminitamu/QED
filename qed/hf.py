"""
QED-HF driver and supporting functions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import psi4

from .linalg_utils import s_inv_sqrt, sym


def density_matrix(C: np.ndarray, nocc: int) -> np.ndarray:
    C_occ = C[:, :nocc]
    return sym(2.0 * (C_occ @ C_occ.conj().T))


def density_matrix_uhf(
    Ca: np.ndarray,
    Cb: np.ndarray,
    nocc_a: int,
    nocc_b: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build α/β/open-shell density matrices from UHF coefficients."""

    Pa = Ca[:, :nocc_a] @ Ca[:, :nocc_a].conj().T
    Pb = Cb[:, :nocc_b] @ Cb[:, :nocc_b].conj().T
    return sym(Pa), sym(Pb), sym(Pa + Pb)


def nelec_from_P(P: np.ndarray, S: np.ndarray) -> float:
    return float(np.trace(P @ S).real)


def build_hc_ao(mux: np.ndarray, muy: np.ndarray, muz: np.ndarray, lam: Tuple[float, float, float]) -> np.ndarray:
    # h_c = lambda · mu
    return lam[0] * mux + lam[1] * muy + lam[2] * muz


def build_hep_ao(hc_ao: np.ndarray, omega: float) -> np.ndarray:
    # h_ep = -sqrt(omega/2) * h_c
    return -np.sqrt(omega / 2.0) * hc_ao


def add_dse_one_electron(H: np.ndarray, S: np.ndarray, hc_ao: np.ndarray, nmode: int = 1) -> np.ndarray:
    # Dipole self-energy 1e term: 1/2 h_c S^{-1} h_c * nmode
    S_inv = np.linalg.inv(S)
    H_dse = 0.5 * (hc_ao @ S_inv @ hc_ao) * nmode
    return sym(H + H_dse)


def add_dse_two_electron(eri: np.ndarray, hc_ao: np.ndarray, nmode: int = 1) -> np.ndarray:
    # Dipole self-energy 2e term: eri += h_c(pq) h_c(rs) * nmode
    return eri + np.einsum("pq,rs->pqrs", hc_ao, hc_ao, optimize=True) * nmode


def build_JK(eri4: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    J and K Integrals Built Directly from Psi4
    J_mn = (mn|ls) P_ls
    K_mn = (ml|ns) P_ls
    """
    J = np.einsum("mnls,ls->mn", eri4, P, optimize=True)
    K = np.einsum("mlns,ls->mn", eri4, P, optimize=True)
    return sym(J), sym(K)


def diagonalize_fock(F: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semi-Eigenvalue Problem Solver
    FC = εSC is solved via orthagonalization X = S^(-0.5)
    """
    F = sym(F)
    F_ortho = sym(X @ F @ X)
    eps, U = np.linalg.eigh(F_ortho)
    C = X @ U
    return eps, C


def qed_hf_energy(
    H_qed: np.ndarray,
    J: np.ndarray,
    K: np.ndarray,
    P: np.ndarray,
    E_nuc: float,
    omega: float,
    z: float,
    A: float,
    add_zpe: bool = True,
) -> float:
    """
    Find QED-HF Energy via the following equation
      E = Tr(P H_qed) + 1/2 Tr[P (J - 1/2 K)] + (2 z A + omega z^2) + E_nuc + (ZPE)
    where A = Tr(g P) and g = h_ep.
    ZPE can be added or ommited.
    """
    E1 = np.einsum("ij,ji->", H_qed, P).real
    E2 = 0.5 * np.einsum("ij,ji->", (J - 0.5 * K), P).real
    Ecoh = (2.0 * z * A + omega * z * z)
    Etot = E1 + E2 + Ecoh + E_nuc
    if add_zpe:
        Etot += 0.5 * omega
    return float(Etot)


def run_qed_hf(
    geom: str,
    basis: str = "6-31g",
    omega: float = 5e-4,
    lam=(1e-4, 0.0, 0.0),
    nmode: int = 1,
    max_iter: int = 200,
    conv_E: float = 1e-8,
    conv_P: float = 1e-7,
    scf_type: str = "pk",
    print_every: int = 10,
) -> Dict[str, object]:
    psi4.core.clean()
    psi4.set_memory("2 GB")
    psi4.set_num_threads(2)

    mol = psi4.geometry(geom)

    psi4.set_options({
        "basis": basis,
        "scf_type": scf_type,       # pk ensures ao_eri available
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    })

    hf_E, scf_wfn = psi4.energy("scf", return_wfn=True)
    nocc = scf_wfn.nalpha()
    mints = psi4.core.MintsHelper(scf_wfn.basisset())

    # Get Integrals
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = sym(T + V)
    E_nuc = mol.nuclear_repulsion_energy()

    mux, muy, muz = [np.asarray(M) for M in mints.ao_dipole()]
    eri = mints.ao_eri().to_array()  # shape (nbf,nbf,nbf,nbf)

    # QED Operators
    lam = tuple(lam)
    hc_ao = sym(build_hc_ao(mux, muy, muz, lam))   # h_c
    g_ao  = sym(build_hep_ao(hc_ao, omega))        # h_ep

    # Add Dipole Self Energy Terms (DSE)
    H_qed   = add_dse_one_electron(H, S, hc_ao, nmode=nmode)
    eri_qed = add_dse_two_electron(eri, hc_ao, nmode=nmode)

    # Initial Density Matrix from Already Converged Hartree-Fock
    P = sym(scf_wfn.Da().to_array() + scf_wfn.Db().to_array())

    # Orthogonalizer
    X = s_inv_sqrt(S)

    # Initialize z (coherent shift) from Tr(g P)
    A = float(np.einsum("ij,ji->", g_ao, P).real)  # Tr(g P)
    z = -A / omega

    E_last = None

    print("\n=== QED-HF (1-mode) ===")
    print(f"HF energy (Psi4): {hf_E: .12f} Ha")
    print(f"Electrons Tr(P S): {nelec_from_P(P, S): .8f} ")
    print(f"omega = {omega:.3e}  lambda = {lam}  nmode = {nmode}")
    print("------------------------")

    for it in range(max_iter):
        J, K = build_JK(eri_qed, P)
        F_e = H_qed + J - 0.5 * K
        F   = F_e + 2.0 * z * g_ao

        eps, C = diagonalize_fock(F, X)
        P_new = density_matrix(C, nocc)

        A_new = float(np.einsum("ij,ji->", g_ao, P_new).real)
        z_new = -A_new / omega

        # energy with updated P_new, z_new
        J_new, K_new = build_JK(eri_qed, P_new)
        E = qed_hf_energy(H_qed, J_new, K_new, P_new, E_nuc, omega, z_new, A_new, add_zpe=False)

        dP = np.linalg.norm(P_new - P)
        dE = abs(E - E_last) if E_last is not None else np.nan

        if it < 5 or (it % print_every == 0):
            print(f"iter {it:3d}  E = {E: .12f}  dE = {dE: .3e}  dP = {dP: .3e}  z = {z_new: .6e}")

        if E_last is not None and dE < conv_E and dP < conv_P:
            print("------------------------")
            print(f"Converged in {it+1} iterations")
            print(f"QED-HF total energy (incl ZPE): {E: .12f} Ha")
            return {
                "E_qed_hf": E,
                "E_hf_psi4": float(hf_E),
                "z": float(z_new),
                "P": P_new,
                "eps": eps,
                "C": C,
                "S": S,
                "H_qed": H_qed,
                "eri_qed": eri_qed,
                "eri_raw": eri,
                "g_ao": g_ao,
                "hc_ao": hc_ao,
                "nocc": nocc,
                "mol": mol,
            }

        P = P_new
        z = z_new
        E_last = E

    raise RuntimeError(f"QED-HF did not converge in {max_iter} iterations.")


def run_qed_uhf(
    geom: str,
    basis: str = "6-31g",
    omega: float = 5e-4,
    lam=(1e-4, 0.0, 0.0),
    nmode: int = 1,
    max_iter: int = 200,
    conv_E: float = 1e-8,
    conv_P: float = 1e-7,
    scf_type: str = "pk",
    print_every: int = 10,
) -> Dict[str, object]:
    """Open-shell QED-HF (UHF-based) calculation with coherent shift."""

    psi4.core.clean()
    psi4.set_memory("2 GB")
    psi4.set_num_threads(2)

    mol = psi4.geometry(geom)

    psi4.set_options({
        "basis": basis,
        "scf_type": scf_type,
        "reference": "uhf",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    })

    hf_E, scf_wfn = psi4.energy("scf", return_wfn=True)
    nocc_a = scf_wfn.nalpha()
    nocc_b = scf_wfn.nbeta()

    mints = psi4.core.MintsHelper(scf_wfn.basisset())

    # Get Integrals
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = sym(T + V)
    E_nuc = mol.nuclear_repulsion_energy()

    mux, muy, muz = [np.asarray(M) for M in mints.ao_dipole()]
    eri = mints.ao_eri().to_array()

    # QED Operators
    lam = tuple(lam)
    hc_ao = sym(build_hc_ao(mux, muy, muz, lam))
    g_ao  = sym(build_hep_ao(hc_ao, omega))

    # Add Dipole Self Energy Terms (DSE)
    H_qed   = add_dse_one_electron(H, S, hc_ao, nmode=nmode)
    eri_qed = add_dse_two_electron(eri, hc_ao, nmode=nmode)

    Pa = scf_wfn.Da().to_array()
    Pb = scf_wfn.Db().to_array()
    P  = sym(Pa + Pb)

    # Orthogonalizer
    X = s_inv_sqrt(S)

    # Initialize z (coherent shift) from Tr(g P)
    A = np.einsum("ij,ji->", g_ao, P).real
    z = -A / omega
    E_last = None

    print("\n=== QED-HF (open-shell, UHF reference) ===")

    Ca = scf_wfn.Ca().to_array()
    Cb = scf_wfn.Cb().to_array()

    for it in range(max_iter):
        J, K = build_JK(eri_qed, P)

        Fa = H_qed + J - K + 2.0 * z * g_ao
        Fb = H_qed + J - K + 2.0 * z * g_ao

        eps_a, Ca = diagonalize_fock(Fa, X)
        eps_b, Cb = diagonalize_fock(Fb, X)

        Pa, Pb, P_new = density_matrix_uhf(Ca, Cb, nocc_a, nocc_b)

        A_new = np.einsum("ij,ji->", g_ao, P_new).real
        z_new = -A_new / omega

        Jn, Kn = build_JK(eri_qed, P_new)
        E = qed_hf_energy(H_qed, Jn, Kn, P_new, E_nuc, omega, z_new, A_new, add_zpe=False)

        dP = np.linalg.norm(P_new - P)
        dE = abs(E - E_last) if E_last is not None else np.nan

        if it < 5 or it % print_every == 0:
            print(f"iter {it:3d}  E = {E:.12f}  dE = {dE:.3e}  dP = {dP:.3e}")

        if E_last is not None and dE < conv_E and dP < conv_P:
            print("Converged.")
            return {
                "E_qed_hf": E,
                "E_hf_psi4": float(hf_E),
                "z": float(z_new),
                "P": P_new,
                "Pa": Pa,
                "Pb": Pb,
                "eps_a": eps_a,
                "eps_b": eps_b,
                "C_a": Ca,
                "C_b": Cb,
                "S": S,
                "H_qed": H_qed,
                "eri_qed": eri_qed,
                "eri_raw": eri,
                "g_ao": g_ao,
                "hc_ao": hc_ao,
                "nocc_a": nocc_a,
                "nocc_b": nocc_b,
                "mol": mol,
            }

        P, z, E_last = P_new, z_new, E

    raise RuntimeError("QED-UHF did not converge")
