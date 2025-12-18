"""
Modular QED-HF / QED-MP2 toolkit.

This package exposes functions that can be imported from Python or used
via the CLI entrypoint in ``main.py``.
"""

from .formatters import ResultFormatter, TextResultFormatter
from .hf import (
    add_dse_one_electron,
    add_dse_two_electron,
    build_JK,
    build_hc_ao,
    build_hep_ao,
    density_matrix,
    density_matrix_uhf,
    diagonalize_fock,
    nelec_from_P,
    qed_hf_energy,
    run_qed_hf,
    run_qed_uhf,
)
from .linalg_utils import s_inv_sqrt, sym
from .mp2 import (
    eri_ao_to_mo_chemist,
    eri_ao_to_mo_uhf,
    mp2_corr_rhf_from_eri_mo,
    mp2_corr_uhf,
    qed_mp2_correction,
    qed_mp2_correction_uhf,
)
from .specs import SimulationResult, SimulationSpec
from .workflow import run_simulation

__all__ = [
    "ResultFormatter",
    "TextResultFormatter",
    "add_dse_one_electron",
    "add_dse_two_electron",
    "build_JK",
    "build_hc_ao",
    "build_hep_ao",
    "density_matrix",
    "density_matrix_uhf",
    "diagonalize_fock",
    "eri_ao_to_mo_chemist",
    "eri_ao_to_mo_uhf",
    "mp2_corr_rhf_from_eri_mo",
    "mp2_corr_uhf",
    "nelec_from_P",
    "qed_hf_energy",
    "qed_mp2_correction",
    "qed_mp2_correction_uhf",
    "run_qed_hf",
    "run_qed_uhf",
    "run_simulation",
    "s_inv_sqrt",
    "sym",
    "SimulationResult",
    "SimulationSpec",
]
