"""
High-level workflow to run QED-HF and optional QED-MP2 from a single spec.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .formatters import ResultFormatter, TextResultFormatter
from .hf import run_qed_hf
from .mp2 import qed_mp2_correction
from .specs import SimulationResult, SimulationSpec


def run_simulation(
    spec: SimulationSpec,
    *,
    formatter: Optional[ResultFormatter] = None,
) -> Tuple[SimulationResult, str]:
    """
    Execute QED-HF followed by optional QED-MP2 using a single specification
    object. Returns both the structured result and a formatted text summary.
    """
    scf_result = run_qed_hf(
        geom=spec.geometry,
        basis=spec.basis,
        omega=spec.omega,
        lam=spec.lam,
        nmode=spec.nmode,
        max_iter=spec.max_iter,
        conv_E=spec.conv_E,
        conv_P=spec.conv_P,
        scf_type=spec.scf_type,
        print_every=spec.print_every,
    )

    mp2_result = None
    if spec.run_mp2:
        mp2_result = qed_mp2_correction(
            eps=scf_result["eps"],
            C=scf_result["C"],
            nocc=scf_result["nocc"],
            eri_qed=scf_result["eri_qed"],
            hc_ao=scf_result["hc_ao"],
            omega=spec.omega,
            spin_factor=spec.spin_factor,
            verbose=True,
        )

    result = SimulationResult(scf=scf_result, mp2=mp2_result, spec=spec)
    formatter = formatter or TextResultFormatter()
    formatted = formatter.format(result)
    return result, formatted
