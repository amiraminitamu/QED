"""
Result formatting helpers so users can easily swap output styles.
"""

from __future__ import annotations

from typing import Optional

from .specs import SimulationResult


class ResultFormatter:
    """Base formatter that can be extended for custom outputs."""

    def format(self, result: SimulationResult) -> str:  # pragma: no cover - simple passthrough
        raise NotImplementedError


class TextResultFormatter(ResultFormatter):
    """
    Human-readable text formatter. Users can subclass and override methods
    to change the printed output without touching the workflow logic.
    """

    def format(self, result: SimulationResult) -> str:
        parts = [
            self._header(result),
            self._format_hf(result),
        ]
        if result.mp2 is not None:
            parts.append(self._format_mp2(result))
        parts.append(self._format_summary(result))
        return "\n".join(parts)

    def _header(self, result: SimulationResult) -> str:
        spec = result.spec
        lam = spec.lam
        lines = [
            "=== QED Calculation ===",
            f"basis: {spec.basis}",
            f"omega: {spec.omega:.6f}  lambda: ({lam[0]:.6f}, {lam[1]:.6f}, {lam[2]:.6f})  nmode: {spec.nmode}",
        ]
        if spec.reference == "uhf":
            lines.append("reference: UHF (open-shell)")
        return "\n".join(lines) + "\n"

    def _format_hf(self, result: SimulationResult) -> str:
        scf = result.scf
        return (
            "=== QED-HF ===\n"
            f"HF (Psi4): {scf['E_hf_psi4']:.12f} Ha\n"
            f"QED-HF (no ZPE): {scf['E_qed_hf']:.12f} Ha\n"
            f"z (coherent shift): {scf['z']:.8e}\n"
        )

    def _format_mp2(self, result: SimulationResult) -> str:
        mp2 = result.mp2 or {}
        return (
            "=== QED-MP2 ===\n"
            f"electronic MP2: {mp2.get('emp2_el', 0.0):.12f} Ha\n"
            f"photon term:    {mp2.get('emp2_ph', 0.0):.12f} Ha\n"
            f"total MP2 corr: {mp2.get('emp2_total', 0.0):.12f} Ha\n"
        )

    def _format_summary(self, result: SimulationResult) -> str:
        lines = ["=== Summary ===", f"QED-HF: {result.qedhf_energy:.12f} Ha"]
        if result.qedmp2_total is not None:
            lines.append(f"QED-MP2 total: {result.qedmp2_total:.12f} Ha")
        return "\n".join(lines)
