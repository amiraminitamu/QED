"""
Simulation specification and result containers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _ensure_lam_tuple(lam: Sequence[float]) -> Tuple[float, float, float]:
    values = tuple(float(x) for x in lam)
    if len(values) != 3:
        raise ValueError("lambda (lam) must have exactly 3 components")
    return values  # type: ignore[return-value]


def _normalize_reference(ref: str) -> str:
    ref_lower = ref.lower()
    if ref_lower not in {"rhf", "uhf"}:
        raise ValueError("reference must be either 'rhf' or 'uhf'")
    return ref_lower


@dataclass
class SimulationSpec:
    """
    Container for all inputs required to run QED-HF and optional QED-MP2.
    """

    geometry: str
    basis: str = "6-31g"
    omega: float = 5e-4
    lam: Tuple[float, float, float] = (1e-4, 0.0, 0.0)
    nmode: int = 1
    max_iter: int = 200
    conv_E: float = 1e-8
    conv_P: float = 1e-7
    scf_type: str = "pk"
    print_every: int = 10
    run_mp2: bool = True
    spin_factor: str = "closed_shell"
    reference: str = "rhf"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SimulationSpec":
        lam = _ensure_lam_tuple(data.get("lam", (1e-4, 0.0, 0.0)))
        return cls(
            geometry=data["geometry"],
            basis=data.get("basis", cls.basis),
            omega=float(data.get("omega", cls.omega)),
            lam=lam,
            nmode=int(data.get("nmode", cls.nmode)),
            max_iter=int(data.get("max_iter", cls.max_iter)),
            conv_E=float(data.get("conv_E", cls.conv_E)),
            conv_P=float(data.get("conv_P", cls.conv_P)),
            scf_type=data.get("scf_type", cls.scf_type),
            print_every=int(data.get("print_every", cls.print_every)),
            run_mp2=bool(data.get("run_mp2", cls.run_mp2)),
            spin_factor=data.get("spin_factor", cls.spin_factor),
            reference=_normalize_reference(str(data.get("reference", cls.reference))),
        )

    @classmethod
    def from_json(cls, path: Path) -> "SimulationSpec":
        with Path(path).open("r", encoding="utf8") as f:
            content = json.load(f)
        return cls.from_mapping(content)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "geometry": self.geometry,
            "basis": self.basis,
            "omega": self.omega,
            "lam": self.lam,
            "nmode": self.nmode,
            "max_iter": self.max_iter,
            "conv_E": self.conv_E,
            "conv_P": self.conv_P,
            "scf_type": self.scf_type,
            "print_every": self.print_every,
            "run_mp2": self.run_mp2,
            "spin_factor": self.spin_factor,
            "reference": self.reference,
        }


@dataclass
class SimulationResult:
    scf: Dict[str, Any]
    mp2: Optional[Dict[str, float]] = None
    spec: SimulationSpec = field(default_factory=SimulationSpec)

    @property
    def qedhf_energy(self) -> float:
        return float(self.scf["E_qed_hf"])

    @property
    def qedmp2_correction(self) -> Optional[float]:
        if self.mp2 is None:
            return None
        return float(self.mp2["emp2_total"])

    @property
    def qedmp2_total(self) -> Optional[float]:
        corr = self.qedmp2_correction
        if corr is None:
            return None
        return self.qedhf_energy + corr
