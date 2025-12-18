"""
CLI entrypoint for QED-HF / QED-MP2 calculations.

The program expects a single input specification (JSON file) that contains geometry
and all numerical settings. The same logic can be imported directly from Python via
``run_simulation``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from qed import SimulationSpec, TextResultFormatter, run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QED-HF and optional QED-MP2")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to JSON file containing the simulation specification",
    )
    parser.add_argument(
        "--no-mp2",
        action="store_true",
        help="Skip the QED-MP2 correction",
    )
    return parser.parse_args()


def load_spec(config_path: Path, *, disable_mp2: bool = False) -> SimulationSpec:
    with config_path.open("r", encoding="utf8") as f:
        data: Dict[str, Any] = json.load(f)
    if disable_mp2:
        data["run_mp2"] = False
    return SimulationSpec.from_mapping(data)


def main() -> None:
    args = parse_args()
    spec = load_spec(args.config, disable_mp2=args.no_mp2)
    _, formatted = run_simulation(spec, formatter=TextResultFormatter())
    print(formatted)
    Path("output.qed").write_text(formatted, encoding="utf8")


if __name__ == "__main__":
    main()
