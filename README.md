# QED-HF / QED-MP2 Toolkit

This repository contains a modular Python implementation of QED-HF and an optional QED-MP2 correction. You can run it either from the command line using a single JSON specification or directly from Python by importing the package.

## Requirements

- Python 3.11+
- [`psi4`](http://www.psicode.org/) and NumPy available in your environment

## Quick start (CLI)

1. Prepare a JSON input file with your geometry and options (see [`examples/sample_config.json`](examples/sample_config.json)).
2. Run:

   ```bash
   python main.py examples/sample_config.json
   ```

3. To skip the MP2 correction:

   ```bash
   python main.py examples/sample_config.json --no-mp2
   ```

The formatted result is printed to the terminal and also written to `output.qed`.

## Quick start (Python)

No file I/O is required—build the specification in Python and run directly:

```python
from qed import SimulationSpec, run_simulation

geom = r"""
O         -2.28150       -1.15110       -0.00080
O         -2.40350        1.11860        0.00060
N          0.29760       -1.18180        0.00040
N          1.83640        1.16300       -0.00030
C         -0.26690        0.04210       -0.00020
C          0.48890        1.19390       -0.00040
C          1.64490       -1.21270        0.00030
C         -1.71670        0.08910        0.00030
C          2.40080       -0.06100        0.00010
H          0.04920        2.18440       -0.00070
H          2.09920       -2.19570        0.00060
H          3.48350       -0.08670        0.00010
H         -3.26150       -1.10530       -0.00100
symmetry C1
"""

spec = SimulationSpec(
    geometry=geom,
    basis="STO-3g",
    omega=0.13,
    lam=(0.01, 0.0, 0.04),
    nmode=1,
    max_iter=1000,
    print_every=5,
    scf_type="pk",
    reference="rhf",        # set to "uhf" for open-shell
    run_mp2=True,          # set False to skip MP2
    spin_factor="closed_shell",
)

result, formatted = run_simulation(spec)

print(formatted)           # human-readable summary
print(result.qedhf_energy) # structured access to QEDHF
print(result.qedmp2_total) # MP2 total (if enabled)
```

To run an open-shell calculation, set `reference="uhf"` in the specification; the workflow
will automatically switch to UHF-based QED-HF and the corresponding open-shell QED-MP2
correction.

You can supply your own formatter by subclassing `qed.TextResultFormatter` and passing it to `run_simulation(..., formatter=...)` to change how results are printed.

## Input specification (JSON)

| Field        | Type            | Default   | Description                                           |
|--------------|-----------------|-----------|-------------------------------------------------------|
| `geometry`   | string          | required  | Molecule geometry block accepted by Psi4.             |
| `basis`      | string          | `6-31g`   | Basis set.                                           |
| `omega`      | float           | `5e-4`    | Photon frequency.                                    |
| `lam`        | array[3]        | `[1e-4,0,0]` | Coupling vector (λx, λy, λz).                    |
| `nmode`      | int             | `1`       | Number of photon modes.                               |
| `max_iter`   | int             | `200`     | Max SCF iterations.                                   |
| `conv_E`     | float           | `1e-8`    | Energy convergence threshold.                         |
| `conv_P`     | float           | `1e-7`    | Density convergence threshold.                        |
| `scf_type`   | string          | `pk`      | SCF type (Psi4 option).                               |
| `reference`  | string          | `rhf`     | SCF reference (`rhf` for closed-shell, `uhf` for open-shell). |
| `print_every`| int             | `10`      | Iteration print frequency.                            |
| `run_mp2`    | bool            | `true`    | Whether to run QED-MP2 after QED-HF.                  |
| `spin_factor`| string          | `closed_shell` | MP2 spin factor (`closed_shell` or `ghf_equivalent`). |

## Example configuration

See [`examples/sample_config.json`](examples/sample_config.json) for a ready-to-run input using the same geometry as the original script.
