# Installation

qcal requires Python 3.11 or later.

## From PyPI

```bash
pip install quantum-calibration
```

The package is installed as `quantum-calibration` and imported as `qcal`:

```python
import qcal as qc
```

## From source

```bash
git clone https://github.com/akelhashim/qcal.git
cd qcal
pip install -e ".[dev]"
```

## Optional dependencies

Some interfaces and hardware backends need extra packages that are not
installed by default:

| Feature | Module | Requires |
|---|---|---|
| QubiC hardware backend | {mod}`qcal.backend.qubic` | QubiC software stack |
| True-Q benchmarking | {mod}`qcal.interface.trueq` | `trueq` |
| Superstaq compilation | {mod}`qcal.interface.superstaq` | `cirq-superstaq` |

## Building the documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.
