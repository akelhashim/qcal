# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**qcal** is a Python package for quantum calibration, characterization, and benchmarking of superconducting quantum systems. It was developed for operating full-stack systems at the Advanced Quantum Testbed (Berkeley Lab). The three primary workflows are: **calibration** (optimize gate parameters), **characterization** (measure system properties), and **benchmarking** (assess performance metrics).

## Code Style

- Limit code lines to **79 characters** and docstring/comment lines to
  **79 characters**, per PEP 8.

## Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Lint and format
ruff check qcal/
ruff format qcal/
```

No CI runs tests — only the PyPI publish pipeline exists (`.github/workflows/publish.yml`, triggered on version tags).

## Architecture

### Circuit Representation

The core data model lives in [qcal/circuit.py](qcal/circuit.py):

- **`Cycle`** — a single clock-cycle; a set of gates acting on disjoint qubits in parallel. This is the fundamental unit of execution on superconducting hardware.
- **`Circuit`** — an ordered sequence of `Cycle`s. Circuits are built by composing cycles, not individual gates.
- **`CircuitSet`** — a DataFrame-backed container for groups of circuits with metadata (experimental variables for sweeps). Most calibration/benchmarking routines produce a `CircuitSet`.

### Gate Hierarchy

Gates are in [qcal/gate/](qcal/gate/). The base `Gate` class ([qcal/gate/gate.py](qcal/gate/gate.py)) holds a numpy unitary matrix and qubit labels. Single-qubit gates ([qcal/gate/single_qubit.py](qcal/gate/single_qubit.py)) include `H`, `X`, `Y`, `Z`, `RX`/`RY`/`RZ`, `X90`/`Y90`, and `Meas`. Two-qubit gates ([qcal/gate/two_qubit.py](qcal/gate/two_qubit.py)) include `CNOT`/`CX`, `CZ`, and `iSWAP`.

Gates support subspace specification (`'GE'` for ground-excited, `'EF'` for including leakage states), which flows through compilation and hardware execution.

### Results

[qcal/results.py](qcal/results.py) defines the `Results` class — the standard output of any measurement. It wraps a dict of bitstring → count and exposes `.counts`, `.probabilities`, `.n_shots`, `.n_qudits`, `.states`, `.ev` (expectation value), `.entropy`, and `.marginalize(qudit_index)`. Readout error correction via confusion matrices is applied through `.apply_readout_correction(confusion_matrix)`.

### Config System

[qcal/config.py](qcal/config.py) loads YAML hardware specification files. The `Config` object is the central store for per-qubit and per-pair parameters (frequencies, amplitudes, phases, native gate sets, readout settings). Calibration protocols read from and write back to this config.

### Calibration / Characterization / Benchmarking Pattern

All three protocol families follow the same abstract structure (see base classes in [qcal/calibration/calibration.py](qcal/calibration/calibration.py) and [qcal/characterization/characterize.py](qcal/characterization/characterize.py)):

1. `generate_circuits()` — produce a `CircuitSet` parametrized by a sweep
2. Execute via `QPU.run()`
3. `analyze()` — fit `Results` to a model (via [qcal/fitting/](qcal/fitting/))
4. `final()` — update `Config` with optimal values
5. `plot()` — visualize

Calibration subclasses cover single-qubit gates, CZ, and readout ([qcal/calibration/](qcal/calibration/)). Benchmarking protocols include RB, cycle benchmarking (CB), interleaved RB, Pauli noise randomization (PNR), XEB, and MCM-specific variants ([qcal/benchmarking/](qcal/benchmarking/)). Characterization covers coherence (T1/T2), spectroscopy, and tomography ([qcal/characterization/](qcal/characterization/)).

### QPU Execution Layer

[qcal/qpu/qpu.py](qcal/qpu/qpu.py) defines the abstract `QPU` base class. It takes a `Config`, compiler, transpiler, and measurement parameters, and exposes `.run(circuit)` → `Results`. Concrete backends live in [qcal/backend/](qcal/backend/) (QUBIC and Rigetti).

### Compilation Pipeline

Circuits are compiled via [qcal/compilation/compiler.py](qcal/compilation/compiler.py) (wraps BQSKit) and transpiled through interface-specific adapters in [qcal/interface/](qcal/interface/):

- **`interface/bqskit/`** — primary compiler; maps native gate sets (Rz, X, X90, CNOT, CZ, iSWAP)
- **`interface/pygsti/`** — Gate Set Tomography (GST) integration; supports Clifford compilation with Pauli randomization
- **`interface/pyquil/`** — Rigetti Quil programs
- **`interface/trueq/`** — True-Q benchmarking integration

### Global Settings

[qcal/settings.py](qcal/settings.py) is a module-level singleton. Key flags: `settings.save_data` (default `False`) and `settings.data_save_path`. Import as `import qcal.settings as settings`.
