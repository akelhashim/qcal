# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- `optuna`/`optunahub` as a qcal dependency
- `plot_error_rates` takes a single dict mapping qubit labels to `ufloat`
  error rates, instead of separate `error_rates` and `uncertainties` dicts
- Renamed the `noise_model` argument and property of
  `DensityMatrixSimulator` to `error_model`
- Renamed the `step()` method of the `Adam`, `CMA`, `LQG`, and `LQR`
  optimizers to `ask()`
- `Calibration` and `Characterize` are abstract base classes: subclasses
  must implement `analyze()` and `generate_circuits()`

### Added

- `Circuit.prepare()` method to prepend a cycle which prepares a Pauli
  eigenstate
- `unitary` property for `Circuit` and `Cycle` classes
- Native Cycle Benchmarking (CB) implementation
- Randomized compiling (RC) support for the PyQuil transpiler, via
  `pyquil._qpu.randomized_compiling`
- Randomized readout: twirl the final pre-measurement single-qubit
  gates, usable independently or alongside randomized compiling
- `cycle_replacement` and `cycles_to_defcircuits` can be combined with
  randomized compiling and randomized readout in the PyQuil transpiler.
  Cycles of X90/SX pulses, two-qubit gates, mid-circuit measurements, or
  terminal measurements can be replaced with DEFCIRCUITs; replacing a
  randomized cycle containing an `Rz`/`VirtualZ` raises a `ValueError`,
  and `cycles_to_defcircuits` leaves such cycles as literal gates
- `decompose_to_zxzxz` parameter for Cycle Benchmarking circuit
  generation
- `RandomizedReadout.fidelity` property: the all-zero state probability
  as a `ufloat` with binomial standard error
- `pauli_to_cycle` utility to convert a Pauli string into a `Circuit`,
  with ZXZXZ decomposition and caching
- `conjugate_pauli` utilities for efficiently tracking Pauli operator
  transformations under conjugation by Clifford gates, cycles, and
  circuits
- `draw_qpu_heatmap` for plotting variable lattice-based values
- `qp_readout_correction` for applying quasi-probabilistic readout correction
- Post-processing of ragged measurement results, where qubits in a circuit
  are measured a different number of times (e.g. ancillas measured
  mid-circuit every round, but data qubits only at the end), via the
  `measurement_cycles` argument of `PostProcessor` and
  `construct_dataframes`
- `circuit_measurement_cycles` utility to find which read of each qubit is
  made in each measurement cycle of a circuit
- `PerShotResults` class for storing the per-shot outcomes of every
  measurement cycle of a circuit (e.g. for decoding), with
  `measurement_record()` and `cycle_results` DataFrame views indexed by the
  original shot numbers
- `PostProcessor.per_shot_results`, and a `Circuit.per_shot_results`
  property (with a setter) for recording the per-shot outcomes of a circuit
- `keep_per_shot_results` argument of `QPU` for keeping the per-shot
  outcomes of every measurement cycle in each circuit's `per_shot_results`
  (defaults to False, since they can be large)
- `aggregate` argument of `PostProcessor` for the pass that aggregates
  per-shot outcomes into counts (defaults to `compute_counts`)
- `OptunaHub` optimizer for using any OptunaHub sampler with `Optimize`
- `global_min` argument of `Optimize` to choose the optimal parameters of
  all qubit labels jointly, from the iteration with the smallest combined
  loss, along with the `opt_loss_global` property

### Changed

- Deferred `Results` DataFrame initialization until first access, to
  avoid unnecessary DataFrame construction for transient results
- Optimized circuit plotting for large circuits by omitting per-gate
  text labels and matrix hover-text details above a node-count
  threshold
- Replaced storing values and their errors in dictionaries with ufloats.
- `PostProcessor` runs per-shot `passes` first, then `aggregate`. An
  aggregation pass at the end of `passes` is moved to `aggregate` for
  backwards compatibility; one elsewhere in `passes` raises a `ValueError`
- `QuantumInstrumentGST` passes `compute_conditional_counts` to
  `PostProcessor` as `aggregate` rather than as the last of its `passes`
- `QPU` raises a `ValueError` instead of an `AssertionError` for an
  unsupported `n_levels`
- `Optimize.run()` only saves the config when
  `settings.Settings.save_data` is enabled
- The `Emulator` stores each circuit's final simulated state object in a
  `'state'` column, instead of a pretty-printed string in a `'states'`
  column; call `.pretty_print()` on an entry for a readable string
- `Gate` equality and hashing include the gate's matrix, so generic gates
  (e.g. from `merge_gates`) with different unitaries are no longer equal
- CB samples exactly `n_decays` distinct, non-identity Pauli decays, using
  all of them when `n_decays` is at least the number available

### Deprecated

### Removed

- `reciprocal_uncertainty`, `uncertainty_of_exp`, `uncertainty_of_product`,
  and `uncertainty_of_sum` from `qcal.math.utils`; use `ufloat` instead

### Fixed

- PyQuil transpiler internal variable name bug
- `DensityMatrixSimulator` returning negative probabilities
- DataFrame concat warning for `config.coherence_times`
- Circuit plotting bug when `MCM` is interpreted as a multi-qubit gate
- `compute_conditional_counts` not relabeling mid-circuit measurement
  outcomes of 2 and 3 (e.g. leakage) as `p2` and `p3`
- Pauli conjugation and CB depth checks returning cached results for a
  different merged gate or cycle on the same qubits
- CB expected signs for cycles whose Pauli conjugation is not an
  involution (e.g. S followed by H, which maps X → Y → Z)
- CB keeping an all-identity Pauli in its decays
- `conjugate_pauli` accepting some non-Clifford gates instead of raising a
  `ValueError`
- `Results.plot` failing with plotly 6
- `readout_correction` truncating instead of rounding corrected counts,
  and skipping results that do not contain every level
- `RandomizedReadout.results` raising before the experiment is run
- `format_gate_text` ignoring `include_matrix`, so large circuit plots
  still included gate matrices in their hover text
- `draw_qpu_heatmap` node hover text showing the figure title instead of
  `cbar_label`
- `draw_circuit` showing the same hover text for different merged gates on
  the same qubits
- PyQuil transpiler raising for phase gates outside the final layer with
  randomized readout and `cycles_to_defcircuits`

### Security

## [5.0.0] - 2026-11-03

### Breaking

- Renamed `QPU.run()`'s `save` kwarg to `auto_save` for clarity
- `Transpiler.transpile()` is now an abstract method (`Transpiler` is an
  `ABC`); the base class and any subclass that doesn't override it can no
  longer be instantiated

### Added

- Comprehensive unit tests

### Changed

- Default plotting in MCB to plotly
- Moved `plot_error_rates` to `qcal/plotting/error_rates.py`

### Removed

- Removed `qcal/characterization/tomography/analysis.py`
- Removed empty, unused `qcal/sequence/sequencer.py`

### Fixed

- `cosine_square()` crashing with a `ZeroDivisionError` for `ramp_fraction=0.0`,
  and silently returning a pulse of the wrong length for `-0.5 <
  ramp_fraction < 0`; a negative `ramp_fraction` (or one `> 0.5`) now
  consistently produces a one-sided ramp
- `XX_N()` partitioning qubits into "even"/"odd" by their position in the
  input list instead of by qubit label parity (unlike `XY_N()`), so e.g.
  `XX_N([1, 3], ...)` incorrectly syncopated two same-parity qubits
- `QPU.acquire()` racing against the next batch's `generate_sequence()` when
  running with `jit=True`, which could overwrite `self._sequence` before
  `acquire()` read it; `acquire()` now reads a race-safe `active_sequence`
  snapshot instead
- `QPU._initialize()` silently overwriting the `exp_id` (and orphaning the
  save folder) that GST/CRB/MCB protocols set up before calling `qpu.run()`;
  added `DataMananger.reserve_exp_id()` so those protocols can protect their
  `exp_id` from the next `generate_exp_id()` call
- `GateMapper.call()` treating a `str` result as a generic `Iterable` and
  crashing with a confusing `AttributeError` instead of the intended
  `TypeError` for an unsupported gate-function return type
- `Config.hardware` crashing (or silently returning an empty table) for any
  config, due to using `pd.DataFrame.from_dict(orient='index')` on a dict of
  opaque values instead of table rows
- `pygsti_pspec()` dropping a two-qubit gate's availability whenever the
  config declared that qubit pair in descending order (e.g. a ring
  topology's wraparound edge `(7, 0)`), since it only matched ascending
  qubit-pair tuples
- `Phase(gate='X', relative_phase=True)` crashing by indexing into the
  first character of the (string, not list) param path; `Phase(gate='X')`'s
  default param path also pointed at a pulse segment that doesn't exist for
  single-segment X pulses, crashing `final()`
- `update_clifford_labels()` crashing on any single-qubit Clifford layer,
  since a pyGSTi `Label` with exactly one component collapses to a form
  whose direct iteration yields raw tuple fields instead of sub-`Label`s
- `TruthTable.analyze()` silently returning an all-zero truth table (no
  error) when `qubits` is a strict superset of `circuit.qubits` (e.g. an
  untouched ancilla included only for state preparation), because it
  marginalized away those extra qubits before looking up the (differently-
  shaped) `self._states` keys

## [4.0.1] - 2026-09-18

### Fixed

- RPE analysis raising a KeyError when encountering zero counts

## [4.0.0] - 2026-09-04

### Breaking

- Added rigetti-quax as a dependency
- Rename `qcal.gate module` to `qcal.gates`
- Rename `Examples` to `examples` and `Tutorials` to `tutorials`

### Added

- `StateVectorSimulator` for simulating ideal circuit outputs
- `DensityMatrixSimulator` for simulating noisy circuits
- Error models module for building error models for simulation
- Single- and two-qutrit gates
- Emulator backend for emulating experiments
- Simulation and Emulation example notebooks

### Changed

- gate.unitary embeds a subspace unitary in qutrit dimension if subspace='EF'

### Fixed

- `circuit.draw()` didn't correctly draw qudit gates
- `plot_mock_sequence` not handling qudits correctly

## [3.2.0] - 2026-08-17

### Added

- `barrier_between_cycles` flag on the QubiC transpilation path
  (`to_qubic()`, `Transpiler`, `QubicQPU`). Defaults to True (unchanged
  behavior). When False, no automatic barriers are inserted between
  cycles: only explicit Barrier objects synchronize qubits, enabling
  desynchronized per-qubit execution (e.g. concurrent measurements with
  per-qubit randomized delays)

## [3.1.1] - 2026-06-15

### Fixed

- Bug: qc.__file__ is None when importing qcal as qc; changed to __file__
- Bug: QuantumInstrumentGST failing when the circuit depth [1] is written as a tuple
- Include default_experiments in pyproject.toml when installing via pip

## [3.1.0] - 2026-06-12

### Added

- Default single- and two-qubit CRB experiment designs
- Default single- and two-qubit GST experiment designs

## [3.0.0] - 2026-06-12

### Breaking

- Changed calculate_cz_frequency to calculate_parametric_cz_frequency

### Added

- Twirled RPE for ZZ phase estimation

### Changed

- JAZZ can measure static and driven ZZs in full qutrit space
- Default ploly save option to png
- Automatically map characterization methods to 0 for EF subspace experimentas by default

## [2.0.0] - 2026-06-12

### Breaking

- Change dynamical decoupling XY name to XY_N

### Added

- Circuit for-loops for pyquil
- Automatic leakage analysis for CRB
- plot_mock_sequence for psuedo-visualization of pulse sequence of circuits
- Fit for decaying cosine with exponential baseline
- __eq__ for Gate, Cycle, Circuit, and CircuitSet
- PyQuil cycles_to_defcircuits kwarg in transpilation
- XX-type (CPMG) dynamical decoupling sequencies
- Make simultaneous DD sequences syncopated for canceling ZZ
- PyQuil helper function for adding DD sequences during targeted operations
- Excited State Promotion (ESP) for PyQuil programs
- Active reset for PyQuil programs
- cycle_replacement option for custom DEFCIRCUIT names in PyQuil transpilation

### Changed

- uncertainty_of_product to account for correlations using the covariance
- T2* EF fit to decaying cosine with exponential baseline
- Default time spacing for coherence measurements to exponential
- Declare individual classical registers for qubit measurements
- 2Q gate definitions now accept either Gate((0, 1)) or Gate(0, 1)

### Removed

### Fixed

- length -> time in example config
- Circuit generation bug for simultaneous single-qubit GST on two qubits

## [1.2.0] - 2026-05-13

### Added

- Expand pyGSTi transpilers to support single-qubit Cliffords
- pyGSTi Pauli generation and measurement grouping utilities
- pyGSTi Pauli randomization for Clifford circuits
- Mirror circuit benchmarking

### Fixed

- Post-processing results for different depth circs

## [1.1.1] - 2026-04-24

### Fixed

- Remove [skip ci] and use the actor check instead

## [1.1.0] - 2026-04-24

### Added

- Just-in-time (JIT) compilation & transpilation in QPU for faster measurements

## [1.0.6] - 2026-03-27

### Added

- pyGSTi support for iSWAP gates

## [1.0.5] - 2026-03-24

### Changed

- Settings default save_data to False

### Fixed

- Frequency plotting now properly handles empty frequency ranges

## [1.0.4] - 2026-03-23

### Changed

- IPython version requirement updated to support older versions
- Faster ReadoutCalibration plotting

### Fixed

- Delay insertion in pyquil programs

## [1.0.3] - 2026-03-19

### Added

- CircuitSet indexing with slices
- Pyquil transpiler support for subspace parameter

### Changed

- save_to_pickle checks for .pkl extension in filename

## [1.0.2] - 2026-03-17

### Added

- Exponential fit with arbitrary base
- Native compiler
- Optional params for single-qubit Amplitude and Frequency calibration
- Pyquil utils for setting parameters in defcals

### Changed

- CRB refactor: plotly plotting, error rate calculation, and faster circuit generation
- BQSKit refactor: parallel compilation and better error handling
- Release script: updated to handle version bumping and changelog updates

### Fixed

- config does not display native gates if 'pulse' is not specified

## [1.0.1] - 2026-03-09

### Added

- Vendored pyRPE dependency

### Removed

- External dependency on pyRPE (not supported on pypi)

## [1.0.0] - 2026-03-06

This release makes pyGSTi and pyRPE dependencies for qcal. It also includes 
support for running pyquil circuits, as well as GST for quantum instruments.

### Added

- GST modes kwarg for optional model fitting
- GST plotting for easier visualization of results
- GST properties for easier access to results
- GST for quantum instruments
- mcm_results property for Circuit object
- qcal -> pyquil transpiler
- PostProcessor class for better handling of post-processing results with optional passes
- Simultaneous GST
- Circuit.join() method for easily joining multiple circuits along cycle number

### Changed

- Updated dependencies (added pygsti and pyrpe)
- RPE, CRB, & GST no longer requires saving to disk

## [0.0.3] - 2025-12-09

This release includes some minor changes and additions, as well as a generic
workflow Directed Acyclical Graph (DAG) for calibration workflows. It also
includes a new section for Example configs and tutorials (in the form of
jupyter notebooks).

### Added

- Workflow DAG
- Generic plotter for error rates for randomized benchmarks
- Super Guassian pulse envelope
- Empty layer to GateMapper in PyGSTiTranspiler
- save_raw_data will save all classified reads in circuits dataframe for qubic
post-processing
- Save GST results object as a pkl file
- examples: config and tutorials

### Changed

- Flexible number of params for CZ gate amplitude calibration
- Default single-qubit GST to smq1Q_XYI (instead of smq1Q_XY)

### Fixed

- Added missing dimension for ZZ coupling in JAZZ
- 'length' -> 'time' in DD XY
- pygsti transpile bug for parallel gate layer
- pygsti dataset post-processing issue with transpiled circuits
- Node adjacency bug in DAG plotting

## [0.0.2] - 2025-11-05

### Added

- Copyright notice
- License agreement

## [0.0.1] - 2025-10-28

### Added

- Initial release
- Basic gate calibration framework
- Single- and two-qubit characterization tools
- Single- and two-qubit benchmarking methods
- Advanced compilation tools

[Unreleased]: https://github.com/akelhashim/qcal/compare/v5.0.0...HEAD
[5.0.0]: https://github.com/akelhashim/qcal/releases/tag/v5.0.0
[4.0.1]: https://github.com/akelhashim/qcal/releases/tag/v4.0.1
[4.0.0]: https://github.com/akelhashim/qcal/releases/tag/v4.0.0
[3.2.0]: https://github.com/akelhashim/qcal/releases/tag/v3.2.0
[3.1.1]: https://github.com/akelhashim/qcal/releases/tag/v3.1.1
[3.1.0]: https://github.com/akelhashim/qcal/releases/tag/v3.1.0
[3.0.0]: https://github.com/akelhashim/qcal/releases/tag/v3.0.0
[2.0.0]: https://github.com/akelhashim/qcal/releases/tag/v2.0.0
[1.2.0]: https://github.com/akelhashim/qcal/releases/tag/v1.2.0
[1.1.1]: https://github.com/akelhashim/qcal/releases/tag/v1.1.1
[1.1.0]: https://github.com/akelhashim/qcal/releases/tag/v1.1.0
[1.0.6]: https://github.com/akelhashim/qcal/releases/tag/v1.0.6
[1.0.5]: https://github.com/akelhashim/qcal/releases/tag/v1.0.5
[1.0.4]: https://github.com/akelhashim/qcal/releases/tag/v1.0.4
[1.0.3]: https://github.com/akelhashim/qcal/releases/tag/v1.0.3
[1.0.2]: https://github.com/akelhashim/qcal/releases/tag/v1.0.2
[1.0.1]: https://github.com/akelhashim/qcal/releases/tag/v1.0.1
[1.0.0]: https://github.com/akelhashim/qcal/releases/tag/v1.0.0
[0.0.3]: https://github.com/akelhashim/qcal/releases/tag/v0.0.3
[0.0.2]: https://github.com/akelhashim/qcal/releases/tag/v0.0.2
[0.0.1]: https://github.com/akelhashim/qcal/releases/tag/v0.0.1
