# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Deprecated

### Removed

- Artificial detuning for T2echo and T2XY experiments

### Fixed

- RPE uncertainty to include uncertainty due to shot noise
- Circuit get_index, remove, replace methods
- CircuitSet subset and sum_results methods

### Security

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
- Examples: config and Tutorials

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

[Unreleased]: https://github.com/akelhashim/qcal/compare/v1.2.0...HEAD
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
