# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

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

[Unreleased]: https://github.com/akelhashim/qcal/compare/v1.0.4...HEAD
[1.0.4]: https://github.com/akelhashim/qcal/releases/tag/v1.0.4
[1.0.3]: https://github.com/akelhashim/qcal/releases/tag/v1.0.3
[1.0.2]: https://github.com/akelhashim/qcal/releases/tag/v1.0.2
[1.0.1]: https://github.com/akelhashim/qcal/releases/tag/v1.0.1
[1.0.0]: https://github.com/akelhashim/qcal/releases/tag/v1.0.0
[0.0.3]: https://github.com/akelhashim/qcal/releases/tag/v0.0.3
[0.0.2]: https://github.com/akelhashim/qcal/releases/tag/v0.0.2
[0.0.1]: https://github.com/akelhashim/qcal/releases/tag/v0.0.1
