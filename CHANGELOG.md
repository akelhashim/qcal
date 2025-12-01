# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Workflow DAG
- Generic plotter for error rates for randomized benchmarks
- Super Guassian pulse envelope
- Empty layer to GateMapper in PyGSTiTranspiler

### Changed

- Flexible number of params for CZ gate amplitude calibration
- Default single-qubit GST to smq1Q_XYI (instead of smq1Q_XY)

### Deprecated

### Removed

### Fixed

- Added missing dimension for ZZ coupling in JAZZ
- 'length' -> 'time' in DD XY
- pygsti transpile bug for parallel gate layer
- pygsti dataset post-processing issue with transpiled circuits
- Node adjacency bug in DAG plotting

### Security

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

[Unreleased]: https://github.com/akelhashim/qcal/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/akelhashim/qcal/releases/tag/v0.0.2
[0.0.1]: https://github.com/akelhashim/qcal/releases/tag/v0.0.1
