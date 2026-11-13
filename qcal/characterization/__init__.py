"""Characterization protocols for superconducting qubits.

This subpackage contains protocols for measuring the properties of a
quantum processor: coherence times
(:mod:`~qcal.characterization.coherence`), readout
(:mod:`~qcal.characterization.readout`), spectroscopy
(:mod:`~qcal.characterization.spectroscopy`), phase estimation
(:mod:`~qcal.characterization.phase_estimation`), and tomography
(:mod:`~qcal.characterization.tomography`). All protocols follow the
common pattern defined by
:class:`~qcal.characterization.characterize.Characterize`: generate
circuits, execute them on a QPU, analyze the results, and write the
characterized values back to the config.
"""
