"""Tests for `_U2Randomization.build_memory_map` per-qubit sources."""
import numpy as np

from qcal.compilation.decompositions import zxzxz_to_unitary
from qcal.interface.pyquil.randomized_compiling import _U2Randomization

# I, X, Y, Z as flat ZXZXZ phases in turns (as in `transpile_circuit`).
SOURCE = [0.0, 0.5, 0.5, -0.25, 0.0, -0.25, 0.5, 0.0, 0.0, 0.25, 0.5, -0.25]
SEEDS = {0: 11, 1: 22}


def _config() -> _U2Randomization:
    return _U2Randomization(
        qubits=[0, 1],
        destination_names=lambda q: f"dest_q{q}",
        seed_names=lambda q: f"seed_q{q}",
        source_names=lambda q: f"src_q{q}",
        source_unitary_phases=SOURCE,
    )


def _unitaries(turns):
    return [
        zxzxz_to_unitary(2 * np.pi * np.asarray(turns[i:i + 3]))
        for i in range(0, len(turns), 3)
    ]


def _equal_up_to_phase(a, b) -> bool:
    return np.isclose(abs(np.trace(a.conj().T @ b)), 2)


def test_default_source_per_qubit():
    memory_map = _config().build_memory_map(SEEDS)
    for q in SEEDS:
        assert memory_map[f"src_q{q}"] == SOURCE
        assert memory_map[f"seed_q{q}"] == [SEEDS[q]]
        assert memory_map[f"dest_q{q}"] == [0.0, 0.5, 0.5]


def test_phases_merged_into_each_qubit_source():
    phases = {0: (0.1, 0.2, 0.3), 1: (0.4, 0.0, -0.2)}
    memory_map = _config().build_memory_map(SEEDS, phases_by_qubit=phases)

    assert memory_map["src_q0"] != memory_map["src_q1"]
    for q, qubit_phases in phases.items():
        provided = zxzxz_to_unitary(2 * np.pi * np.asarray(qubit_phases))
        actual = _unitaries(memory_map[f"src_q{q}"])
        expected = [r @ provided for r in _unitaries(SOURCE)]
        assert len(actual) == len(expected) == 4
        assert all(map(_equal_up_to_phase, actual, expected))
