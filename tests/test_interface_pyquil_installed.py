"""Tests for qcal.interface.pyquil that require pyquil to be installed.

These complement test_interface_pyquil.py, which only covers the
pyquil-not-installed code path (and is skipped when pyquil is present).
Every test here transpiles to real pyquil objects, so the whole module
is skipped when pyquil is not installed.
"""
import numpy as np
import pytest

pytest.importorskip('pyquil')

from pyquil import Program  # noqa: E402
from pyquil.quilbase import Gate as QuilGate  # noqa: E402

from qcal.circuit import Circuit, CircuitSet, Cycle  # noqa: E402
from qcal.gates.single_qubit import (  # noqa: E402
    SX,
    SY,
    X90,
    Y90,
    H,
    Meas,
    Rz,
    SXdag,
    SYdag,
    X,
    Y,
    rx,
    rz,
)
from qcal.gates.two_qubit import CZ  # noqa: E402
from qcal.interface.pyquil.transpiler import (  # noqa: E402
    DEFAULT_GATEMAPPER,
    PyQuilTranspiler,
    to_pyquil,
)

ANGLES = (0.1, 0.2, 0.3)
RX90 = f'RX({np.pi / 2!r}) 0'


class TestSingleQubitGateUnitaries:
    """The emitted RZ/RX (or RZ_F12/RX_F12) sequences, in time order,
    implement the intended unitaries up to a global phase.
    """

    @pytest.mark.parametrize('gate_cls', [
        H, SX, SXdag, SY, SYdag, X, X90, Y, Y90
    ])
    @pytest.mark.parametrize('subspace', ['GE', 'EF'])
    def test_matches_gate_matrix(self, gate_cls, subspace):
        gate = gate_cls(0)
        insts = list(
            DEFAULT_GATEMAPPER[gate.name](0, subspace=subspace)
        )
        assert all(isinstance(inst, QuilGate) for inst in insts)

        suffix = '_F12' if subspace == 'EF' else ''
        rotation = {f'RX{suffix}': rx, f'RZ{suffix}': rz}
        unitary = np.eye(2)
        for inst in insts:
            unitary = rotation[inst.name](inst.params[0]) @ unitary

        overlap = np.trace(gate.matrix.conj().T @ unitary) / 2
        assert np.isclose(abs(overlap), 1)


class TestToPyquil:

    def test_plain_circuit(self):
        circuit = Circuit([
            Cycle({X90(0)}),
            Cycle({CZ(0, 1)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        program = to_pyquil(circuit, DEFAULT_GATEMAPPER)

        assert isinstance(program, Program)
        assert _lines(program) == [
            'DECLARE ro0 BIT[1]',
            'DECLARE ro1 BIT[1]',
            'FENCE',
            RX90,
            'FENCE',
            'CZ 0 1',
            'FENCE',
            'MEASURE 0 ro0[0]',
            'MEASURE 1 ro1[0]',
        ]

    def test_no_fence_between_cycles(self):
        circuit = Circuit([Cycle({X90(0)}), Cycle({Meas(0)})])
        program = to_pyquil(
            circuit, DEFAULT_GATEMAPPER, fence_between_cycles=False
        )
        assert 'FENCE' not in _lines(program)

    def test_cycle_replacement(self):
        entangle = Cycle({CZ(0, 1)})
        circuit = Circuit([entangle, Cycle({Meas(0), Meas(1)})])
        program = to_pyquil(
            circuit, DEFAULT_GATEMAPPER,
            cycle_replacement={entangle: 'ENTANGLE'},
        )
        lines = _lines(program)
        assert 'ENTANGLE 0 1' in lines
        assert 'CZ 0 1' not in lines

    def test_cycles_to_defcircuits_reuses_repeated_cycles(self):
        circuit = Circuit([
            Cycle({X90(0)}),
            Cycle({CZ(0, 1)}),
            Cycle({X90(0)}),
            Cycle({Meas(0), Meas(1)}),
        ])
        program = to_pyquil(
            circuit, DEFAULT_GATEMAPPER, cycles_to_defcircuits=True
        )
        text = str(program)
        assert text.count('DEFCIRCUIT') == 3
        assert [
            line for line in _lines(program) if line.startswith('Cycle_')
        ] == ['Cycle_0 0', 'Cycle_1 0 1', 'Cycle_0 0', 'Cycle_3 0 1']

    def test_circuit_for_loop(self):
        circuit = Circuit(
            [Cycle({X90(0)}), Cycle({CZ(0, 1)})] * 3
            + [Cycle({Meas(0), Meas(1)})]
        )
        program = to_pyquil(
            circuit, DEFAULT_GATEMAPPER, circuit_for_loop=True
        )
        lines = _lines(program)
        assert 'MOVE counter0[0] 3' in lines
        # The repeated body is written once, inside the loop.
        assert lines.count(RX90) == 1
        assert lines.count('CZ 0 1') == 1

    def test_randomized_compiling(self):
        circuit = _rc_circuit()
        program = to_pyquil(
            circuit, DEFAULT_GATEMAPPER, randomized_compiling=True
        )
        lines = _lines(program)

        assert type(program.rc_configuration).__name__ == (
            'RandomizedCompilingConfiguration'
        )
        assert program.source_phases['source_phases_q0'] == (
            pytest.approx([a / (2 * np.pi) for a in ANGLES * 2])
        )
        # Every Rz is replaced by its randomized unitary angle; the X90
        # pulses stay literal.
        rz = [line for line in lines if line.startswith('RZ')]
        assert rz == [
            f'RZ((2*pi)*unitaries_q0[{i}]) 0' for i in range(6)
        ]
        assert lines.count(RX90) == 4

    def test_randomized_compiling_rejects_non_zxzxz_gates(self):
        circuit = Circuit([
            Cycle({Y90(0)}),
            Cycle({CZ(0, 1)}),
            *_zxzxz(0),
            Cycle({Meas(0), Meas(1)}),
        ])
        with pytest.raises(ValueError, match="'Y90'.*ZXZXZ"):
            to_pyquil(
                circuit, DEFAULT_GATEMAPPER, randomized_compiling=True
            )

    def test_randomized_compiling_with_for_loop_raises(self):
        with pytest.raises(ValueError, match='for-loops'):
            to_pyquil(
                _rc_circuit(), DEFAULT_GATEMAPPER,
                randomized_compiling=True, circuit_for_loop=True,
            )

    def test_randomized_readout_only_randomizes_final_layer(self):
        program = to_pyquil(
            _rc_circuit(), DEFAULT_GATEMAPPER, randomized_readout=True
        )
        rz = [line for line in _lines(program) if line.startswith('RZ')]

        assert program.readout_source_phases[0] == (
            pytest.approx([a / (2 * np.pi) for a in ANGLES])
        )
        assert program.source_phases is program.readout_source_phases
        assert rz == [
            'RZ(0.1) 0', 'RZ(0.2) 0', 'RZ(0.3) 0',
            *(
                f'RZ((2*pi)*measurement_unitary_q0[{i}]) 0'
                for i in range(3)
            ),
        ]


class TestPyQuilTranspiler:

    def test_transpiles_every_circuit(self):
        circuits = [
            _rc_circuit(),
            Circuit([Cycle({X90(1)}), Cycle({Meas(1)})]),
        ]
        programs = PyQuilTranspiler().transpile(circuits)

        assert isinstance(programs, CircuitSet)
        assert programs.n_circuits == 2
        assert all(
            isinstance(p, Program) for p in programs.circuit
        )


def _lines(program: Program) -> list[str]:
    """Non-empty lines of a Program's Quil text.

    Args:
        program (Program): PyQuil Program.

    Returns:
        list[str]: Quil instructions, one per line.
    """
    return [line for line in str(program).splitlines() if line.strip()]


def _rc_circuit() -> Circuit:
    """A ZXZXZ template, a CZ, a final template, and measurements.

    Returns:
        Circuit: circuit compiled to ZXZXZ form.
    """
    return Circuit([
        *_zxzxz(0),
        Cycle({CZ(0, 1)}),
        *_zxzxz(0),
        Cycle({Meas(0), Meas(1)}),
    ])


def _zxzxz(q: int) -> list[Cycle]:
    """Cycles of a ZXZXZ template on qubit `q` with angles `ANGLES`.

    Args:
        q (int): qubit label.

    Returns:
        list[Cycle]: the five Rz/X90 cycles.
    """
    return [
        Cycle({Rz(q, ANGLES[0])}), Cycle({X90(q)}),
        Cycle({Rz(q, ANGLES[1])}), Cycle({X90(q)}),
        Cycle({Rz(q, ANGLES[2])}),
    ]
