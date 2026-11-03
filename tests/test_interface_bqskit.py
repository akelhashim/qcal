"""Unit tests for qcal.interface.bqskit.transpiler.

bqskit is a hard dependency (always installed), so unlike
test_interface_{trueq,superstaq,pyquil}.py this isn't about optional-
import behavior -- it's direct coverage of to_bqskit()/to_qcal()/
BQSKitTranspiler, which qcal.compilation.compiler.Compiler
(tests/test_compiler.py) already exercises indirectly through a full
compile() pipeline.

to_bqskit() always wraps each qcal gate as a generic
ConstantUnitaryGate(matrix) (confirmed: str(ConstantUnitaryGate(...))
is always 'ConstantUnitaryGate', regardless of the original gate), so
it is not meant to round-trip directly back through to_qcal() without
an actual bqskit synthesis pass in between -- that's the real pipeline
in qcal/interface/bqskit/compiler.py. to_qcal() instead expects a
circuit built from bqskit's *named* native gates (RZGate, SqrtXGate,
CNOTGate, ...), which is what a real compile() call produces. Each
direction is tested against the input form it actually expects.
"""
import numpy as np
import pytest
from bqskit.ir import Circuit as bqCircuit
from bqskit.ir.gates import (
    BarrierPlaceholder,
    CNOTGate,
    MeasurementPlaceholder,
    RZGate,
    SqrtXGate,
)
from bqskit.ir.gates import ConstantUnitaryGate as UGate

from qcal.circuit import Barrier, Circuit, Cycle
from qcal.gates.single_qubit import X90, H, Meas, Rz
from qcal.gates.two_qubit import CNOT, CZ
from qcal.interface.bqskit.transpiler import (
    BQSKitTranspiler,
    to_bqskit,
    to_qcal,
)
from qcal.transpilation.transpiler import Transpiler
from qcal.transpilation.utils import GateMapper


class TestToBqskit:

    def test_single_qubit_gate_becomes_constant_unitary_gate(self):
        circuit = Circuit([Cycle({H(0)})])
        bqc = to_bqskit(circuit)

        assert bqc.num_qudits == 1
        ops = list(bqc.operations_with_cycles())
        assert len(ops) == 1
        _, op = ops[0]
        assert str(op.gate) == 'ConstantUnitaryGate'
        assert op.location == (0,)

    def test_preserves_unitary(self):
        circuit = Circuit([Cycle({H(0)})])
        bqc = to_bqskit(circuit)
        np.testing.assert_allclose(
            bqc.get_unitary().numpy, circuit.unitary, atol=1e-10
        )

    def test_two_qubit_gate_preserves_unitary_and_location(self):
        circuit = Circuit([Cycle(CZ(0, 1))])
        bqc = to_bqskit(circuit)

        assert bqc.num_qudits == 2
        _, op = next(bqc.operations_with_cycles())
        assert tuple(op.location) == (0, 1)
        np.testing.assert_allclose(
            bqc.get_unitary().numpy, circuit.unitary, atol=1e-10
        )

    def test_measurement_becomes_measurement_placeholder(self):
        circuit = Circuit([Cycle({Meas(0)})])
        bqc = to_bqskit(circuit)

        _, op = next(bqc.operations_with_cycles())
        assert str(op.gate) == 'measurement'
        assert op.location == (0,)

    def test_barrier_cycle_becomes_barrier_placeholder(self):
        circuit = Circuit([Cycle({H(0)}), Barrier((0,))])
        bqc = to_bqskit(circuit)

        ops = list(bqc.operations_with_cycles())
        assert str(ops[1][1].gate) == 'barrier'

    def test_multi_cycle_circuit_preserves_cycle_order(self):
        circuit = Circuit([Cycle({H(0)}), Cycle({H(0)})])
        bqc = to_bqskit(circuit)

        cycles = [cyc for cyc, _ in bqc.operations_with_cycles()]
        assert cycles == [0, 1]


class TestToQcal:

    def _gate_mapper(self):
        return GateMapper({
            'barrier':     Barrier,
            'measurement': Meas,
            'CNOTGate':    CNOT,
            'RZGate':      Rz,
            'SqrtXGate':   X90,
        })

    def test_maps_native_gates_by_name(self):
        bqc = bqCircuit(1)
        bqc.append_gate(SqrtXGate(), (0,))
        bqc.append_gate(RZGate(), (0,), [0.5])

        tc = to_qcal(bqc, self._gate_mapper())

        assert len(tc) == 2
        assert tc[0] == Cycle({X90(0)})
        assert tc[1] == Cycle({Rz(0, 0.5)})

    def test_two_qubit_gate_with_no_params(self):
        bqc = bqCircuit(2)
        bqc.append_gate(CNOTGate(), (0, 1))

        tc = to_qcal(bqc, self._gate_mapper())

        assert len(tc) == 1
        assert tc[0].qubits == (0, 1)

    def test_measurement_appends_measure_cycle(self):
        bqc = bqCircuit(1)
        bqc.append_gate(SqrtXGate(), (0,))
        bqc.append_gate(
            MeasurementPlaceholder([('0', 1)], {0: ('0', 0)}), (0,)
        )

        tc = to_qcal(bqc, self._gate_mapper())

        assert tc[-1] == Cycle({Meas(0)})

    def test_barrier_uses_gate_mapper(self):
        bqc = bqCircuit(2)
        bqc.append_gate(SqrtXGate(), (0,))
        bqc.append_gate(BarrierPlaceholder(2), (0, 1))
        bqc.append_gate(SqrtXGate(), (0,))

        tc = to_qcal(bqc, self._gate_mapper())

        assert any(cyc.is_barrier for cyc in tc)

    def test_unmapped_gate_raises(self):
        bqc = bqCircuit(1)
        bqc.append_gate(UGate(np.eye(2)), (0,))

        with pytest.raises(Exception, match='non-native gate'):
            to_qcal(bqc, self._gate_mapper())

    def test_same_cycle_ops_are_grouped_together(self):
        bqc = bqCircuit(2)
        bqc.append_gate(SqrtXGate(), (0,))
        bqc.append_gate(SqrtXGate(), (1,))

        tc = to_qcal(bqc, self._gate_mapper())

        assert len(tc) == 1
        assert tc[0].n_gates == 2


class TestBQSKitTranspilerConstruction:

    def test_is_a_transpiler(self):
        assert isinstance(BQSKitTranspiler(), Transpiler)

    def test_default_to_qcal_gate_mapper(self):
        t = BQSKitTranspiler()
        for key in (
            'barrier', 'measurement', 'CNOTGate', 'CZGate', 'ISwapGate',
            'RZGate', 'SqrtXGate', 'XGate',
        ):
            assert key in t.gate_mapper

    def test_to_bqskit_defaults_to_empty_gate_mapper(self):
        t = BQSKitTranspiler(to_bqskit=True)
        assert dict(t.gate_mapper) == {}

    def test_custom_dict_gate_mapper_is_wrapped(self):
        t = BQSKitTranspiler(gate_mapper={'XGate': lambda q: None})
        assert isinstance(t.gate_mapper, GateMapper)


class TestBQSKitTranspilerTranspile:

    def test_to_bqskit_direction(self):
        t = BQSKitTranspiler(to_bqskit=True)
        circuit = Circuit([Cycle({H(0)})])

        result = t.transpile(circuit)

        assert result.n_circuits == 1
        assert isinstance(result[0], bqCircuit)

    def test_to_qcal_direction(self):
        bqc = bqCircuit(1)
        bqc.append_gate(SqrtXGate(), (0,))

        t = BQSKitTranspiler()
        result = t.transpile([bqc])

        assert result.n_circuits == 1
        assert isinstance(result[0], Circuit)
        assert result[0][0] == Cycle({X90(0)})
