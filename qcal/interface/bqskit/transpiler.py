"""Submodule for handling transpilation to/from Bqskit.

See: https://github.com/BQSKit/bqskit
"""
import logging
from typing import Dict, List

from bqskit.ir import Circuit as bqCircuit
from bqskit.ir.gates import BarrierPlaceholder as bqBarrier
from bqskit.ir.gates import ConstantUnitaryGate as UGate
from bqskit.ir.gates import MeasurementPlaceholder as bqMeas

from qcal.circuit import Barrier, Circuit, CircuitSet, Cycle
from qcal.gate.single_qubit import X90, Meas, Rz, X
from qcal.gate.two_qubit import CX, CZ, iSWAP
from qcal.transpilation.transpiler import Transpiler
from qcal.transpilation.utils import GateMapper

logger = logging.getLogger(__name__)


def to_bqskit(circuit: Circuit):
    """Compile a qcal circuit to a BQSKit circuit.

    Args:
        circuit (Circuit): qcal circuit.

    Returns:
        bqskit.ir.Circuit: BQSKit circuit.
    """

    tcircuit = bqCircuit(circuit.n_qubits)
    for cycle in circuit:
        if cycle.is_barrier:
            tcircuit.append_gate(bqBarrier(len(cycle.qubits)), cycle.qubits)
        else:
            for gate in cycle:

                if gate.is_measurement:
                    tcircuit.append_gate(
                        bqMeas(
                            [(str(gate.qubits[0]), 1)],
                            {gate.qubits[0]: (str(gate.qubits[0]), 0)}
                        ),
                        gate.qubits[0]
                    )

                elif not gate.is_measurement:
                    tcircuit.append_gate(
                        UGate(gate.matrix), gate.qubits
                    )

    return tcircuit


def to_qcal(circuit, gate_mapper: GateMapper | Dict) -> Circuit:
    """Compile a BQSKit circuit to a qcal circuit.

    Args:
        circuit (bqskit.ir.Circuit): BQSKit circuit.
        gate_mapper (GateMapper | Dict): map between BQSKit to qcal gates.

    Returns:
        Circuit: qcal circuit.
    """
    c = 0
    tcycle = Cycle()
    tcircuit = Circuit()
    for cyc, op in circuit.operations_with_cycles():

        if cyc > c:
            if tcycle.n_gates > 0:
                tcircuit.append(tcycle)
            c = cyc
            tcycle = Cycle()

        if str(op.gate) == 'barrier':
            tcircuit.append(gate_mapper[str(op.gate)](tuple(op.location)))
        elif str(op.gate) == 'measurement':
            tcircuit.measure(tuple(op.location))
        else:
            tcycle.append(
                gate_mapper[str(op.gate)](tuple(op.location)) if
                len(op.params) == 0 else
                gate_mapper[str(op.gate)](tuple(op.location), op.params[0])
            )
    if tcycle.n_gates > 0:
        tcircuit.append(tcycle)

    return tcircuit


class BQSKitTranspiler(Transpiler):
    """BQSKit Transpiler."""

    __slots__ = ('_gate_mapper', '_to_bqskit')

    def __init__(
        self,
        gate_mapper: GateMapper | Dict | None = None,
        to_bqskit:   bool = False
    ) -> None:
        """Initialize with gate_mapper.

        Args:
            gate_mapper (GateMapper | Dict | None, optional): dictionary which
                maps str names of BQSKit gates to qcal gates. Defaults to None.
            to_bqskit (bool): whether to transpile from qcal to BSQKit.
                Defaults to False.
        """
        if gate_mapper is None and to_bqskit is False:
            self._gate_mapper = GateMapper(
                {
                    'barrier':     Barrier,
                    'measurement': Meas,
                    'CNOTGate':    CX,
                    'CZGate':      CZ,
                    'ISwapGate':   iSWAP,
                    'RZGate':      Rz,
                    'SqrtXGate':   X90,
                    'XGate':       X,
                }
            )
        elif gate_mapper is None and to_bqskit is True:
            self._gate_mapper = GateMapper()
        elif isinstance(gate_mapper, dict):
            self._gate_mapper = GateMapper(gate_mapper)
        else:
            self._gate_mapper = gate_mapper
        super().__init__(gate_mapper=self._gate_mapper)

        self._to_bqskit = to_bqskit

    def transpile(
        self, circuits: Circuit | CircuitSet | List
    ) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (Circuit | CircuitSet | List): circuits to transpile.

        Returns:
            CircuitSet: transpiled circuits.
        """
        if not isinstance(circuits, CircuitSet):
            circuits = CircuitSet(circuits=circuits)

        tcircuits = []
        for circuit in circuits:
            if self._to_bqskit:
                tcircuits.append(to_bqskit(circuit))
            else:
                tcircuits.append(to_qcal(circuit, self._gate_mapper))

        tcircuits = CircuitSet(circuits=tcircuits)
        return tcircuits
