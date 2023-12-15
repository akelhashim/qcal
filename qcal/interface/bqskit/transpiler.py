"""Submodule for handling transpilation to/from Bqskit.

See: https://github.com/BQSKit/bqskit
"""
from qcal.circuit import Barrier, Cycle, Circuit, CircuitSet
from qcal.gate.single_qubit import Meas, Rz, X90, X
from qcal.gate.two_qubit import CX, CZ
from qcal.transpilation.transpiler import Transpiler

import logging

from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


def to_bqskit(circuit: Circuit):
    """Compile a qcal circuit to a BQSKit circuit.

    Args:
        circuit (Circuit): 

    Returns:
        bqskit.ir.Circuit: BQSKit circuit.
    """
    from bqskit.ir import Circuit as bqCircuit
    from bqskit.ir.gates import BarrierPlaceholder as bqBarrier
    from bqskit.ir.gates import ConstantUnitaryGate as UGate
    from bqskit.ir.gates import MeasurementPlaceholder as bqMeas

    tcircuit = bqCircuit(circuit.n_qubits)
    for cycle in circuit:
        if cycle.is_barrier:
            tcircuit.append_gate(bqBarrier(len(cycle.qubits)), cycle.qubits)
            pass
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


def to_qcal(circuit, gate_mapper: defaultdict) -> Circuit:
    """Compile a BQSKit circuit to a qcal circuit.

    Args:
        circuit (bqskit.ir.Circuit): BQSKit circuit.
        gate_mapper (defaultdict): map between BQSKit to qcal gates.

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
                gate_mapper[str(op.gate)](op.params[0], tuple(op.location))
            )
    if tcycle.n_gates > 0:
        tcircuit.append(tcycle)
        
    return tcircuit


def transpilation_error(*args):
    """Generic transpilation error.

    Raises:
        Exception: transpilation error for non-native gate.
    """
    print(args)
    raise Exception(
        f'Cannot transpile {str(args)} (non-native gate)!'
    ) 


class Transpiler(Transpiler):
    """BQSKit Transpiler."""

    # __slots__ = ('_config', '_gate_mapper', '_to_bqskit', '_to_qcal')

    def __init__(self, 
            gate_mapper:  defaultdict | Dict | None = None,
            to_bqskit:    bool = False,
            to_qcal:      bool = False
        ) -> None:
        """Initialize with gate_mapper.

        Args:
            gate_mapper (defaultdict | Dict | None, optional): dictionary which 
                maps str names of BQSKit gates to qcal gates. Defaults to None.
            to_bqskit (bool): whether to transpile from qcal to BSQKit.
                Defaults to False.
            to_qcal (bool): whether to transpile from BSQKit to qcal.
                Defaults to False.
        """
        assert any((to_bqskit, to_qcal)), (
            "One of 'to_bqskit' or 'to_qcal' must be True!"
        )
        assert not all((to_bqskit, to_qcal)), (
            "One of 'to_bqskit' or 'to_qcal' must be True (not both)!"
        )
        
        if gate_mapper is None and to_qcal is True:
            self._gate_mapper = defaultdict(lambda: transpilation_error,
                {'barrier':     Barrier,
                 'measurement': Meas,
                 'CNOTGate':    CX,
                 'CZGate':      CZ,
                 'RZGate':      Rz,
                 'SqrtXGate':   X90,
                 'XGate':       X
                }
            )
        else:
            self._gate_mapper = gate_mapper
        super().__init__(gate_mapper=self._gate_mapper)

        self._to_bqskit = to_bqskit
        self._to_qcal = to_qcal

    def transpile(
            self, circuits: Circuit | CircuitSet | List
        ) -> CircuitSet:
        """Transpile all circuits.

        Args:
            circuits (Circuit | CircuitSet | List): circuits to 
                transpile.

        Returns:
            CircuitSet: transpiled circuits.
        """
        if not isinstance(circuits, CircuitSet):
            circuits = CircuitSet(circuits=circuits)

        tcircuits = []
        for circuit in circuits:
            if self._to_bqskit:
                tcircuits.append(to_bqskit(circuit))
            elif self._to_qcal:
                tcircuits.append(to_qcal(circuit, self._gate_mapper))
    
        tcircuits = CircuitSet(circuits=tcircuits)
        return tcircuits