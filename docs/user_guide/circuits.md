# Circuits

The circuit data model lives in {mod}`qcal.circuit`.

## Gates

Gates are defined in {mod}`qcal.gates`. Each {class}`~qcal.gates.gate.Gate`
stores a unitary matrix and the qubit labels it acts on.

| Module | Gates |
|---|---|
| {mod}`qcal.gates.single_qubit` | `H`, `X`, `Y`, `Z`, `X90`, `Y90`, `Rx`, `Ry`, `Rz`, `Meas`, ... |
| {mod}`qcal.gates.two_qubit` | `CNOT`/`CX`, `CZ`, `iSWAP`, ... |
| {mod}`qcal.gates.single_qutrit` | `H3`, `X01`, `X12`, `Rx12`, ... |
| {mod}`qcal.gates.two_qutrit` | two-qutrit gates |

All gates are re-exported from the top-level `qcal` namespace:

```python
import qcal as qc

qc.X90(0)
qc.CZ(0, 1)
```

Many gates take a `subspace` argument: `'GE'` for the ground-excited
transition or `'EF'` for the excited-second-excited transition. This
setting carries through compilation and hardware execution.

## Cycles

A {class}`~qcal.circuit.Cycle` is a set of gates on **disjoint** qubits
that run in parallel during one clock cycle. This is the basic unit of
execution on superconducting hardware.

```python
cycle = qc.Cycle({qc.X90(0), qc.X90(1)})
cycle.qubits    # (0, 1)
```

## Circuits

A {class}`~qcal.circuit.Circuit` is an ordered sequence of cycles. You
build circuits by composing cycles, not individual gates:

```python
circuit = qc.Circuit([
    qc.Cycle({qc.H(0)}),
    qc.Cycle({qc.CNOT(0, 1)}),
])
circuit.measure()
circuit.n_cycles        # 3
circuit.circuit_depth   # 3
```

## Circuit sets

A {class}`~qcal.circuit.CircuitSet` is a DataFrame-backed collection of
circuits. Extra columns can hold metadata, such as the values of a
parameter sweep. Most protocols generate a `CircuitSet`, and the compiler
returns one too.

```python
cs = qc.CircuitSet([circuit, circuit.copy()])
cs.n_circuits      # 2
cs[0].draw()
```
