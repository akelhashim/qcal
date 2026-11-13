# Simulation

{mod}`qcal.simulation` provides two simulators. Both support qubit and
qutrit circuits.

- {class}`~qcal.simulation.simulators.StateVectorSimulator`: exact,
  noise-free pure-state simulation. Supports coherent (unitary) errors
  when you override a gate's unitary.
- {class}`~qcal.simulation.simulators.DensityMatrixSimulator`:
  mixed-state simulation with arbitrary noise channels.

```python
import qcal as qc
from qcal.simulation import DensityMatrixSimulator, DepolarizingNoise

circuit = qc.Circuit([qc.Cycle({qc.H(0)}), qc.Cycle({qc.CNOT(0, 1)})])
circuit.measure()

sim = DensityMatrixSimulator(
    error_model=DepolarizingNoise(single_qubit=0.01, two_qubit=0.05)
)
sim.run(circuit, n_shots=1000)
circuit.results
```

## Error models

Built-in error models are in {mod}`qcal.simulation.error_models`:
`DepolarizingNoise`, `AmplitudeDamping`, `DephasingNoise`,
`RelaxationNoise` (T1 + Tφ, set with `RelaxationParams`), `LeakageNoise`,
`SeepageNoise`, `UnitaryError`, and `CustomErrorModel` for combining
several models. You can also pass a dictionary that maps gate names to
`quax` channels.

See the {doc}`../tutorials/Simulation` tutorial for a full walkthrough.
