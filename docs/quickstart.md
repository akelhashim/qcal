# Quickstart

This page walks through the core qcal objects. Everything here runs on a
laptop: no quantum hardware is needed.

## Build a circuit

Circuits in qcal are sequences of **cycles**. A {class}`~qcal.circuit.Cycle`
is a set of gates acting in parallel on disjoint qubits during one clock
cycle.

```python
import qcal as qc

circuit = qc.Circuit([
    qc.Cycle({qc.H(0)}),
    qc.Cycle({qc.CNOT(0, 1)}),
])
circuit.measure()  # appends a measurement cycle on all qubits
circuit.draw()
```

## Simulate it

```python
from qcal.simulation import (
    DensityMatrixSimulator,
    DepolarizingNoise,
    StateVectorSimulator,
)

# Exact, noiseless probabilities
sim = StateVectorSimulator()
sim.run(circuit)
print(circuit.results)

# Shot-based sampling with depolarizing noise
noise = DepolarizingNoise(single_qubit=0.01, two_qubit=0.05)
sim = DensityMatrixSimulator(error_model=noise)
sim.run(circuit, n_shots=1000)
print(circuit.results)
```

## Work with results

Every measurement is returned as a {class}`~qcal.results.Results` object:

```python
results = qc.Results({'00': 480, '01': 12, '10': 9, '11': 499})

results.n_shots          # 1000
results.probabilities    # pandas Series of bitstring -> probability
results.marginalize(0)   # Results on qubit 0 only
```

## Load a hardware configuration

A {class}`~qcal.config.Config` holds all the per-qubit and per-pair
parameters of a device. It is loaded from a YAML file. qcal ships an
example configuration for 8 qubits:

```python
from qcal.backend.emulator import EXAMPLE_CONFIG_PATH

cfg = qc.Config(EXAMPLE_CONFIG_PATH)
cfg.qubits                      # (0, 1, ..., 7)
cfg['single_qubit/0/GE/freq']   # qubit 0 frequency
cfg.native_gates['set']         # {'X90', 'X', 'CZ'}
```

## Run a protocol on the emulator

Calibration, characterization, and benchmarking protocols take a QPU
class and a config. The {class}`~qcal.backend.emulator.Emulator` is a
drop-in QPU that runs a noisy simulation instead of real hardware, so it
is a good place to try out protocols:

```python
from qcal.backend.emulator import Emulator
from qcal.benchmarking.rb import CRB
from qcal.characterization.coherence import T1

t1 = T1(Emulator, cfg, qubits=[0, 1])
t1.run()

crb = CRB(qpu=Emulator, config=cfg, qubit_labels=[0])
crb.run()
crb.process_infidelity
```

On real hardware, replace `Emulator` with a hardware backend such as
{class}`~qcal.backend.qubic.qpu.QubicQPU`.

## Next steps

- {doc}`user_guide/circuits`: cycles, circuits, and circuit sets
- {doc}`user_guide/protocols`: the calibration, characterization, and
  benchmarking workflow
- {doc}`api/index`: full API reference
