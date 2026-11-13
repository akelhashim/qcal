# Compilation and transpilation

Running a circuit on hardware happens in two stages:

1. **Compilation** rewrites a circuit into the device's native gate set
   (read from the config). {class}`~qcal.compilation.compiler.Compiler`
   wraps [BQSKit](https://bqskit.lbl.gov/) through
   {class}`~qcal.interface.bqskit.compiler.BQSKitCompiler`.
2. **Transpilation** converts qcal circuits into the instruction format
   of a specific control system. Transpilers subclass
   {class}`~qcal.transpilation.transpiler.Transpiler`.

```python
import qcal as qc
from qcal.backend.emulator import EXAMPLE_CONFIG_PATH

cfg = qc.Config(EXAMPLE_CONFIG_PATH)
circuit = qc.Circuit([qc.Cycle({qc.H(0)}), qc.Cycle({qc.CNOT(0, 1)})])
circuit.measure()

compiled = qc.Compiler(cfg).compile(circuit)   # returns a CircuitSet
compiled[0].draw()                             # X90 / X / CZ / Rz gates
```

A {class}`~qcal.qpu.qpu.QPU` runs both stages automatically. You can
pass a custom `compiler` and `transpiler` to the QPU constructor.

## Interfaces to other packages

| Package | Module | Purpose |
|---|---|---|
| BQSKit | {mod}`qcal.interface.bqskit` | default compiler |
| pyGSTi | {mod}`qcal.interface.pygsti` | GST, Clifford compilation with Pauli randomization |
| pyQuil | {mod}`qcal.interface.pyquil` | Rigetti Quil programs, randomized compiling |
| True-Q | {mod}`qcal.interface.trueq` | True-Q benchmarking circuits |
| Superstaq | {mod}`qcal.interface.superstaq` | Superstaq compilation |

Circuit-level passes such as Pauli conjugation, gate decompositions, and
cycle merging are in {mod}`qcal.compilation`.
