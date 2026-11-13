# Calibration, characterization, and benchmarking

All three protocol families follow the same structure:

1. **Generate circuits**: build a {class}`~qcal.circuit.CircuitSet`,
   usually parametrized by a sweep.
2. **Run**: execute the circuits on a QPU.
3. **Analyze**: fit the {class}`~qcal.results.Results` to a model (see
   {mod}`qcal.fitting`).
4. **Finalize**: for calibrations, write the optimal values back to the
   {class}`~qcal.config.Config`.
5. **Plot**: visualize the data and fits.

Protocols are **factory functions**. Each takes a QPU *class* (not an
instance) and a config, and returns an object that inherits from both the
QPU and the protocol base class. Calling `.run()` performs all of the
steps above:

```python
from qcal.calibration.single_qubit import Amplitude

cal = Amplitude(QPU, cfg, qubits=[0, 1, 2])
cal.run()
```

Any extra keyword arguments go to the QPU constructor, for example
`n_shots` or hardware-specific options such as an IP address.

## Calibration

{mod}`qcal.calibration`

| Protocol | Module |
|---|---|
| Single-qubit amplitude, frequency, phase | {mod}`qcal.calibration.single_qubit` |
| CZ amplitude, frequency, relative phase, local phases | {mod}`qcal.calibration.cz` |
| Readout classification, separation, fidelity | {mod}`qcal.calibration.readout` |
| Leakage | {mod}`qcal.calibration.leakage` |
| Generic parameter sweep | {mod}`qcal.calibration.sweep` |

## Characterization

{mod}`qcal.characterization`

| Protocol | Module |
|---|---|
| T1, T2 (Ramsey/echo), T2 with dynamical decoupling | {mod}`qcal.characterization.coherence` |
| Qubit and resonator spectroscopy | {mod}`qcal.characterization.spectroscopy` |
| Gate set tomography (GST) | {mod}`qcal.characterization.tomography.gst` |
| Robust phase estimation (RPE), ZZ characterization | {mod}`qcal.characterization.phase_estimation` |

## Benchmarking

{mod}`qcal.benchmarking`

| Protocol | Module |
|---|---|
| Clifford / standard randomized benchmarking | {mod}`qcal.benchmarking.rb` |
| Interleaved RB | {mod}`qcal.benchmarking.irb` |
| Cycle benchmarking (CB) | {mod}`qcal.benchmarking.cb` |
| Mid-circuit measurement CB | {mod}`qcal.benchmarking.mcb` |
| Pauli noise reconstruction | {mod}`qcal.benchmarking.pnr` |
| Cross-entropy benchmarking (XEB) | {mod}`qcal.benchmarking.xeb` |
| Readout fidelity | {mod}`qcal.benchmarking.readout` |

## QPU backends

{class}`~qcal.qpu.qpu.QPU` is the base class for all backends. It handles
compilation, transpilation, execution, and post-processing.

- {class}`~qcal.backend.emulator.Emulator`: a noisy simulator that acts
  like a QPU. Useful for developing and testing protocols.
- {mod}`qcal.backend.qubic`: the QubiC control-system backend.

See the {doc}`../tutorials/Emulation` tutorial for complete examples.
