# qcal

**qcal** is a Python package for quantum calibration, characterization,
and benchmarking of superconducting quantum systems. It was developed to
operate full-stack superconducting quantum processors at the
[Advanced Quantum Testbed](https://aqt.lbl.gov/) at Lawrence Berkeley
National Laboratory.

qcal supports three primary workflows:

- **Calibration**: optimize gate and readout parameters and write them
  back to your hardware configuration.
- **Characterization**: measure physical properties of the system, such
  as coherence times, spectroscopy, and process tomography.
- **Benchmarking**: estimate performance metrics with protocols such as
  randomized benchmarking (RB) and cycle benchmarking (CB).

```python
import qcal as qc

circuit = qc.Circuit([
    qc.Cycle({qc.H(0)}),
    qc.Cycle({qc.CNOT(0, 1)}),
])
circuit.measure()
circuit.draw()
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/circuits
user_guide/config
user_guide/results
user_guide/protocols
user_guide/simulation
user_guide/compilation
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/1_Configuration_Management
tutorials/2_Spectroscopy
tutorials/3_Readout_Calibration
tutorials/Simulation
tutorials/Emulation
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
changelog
citation
```
