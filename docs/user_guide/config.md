# Configuration

{class}`~qcal.config.Config` loads a YAML hardware specification. It is
the central store for per-qubit and per-pair parameters: frequencies,
amplitudes, phases, native gate set, readout settings, and so on.
Calibration protocols read from the config and write their results back
to it.

It is strongly recommended that you track your `config.yaml` with git, so
that you can see how calibrations change over time and revert a bad
calibration.

## Loading and saving

```python
import qcal as qc
import qcal.settings as settings

settings.Settings.config_path = '/path/to/configs/'

cfg = qc.Config()                     # loads <config_path>/config.yaml
cfg = qc.Config('/path/to/other.yaml')

cfg.load()   # reload from disk
cfg.save()   # write local changes back to disk
```

## Reading and writing parameters

Parameters can be addressed with a list of nested keys, or with a single
slash-separated string:

```python
cfg.set(['single_qubit', 0, 'GE', 'freq'], 5.5e9)
cfg['single_qubit/0/GE/freq'] = 5.4e9

cfg.get(['single_qubit', 0, 'GE', 'freq'])
cfg['single_qubit/0/GE/freq']
```

## Useful properties and tools

```python
cfg.qubits          # all qubit labels
cfg.qubit_pairs     # connectivity, from the two-qubit gates
cfg.native_gates    # native gate set
cfg.parameters      # human-readable DataFrame view

cfg.draw_qpu()          # interactive connectivity graph (Plotly)
cfg.plot_qubit_freqs()
cfg.plot_readout_freqs()
```

## Global settings

{mod}`qcal.settings` holds a module-level `Settings` singleton:

| Attribute | Meaning |
|---|---|
| `Settings.config_path` | directory containing `config.yaml` |
| `Settings.save_data` | whether protocols save their data automatically |
| `Settings.data_save_path` | base directory for saved data |
| `Settings.plot_renderer` | Plotly renderer for plots in notebooks |

See the {doc}`../tutorials/1_Configuration_Management` tutorial for a
full walkthrough.
