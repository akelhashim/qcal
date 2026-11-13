# Results

{class}`~qcal.results.Results` is the standard output of every
measurement. It wraps a dictionary that maps bitstrings to counts.

```python
import qcal as qc

results = qc.Results({'00': 480, '01': 12, '10': 9, '11': 499})
```

| Attribute / method | Description |
|---|---|
| `counts` | bitstring → count |
| `probabilities` | bitstring → probability |
| `n_shots` | total number of shots |
| `n_qudits` | number of measured qudits |
| `states` | measured bitstrings |
| `ev` | expectation value |
| `entropy` | classical (Shannon) entropy of the distribution |
| `marginalize(idx)` | results for a subset of qudits |
| `apply_readout_correction(cmat)` | correct readout errors with a confusion matrix |

After a circuit runs on a QPU or simulator, its results are attached to
the circuit as `circuit.results`.

## Readout correction

To mitigate readout errors, pass a confusion matrix, either when
constructing the object or afterwards:

```python
results = qc.Results(counts, confusion_matrix=cmat)
# or
results.apply_readout_correction(cmat)
```

A confusion matrix can be measured with
{func}`~qcal.benchmarking.readout.ReadoutFidelity`.
