"""Tests for aligning (ragged) measurement results by circuit layer."""
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from qcal.circuit import Barrier, Circuit, Cycle
from qcal.gates.single_qubit import MCM, H, Meas, X
from qcal.gates.two_qubit import CZ
from qcal.post_processing.passes import (
    compute_conditional_counts,
    compute_counts,
    discard_heralded_shots,
    relabel_esp,
)
from qcal.post_processing.post_process import PostProcessor
from qcal.post_processing.utils import (
    MeasurementCycles,
    circuit_measurement_cycles,
    construct_dataframes,
)
from qcal.results import PerShotResults

N_SHOTS = 4


class _Config(dict):
    """Minimal stand-in for the parts of `Config` used in post-processing."""

    def __init__(self, herald: bool = False, n_resets: int = 0) -> None:
        super().__init__({
            'readout/herald': herald,
            'readout/esp/enable': False,
        })
        self.parameters = {
            'reset': {
                'active': {'enable': n_resets > 0, 'n_resets': n_resets}
            }
        }


def _qec_circuit() -> Circuit:
    """Ancilla 1 is measured mid-circuit twice; data qubit 0 only at the end.
    """
    return Circuit([
        Cycle({X(0)}),
        Cycle({MCM(1)}),
        Barrier((0, 1)),
        Cycle({MCM(1)}),
        Cycle({Meas(0)}),
    ])


def _reads(*columns: int) -> np.ndarray:
    """(n_shots, n_reads) array where read j is `columns[j]` on every shot."""
    return np.tile(np.array(columns), (N_SHOTS, 1))


def _repeated_qec_circuit(n_rounds: int = 10) -> Circuit:
    """Ancillas 1 and 3 are measured every round; all qubits at the end."""
    qubits = [0, 1, 2, 3, 4]
    round_ = Circuit([
        Cycle({H(q) for q in qubits}),
        Cycle({CZ(0, 1), CZ(2, 3)}),
        Barrier((4,)),
        Cycle({H(q) for q in qubits}),
        Cycle({CZ(1, 2), CZ(3, 4)}),
        Barrier((0,)),
        Cycle({H(q) for q in qubits}),
        Cycle({MCM((1, 3))}),
    ])
    circuit = Circuit(round_.cycles * n_rounds)
    circuit.measure()
    return circuit


def _synthetic_reads(
    measurement_cycles: MeasurementCycles,
    n_shots:            int,
    start_idx:          int = 0,
    probabilities:      Sequence[float] = (0.5, 0.5),
    p_preceding:        float = 0.01,
    seed:               int | None = None,
) -> dict[int, NDArray[np.int8]]:
    """Generate synthetic per-shot reads for a single circuit.

    This is useful for testing post-processing without hardware. Each read
    is sampled independently from ``probabilities``, so there are no
    correlations between reads. Qubits measured a different number of times
    get a different number of reads, as on hardware.

    For example, for a circuit with measurement cycles
    ``[{1: 0}, {1: 1}, {0: 0}]`` and ``start_idx=1``, qubit 0 has 2 reads and
    qubit 1 has 3 reads per shot.

    Args:
        measurement_cycles (MeasurementCycles): measurement cycles of the
            circuit (see ``circuit_measurement_cycles``).
        n_shots (int): number of shots.
        start_idx (int, optional): number of reads preceding the circuit
            (e.g. heralding). Defaults to 0.
        probabilities (Sequence[float], optional): probability of reading
            out each level. Defaults to (0.5, 0.5).
        p_preceding (float, optional): probability that each read preceding
            the circuit is 1 instead of 0 (e.g. a failed herald). Defaults
            to 0.01.
        seed (int | None, optional): random seed. Defaults to None.

    Returns:
        dict[int, NDArray[np.int8]]: reads of each qubit, with shape
            ``(n_shots, start_idx + n_reads)``.
    """
    n_reads = defaultdict(int)
    for cycle in measurement_cycles:
        for q in cycle:
            n_reads[q] += 1

    rng = np.random.default_rng(seed)
    return {
        q: np.hstack([
            (rng.random((n_shots, start_idx)) < p_preceding).astype(np.int8),
            rng.choice(
                len(probabilities), size=(n_shots, n), p=probabilities
            ).astype(np.int8),
        ])
        for q, n in sorted(n_reads.items())
    }


def test_circuit_measurement_cycles():
    assert circuit_measurement_cycles(_qec_circuit()) == [
        {1: 0}, {1: 1}, {0: 0}
    ]


def test_circuit_measurement_cycles_with_simultaneous_measurements():
    circuit = Circuit([
        Cycle({MCM((0, 1))}),
        Cycle({Meas(0), Meas(1), Meas(2)}),
    ])
    assert circuit_measurement_cycles(circuit) == [
        {0: 0, 1: 0},
        {0: 1, 1: 1, 2: 0},
    ]


def test_construct_dataframes_aligns_reads_by_layer():
    # A herald read (0) precedes the circuit's reads on both qubits.
    results = [{0: _reads(0, 1), 1: _reads(0, 1, 0)}]

    dfs = construct_dataframes(
        results, 1, [circuit_measurement_cycles(_qec_circuit())]
    )

    assert [list(df.columns) for df in dfs[0]] == [[1], [1], [0]]
    assert [df.iloc[0].tolist() for df in dfs[0]] == [[1], [0], [1]]


def test_construct_dataframes_drops_layers_without_results():
    # Only the data qubit is post-processed.
    results = [{0: _reads(1)}]

    dfs = construct_dataframes(
        results, 0, [circuit_measurement_cycles(_qec_circuit())]
    )

    assert [list(df.columns) for df in dfs[0]] == [[0]]


def test_construct_dataframes_rejects_mismatched_layers():
    results = [{0: _reads(1), 1: _reads(0)}]
    with pytest.raises(ValueError, match="Qubit 1 in circuit 0 has 1 reads"):
        construct_dataframes(
            results, 0, [circuit_measurement_cycles(_qec_circuit())]
        )


def test_construct_dataframes_without_layers_requires_uniform_reads():
    results = [{0: _reads(1), 1: _reads(0, 1)}]
    with pytest.raises(ValueError, match="pass `measurement_cycles`"):
        construct_dataframes(results, 0)

    dfs = construct_dataframes([{0: _reads(0, 1), 1: _reads(1, 0)}], 0)
    assert [df.iloc[0].tolist() for df in dfs[0]] == [[0, 1], [1, 0]]


def test_post_processor_with_ragged_reads():
    results = [{0: _reads(0, 1), 1: _reads(0, 1, 0)}]
    pp = PostProcessor(
        _Config(herald=True),
        results,
        measurement_cycles=[circuit_measurement_cycles(_qec_circuit())],
    )
    pp.run()

    assert pp.tm_results == [{'1': N_SHOTS}]
    assert pp.mcm_results == [[{'1': N_SHOTS}, {'0': N_SHOTS}]]


def test_heralding_discards_shots_from_every_layer():
    # Herald (read 0) fails on shot 0 for the data qubit, and on shot 1 for
    # the ancilla, so both shots are discarded from every layer, including
    # layers that don't contain the qubit whose herald failed.
    q0 = _reads(0, 1)
    q0[0, 0] = 1
    q1 = _reads(0, 1, 0)
    q1[1, 0] = 1
    pp = PostProcessor(
        _Config(herald=True),
        [{0: q0, 1: q1}],
        measurement_cycles=[circuit_measurement_cycles(_qec_circuit())],
    )
    pp.run()

    assert pp.tm_results == [{'1': N_SHOTS - 2}]
    assert pp.mcm_results == [[{'1': N_SHOTS - 2}, {'0': N_SHOTS - 2}]]


def test_heralding_with_active_reset_and_ragged_reads():
    # Two active reset reads, then the herald, then the circuit's reads.
    q0 = _reads(0, 0, 0, 1)
    q1 = _reads(1, 0, 0, 1, 0)
    q1[2, 2] = 1  # Herald fails on shot 2
    pp = PostProcessor(
        _Config(herald=True, n_resets=2),
        [{0: q0, 1: q1}],
        measurement_cycles=[circuit_measurement_cycles(_qec_circuit())],
    )
    pp.run()

    assert pp.tm_results == [{'1': N_SHOTS - 1}]
    assert pp.mcm_results == [[{'1': N_SHOTS - 1}, {'0': N_SHOTS - 1}]]


def _heralded_qec_post_processor(**kwargs) -> PostProcessor:
    """Herald fails on shot 1 of the ancilla; ancilla reads differ by shot."""
    q0 = _reads(0, 1)
    q1 = _reads(0, 1, 0)
    q1[1, 0] = 1
    q1[:, 1] = np.arange(N_SHOTS) % 2
    return PostProcessor(
        _Config(herald=True),
        [{0: q0, 1: q1}],
        measurement_cycles=[circuit_measurement_cycles(_qec_circuit())],
        **kwargs,
    )


def test_per_shot_results_are_only_kept_on_request():
    pp = _heralded_qec_post_processor()
    pp.run()
    assert pp.per_shot_results is None


def test_per_shot_results_are_kept_after_aggregation():
    pp = _heralded_qec_post_processor(keep_per_shot_results=True)
    pp.run()

    per_shot = pp.per_shot_results[0]
    assert per_shot.outcomes.dtype == np.int8
    assert per_shot.outcomes.shape == (3, 3)
    assert per_shot.columns == ((0, 1), (1, 1), (2, 0))
    assert per_shot.n_measurement_cycles == 3
    # Shot 1 failed heralding; the rest keep their original shot numbers.
    assert per_shot.shot_index.tolist() == [0, 2, 3]
    assert per_shot.n_shots == 3

    cycles = per_shot.cycle_results
    assert [list(df.columns) for df in cycles] == [[1], [1], [0]]
    assert all(list(df.index) == [0, 2, 3] for df in cycles)
    assert cycles[0][1].tolist() == [0, 0, 1]
    # The counts are still aggregated from the same shots.
    assert pp.mcm_results[0][0] == {'0': 2, '1': 1}


def test_measurement_record():
    pp = _heralded_qec_post_processor(keep_per_shot_results=True)
    pp.run()

    record = pp.per_shot_results[0].measurement_record()
    assert list(record.columns) == [(0, 1), (1, 1), (2, 0)]
    assert record.columns.names == ['cycle', 'qubit']
    assert list(record.index) == [0, 2, 3]
    assert record.loc[3].tolist() == [1, 0, 1]

    mcms = pp.per_shot_results[0].measurement_record(include_final=False)
    assert list(mcms.columns) == [(0, 1), (1, 1)]


def test_per_shot_results_keep_dtype_when_int8_is_lossy():
    per_shot = PerShotResults.from_cycles([pd.DataFrame({0: [0.5, 1.0]})])
    assert per_shot.outcomes.dtype == np.float64


def test_per_shot_results_reject_misaligned_cycles():
    cycles = [pd.DataFrame({0: [0, 1]}), pd.DataFrame({1: [0]})]
    with pytest.raises(ValueError, match="same shots"):
        PerShotResults.from_cycles(cycles)


def test_per_shot_results_without_layers():
    per_shot = PerShotResults.from_cycles([])
    assert per_shot.n_shots == 0
    assert per_shot.cycle_results == []
    assert per_shot.measurement_record().empty


def test_synthetic_reads_match_measurement_cycles():
    n_rounds = 10
    reads = _synthetic_reads(
        circuit_measurement_cycles(_repeated_qec_circuit(n_rounds)),
        n_shots=100,
        start_idx=1,
        p_preceding=0,
        seed=0,
    )

    # Ancillas: one read per round plus the final read; data: final only
    assert {q: r.shape for q, r in reads.items()} == {
        0: (100, 2),
        1: (100, n_rounds + 2),
        2: (100, 2),
        3: (100, n_rounds + 2),
        4: (100, 2),
    }
    assert all(np.all(r[:, 0] == 0) for r in reads.values())


def test_construct_dataframes_aligns_random_reads():
    start_idx = 1
    meas_cycles = circuit_measurement_cycles(_repeated_qec_circuit())
    reads = _synthetic_reads(meas_cycles, 100, start_idx, seed=1)

    dfs = construct_dataframes([reads], start_idx, [meas_cycles])[0]

    assert len(dfs) == len(meas_cycles)
    for df, cycle in zip(dfs, meas_cycles, strict=True):
        assert set(df.columns) == set(cycle)
        for q, j in cycle.items():
            np.testing.assert_array_equal(df[q], reads[q][:, start_idx + j])


def test_heralding_discards_random_failures_from_every_layer():
    meas_cycles = circuit_measurement_cycles(_repeated_qec_circuit())
    reads = _synthetic_reads(
        meas_cycles, 200, start_idx=1, p_preceding=0.05, seed=2
    )

    pp = PostProcessor(
        _Config(herald=True),
        [reads],
        measurement_cycles=[meas_cycles],
        keep_per_shot_results=True,
    )
    pp.run()

    passed = np.flatnonzero(
        np.all([r[:, 0] == 0 for r in reads.values()], axis=0)
    )
    assert 0 < len(passed) < 200
    assert pp.per_shot_results[0].shot_index.tolist() == passed.tolist()
    assert sum(pp.tm_results[0].values()) == len(passed)
    assert all(
        sum(counts.values()) == len(passed)
        for counts in pp.mcm_results[0]
    )


def test_per_shot_results_are_consistent_with_counts():
    meas_cycles = circuit_measurement_cycles(_repeated_qec_circuit())
    pp = PostProcessor(
        _Config(),
        [_synthetic_reads(meas_cycles, 200, seed=4)],
        measurement_cycles=[meas_cycles],
        keep_per_shot_results=True,
    )
    pp.run()

    record = pp.per_shot_results[0].measurement_record()
    assert list(record.columns) == [
        (i, q) for i, cycle in enumerate(meas_cycles) for q in sorted(cycle)
    ]

    counts = pp.mcm_results[0] + pp.tm_results
    for i, cycle_counts in enumerate(counts):
        bitstrings = record[i].astype(str).agg(''.join, axis=1)
        assert bitstrings.value_counts().to_dict() == cycle_counts


def test_aggregation_at_end_of_passes_is_moved_to_aggregate():
    pp = _heralded_qec_post_processor(
        passes=[
            discard_heralded_shots, relabel_esp, compute_conditional_counts
        ]
    )
    assert pp.passes == [discard_heralded_shots, relabel_esp]
    assert pp.aggregate is compute_conditional_counts

    pp.run()
    assert pp.tm_results == [{'p0:p0:1': 2, 'p1:p0:1': 1}]


def test_aggregation_in_middle_of_passes_is_rejected():
    with pytest.raises(ValueError, match="`aggregate`"):
        _heralded_qec_post_processor(passes=[compute_counts, relabel_esp])
