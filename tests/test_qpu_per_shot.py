"""Unit tests for the QPU's per-shot and QP readout correction kwargs.

QPU.__init__ accepts ``rcorr_rr_results`` (randomized readout
characterization results for quasi-probabilistic readout correction)
and ``keep_per_shot_results`` (whether to keep every circuit's per-shot
outcomes). The base QPU only validates and stores them; consuming them
is up to each backend's process(). _PerShotQPU below is a minimal
backend that does so with the real PostProcessor, Results and
PerShotResults, feeding synthetic reads with known outcomes so that
end-to-end behavior can be checked by hand.
"""
import numpy as np
import pandas as pd
import pytest

from qcal.backend.emulator import Emulator
from qcal.circuit import Circuit, Cycle
from qcal.gates.single_qubit import MCM, Meas, X
from qcal.post_processing.post_process import PostProcessor
from qcal.post_processing.utils import circuit_measurement_cycles
from qcal.qpu.qpu import QPU
from qcal.results import PerShotResults, Results

N_SHOTS = 1000


def _cmat() -> pd.DataFrame:
    """Single-qubit confusion matrix.

    Returns:
        pd.DataFrame: confusion matrix.
    """
    return pd.DataFrame(
        [[0.9, 0.1], [0.1, 0.9]],
        index=[0, 1],
        columns=pd.MultiIndex.from_tuples(
            [('Meas State', 0), ('Meas State', 1)]
        ),
    )


class _PerShotQPU(QPU):
    """QPU that post-processes synthetic reads using the new kwargs.

    Attributes:
        synthetic_reads (list): one {qubit: (n_shots, n_reads) array} per
            circuit, returned by acquire().
    """

    def __init__(self, *args, synthetic_reads=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.synthetic_reads = synthetic_reads or []

    def acquire(self):
        self._measurements.extend(self.synthetic_reads)

    def process(self):
        circuits = list(self._circuits)
        pp = PostProcessor(
            self._config,
            self._measurements,
            measurement_cycles=[
                circuit_measurement_cycles(c) for c in circuits
            ],
            keep_per_shot_results=self._keep_per_shot_results,
        )
        pp.run()
        for i, circuit in enumerate(circuits):
            circuit.results = pp.tm_results[i]
            if self._rcorr_rr_results is not None:
                circuit.results.apply_qp_readout_correction(
                    self._rcorr_rr_results
                )
            if self._keep_per_shot_results:
                circuit.per_shot_results = pp.per_shot_results[i]


def _mcm_circuit() -> Circuit:
    """Ancilla 1 is measured mid-circuit, then both qubits at the end.

    Returns:
        Circuit: circuit.
    """
    return Circuit([
        Cycle({X(0)}),
        Cycle({MCM(1)}),
        Cycle({Meas(0), Meas(1)}),
    ])


def _mcm_reads(n_failed_herald: int = 0) -> dict:
    """Synthetic heralded reads for ``_mcm_circuit``.

    Qubit 0 always reads 1. Qubit 1 alternates 0/1 in its MCM and reads
    the opposite at the end. The first ``n_failed_herald`` shots fail
    heralding on qubit 1.

    Args:
        n_failed_herald (int, optional): number of failed heralds.
            Defaults to 0.

    Returns:
        dict: {qubit: (n_shots, n_reads)} reads, herald first.
    """
    q0 = np.zeros((N_SHOTS, 2), dtype=int)
    q0[:, 1] = 1
    q1 = np.zeros((N_SHOTS, 3), dtype=int)
    q1[:n_failed_herald, 0] = 1
    q1[:, 1] = np.arange(N_SHOTS) % 2
    q1[:, 2] = 1 - q1[:, 1]
    return {0: q0, 1: q1}


class TestQPUInitKwargs:

    def test_defaults(self, config):
        qpu = QPU(config)
        assert qpu._rcorr_cmat is None
        assert qpu._rcorr_rr_results is None
        assert qpu._keep_per_shot_results is False

    def test_stores_rr_results_and_keep_flag(self, config):
        rr = {'0': 900, '1': 100}
        qpu = QPU(config, rcorr_rr_results=rr, keep_per_shot_results=True)
        assert qpu._rcorr_rr_results is rr
        assert qpu._keep_per_shot_results is True

    def test_accepts_results_object_for_rr_results(self, config):
        rr = Results({'0': 900, '1': 100})
        qpu = QPU(config, rcorr_rr_results=rr)
        assert qpu._rcorr_rr_results is rr

    def test_cmat_alone_is_allowed(self, config):
        cmat = _cmat()
        qpu = QPU(config, rcorr_cmat=cmat)
        assert qpu._rcorr_cmat is cmat

    def test_cmat_and_rr_results_are_mutually_exclusive(self, config):
        with pytest.raises(ValueError, match='cannot both be passed'):
            QPU(
                config,
                rcorr_cmat=_cmat(),
                rcorr_rr_results={'0': 900, '1': 100},
            )

    def test_emulator_forwards_new_kwargs(self, config):
        qpu = Emulator(
            config, rcorr_rr_results={'0': 1}, keep_per_shot_results=True
        )
        assert qpu._rcorr_rr_results == {'0': 1}
        assert qpu._keep_per_shot_results is True

    def test_emulator_rejects_both_corrections(self, config):
        with pytest.raises(ValueError, match='cannot both be passed'):
            Emulator(
                config,
                rcorr_cmat=_cmat(),
                rcorr_rr_results={'0': 900, '1': 100},
            )


class TestKeepPerShotResults:

    def test_not_kept_by_default(self, config):
        circuit = _mcm_circuit()
        qpu = _PerShotQPU(config, synthetic_reads=[_mcm_reads()])
        qpu.run(circuit)

        assert circuit.per_shot_results is None
        assert dict(circuit.results.dict) == {'10': 500, '11': 500}

    def test_kept_on_request(self, config):
        circuit = _mcm_circuit()
        qpu = _PerShotQPU(
            config,
            synthetic_reads=[_mcm_reads()],
            keep_per_shot_results=True,
        )
        qpu.run(circuit)

        per_shot = circuit.per_shot_results
        assert isinstance(per_shot, PerShotResults)
        assert per_shot.columns == ((0, 1), (1, 0), (1, 1))
        assert per_shot.n_shots == N_SHOTS
        assert per_shot.n_measurement_cycles == 2

    def test_discarded_shots_keep_original_shot_numbers(self, config):
        n_failed = 10
        circuit = _mcm_circuit()
        qpu = _PerShotQPU(
            config,
            synthetic_reads=[_mcm_reads(n_failed)],
            keep_per_shot_results=True,
        )
        qpu.run(circuit)

        per_shot = circuit.per_shot_results
        assert per_shot.kept.tolist() == (
            [False] * n_failed + [True] * (N_SHOTS - n_failed)
        )
        assert per_shot.shot_index[0] == n_failed
        assert circuit.results.n_shots == N_SHOTS - n_failed

    def test_per_shot_record_reproduces_counts(self, config):
        circuit = _mcm_circuit()
        qpu = _PerShotQPU(
            config,
            synthetic_reads=[_mcm_reads(7)],
            keep_per_shot_results=True,
        )
        qpu.run(circuit)

        record = circuit.per_shot_results.measurement_record()
        final = record[1].astype(str).agg(''.join, axis=1)
        assert final.value_counts().to_dict() == dict(circuit.results.dict)

        # The MCM is anti-correlated with qubit 1's final read.
        np.testing.assert_array_equal(
            record[(0, 1)].to_numpy(), 1 - record[(1, 1)].to_numpy()
        )

    def test_one_per_shot_result_per_circuit(self, config):
        circuits = [_mcm_circuit(), _mcm_circuit()]
        qpu = _PerShotQPU(
            config,
            synthetic_reads=[_mcm_reads(0), _mcm_reads(5)],
            keep_per_shot_results=True,
            n_circs_per_seq=2,
        )
        qpu.run(circuits)

        assert [c.per_shot_results.n_shots for c in qpu.circuits] == [
            N_SHOTS, N_SHOTS - 5
        ]


class TestRcorrRRResults:

    def test_qp_correction_applied_to_results(self, config):
        # Qubit 0 is prepared in |1>, and 10% of reads flip to 0.
        reads = np.ones((N_SHOTS, 2), dtype=int)
        reads[:, 0] = 0  # herald
        reads[:N_SHOTS // 10, 1] = 0
        rr = {'0': 900, '1': 100}

        circuit = Circuit([Cycle({X(0)}), Cycle({Meas(0)})])
        qpu = _PerShotQPU(
            config, synthetic_reads=[{0: reads}], rcorr_rr_results=rr
        )
        qpu.run(circuit)

        assert circuit.results.raw_results == {'0': 100, '1': 900}
        assert dict(circuit.results.dict) == {'1': 1000}
        assert circuit.results.rr_char_results is rr

    def test_per_shot_results_are_uncorrected(self, config):
        reads = np.ones((N_SHOTS, 2), dtype=int)
        reads[:, 0] = 0
        reads[:N_SHOTS // 10, 1] = 0

        circuit = Circuit([Cycle({X(0)}), Cycle({Meas(0)})])
        qpu = _PerShotQPU(
            config,
            synthetic_reads=[{0: reads}],
            rcorr_rr_results={'0': 900, '1': 100},
            keep_per_shot_results=True,
        )
        qpu.run(circuit)

        outcomes = circuit.per_shot_results.outcomes[:, 0]
        assert (outcomes == 0).sum() == N_SHOTS // 10
