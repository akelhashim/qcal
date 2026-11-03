"""Unit tests for qcal.characterization.readout.RandomizedReadout.

RandomizedReadout measures the all-zero state (prepared by a ZXZXZ
decomposition of the identity) and reports the probability of reading
out all zeros as its fidelity. Under randomized readout this
characterizes the stochastic bit-flip channel used by quasi-
probabilistic readout correction (QPRC).

The Emulator does not implement randomized readout itself, but its
**kwargs signature passes RandomizedReadout's backend check. Tests
inject a DensityMatrixSimulator whose only noise is a known per-qubit
readout confusion matrix, so the expected all-zero probability is the
product of each qubit's P(0 | 0).

Results.plot() currently raises with plotly >= 6 (see the xfail in
TestRandomizedReadoutPlot and in tests/test_results.py), and run()
calls plot(), so most tests stub it out with the ``no_plot`` fixture.
"""
import numpy as np
import pytest

from qcal.backend.emulator import (
    DEFAULT_READOUT_P0,
    Emulator,
)
from qcal.characterization.readout import RandomizedReadout
from qcal.circuit import Circuit, Cycle
from qcal.gates.single_qubit import Meas, X
from qcal.qpu.qpu import QPU
from qcal.results import Results
from qcal.simulation.error_models import DepolarizingNoise
from qcal.simulation.simulators import (
    DensityMatrixSimulator,
    StateVectorSimulator,
)

# P(read 1 | prepared 0) on qubit 0; qubit 1 has perfect readout.
FLIP_Q0 = 0.1


@pytest.fixture
def no_plot(monkeypatch):
    """Stub out Results.plot, which RandomizedReadout.run() calls."""
    monkeypatch.setattr(Results, 'plot', lambda self, *a, **k: None)


def _readout_only_simulator(cmats: dict) -> DensityMatrixSimulator:
    """Density-matrix simulator whose only noise is readout error.

    Args:
        cmats (dict): mapping of qubit label to 2x2 row-stochastic
            confusion matrix, C[prep, meas] = P(meas | prep).

    Returns:
        DensityMatrixSimulator: simulator.
    """
    noise = DepolarizingNoise(single_qubit=0., two_qubit=0.)
    for q, cmat in cmats.items():
        noise.add_readout_noise(q, np.asarray(cmat))
    return DensityMatrixSimulator(error_model=noise)


def _asymmetric_q0_simulator() -> DensityMatrixSimulator:
    """Readout noise only on qubit 0, flipping 0 -> 1 w.p. FLIP_Q0.

    Returns:
        DensityMatrixSimulator: simulator.
    """
    return _readout_only_simulator({
        0: [[1 - FLIP_Q0, FLIP_Q0], [0., 1.]],
    })


class _ExplicitRRQPU(QPU):
    """QPU that accepts randomized_readout explicitly (no **kwargs)."""

    def __init__(self, config, n_shots=1024, randomized_readout=False):
        super().__init__(config=config, n_shots=n_shots)
        self.randomized_readout = randomized_readout


class TestRandomizedReadoutInit:

    def test_rejects_qpu_without_randomized_readout(self, config):
        with pytest.raises(ValueError, match='randomized_readout'):
            RandomizedReadout(QPU, config, [0])

    def test_accepts_qpu_with_explicit_kwarg(self, config):
        ro = RandomizedReadout(
            _ExplicitRRQPU, config, [0], randomized_readout=True
        )
        assert ro.randomized_readout is True

    def test_accepts_qpu_with_var_kwargs(self, config):
        ro = RandomizedReadout(Emulator, config, [0])
        assert isinstance(ro, Emulator)

    def test_kwargs_are_forwarded_to_qpu(self, config):
        ro = RandomizedReadout(Emulator, config, [0], n_shots=123)
        assert ro._n_shots == 123

    def test_qubits_are_sorted(self, config):
        ro = RandomizedReadout(Emulator, config, [2, 0, 1])
        assert ro._qubits == [0, 1, 2]


class TestRandomizedReadoutCircuits:

    def test_generates_a_single_circuit(self, config):
        ro = RandomizedReadout(Emulator, config, [0, 1])
        ro.generate_circuits()
        assert ro.circuits.n_circuits == 1

    def test_ends_with_a_measurement_of_every_qubit(self, config):
        ro = RandomizedReadout(Emulator, config, [1, 0])
        ro.generate_circuits()
        circuit = ro.circuits[0]

        final = circuit.cycles[-1]
        assert all(gate.is_measurement for gate in final)
        assert sorted(q for gate in final for q in gate.qubits) == [0, 1]
        assert circuit.cycles[-2].is_barrier

    def test_state_preparation_is_the_identity(self, config):
        ro = RandomizedReadout(Emulator, config, [0, 1])
        ro.generate_circuits()
        prep = Circuit(
            [c for c in ro.circuits[0].cycles[:-1] if not c.is_barrier]
        )
        unitary = prep.unitary
        # Identity up to a global phase.
        assert abs(np.trace(unitary)) / unitary.shape[0] == (
            pytest.approx(1.)
        )


@pytest.mark.usefixtures('no_plot')
class TestRandomizedReadoutRun:

    def test_noiseless_fidelity_is_one(self, config):
        ro = RandomizedReadout(
            Emulator, config, [0, 1],
            simulator=StateVectorSimulator(), n_shots=200,
        )
        ro.run()

        assert dict(ro.results.dict) == {'00': 200}
        assert ro.fidelity.nominal_value == 1.
        assert ro.fidelity.std_dev == 0.

    def test_fidelity_matches_readout_error(self, config):
        n_shots = 20000
        ro = RandomizedReadout(
            Emulator, config, [0, 1],
            simulator=_asymmetric_q0_simulator(), n_shots=n_shots,
        )
        ro.run()

        sigma = np.sqrt(FLIP_Q0 * (1 - FLIP_Q0) / n_shots)
        assert ro.fidelity.nominal_value == pytest.approx(
            1 - FLIP_Q0, abs=5 * sigma
        )

    def test_bit_order_matches_sorted_qubits(self, config):
        # Only qubit 0 (the leftmost bit) has readout errors.
        ro = RandomizedReadout(
            Emulator, config, [1, 0],
            simulator=_asymmetric_q0_simulator(), n_shots=2000,
        )
        ro.run()

        assert set(ro.results.states) == {'00', '10'}

    def test_fidelity_uncertainty_is_binomial_standard_error(self, config):
        n_shots = 4000
        ro = RandomizedReadout(
            Emulator, config, [0, 1],
            simulator=_asymmetric_q0_simulator(), n_shots=n_shots,
        )
        ro.run()

        p0 = ro.results.dict['00'] / n_shots
        assert ro.results.n_shots == n_shots
        assert ro.fidelity.nominal_value == pytest.approx(p0)
        assert ro.fidelity.std_dev == pytest.approx(
            np.sqrt(p0 * (1 - p0) / n_shots)
        )

    def test_default_emulator_fidelity(self, config):
        # Default Emulator readout: P(0 | 0) = DEFAULT_READOUT_P0 per
        # qubit, plus negligible single-qubit depolarizing noise.
        ro = RandomizedReadout(Emulator, config, [0, 1], n_shots=8192)
        ro.run()

        assert ro.fidelity.nominal_value == pytest.approx(
            DEFAULT_READOUT_P0 ** 2, abs=0.01
        )

    def test_results_are_the_circuit_results(self, config):
        ro = RandomizedReadout(
            Emulator, config, [0], simulator=StateVectorSimulator(),
            n_shots=50,
        )
        ro.run()
        assert ro.results is ro.circuits[0].results

    def test_exp_id_is_tagged_with_qubits(self, config):
        ro = RandomizedReadout(
            Emulator, config, [1, 0], simulator=StateVectorSimulator(),
            n_shots=10,
        )
        ro.run()
        assert ro.data_manager._exp_id.endswith(
            '_randomized_readout_Q0Q1'
        )

    def test_results_correct_a_symmetric_bit_flip_channel(self, config):
        # A symmetric confusion matrix is a stochastic bit-flip channel,
        # which is what randomized readout tailors readout noise into.
        # The RR characterization should then undo it via QPRC.
        flip = 0.1
        n_shots = 20000
        sim = _readout_only_simulator({
            0: [[1 - flip, flip], [flip, 1 - flip]],
        })

        ro = RandomizedReadout(
            Emulator, config, [0], simulator=sim, n_shots=n_shots,
        )
        ro.run()

        qpu = Emulator(config, simulator=sim, n_shots=n_shots)
        circuit = Circuit([Cycle({X(0)}), Cycle({Meas(0)})])
        qpu.run(circuit)
        measured = circuit.results
        assert measured.populations['0'] == pytest.approx(flip, abs=0.02)

        corrected = Results(measured.dict, rr_char_results=ro.results)
        assert corrected.rr_char_results is ro.results
        assert corrected.populations['1'] == pytest.approx(1., abs=0.02)


class TestRandomizedReadoutBeforeRun:

    @pytest.mark.xfail(
        reason='RandomizedReadout.results indexes self._circuits[0] '
               'before checking it, so it raises TypeError instead of '
               'returning an empty Results before generate_circuits() '
               '(qcal/characterization/readout.py:110).',
        raises=TypeError,
        strict=True,
    )
    def test_results_before_run_is_empty(self, config):
        ro = RandomizedReadout(Emulator, config, [0])
        assert dict(ro.results.dict) == {}


class TestRandomizedReadoutPlot:

    @pytest.mark.xfail(
        reason="run() calls Results.plot, which passes 'titlefont_size' "
               'to plotly; plotly >= 6 rejects it '
               '(qcal/results.py:464).',
        raises=ValueError,
        strict=True,
    )
    def test_run_with_plotting(self, config):
        ro = RandomizedReadout(
            Emulator, config, [0], simulator=StateVectorSimulator(),
            n_shots=10,
        )
        ro.run()
