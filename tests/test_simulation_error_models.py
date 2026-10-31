"""Unit tests for qcal.simulation.error_models.

Each concrete ErrorModel's channel_for() dispatch is tested directly;
a few representative physical effects are verified end-to-end through
DensityMatrixSimulator (see also test_simulators.py), since that is
the only way to observe what a quax channel actually does to a state.
"""
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import quax

import qcal as qc
from qcal.simulation.error_models import (
    AmplitudeDamping,
    BitFlipNoise,
    CustomErrorModel,
    DephasingNoise,
    DepolarizingNoise,
    ErrorModel,
    LeakageNoise,
    PhaseFlipNoise,
    RelaxationNoise,
    RelaxationParams,
    SeepageNoise,
    UnitaryError,
    gate_category,
)
from qcal.simulation.simulators import DensityMatrixSimulator


class TestGateCategory:

    def test_single_qubit_gate(self):
        assert gate_category('X') == 'single_qubit'

    def test_two_qubit_gate(self):
        assert gate_category('CNOT') == 'two_qubit'

    def test_single_qutrit_gate(self):
        assert gate_category('H3') == 'single_qutrit'

    def test_two_qutrit_gate(self):
        assert gate_category('CZ3') == 'two_qutrit'

    def test_unknown_gate_returns_none(self):
        assert gate_category('NotARealGate') is None


class TestRelaxationParams:

    def test_defaults(self):
        params = RelaxationParams(t1=50e-6, tphi=30e-6)
        assert params.p1 == 0.0
        assert params.t == 1.0

    def test_custom_values(self):
        params = RelaxationParams(t1=50e-6, tphi=30e-6, p1=0.02, t=20e-9)
        assert params.t1 == 50e-6
        assert params.tphi == 30e-6
        assert params.p1 == 0.02
        assert params.t == 20e-9


class _NoOpModel(ErrorModel):
    """Minimal concrete ErrorModel for exercising the base-class API."""

    def channel_for(self, gate_name):
        return None


class TestErrorModelBaseClass:

    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            ErrorModel()

    def test_channel_for_gate_defaults_to_none(self):
        model = _NoOpModel()
        assert model.channel_for_gate(qc.X(0)) is None

    def test_confusion_matrix_for_defaults_to_none(self):
        model = _NoOpModel()
        assert model.confusion_matrix_for('Q0') is None

    def test_instrument_for_defaults_to_none(self):
        model = _NoOpModel()
        assert model.instrument_for('Q0') is None

    def test_add_readout_noise_transposes_to_column_stochastic(self):
        model = _NoOpModel()
        # Row-stochastic: C[prep, meas].
        cmat = np.array([[0.9, 0.1], [0.2, 0.8]])
        model.add_readout_noise('Q0', cmat)
        np.testing.assert_allclose(model.confusion_matrix_for('Q0'), cmat.T)

    def test_add_readout_noise_accepts_integer_qudit_label(self):
        model = _NoOpModel()
        model.add_readout_noise(0, np.eye(2))
        assert model.confusion_matrix_for('Q0') is not None

    def test_add_readout_noise_accepts_dataframe(self):
        model = _NoOpModel()
        cmat = pd.DataFrame([[0.9, 0.1], [0.2, 0.8]])
        model.add_readout_noise('Q0', cmat)
        np.testing.assert_allclose(
            model.confusion_matrix_for('Q0'), cmat.to_numpy().T
        )

    def test_instrument_for_builds_instrument_once_registered(self):
        model = _NoOpModel()
        model.add_readout_noise('Q0', np.eye(2))
        instrument = model.instrument_for('Q0')
        assert isinstance(instrument, quax.QuantumInstrument)


class TestDepolarizingNoise:

    def test_zero_rates_give_none_for_every_category(self):
        noise = DepolarizingNoise()
        assert noise.channel_for('X') is None
        assert noise.channel_for('CNOT') is None
        assert noise.channel_for('H3') is None
        assert noise.channel_for('CZ3') is None

    def test_single_qubit_rate_only_affects_single_qubit_category(self):
        noise = DepolarizingNoise(single_qubit=0.01)
        assert noise.channel_for('X') is not None
        assert noise.channel_for('CNOT') is None

    def test_two_qubit_rate_is_a_joint_channel_on_the_gate_body(self):
        noise = DepolarizingNoise(two_qubit=0.01)
        channel = noise.channel_for('CNOT')
        assert channel is not None
        assert channel.dims == ((2, 2), (2, 2))

    def test_unknown_gate_returns_none(self):
        noise = DepolarizingNoise(single_qubit=0.01)
        assert noise.channel_for('NotARealGate') is None

    def test_qutrit_categories_build_without_error(self):
        noise = DepolarizingNoise(single_qutrit=0.01, two_qutrit=0.01)
        assert noise.channel_for('H3') is not None
        assert noise.channel_for('CZ3') is not None


class TestAmplitudeDamping:

    def test_zero_rates_give_none(self):
        noise = AmplitudeDamping()
        assert noise.channel_for('X') is None

    def test_decays_excited_population_toward_ground_state(self):
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        noise = AmplitudeDamping(single_qubit=0.2)
        sim = DensityMatrixSimulator(noise_model=noise)
        sim.run(circuit)
        results = dict(circuit.results.dict)
        assert 0.0 < results['1'] < 1.0

    def test_two_qubit_channel_is_applied_per_qudit_not_jointly(self):
        # Unlike DepolarizingNoise, AmplitudeDamping's two_qubit channel
        # acts on a single qudit (dims=(2,)), broadcast independently.
        noise = AmplitudeDamping(two_qubit=0.1)
        channel = noise.channel_for('CNOT')
        assert channel.dims == ((2,), (2,))


class TestDephasingNoise:

    def test_zero_rates_give_none(self):
        noise = DephasingNoise()
        assert noise.channel_for('X') is None
        assert noise.channel_for('H3') is None

    def test_single_qubit_rate_does_not_change_populations(self):
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        noise = DephasingNoise(single_qubit=0.5)
        sim = DensityMatrixSimulator(noise_model=noise)
        sim.run(circuit)
        # Pure dephasing kills coherence but not populations: X(0)
        # deterministically prepares |1>, so P(1) stays 1 regardless.
        assert dict(circuit.results.dict) == pytest.approx({'1': 1.0})

    def test_single_qutrit_channel_needs_either_qubit_or_qutrit_rate(self):
        assert DephasingNoise(single_qubit=0.1).channel_for('H3') is not None
        assert DephasingNoise(single_qutrit=0.1).channel_for('H3') is not None
        assert DephasingNoise().channel_for('H3') is None

    def test_two_qubit_rate_is_a_joint_channel_on_the_gate_body(self):
        noise = DephasingNoise(two_qubit=0.1)
        channel = noise.channel_for('CNOT')
        assert channel is not None
        assert channel.dims == ((2, 2), (2, 2))


class TestBitFlipNoise:

    def test_zero_rates_give_none(self):
        assert BitFlipNoise().channel_for('X') is None

    def test_single_qubit_category(self):
        assert BitFlipNoise(single_qubit=0.1).channel_for('X') is not None

    def test_single_qutrit_channel_needs_either_qubit_or_qutrit_rate(self):
        assert BitFlipNoise(single_qubit=0.1).channel_for('H3') is not None
        assert BitFlipNoise(single_qutrit=0.1).channel_for('H3') is not None
        assert BitFlipNoise().channel_for('H3') is None


class TestPhaseFlipNoise:

    def test_zero_rates_give_none(self):
        assert PhaseFlipNoise().channel_for('X') is None

    def test_single_qubit_category(self):
        assert PhaseFlipNoise(single_qubit=0.1).channel_for('X') is not None

    def test_phase_flip_does_not_change_populations(self):
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        noise = PhaseFlipNoise(single_qubit=0.5)
        sim = DensityMatrixSimulator(noise_model=noise)
        sim.run(circuit)
        assert dict(circuit.results.dict) == pytest.approx({'1': 1.0})


class TestRelaxationNoise:

    def test_none_params_give_none_for_every_category(self):
        noise = RelaxationNoise()
        assert noise.channel_for('X') is None
        assert noise.channel_for('H3') is None

    def test_single_qubit_params_build_a_channel(self):
        params = RelaxationParams(t1=50e-6, tphi=30e-6, t=50e-9)
        noise = RelaxationNoise(single_qubit=params)
        assert noise.channel_for('X') is not None
        assert noise.channel_for('CNOT') is None

    def test_qutrit_category_falls_back_to_qubit_params_for_ge(self):
        params = RelaxationParams(t1=50e-6, tphi=30e-6, t=50e-9)
        noise = RelaxationNoise(single_qubit=params)
        assert noise.channel_for('H3') is not None

    def test_decays_excited_population_toward_ground_state(self):
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        params = RelaxationParams(t1=20e-9, tphi=30e-6, t=50e-9)
        noise = RelaxationNoise(single_qubit=params)
        sim = DensityMatrixSimulator(noise_model=noise)
        sim.run(circuit)
        assert dict(circuit.results.dict)['1'] < 1.0


class TestLeakageNoise:

    def test_qubit_categories_are_always_none(self):
        noise = LeakageNoise(single_qutrit=0.1, two_qutrit=0.1)
        assert noise.channel_for('X') is None
        assert noise.channel_for('CNOT') is None

    def test_single_qutrit_category(self):
        assert LeakageNoise(single_qutrit=0.1).channel_for('H3') is not None

    def test_zero_rate_gives_none(self):
        assert LeakageNoise().channel_for('H3') is None


class TestSeepageNoise:

    def test_qubit_categories_are_always_none(self):
        noise = SeepageNoise(single_qutrit=0.1, two_qutrit=0.1)
        assert noise.channel_for('X') is None
        assert noise.channel_for('CNOT') is None

    def test_single_qutrit_category(self):
        assert SeepageNoise(single_qutrit=0.1).channel_for('H3') is not None


class TestUnitaryError:

    def test_none_matrix_gives_none_channel(self):
        assert UnitaryError().channel_for('X') is None

    def test_identity_matrix_collapses_to_none(self):
        noise = UnitaryError(single_qubit=np.eye(2))
        assert noise.channel_for('X') is None

    def test_non_identity_matrix_builds_a_channel(self):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        noise = UnitaryError(single_qubit=Z)
        assert noise.channel_for('X') is not None

    def test_unsupported_matrix_size_raises_value_error(self):
        non_identity_5x5 = np.diag([1, 2, 3, 4, 5]).astype(complex)
        with pytest.raises(ValueError, match='unsupported matrix size'):
            UnitaryError(single_qubit=non_identity_5x5)

    def test_per_gate_instance_channel(self):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        noise = UnitaryError(gate_unitaries={qc.X(0): Z})
        assert noise.channel_for_gate(qc.X(0)) is not None
        assert noise.channel_for_gate(qc.X(1)) is None

    def test_per_gate_instance_takes_priority_over_category(self):
        # Category-level error would flip the population; the per-gate
        # override (a pure phase flip) must not. This is only
        # observable by actually running the simulator (the
        # category-vs-instance priority is the simulator's job).
        X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
        noise = UnitaryError(
            single_qubit=X_mat, gate_unitaries={qc.X(0): Z_mat}
        )
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        sim = DensityMatrixSimulator(noise_model=noise)
        sim.run(circuit)
        assert dict(circuit.results.dict) == pytest.approx({'1': 1.0})


class TestCustomErrorModel:

    def test_empty_composite_returns_none_for_everything(self):
        noise = CustomErrorModel()
        assert noise.channel_for('X') is None
        assert noise.confusion_matrix_for('Q0') is None
        assert noise.instrument_for('Q0') is None

    def test_returns_the_single_submodel_channel_unchanged(self):
        sub = DepolarizingNoise(single_qubit=0.01)
        noise = CustomErrorModel(sub)
        assert noise.channel_for('X') is sub.channel_for('X')

    def test_skips_submodels_with_no_channel_for_the_gate(self):
        noise = CustomErrorModel(
            DepolarizingNoise(single_qubit=0.01),  # no two_qubit rate
            RelaxationNoise(
                two_qubit=RelaxationParams(t1=50e-6, tphi=30e-6)
            ),
        )
        assert noise.channel_for('CNOT') is not None
        assert noise.channel_for('NotARealGate') is None

    def test_channels_compose_in_submodel_order(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        flip_kraus = quax.KrausMap.from_matrix(
            jnp.array(X[np.newaxis]), dims=((2,), (2,))
        )

        class _FlipModel(ErrorModel):
            def channel_for(self, gate_name):
                return flip_kraus if gate_name == 'X' else None

        damp_channel = quax.channels.amplitude_damping(1.0)

        class _DampModel(ErrorModel):
            def channel_for(self, gate_name):
                return damp_channel if gate_name == 'X' else None

        def run_order(*models):
            noise = CustomErrorModel(*models)
            circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
            circuit.measure()
            sim = DensityMatrixSimulator(noise_model=noise)
            sim.run(circuit)
            return dict(circuit.results.dict)

        # flip-then-damp: X(|0>-> |1>) -> flip -> |0> -> damp (no-op on
        # ground state) -> |0>.
        flip_then_damp = run_order(_FlipModel(), _DampModel())
        damp_then_flip = run_order(_DampModel(), _FlipModel())

        assert flip_then_damp == pytest.approx({'0': 1.0})
        assert damp_then_flip != pytest.approx(flip_then_damp)

    def test_own_confusion_matrix_overrides_submodel(self):
        sub = _NoOpModel()
        sub.add_readout_noise('Q0', np.eye(2))
        noise = CustomErrorModel(sub)
        own_cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        noise.add_readout_noise('Q0', own_cmat)
        np.testing.assert_allclose(
            noise.confusion_matrix_for('Q0'), own_cmat.T
        )

    def test_confusion_matrix_falls_through_to_submodel(self):
        sub = _NoOpModel()
        sub_cmat = np.array([[0.9, 0.1], [0.2, 0.8]])
        sub.add_readout_noise('Q0', sub_cmat)
        noise = CustomErrorModel(sub)
        np.testing.assert_allclose(
            noise.confusion_matrix_for('Q0'), sub_cmat.T
        )

    def test_channel_for_gate_falls_through_to_submodel(self):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        sub = UnitaryError(gate_unitaries={qc.X(0): Z})
        noise = CustomErrorModel(sub)
        assert noise.channel_for_gate(qc.X(0)) is not None
        assert noise.channel_for_gate(qc.X(1)) is None

    def test_nested_custom_error_model(self):
        inner = CustomErrorModel(DepolarizingNoise(single_qubit=0.01))
        outer = CustomErrorModel(inner)
        assert outer.channel_for('X') is not None


class TestCustomErrorModelFromPTM:

    def test_identity_ptm_leaves_state_unchanged(self):
        ptm_identity = np.diag([1.0, 1.0, 1.0, 1.0])
        noise = CustomErrorModel.from_PTM({qc.X(0): ptm_identity})
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        sim = DensityMatrixSimulator(noise_model=noise)
        sim.run(circuit)
        assert dict(circuit.results.dict) == pytest.approx({'1': 1.0})

    def test_only_matches_the_exact_gate_instance(self):
        noise = CustomErrorModel.from_PTM(
            {qc.X(0): np.diag([1.0, 1.0, 1.0, 1.0])}
        )
        assert noise.channel_for_gate(qc.X(0)) is not None
        assert noise.channel_for_gate(qc.X(1)) is None

    def test_qutrit_gate_raises_value_error(self):
        with pytest.raises(ValueError, match='power-of-2'):
            CustomErrorModel.from_PTM({qc.H3(0): np.eye(9)})


class TestCustomErrorModelFromPNR:

    def test_pure_pauli_z_does_not_change_populations(self):
        noise = CustomErrorModel.from_PNR({qc.X(0): {'Z': 1.0}})
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        sim = DensityMatrixSimulator(noise_model=noise)
        sim.run(circuit)
        assert dict(circuit.results.dict) == pytest.approx({'1': 1.0})

    def test_pure_pauli_x_flips_population(self):
        noise = CustomErrorModel.from_PNR({qc.X(0): {'X': 1.0}})
        circuit = qc.Circuit([qc.Cycle({qc.X(0)})])
        circuit.measure()
        sim = DensityMatrixSimulator(noise_model=noise)
        sim.run(circuit)
        assert dict(circuit.results.dict) == pytest.approx({'0': 1.0})

    def test_mismatched_pauli_length_raises_value_error(self):
        with pytest.raises(ValueError, match='acts on 2 qudit'):
            CustomErrorModel.from_PNR({qc.CNOT(0, 1): {'Z': 0.1}})

    def test_negative_probability_raises_value_error(self):
        with pytest.raises(ValueError, match='negative'):
            CustomErrorModel.from_PNR({qc.X(0): {'Z': -0.1}})

    def test_probabilities_summing_over_one_raises_value_error(self):
        with pytest.raises(ValueError, match=r'> 1'):
            CustomErrorModel.from_PNR({qc.X(0): {'Z': 0.6, 'X': 0.6}})

    def test_empty_pauli_dict_is_skipped(self):
        noise = CustomErrorModel.from_PNR({qc.X(0): {}})
        assert noise.channel_for_gate(qc.X(0)) is None
