"""Unit tests for qcal.interface.pygsti.compiler.

pygsti is a hard dependency (always installed), so this is direct
functional coverage, not an optional-import test like
test_interface_{trueq,superstaq,pyquil}.py.
"""
import numpy as np
import pygsti
import pytest
from pygsti.baseobjs import Label

from qcal.interface.pygsti.compiler import (
    CLIFFORD_UNITARIES,
    CLIFFORD_UPDATE_TABLE,
    decompose_CZ,
    decompose_X90,
    get_clifford_from_unitary,
    pad_layer_clifford,
    pauli_randomize_clifford_circuit,
    pauli_vector_to_clifford_layer,
    update_clifford_labels,
)


class TestDecomposeX90:

    def test_two_phases_gives_zxz(self):
        layers = decompose_X90({0: [0.1, 0.2]})
        assert layers == [
            [Label('Gzr', 0, args=(0.1,))],
            [('Gxpi2', 0)],
            [Label('Gzr', 0, args=(0.2,))],
        ]

    def test_single_phase_gives_only_z_layer(self):
        layers = decompose_X90({0: [0.1]})
        assert layers == [[Label('Gzr', 0, args=(0.1,))]]

    def test_multiple_qubits_grouped_per_layer(self):
        layers = decompose_X90({0: [0.1, 0.2], 1: [0.3, 0.4]})
        assert len(layers) == 3
        assert set(layers[1]) == {('Gxpi2', 0), ('Gxpi2', 1)}


class TestDecomposeCZ:

    def test_two_layers_cphase_then_z(self):
        layers = decompose_CZ({(0, 1): [0.1, 0.2]})
        assert layers == [
            [('Gcphase', 0, 1)],
            [Label('Gzr', 0, args=(0.1,)), Label('Gzr', 1, args=(0.2,))],
        ]


class TestGetCliffordFromUnitary:

    @pytest.mark.parametrize('name', sorted(CLIFFORD_UNITARIES.keys()))
    def test_recovers_name_exactly(self, name):
        U = CLIFFORD_UNITARIES[name]
        assert get_clifford_from_unitary(U) == name

    @pytest.mark.parametrize('phase', [1, -1, 1j, -1j])
    def test_recovers_name_up_to_global_phase(self, phase):
        name = 'Gc1'
        U = phase * CLIFFORD_UNITARIES[name]
        assert get_clifford_from_unitary(U) == name

    def test_unrecognized_unitary_raises(self):
        with pytest.raises(ValueError):
            get_clifford_from_unitary(np.array([[1., 2.], [3., 4.]]))


class TestPadLayerClifford:

    def test_adds_identity_for_unused_qubits(self):
        padded = pad_layer_clifford([Label('Gc1', 0)], [0, 1])
        names = {c.name: c.qubits for c in padded.components}
        assert names == {'Gc1': (0,), 'Gc0': (1,)}

    def test_no_padding_needed_when_all_qubits_used(self):
        padded = pad_layer_clifford(
            [Label('Gc1', 0), Label('Gc2', 1)], [0, 1]
        )
        names = {c.name for c in padded.components}
        assert names == {'Gc1', 'Gc2'}


class TestPauliVectorToCliffordLayer:

    def test_matches_pauli_to_clifford_mapping(self):
        p_vec = np.array([2, 0, 0, 2])  # qubit0 -> Z, qubit1 -> X
        layer = pauli_vector_to_clifford_layer(p_vec, [0, 1])
        names = {c.name: c.qubits for c in layer.components}
        assert names == {'Gc9': (0,), 'Gc3': (1,)}


class TestUpdateCliffordLabels:

    def test_applies_clifford_update_table(self):
        layer = [Label('Gc1', 0)]
        p_vec = np.array([0, 0])
        q_vec = np.array([2, 0])
        updated = update_clifford_labels(layer, p_vec, q_vec, [0])

        expected_name = CLIFFORD_UPDATE_TABLE[('Gc0', 'Gc1', 'Gc9')]
        assert len(updated.components) == 1
        assert updated.components[0].name == expected_name

    def test_raises_when_layer_omits_a_qubit(self):
        layer = [Label('Gc1', 0)]
        p_vec = np.zeros(4, int)
        q_vec = np.zeros(4, int)
        with pytest.raises(ValueError, match='Missing'):
            update_clifford_labels(layer, p_vec, q_vec, [0, 1])


class TestPauliRandomizeCliffordCircuit:

    def _unitary(self, circuit):
        U = np.eye(2 ** circuit.width, dtype=complex)
        for i in range(circuit.depth):
            layer_U = None
            for c in circuit.layer_label(i).components:
                g = CLIFFORD_UNITARIES[c.name]
                layer_U = g if layer_U is None else np.kron(layer_U, g)
            U = layer_U @ U
        return U

    def test_two_qubit_circuit_implements_same_unitary_up_to_final_pauli(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            np.random, 'randint',
            lambda low, high, size: np.zeros(size, dtype=int)
        )
        circuit = pygsti.circuits.Circuit(
            [
                Label([Label('Gc1', 0), Label('Gc2', 1)]),
                Label([Label('Gc5', 0), Label('Gc3', 1)]),
            ],
            line_labels=(0, 1),
        )

        rc, bs, q_vec = pauli_randomize_clifford_circuit(circuit)

        assert rc.width == 2
        assert rc.depth == circuit.depth

        # With the Pauli-frame draws stubbed to all-zero (identity), the
        # randomized circuit's gates should be unchanged from the
        # original Clifford dressing (no actual Pauli randomization
        # applied), so the two circuits implement the same unitary
        # exactly.
        np.testing.assert_allclose(
            self._unitary(rc), self._unitary(circuit), atol=1e-10
        )
        assert bs == '00'
        np.testing.assert_array_equal(q_vec, np.zeros(4, dtype=int))

    def test_two_qubit_circuit_matches_unitary_up_to_tracked_pauli(self):
        np.random.seed(0)
        circuit = pygsti.circuits.Circuit(
            [
                Label([Label('Gc1', 0), Label('Gc2', 1)]),
                Label([Label('Gc5', 0), Label('Gc3', 1)]),
            ],
            line_labels=(0, 1),
        )

        rc, bs, q_vec = pauli_randomize_clifford_circuit(circuit)

        final_pauli_name = pauli_vector_to_clifford_layer(
            q_vec, [0, 1]
        )
        P_final = None
        for c in final_pauli_name.components:
            g = CLIFFORD_UNITARIES[c.name]
            P_final = g if P_final is None else np.kron(P_final, g)

        U_orig = self._unitary(circuit)
        U_rand = self._unitary(rc)
        expected = P_final @ U_orig

        idx = np.unravel_index(np.argmax(np.abs(expected)), expected.shape)
        phase = U_rand[idx] / expected[idx]
        np.testing.assert_allclose(
            expected * phase, U_rand, atol=1e-8
        )


class TestKnownIssues:

    @pytest.mark.xfail(
        reason=(
            "pad_layer_clifford()'s `return Label(padded_layer)` "
            "collapses to a plain atomic LabelTup (not a compound "
            "label) whenever padded_layer ends up with exactly one "
            "component -- i.e. whenever a circuit only involves a "
            "single qubit. Iterating that atomic Label then yields its "
            "raw (name, qubit) tuple contents instead of sub-Label "
            "objects, so pauli_randomize_clifford_circuit's "
            "update_clifford_labels(padded_layer, ...) crashes with "
            "AttributeError: 'str' object has no attribute 'qubits' "
            "for any single-qubit Clifford circuit."
        ),
        raises=AttributeError,
        strict=True,
    )
    def test_pauli_randomize_single_qubit_circuit(self):
        circuit = pygsti.circuits.Circuit(
            [Label('Gc1', 0), Label('Gc5', 0)], line_labels=(0,)
        )
        pauli_randomize_clifford_circuit(circuit)
