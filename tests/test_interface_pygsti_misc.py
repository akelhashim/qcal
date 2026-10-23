"""Unit tests for qcal.interface.pygsti.{circuits,datasets,processor_spec}.

pygsti is a hard dependency (always installed), so these are direct
functional tests, not optional-import tests.
"""
import pytest
from pygsti.circuits import Circuit as PygstiCircuit
from pygsti.data import DataSet
from pygsti.io import write_circuit_list
from pygsti.processors import QubitProcessorSpec

from qcal.circuit import CircuitSet
from qcal.config import Config
from qcal.interface.pygsti.circuits import load_circuits
from qcal.interface.pygsti.datasets import generate_pygsti_dataset
from qcal.interface.pygsti.processor_spec import pygsti_pspec

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'


class TestLoadCircuits:

    def test_from_list(self):
        c1 = PygstiCircuit([('Gxpi2', 0)], line_labels=(0,))
        c2 = PygstiCircuit([('Gypi2', 0)], line_labels=(0,))

        cs = load_circuits([c1, c2])

        assert isinstance(cs, CircuitSet)
        assert cs.n_circuits == 2
        assert list(cs['pygsti_circuit']) == [c1.str, c2.str]

    def test_from_file(self, tmp_path):
        c1 = PygstiCircuit([('Gxpi2', 0)], line_labels=(0,))
        c2 = PygstiCircuit([('Gypi2', 0)], line_labels=(0,))
        path = str(tmp_path / 'circuits.txt')
        write_circuit_list(path, [c1, c2])

        cs = load_circuits(path)

        assert cs.n_circuits == 2
        assert list(cs['pygsti_circuit']) == [c1.str, c2.str]


class TestGeneratePygstiDataset:

    def test_builds_dataset_with_correct_counts(self):
        c1 = PygstiCircuit([('Gxpi2', 0)], line_labels=(0,))
        c2 = PygstiCircuit([('Gypi2', 0)], line_labels=(0,))
        cs = CircuitSet(circuits=[c1, c2])
        cs['results'] = [{'0': 80, '1': 20}, {'0': 50, '1': 50}]

        ds = generate_pygsti_dataset(cs)

        assert isinstance(ds, DataSet)
        assert ds[c1][('0',)] == 80
        assert ds[c1][('1',)] == 20
        assert ds[c2][('0',)] == 50

    def test_missing_results_column_raises(self):
        c1 = PygstiCircuit([('Gxpi2', 0)], line_labels=(0,))
        cs = CircuitSet(circuits=[c1])

        with pytest.raises(ValueError, match='results'):
            generate_pygsti_dataset(cs)

    def test_save_path_writes_dataset_file(self, tmp_path):
        c1 = PygstiCircuit([('Gxpi2', 0)], line_labels=(0,))
        cs = CircuitSet(circuits=[c1])
        cs['results'] = [{'0': 80, '1': 20}]

        save_path = str(tmp_path) + '/'
        generate_pygsti_dataset(cs, save_path=save_path)

        assert (tmp_path / 'dataset.txt').exists()


@pytest.fixture
def config():
    return Config(EXAMPLE_CONFIG_PATH)


class TestPygstiPspec:

    def test_returns_processor_spec(self, config):
        pspec = pygsti_pspec(config, qubits=(0, 1))
        assert isinstance(pspec, QubitProcessorSpec)
        assert pspec.qubit_labels == ('Q0', 'Q1')

    def test_translates_gate_names_via_gate_mapper(self, config):
        pspec = pygsti_pspec(
            config, qubits=(0, 1), native_gates=('X90', 'CZ')
        )
        assert pspec.gate_names == ('Gxpi2', 'Gcphase')

    def test_passthrough_for_already_pygsti_named_gates(self, config):
        pspec = pygsti_pspec(
            config, qubits=(0,), native_gates=('Gxpi2',),
            availability={'Gxpi2': [('Q0',)]},
        )
        assert pspec.gate_names == ('Gxpi2',)

    def test_auto_generated_availability_for_single_qubit_gate(self, config):
        pspec = pygsti_pspec(config, qubits=(0, 1), native_gates=('X90',))
        assert set(pspec.availability['Gxpi2']) == {('Q0',), ('Q1',)}

    def test_auto_generated_availability_for_two_qubit_gate_respects_config(
        self, config
    ):
        pspec = pygsti_pspec(
            config, qubits=(0, 1, 2), native_gates=('CZ',)
        )
        assert set(pspec.availability['Gcphase']) == {
            ('Q0', 'Q1'), ('Q1', 'Q2')
        }

    def test_explicit_availability_overrides_auto_generation(self, config):
        pspec = pygsti_pspec(
            config, qubits=(0, 1), native_gates=('X90',),
            availability={'Gxpi2': [('Q0',)]},
        )
        assert pspec.availability['Gxpi2'] == [('Q0',)]


class TestDunderAll:

    def test_datasets_all_is_a_tuple(self):
        import qcal.interface.pygsti.datasets as d
        assert d.__all__ == ('generate_pygsti_dataset',)

    def test_processor_spec_all_is_a_tuple(self):
        import qcal.interface.pygsti.processor_spec as p
        assert p.__all__ == ('pygsti_pspec',)


class TestKnownIssues:

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    def test_wraparound_qubit_pair_availability_is_dropped(self, config):
        # pygsti_pspec() builds `pairings` via
        # itertools.combinations(qubits, 2), which always yields
        # ascending-sorted tuples, then checks
        # `qubit_pair in pairings` against config.native_gates
        # ['two_qubit'].keys() verbatim. The example config's ring
        # topology declares its wraparound edge as the descending
        # tuple (7, 0) (not (0, 7)), so it never matches any ascending
        # `pairings` entry and is silently dropped from availability
        pspec = pygsti_pspec(
            config, qubits=(0, 1, 7), native_gates=('CZ',)
        )
        assert ('Q7', 'Q0') in pspec.availability['Gcphase']
