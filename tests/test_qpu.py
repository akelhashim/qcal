"""Unit tests for qcal.qpu.qpu.QPU.

QPU is an abstract base class: generate_sequence()/write()/acquire()/
process() are no-op stubs meant to be overridden by hardware-specific
subclasses. FakeQPU below overrides them with simple instrumented
stand-ins so the orchestration logic in measure()/_batch_measurements()
/run()/save() can be exercised without any real hardware.

Note on the JIT batching path: _batch_measurements() submits acquire()
to a background thread while the main thread concurrently prepares the
*next* batch, mutating self._exp_circuits and self._sequence. A real
acquire() must not read either of those directly -- only
self.active_sequence, a snapshot taken synchronously right after
write() and only ever reassigned after the previous batch's acquire()
has already returned (see QPU.active_sequence's docstring).
FakeQPU.write() records circuit counts via self._exp_circuits (safe,
always runs synchronously before submission, never inside acquire());
FakeQPU.acquire() only increments a counter, to avoid a flaky
dependency on exp_circuits from a background thread.
"""
import pathlib

import pytest

import qcal.settings as settings
from qcal.circuit import Circuit, CircuitSet, Cycle
from qcal.config import Config
from qcal.gates.single_qubit import X90
from qcal.managers.classification_manager import ClassificationManager
from qcal.managers.data_manager import DataMananger
from qcal.qpu.qpu import QPU
from qcal.utils import save_to_pickle

EXAMPLE_CONFIG_PATH = 'examples/config/config.yaml'


class FakeQPU(QPU):
    """Instrumented QPU subclass for testing orchestration logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_sequence_calls = 0
        self.write_sizes = []
        self.acquire_calls = 0
        self.process_calls = 0

    def generate_sequence(self):
        self.generate_sequence_calls += 1

    def write(self):
        self.write_sizes.append(self._exp_circuits.n_circuits)

    def acquire(self):
        self.acquire_calls += 1
        self._measurements.append('meas')

    def process(self):
        self.process_calls += 1


def _circuits(n=1):
    return [Circuit([Cycle({X90(0)})]) for _ in range(n)]


@pytest.fixture(autouse=True)
def isolate_settings(tmp_path, monkeypatch):
    """Keep QPU's classifier-pickle lookup and data saving out of the
    user's real config_path/data_save_path, and restore settings.
    """
    original_config_path = settings.Settings.config_path
    original_data_save_path = settings.Settings.data_save_path
    original_save_data = settings.Settings.save_data

    settings.Settings.config_path = str(tmp_path) + '/nonexistent/'
    settings.Settings.data_save_path = str(tmp_path) + '/'
    settings.Settings.save_data = False

    yield

    settings.Settings.config_path = original_config_path
    settings.Settings.data_save_path = original_data_save_path
    settings.Settings.save_data = original_save_data


@pytest.fixture
def config():
    return Config(EXAMPLE_CONFIG_PATH)


class TestConstruction:

    def test_defaults(self, config):
        qpu = QPU(config)

        assert qpu.config is config
        assert qpu.circuits is None
        assert qpu.compiled_circuits is None
        assert qpu.transpiled_circuits is None
        assert qpu.classifier is None
        assert qpu._n_shots == 1024
        assert qpu._n_batches == 1
        assert qpu.jit is True

    def test_n_levels_above_three_raises(self, config):
        with pytest.raises(AssertionError, match='n_levels'):
            QPU(config, n_levels=4)

    def test_classifier_n_levels_mismatch_raises(self, config):
        classifier = ClassificationManager(qubits=(0, 1), n_levels=2)

        with pytest.raises(AssertionError, match='n_levels'):
            QPU(config, classifier=classifier, n_levels=3)

    def test_explicit_classifier_is_used_as_is(self, config):
        classifier = ClassificationManager(qubits=(0, 1), n_levels=2)

        qpu = QPU(config, classifier=classifier, n_levels=2)

        assert qpu.classifier is classifier

    def test_falls_back_to_pickled_classifier_at_config_path(self, config):
        pathlib.Path(settings.Settings.config_path).mkdir(
            parents=True, exist_ok=True
        )
        classifier = ClassificationManager(qubits=(0, 1), n_levels=2)
        save_to_pickle(
            classifier,
            settings.Settings.config_path + 'ClassificationManager',
        )

        qpu = QPU(config)

        assert qpu.classifier is not None

    def test_no_classifier_available_is_not_an_error(self, config):
        qpu = QPU(config)
        assert qpu.classifier is None


class TestInitialize:

    def test_wraps_a_list_of_circuits_in_a_circuitset(self, config):
        qpu = QPU(config)

        qpu._initialize(_circuits(3))

        assert isinstance(qpu.circuits, CircuitSet)
        assert qpu.circuits.n_circuits == 3
        assert qpu.compiled_circuits.n_circuits == 0
        assert qpu.exp_circuits is qpu.circuits

    def test_uses_an_existing_circuitset_directly(self, config):
        qpu = QPU(config)
        cs = CircuitSet(circuits=_circuits(2))

        qpu._initialize(cs)

        assert qpu.circuits is cs

    def test_n_shots_and_n_batches_only_override_when_given(self, config):
        qpu = QPU(config, n_shots=100, n_batches=2)

        qpu._initialize(_circuits(1))
        assert qpu._n_shots == 100 and qpu._n_batches == 2

        qpu._initialize(_circuits(1), n_shots=500, n_batches=5)
        assert qpu._n_shots == 500 and qpu._n_batches == 5

    def test_runtime_dataframe_starts_at_zero(self, config):
        qpu = QPU(config)

        qpu._initialize(_circuits(1))

        expected_cols = {
            'Compile', 'Transpile', 'Sequencing', 'Write', 'Measure',
            'Process', 'Total',
        }
        assert set(qpu.runtime.columns) == expected_cols
        assert (qpu.runtime.iloc[0] == 0.0).all()

    def test_generates_a_new_exp_id(self, config, monkeypatch):
        qpu = QPU(config)
        calls = []
        monkeypatch.setattr(
            DataMananger, 'generate_exp_id', lambda self: calls.append(1)
        )

        qpu._initialize(_circuits(1))

        assert calls == [1]


class TestCompileTranspile:

    class _StubPass:
        def __init__(self):
            self.calls = []

        def compile(self, circuits):
            self.calls.append(circuits)
            return circuits

        transpile = compile

    def test_single_compiler_is_applied_once(self, config):
        stub = self._StubPass()
        qpu = QPU(config, compiler=stub)
        qpu._initialize(_circuits(1))

        qpu.compile()

        assert len(stub.calls) == 1
        assert qpu.compiled_circuits.n_circuits == 1

    def test_list_of_compilers_are_applied_in_sequence(self, config):
        stub_a, stub_b = self._StubPass(), self._StubPass()
        qpu = QPU(config, compiler=[stub_a, stub_b])
        qpu._initialize(_circuits(1))

        qpu.compile()

        assert len(stub_a.calls) == 1
        assert len(stub_b.calls) == 1

    def test_single_transpiler_is_applied_once(self, config):
        stub = self._StubPass()
        qpu = QPU(config, transpiler=stub)
        qpu._initialize(_circuits(1))

        qpu.transpile()

        assert len(stub.calls) == 1
        assert qpu.transpiled_circuits.n_circuits == 1


class TestMeasure:

    def test_prepares_writes_and_acquires_in_order(self, config):
        qpu = FakeQPU(config)
        qpu._initialize(_circuits(2))

        qpu.measure()

        assert qpu.generate_sequence_calls == 1
        assert qpu.write_sizes == [2]
        assert qpu.acquire_calls == 1


class TestBatchMeasurements:

    def test_no_batching_when_within_n_circs_per_seq(self, config):
        qpu = FakeQPU(config, n_circs_per_seq=10)
        qpu._initialize(_circuits(5))

        qpu._batch_measurements()

        assert qpu.write_sizes == [5]
        assert qpu.acquire_calls == 1
        assert qpu.process_calls == 1

    def test_non_jit_batches_sequentially(self, config):
        qpu = FakeQPU(config, n_circs_per_seq=2, jit=False)
        qpu._initialize(_circuits(5))

        qpu._batch_measurements()

        assert qpu.write_sizes == [2, 2, 1]
        assert qpu.generate_sequence_calls == 3
        assert qpu.acquire_calls == 3
        assert qpu.process_calls == 1

    def test_jit_batches_overlap_prep_with_acquire(self, config):
        qpu = FakeQPU(config, n_circs_per_seq=2, jit=True)
        qpu._initialize(_circuits(5))

        qpu._batch_measurements()

        assert qpu.write_sizes == [2, 2, 1]
        assert qpu.generate_sequence_calls == 3
        assert qpu.acquire_calls == 3
        assert qpu.process_calls == 1
        assert len(qpu.measurements) == 3


class TestActiveSequence:

    class _SequencingQPU(QPU):
        """Sets self._sequence to a batch-identifiable value, and
        records self.active_sequence at the moment acquire() runs --
        the regression check for the active_sequence snapshot.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.active_sequence_at_acquire = []

        def generate_sequence(self):
            self._sequence = f'seq-for-{self._exp_circuits.n_circuits}'

        def acquire(self):
            self.active_sequence_at_acquire.append(self.active_sequence)
            self._measurements.append('meas')

    def test_is_none_before_any_batch_is_written(self, config):
        qpu = self._SequencingQPU(config)
        assert qpu.active_sequence is None

    def test_matches_sequence_at_write_time(self, config):
        qpu = self._SequencingQPU(config)
        qpu._initialize(_circuits(2))

        qpu.measure()

        assert qpu.active_sequence == 'seq-for-2'

    def test_each_jit_batch_acquires_its_own_sequence(self, config):
        # Regression test: acquire() used to read self._sequence
        # directly, which generate_sequence() for the *next* batch can
        # overwrite concurrently with acquire() running in the JIT
        # path's background thread. Each acquire() call must see the
        # sequence generated for its own batch, not a later one.
        qpu = self._SequencingQPU(config, n_circs_per_seq=2, jit=True)
        qpu._initialize(_circuits(5))

        qpu._batch_measurements()

        assert qpu.active_sequence_at_acquire == [
            'seq-for-2', 'seq-for-2', 'seq-for-1',
        ]


class TestSave:

    def test_saves_circuits_measurements_runtime_and_config(self, config):
        qpu = FakeQPU(config)
        qpu.run(_circuits(1), auto_save=False)

        qpu.save()

        saved = {
            p.name
            for p in pathlib.Path(qpu.data_manager.save_path).iterdir()
        }
        assert {
            'circuits.qc', 'measurements.pkl', 'runtime.csv', 'config.yaml',
        } <= saved

    def test_skips_empty_compiled_and_transpiled_circuits(self, config):
        qpu = FakeQPU(config)
        qpu.run(_circuits(1), auto_save=False)

        qpu.save()

        saved = {
            p.name
            for p in pathlib.Path(qpu.data_manager.save_path).iterdir()
        }
        assert 'compiled_circuits.qc' not in saved
        assert 'transpiled_circuits.qc' not in saved

    def test_saves_classifier_when_present(self, config):
        classifier = ClassificationManager(qubits=(0,), n_levels=2)
        qpu = FakeQPU(config, classifier=classifier)
        qpu.run(_circuits(1), auto_save=False)

        qpu.save()

        saved = {
            p.name
            for p in pathlib.Path(qpu.data_manager.save_path).iterdir()
        }
        assert 'ClassificationManager.pkl' in saved


class TestRun:

    def test_updates_total_runtime(self, config):
        qpu = FakeQPU(config)

        qpu.run(_circuits(1), auto_save=False)

        assert qpu.runtime['Total'].iloc[0] >= 0.0

    def test_auto_save_true_and_save_data_true_saves(self, config):
        settings.Settings.save_data = True
        qpu = FakeQPU(config)

        qpu.run(_circuits(1), auto_save=True)

        assert qpu.data_manager.save_path is not None

    def test_auto_save_false_skips_saving_even_if_save_data_true(
        self, config
    ):
        settings.Settings.save_data = True
        qpu = FakeQPU(config)

        qpu.run(_circuits(1), auto_save=False)

        assert qpu.data_manager.save_path is None

    def test_save_data_false_skips_saving_even_if_auto_save_true(
        self, config
    ):
        settings.Settings.save_data = False
        qpu = FakeQPU(config)

        qpu.run(_circuits(1), auto_save=True)

        assert qpu.data_manager.save_path is None
