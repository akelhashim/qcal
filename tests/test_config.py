"""Unit tests for qcal.config.Config, using the example config.yaml."""
import pandas as pd
import pytest

from qcal.config import Config


class TestLoading:

    def test_filename(self, config):
        assert config.filename.endswith('config.yaml')

    def test_top_level_keys(self, config):
        assert set(config.parameters.keys()) == {
            'single_qubit', 'two_qubit', 'readout', 'reset', 'hardware',
            'initialize',
        }

    def test_len(self, config):
        assert len(config) == 6

    def test_reload_restores_mutated_value(self, config):
        original = config['single_qubit/0/GE/freq']
        config['single_qubit/0/GE/freq'] = 1.0
        config.reload()
        assert config['single_qubit/0/GE/freq'] == original

    def test_missing_file_falls_back_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = Config()
        assert cfg.filename is None
        assert cfg.parameters == {}
        assert cfg['single_qubit/0/GE/freq'] is None


class TestQubitsAndPairs:

    def test_qubits(self, config):
        assert config.qubits == (0, 1, 2, 3, 4, 5, 6, 7)

    def test_n_qubits(self, config):
        assert config.n_qubits == 8

    def test_qubits_even_odd(self, config):
        assert config.qubits_even == (0, 2, 4, 6)
        assert config.qubits_odd == (1, 3, 5, 7)

    def test_qubit_pairs(self, config):
        assert config.qubit_pairs == [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (4, 5), (5, 6), (6, 7), (7, 0),
        ]


class TestGetSetItem:

    def test_getitem_nested_path(self, config):
        assert config['single_qubit/0/GE/freq'] == 5459280156.35984

    def test_get_with_list_path_matches_getitem(self, config):
        assert (
            config.get(['single_qubit', 0, 'GE', 'freq'])
            == config['single_qubit/0/GE/freq']
        )

    def test_getitem_missing_param_returns_none(self, config):
        assert config['single_qubit/0/GE/bogus'] is None

    def test_setitem_updates_value(self, config):
        config['single_qubit/0/GE/T2DD'] = 1e-05
        assert config['single_qubit/0/GE/T2DD'] == 1e-05

    def test_setitem_rounds_large_float_to_5_decimals(self, config):
        config['single_qubit/0/GE/freq'] = 123.456789123
        assert config['single_qubit/0/GE/freq'] == 123.45679

    def test_setitem_rounds_small_float_to_8_sig_figs(self, config):
        config['single_qubit/0/GE/T1'] = 0.000123456789123
        assert config['single_qubit/0/GE/T1'] == 0.00012345679

    def test_setitem_zero_float_is_preserved(self, config):
        config['single_qubit/0/GE/freq'] = 0.0
        assert config['single_qubit/0/GE/freq'] == 0.0


class TestNativeGates:

    def test_single_qubit_native_gates(self, config):
        gates = config.native_gates['single_qubit']
        assert gates[0] == {'GE': ['X90', 'X'], 'EF': ['X90', 'X']}

    def test_two_qubit_native_gates(self, config):
        gates = config.native_gates['two_qubit']
        assert gates[(0, 1)] == ['CZ']

    def test_native_gate_set(self, config):
        assert config.native_gates['set'] == {'X90', 'X', 'CZ'}


class TestTables:

    def test_single_qubit_table(self, config):
        df = config.single_qubit
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == list(config.qubits)

    def test_two_qubit_table(self, config):
        df = config.two_qubit
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == config.qubit_pairs

    def test_readout_table(self, config):
        df = config.readout
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == config.n_qubits

    def test_reset_table(self, config):
        df = config.reset
        assert list(df['reset']) == [True, 0.0005, False, 1]

    def test_coherence_times(self, config):
        df = config.coherence_times
        row = df[(df['Qubit'] == 0) & (df['Coherence Type'] == 'T1 GE')]
        assert row['Coherence Time'].iloc[0] == 5.4e-05


class TestCopy:

    def test_shallow_copy_shares_nested_state(self, config):
        shallow = config.__copy__()
        shallow['single_qubit'][0]['GE']['T1'] = 999
        assert config['single_qubit/0/GE/T1'] == 999

    def test_deep_copy_is_independent(self, config):
        deep = config.__copy__(deep_copy=True)
        deep['single_qubit'][0]['GE']['T1'] = 999
        assert config['single_qubit/0/GE/T1'] != 999


class TestKnownIssues:

    @pytest.mark.xfail(
        reason=(
            "Config.hardware crashes on configs where the 'hardware' "
            "section mixes dict-valued keys (e.g. sample_rate) with "
            "None-valued keys (e.g. readout_LO): pandas.DataFrame.from_dict"
            "(orient='index') calls .items() on the None and raises "
            "AttributeError."
        ),
        raises=AttributeError,
        strict=True,
    )
    def test_hardware_table(self, config):
        df = config.hardware
        assert df.loc['sample_rate', 'hardware'] == {
            'ADC': 2000000000.0, 'DAC': 8000000000.0
        }


class TestSaveLoad:

    def test_save_and_load_roundtrip(self, config, tmp_path):
        path = tmp_path / 'saved_config.yaml'
        config.save(str(path))
        reloaded = Config(str(path))
        assert reloaded.qubits == config.qubits
        assert (
            reloaded['single_qubit/0/GE/freq']
            == config['single_qubit/0/GE/freq']
        )

    def test_save_without_filename_warns_and_skips(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        cfg = Config()
        cfg.save()
        assert list(tmp_path.iterdir()) == []


class TestDunderMethods:

    def test_call_returns_parameters(self, config):
        assert config() == config.parameters

    def test_repr_and_str_contain_top_level_keys(self, config):
        assert 'single_qubit' in repr(config)
        assert 'single_qubit' in str(config)

    def test_items_matches_parameters(self, config):
        assert dict(config.items()) == config.parameters
