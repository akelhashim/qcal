"""Tests for renamed APIs and their deprecated aliases."""
import inspect
import pickle

import pandas as pd
import pytest

from qcal.benchmarking.cb import CB1, SC
from qcal.benchmarking.irb import IRB
from qcal.benchmarking.pnr import KNR
from qcal.dag.dag import NodeStatus
from qcal.fitting.fit import FitLinear
from qcal.fitting.fit_functions import linear
from qcal.managers.data_manager import DataManager, DataMananger
from qcal.utils import pop_deprecated_kwarg, save_to_pickle


class TestPopDeprecatedKwarg:

    def test_returns_value_when_old_name_absent(self):
        kwargs = {'other': 1}
        assert pop_deprecated_kwarg(kwargs, 'old', 'new', 5) == 5
        assert kwargs == {'other': 1}

    def test_uses_old_value_and_warns(self):
        kwargs = {'old': True, 'other': 1}
        with pytest.warns(DeprecationWarning, match="'old' is deprecated"):
            value = pop_deprecated_kwarg(
                kwargs, 'old', 'new', False, default=False
            )
        assert value is True
        assert kwargs == {'other': 1}  # old name is not forwarded

    def test_conflicting_values_raise(self):
        with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
            pop_deprecated_kwarg(
                {'old': False}, 'old', 'new', True, default=False
            )

    def test_matching_values_are_allowed(self):
        with pytest.warns(DeprecationWarning):
            assert pop_deprecated_kwarg(
                {'old': True}, 'old', 'new', True, default=False
            ) is True


@pytest.mark.parametrize('factory', [CB1, SC, IRB, KNR])
def test_factories_use_correct_propagate_correction_spelling(factory):
    params = inspect.signature(factory).parameters
    assert 'propagate_correction' in params
    assert 'propogate_correction' not in params
    assert params['propagate_correction'].default is False


class TestFitFunctionProperty:

    def test_fit_function(self):
        assert FitLinear().fit_function is linear

    def test_misspelled_alias_warns(self):
        with pytest.warns(DeprecationWarning, match='fit_function'):
            assert FitLinear().fit_fuction is linear


def test_data_manager_alias_is_same_class():
    assert DataMananger is DataManager


def test_node_status_completed_has_no_trailing_space():
    assert NodeStatus.COMPLETED.value == 'COMPLETED'


class TestSaveToPickle:

    def test_dataframe_round_trip(self, tmp_path):
        df = pd.DataFrame({'a': [1, 2]})
        save_to_pickle(df, str(tmp_path / 'df'))
        assert pd.read_pickle(tmp_path / 'df.pkl').equals(df)

    def test_non_dataframe_round_trip(self, tmp_path):
        save_to_pickle({'k': 1}, str(tmp_path / 'd.pkl'))
        with open(tmp_path / 'd.pkl', 'rb') as f:
            assert pickle.load(f) == {'k': 1}
