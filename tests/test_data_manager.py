"""Unit tests for qcal.managers.data_manager.DataMananger."""
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import qcal.settings as settings
from qcal import __version__
from qcal.managers.data_manager import DataMananger

EXP_ID_PATTERN = re.compile(
    r'^\d{8}_\d{6}_v' + re.escape(__version__) + r'$'
)


@pytest.fixture(autouse=True)
def restore_data_save_path():
    """DataMananger reads the shared Settings singleton -- restore it
    so a test's override doesn't leak into others.
    """
    original = settings.Settings.data_save_path
    yield
    settings.Settings.data_save_path = original


class TestInit:

    def test_date_matches_today(self):
        dm = DataMananger()
        assert dm.date == datetime.today().strftime('%Y-%m-%d')

    def test_exp_id_includes_version(self):
        dm = DataMananger()
        assert EXP_ID_PATTERN.match(dm.exp_id)

    def test_save_path_starts_as_none(self):
        assert DataMananger().save_path is None


class TestRepr:

    def test_repr_includes_all_fields(self):
        dm = DataMananger()
        string = repr(dm)
        assert f'Date: {dm.date}' in string
        assert f'Exp id: {dm.exp_id}' in string
        assert f'Save path: {dm.save_path}' in string


class TestGenerateExpId:

    def test_new_exp_id_still_matches_pattern(self):
        dm = DataMananger()
        dm.generate_exp_id()
        assert EXP_ID_PATTERN.match(dm.exp_id)

    def test_without_reserve_exp_id_always_regenerates(self):
        # Regression guard: protocols that loop run()+save() on the
        # same instance (e.g. calibration/readout.py's Fidelity/
        # Separation) rely on every generate_exp_id() call producing
        # a fresh id, even after save_path has already been set by a
        # prior iteration's save(). Only an explicit reserve_exp_id()
        # call should ever suppress regeneration. generate_exp_id()
        # has 1-second timestamp resolution, so datetime.now() is
        # mocked to force two distinct timestamps.
        dm = DataMananger()
        dm.create_data_path()
        first = dm.exp_id

        later = datetime.now() + timedelta(seconds=1)
        with patch(
            'qcal.managers.data_manager.datetime'
        ) as mock_datetime:
            mock_datetime.now.return_value = later
            dm.generate_exp_id()

        assert dm.exp_id != first


class TestReserveExpId:

    def test_suppresses_the_next_generate_exp_id_call(self):
        dm = DataMananger()
        dm._exp_id += '_custom_suffix'
        custom = dm.exp_id

        dm.reserve_exp_id()
        dm.generate_exp_id()

        assert dm.exp_id == custom

    def test_is_one_shot_not_sticky(self):
        dm = DataMananger()
        dm.reserve_exp_id()
        dm.generate_exp_id()
        reserved = dm.exp_id

        later = datetime.now() + timedelta(seconds=1)
        with patch(
            'qcal.managers.data_manager.datetime'
        ) as mock_datetime:
            mock_datetime.now.return_value = later
            dm.generate_exp_id()

        assert dm.exp_id != reserved

    def test_save_path_created_before_reserve_is_preserved(self, tmp_path):
        # The scenario this exists for: a caller appends a suffix,
        # creates the save path for it, then reserves it right before
        # QPU.run() (whose _initialize() unconditionally calls
        # generate_exp_id()).
        settings.Settings.data_save_path = str(tmp_path) + '/'
        dm = DataMananger()
        dm._exp_id += '_GST_Q0Q1'
        dm.create_data_path()
        reserved_path = dm.save_path
        dm.reserve_exp_id()

        dm.generate_exp_id()  # simulates QPU._initialize()

        assert dm.save_path == reserved_path
        assert dm.exp_id.endswith('_GST_Q0Q1')


class TestCreateDataPath:

    def test_creates_directory_tree(self, tmp_path):
        settings.Settings.data_save_path = str(tmp_path) + '/'
        dm = DataMananger()

        dm.create_data_path()

        expected = (
            str(tmp_path) + '/' + f'{dm.date}/' + f'{dm.exp_id}/'
        )
        assert dm.save_path == expected
        assert Path(dm.save_path).is_dir()

    def test_mkdir_failure_is_caught_and_logged(self, tmp_path, caplog):
        settings.Settings.data_save_path = str(tmp_path) + '/'
        dm = DataMananger()

        with patch(
            'pathlib.Path.mkdir', side_effect=OSError('permission denied')
        ):
            with caplog.at_level('WARNING'):
                dm.create_data_path()

        assert 'permission denied' in caplog.text


class TestSaveToCsv:

    def test_writes_csv_file_at_save_path(self, tmp_path):
        settings.Settings.data_save_path = str(tmp_path) + '/'
        dm = DataMananger()
        dm.create_data_path()
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        dm.save_to_csv(df, 'results')

        saved = pd.read_csv(dm.save_path + 'results.csv', index_col=0)
        pd.testing.assert_frame_equal(saved, df)


class TestSaveToPickle:

    def test_writes_pickle_file_at_save_path(self, tmp_path):
        settings.Settings.data_save_path = str(tmp_path) + '/'
        dm = DataMananger()
        dm.create_data_path()

        dm.save_to_pickle({'x': 1}, 'results')

        import pickle
        with open(dm.save_path + 'results.pkl', 'rb') as f:
            assert pickle.load(f) == {'x': 1}
