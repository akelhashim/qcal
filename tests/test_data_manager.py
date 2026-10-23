"""Unit tests for qcal.managers.data_manager.DataMananger."""
import re
from datetime import datetime
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
