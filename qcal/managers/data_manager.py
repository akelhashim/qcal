"""Submodule for data management.

The saving of data is handled by the DataManager class.
"""
import logging
import pathlib
from datetime import datetime
from typing import Any

import pandas as pd

import qcal.settings as settings
from qcal.utils import save_to_csv, save_to_pickle

logger = logging.getLogger(__name__)


class DataMananger:

    __slots__ = (
        '_date',
        '_exp_id',
        '_save_path'
    )

    def __init__(self) -> None:
        """Initialize a DataManager.
        """
        self._date = datetime.today().strftime('%Y-%m-%d')
        self._exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_path = None

    def __repr__(self) -> str:
        string = f'Date: {self._date}\n'
        string += f'Exp id: {self._exp_id}\n'
        string += f'Save path: {self._save_path}'
        return string

    @property
    def date(self) -> str:
        """Today's date.

        Returns:
            str: today's date.
        """
        return self._date

    @property
    def exp_id(self) -> str:
        """Experiment id.

        This is the current state time in hrs, mins, secs.

        Returns:
            str: experiment id.
        """
        return self._exp_id

    @property
    def save_path(self) -> str:
        """Save path.

        Returns:
            str: path where data is saved.
        """
        return self._save_path

    def generate_exp_id(self) -> None:
        """Generate a new experimental id."""
        self._exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        # self._exp_id = datetime.now().strftime('%H%M%S')

    def create_data_path(self) -> None:
        """Create a directory for data if none exists."""
        base_dir = settings.Settings.data_path
        self._save_path = base_dir + f'{self._date}/' + f'{self._exp_id}/'
        try:
            path = pathlib.Path(self._save_path)
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f' {e.message}')

    def save_to_csv(self, data: pd.DataFrame, filename: str) -> None:
        """Save a dataframe to a csv file.

        Args:
            data (pd.DataFrame): data in a pandas DataFrame.
            filename (str): filename for the saved data.
        """
        save_to_csv(data, self._save_path + filename)

    def save_to_pickle(self, data: Any, filename: str) -> None:
        """Save data to a pickle file in the save_path directory.

        Args:
            data (Any): data to save.
            filename (str): filename for the data.
        """
        save_to_pickle(data, self._save_path + filename)
