"""Submodule for data management.

The saving of data is handled by the DataManager class.
"""
import qcal.settings as settings
from qcal.utils import save

import logging
import pathlib

from datetime import datetime
from typing import Any

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
        self._exp_id = datetime.now().strftime('%H%M%S')
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
    
    def generate_exp_id(self) -> None:
        """Generate a new experimental id."""
        self._exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def create_data_save_path(self) -> None:
        """Create a directory for data if none exists."""
        base_dir = settings.Settings.data_save_path
        self._save_path = (
            base_dir + f'{self._date}/' + self._date.replace('-', '') +
            f'{self._exp_id}/' + self._exp_id
        )
        
        path = pathlib.Path(self._save_path)
        path.mkdir(parents=True, exist_ok=True)

    def save(self, data: Any, filename: str) -> None:
        """Save data to the save_path directory.

        Args:
            data (Any): data to save.
            filename (str): filename for the data.
        """
        save(data, self._save_path + '_' + filename)