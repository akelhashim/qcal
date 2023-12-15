"""Submodule for storing basic qcal settings for experiments."""

__all__ = ['Settings']

class _Settings:

    def __init__(self) -> None:
        """Initialize the settings attributes."""
        self._config_path = ''
        self._save_data = True
        self._data_save_path = ''

    @property
    def config_path(self) -> str:
        """Path where the config is located.

        Returns:
            str: config path.
        """
        return self._config_path
    
    @property
    def save_data(self) -> bool:
        """Save data automatically.

        Returns:
            bool: whether to save data or not.
        """
        return self._save_data

    @property
    def data_save_path(self) -> str:
        """Path where data is saved.

        Returns:
            str: data save path.
        """
        return self._data_save_path
    
    @config_path.setter
    def config_path(self, value):
        """Setter for config_path."""
        assert isinstance(value, str), "The passed value must be a string!"
        self._config_path = value
    
    @save_data.setter
    def save_data(self, value):
        """Setter for save_data."""
        assert isinstance(value, bool), "The passed value must be a boolean!"
        self._save_data = value

    @data_save_path.setter
    def data_save_path(self, value):
        """Setter for data_save_path."""
        assert isinstance(value, str), "The passed value must be a string!"
        self._data_save_path = value
    

Settings = _Settings()