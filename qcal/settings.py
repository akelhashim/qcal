"""Submodule for storing basic qcal settings for experiments."""

__all__ = ['Settings']

class _Settings:

    def __init__(self) -> None:
        """Initialize the settings attributes."""
        self._config_directory = ''
        self._save_data = True
        self._data_save_directory = ''

    @property
    def config_directory(self) -> str:
        """Directory where the config is located.

        Returns:
            str: config directory.
        """
        return self._config_directory
    
    @property
    def save_data(self) -> bool:
        """Save data automatically.

        Returns:
            bool: whether to save data or not.
        """
        return self._save_data

    @property
    def data_save_directory(self) -> str:
        """Directory where data is saved.

        Returns:
            str: data save directory.
        """
        return self._data_save_directory
    
    @config_directory.setter
    def config_directory(self, value):
        """Setter for config_directory."""
        assert isinstance(value, str), "The passed value must be a string!"
        self._config_directory = value
    
    @save_data.setter
    def save_data(self, value):
        """Setter for save_data."""
        assert isinstance(value, bool), "The passed value must be a boolean!"
        self._save_data = value

    @data_save_directory.setter
    def data_save_directory(self, value):
        """Setter for data_save_directory."""
        assert isinstance(value, str), "The passed value must be a string!"
        self._data_save_directory = value
    

Settings = _Settings()