"""Submodule for storing basic qcal settings for experiments."""
import plotly.io as pio

__all__ = ['Settings']

_DEFAULT_PLOT_RENDERER = pio.renderers.default

class _Settings:

    def __init__(self) -> None:
        """Initialize the settings attributes."""
        self._config_path = ''
        self._save_data = False
        self._data_save_path = ''
        # Plotly auto-detects the right renderer for the current
        # environment (Jupyter, JupyterLab, VS Code, Colab, or a plain
        # script/CLI, where it falls back to opening a new browser
        # tab) as soon as it's imported. Remember that auto-detected
        # value so `plot_renderer = None` can restore it later.
        self._plot_renderer = None

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

    @property
    def plot_renderer(self) -> str | None:
        """Plotly renderer used to display figures.

        Returns:
            str | None: renderer name (e.g. 'browser', 'notebook',
                'vscode', 'colab', 'iframe', 'png'), or ``None`` to use
                plotly's own auto-detected default for the current
                environment. Defaults to ``None``.
        """
        return self._plot_renderer

    @config_path.setter
    def config_path(self, value: str):
        """Setter for config_path."""
        if not isinstance(value, str):
            raise TypeError("The passed value must be a string!")
        self._config_path = value

    @save_data.setter
    def save_data(self, value: bool):
        """Setter for save_data."""
        if not isinstance(value, bool):
            raise TypeError("The passed value must be a boolean!")
        self._save_data = value

    @data_save_path.setter
    def data_save_path(self, value: str):
        """Setter for data_save_path."""
        if not isinstance(value, str):
            raise TypeError("The passed value must be a string!")
        self._data_save_path = value

    @plot_renderer.setter
    def plot_renderer(self, value: str | None) -> None:
        """Setter for plot_renderer.

        Args:
            value (str | None): a valid plotly renderer name (see
                ``plotly.io.renderers``), or ``None`` to restore
                plotly's auto-detected default.
        """
        if value is not None and not isinstance(value, str):
            raise TypeError(
                "The passed value must be a string or None!"
            )
        self._plot_renderer = value
        pio.renderers.default = (
            value if value is not None else _DEFAULT_PLOT_RENDERER
        )


Settings = _Settings()
