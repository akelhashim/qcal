"""Unit tests for qcal.settings.Settings."""
import plotly.io as pio
import pytest

import qcal.settings as settings
from qcal.settings import _DEFAULT_PLOT_RENDERER, _Settings


@pytest.fixture(autouse=True)
def restore_plot_renderer():
    """Ensure a test's renderer override doesn't leak into others."""
    original = settings.Settings.plot_renderer
    yield
    settings.Settings.plot_renderer = original


class TestConfigPath:

    def test_default_is_empty_string(self):
        assert isinstance(settings.Settings.config_path, str)

    def test_setter_rejects_non_string(self):
        with pytest.raises(TypeError):
            settings.Settings.config_path = 123


class TestSaveData:

    def test_setter_rejects_non_bool(self):
        with pytest.raises(TypeError):
            settings.Settings.save_data = 'yes'

    def test_setter_round_trip(self):
        original = settings.Settings.save_data
        settings.Settings.save_data = not original
        assert settings.Settings.save_data is not original
        settings.Settings.save_data = original


class TestPlotRenderer:

    def test_fresh_instance_defaults_to_none(self):
        # None means "use plotly's own auto-detected renderer" -- it
        # should not be forced to a notebook-specific value like
        # 'colab' just by importing qcal. Constructing a fresh
        # instance (rather than reading the shared Settings
        # singleton) keeps this independent of what other tests may
        # have set on the singleton.
        assert _Settings().plot_renderer is None

    def test_setting_a_renderer_updates_plotly(self):
        settings.Settings.plot_renderer = 'json'
        assert settings.Settings.plot_renderer == 'json'
        assert pio.renderers.default == 'json'

    def test_resetting_to_none_restores_module_default(self):
        # _DEFAULT_PLOT_RENDERER is captured once, when qcal.settings
        # is first imported -- i.e. plotly's true auto-detected value,
        # from before anything had a chance to override it. Every
        # _Settings instance shares that same fallback target.
        fresh = _Settings()
        fresh.plot_renderer = 'colab'
        assert pio.renderers.default == 'colab'
        fresh.plot_renderer = None
        assert fresh.plot_renderer is None
        assert pio.renderers.default == _DEFAULT_PLOT_RENDERER

    def test_setter_rejects_non_string_non_none(self):
        with pytest.raises(TypeError):
            settings.Settings.plot_renderer = 123
