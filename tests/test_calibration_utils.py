"""Unit tests for qcal.calibration.utils (find_pulse_index, in_range).

find_pulse_index() is exercised here against synthetic pulse lists
written directly into config._parameters (a plain nested dict), rather
than the real pulses in the example config, so that the string-
pre-pulse-reference and all-virtualz edge cases can be constructed
explicitly instead of relying on them happening to occur somewhere in
config.yaml.
"""
import numpy as np

from qcal.calibration.utils import find_pulse_index, in_range


class TestFindPulseIndex:

    def test_finds_first_physical_pulse(self, config):
        # single_qubit/0/GE/X90/pulse in the example config is
        # [virtualz, DRAG, virtualz] -- the first physical pulse is at
        # index 1.
        assert find_pulse_index(config, 'single_qubit/0/GE/X90/pulse') == 1

    def test_single_physical_pulse_is_index_zero(self, config):
        # single_qubit/0/GE/X/pulse in the example config has a single
        # physical (cosine_square) pulse, at index 0.
        assert find_pulse_index(config, 'single_qubit/0/GE/X/pulse') == 0

    def test_skips_leading_string_pulse_references(self, config):
        config._parameters['dummy'] = {'pulse': [
            'some_other_pulse_reference',
            {'env': 'virtualz', 'kwargs': {'phase': 0.1}},
            {'env': 'DRAG', 'kwargs': {'amp': 0.5}},
        ]}
        assert find_pulse_index(config, 'dummy/pulse') == 2

    def test_returns_none_when_every_pulse_is_virtualz(self, config):
        config._parameters['dummy'] = {'pulse': [
            {'env': 'virtualz', 'kwargs': {'phase': 0.1}},
            {'env': 'virtualz', 'kwargs': {'phase': -0.1}},
        ]}
        assert find_pulse_index(config, 'dummy/pulse') is None


class TestInRange:

    def test_number_strictly_inside_range_is_in_range(self):
        assert in_range(0.5, [0.0, 1.0])

    def test_number_at_lower_bound_is_in_range(self):
        assert in_range(0.0, [0.0, 1.0])

    def test_number_at_upper_bound_is_in_range(self):
        assert in_range(1.0, [0.0, 1.0])

    def test_number_outside_range_is_not_in_range(self):
        assert not in_range(1.5, [0.0, 1.0])
        assert not in_range(-0.5, [0.0, 1.0])

    def test_range_order_does_not_matter(self):
        # Only the array's min/max define the range, not element order.
        assert in_range(0.5, [1.0, 0.0])

    def test_uses_array_min_and_max_not_just_endpoints(self):
        # An unsorted array with interior extrema: the range is still
        # defined by min/max, not by the first/last elements.
        assert in_range(0.5, [1.0, -1.0, 0.0])
        assert not in_range(-2.0, [1.0, -1.0, 0.0])

    def test_returns_numpy_bool(self):
        assert isinstance(in_range(0.5, [0.0, 1.0]), np.bool_)
