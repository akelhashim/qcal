"""Unit tests for qcal.machine_learning.utils.find_mapping."""
import numpy as np
import pytest

from qcal.machine_learning.utils import find_mapping


class TestFindMapping:

    def test_maps_mismatched_elements(self):
        mapping = find_mapping([0, 1, 2], [1, 1, 0])
        assert mapping == {0: 1, 2: 0}

    def test_identical_arrays_give_empty_mapping(self):
        assert find_mapping([0, 1, 2], [0, 1, 2]) == {}

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match='same length'):
            find_mapping([0, 1], [0, 1, 2])

    def test_works_with_numpy_arrays(self):
        mapping = find_mapping(np.array([0, 1]), np.array([1, 0]))
        assert mapping == {0: 1, 1: 0}

    def test_duplicate_source_value_keeps_last_target(self):
        # Both array1 entries are 0, so the second (num1, num2) pair
        # overwrites the first in the output dict.
        mapping = find_mapping([0, 0], [1, 2])
        assert mapping == {0: 2}
