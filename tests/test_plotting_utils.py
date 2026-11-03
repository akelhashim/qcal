"""Unit tests for qcal.plotting.utils.calculate_nrows_ncols."""
from qcal.plotting.utils import calculate_nrows_ncols


class TestCalculateNrowsNcols:

    def test_zero_elements_does_not_crash(self):
        # Regression test: ncols = min(0, max_ncols) == 0, and the
        # original implementation divided by ncols unconditionally,
        # raising ZeroDivisionError. This function is called via
        # len(self._qubits) across many .plot() methods, so any class
        # with zero qubits would crash.
        assert calculate_nrows_ncols(0) == (0, 0)

    def test_single_element(self):
        assert calculate_nrows_ncols(1) == (1, 1)

    def test_fits_within_max_ncols(self):
        assert calculate_nrows_ncols(4) == (1, 4)

    def test_exceeds_max_ncols_wraps_to_new_row(self):
        assert calculate_nrows_ncols(5) == (2, 4)

    def test_exact_multiple_of_max_ncols(self):
        assert calculate_nrows_ncols(8) == (2, 4)

    def test_custom_max_ncols(self):
        assert calculate_nrows_ncols(9, max_ncols=3) == (3, 3)
