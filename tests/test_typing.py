"""Tests for the NUMERIC protocol and related type aliases."""

import numpy as np
import pytest

from conjugate._typing import NUMERIC, Natural, PositiveReal, Probability, Real


class TestNUMERICProtocol:
    """Test that the NUMERIC protocol works with various numeric types."""

    def test_int_conforms_to_numeric(self):
        """Test that int conforms to NUMERIC protocol."""
        assert isinstance(5, NUMERIC)
        assert isinstance(-5, NUMERIC)
        assert isinstance(0, NUMERIC)

    def test_float_conforms_to_numeric(self):
        """Test that float conforms to NUMERIC protocol."""
        assert isinstance(5.0, NUMERIC)
        assert isinstance(-5.0, NUMERIC)
        assert isinstance(0.0, NUMERIC)
        assert isinstance(3.14159, NUMERIC)

    def test_complex_conforms_to_numeric(self):
        """Test that complex does NOT fully conform to NUMERIC protocol.

        Complex numbers don't support floor division and modulo operations,
        so they don't conform to the full NUMERIC protocol.
        """
        # Complex doesn't support //, %, so it doesn't conform
        assert not isinstance(5 + 3j, NUMERIC)
        assert not isinstance(1j, NUMERIC)

    def test_numpy_scalar_conforms_to_numeric(self):
        """Test that numpy scalars conform to NUMERIC protocol."""
        assert isinstance(np.int32(5), NUMERIC)
        assert isinstance(np.int64(5), NUMERIC)
        assert isinstance(np.float32(5.0), NUMERIC)
        assert isinstance(np.float64(5.0), NUMERIC)

    def test_numpy_array_conforms_to_numeric(self):
        """Test that numpy arrays conform to NUMERIC protocol."""
        assert isinstance(np.array([1, 2, 3]), NUMERIC)
        assert isinstance(np.array([[1, 2], [3, 4]]), NUMERIC)
        assert isinstance(np.array([1.0, 2.0, 3.0]), NUMERIC)

    def test_string_does_not_conform_to_numeric(self):
        """Test that strings do not conform to NUMERIC protocol."""
        assert not isinstance("hello", NUMERIC)
        assert not isinstance("123", NUMERIC)

    def test_list_does_not_conform_to_numeric(self):
        """Test that lists do not conform to NUMERIC protocol."""
        assert not isinstance([1, 2, 3], NUMERIC)

    def test_arithmetic_operations(self):
        """Test that arithmetic operations work with NUMERIC types."""

        def perform_arithmetic(x: NUMERIC) -> NUMERIC:
            result = x + 1
            result = result - 1
            result = result * 2
            result = result / 2
            result = result // 2
            result = result % 2
            result = result**2
            return result

        # Test with int (note: / operator returns float)
        result = perform_arithmetic(5)
        # After division, result becomes float
        assert isinstance(result, (int, float))

        # Test with float
        result = perform_arithmetic(5.0)
        assert isinstance(result, float)

        # Test with numpy array
        result = perform_arithmetic(np.array([1, 2, 3]))
        assert isinstance(result, np.ndarray)

    def test_reverse_arithmetic_operations(self):
        """Test that reverse arithmetic operations work with NUMERIC types."""

        def perform_reverse_arithmetic(x: NUMERIC) -> NUMERIC:
            result = 1 + x
            result = 1 - x
            result = 2 * x
            result = 2 / x
            result = 2 // x
            result = 2 % x
            result = 2**x
            return result

        # Test with int
        result = perform_reverse_arithmetic(2)
        assert isinstance(result, int)

        # Test with float
        result = perform_reverse_arithmetic(2.0)
        assert isinstance(result, float)

    def test_unary_operations(self):
        """Test that unary operations work with NUMERIC types."""

        def perform_unary(x: NUMERIC) -> tuple:
            return -x, +x, abs(x)

        # Test with int
        neg, pos, abs_val = perform_unary(5)
        assert neg == -5
        assert pos == 5
        assert abs_val == 5

        # Test with float
        neg, pos, abs_val = perform_unary(-5.0)
        assert neg == 5.0
        assert pos == -5.0
        assert abs_val == 5.0

    def test_comparison_operations(self):
        """Test that comparison operations work with NUMERIC types."""

        def perform_comparisons(x: NUMERIC, y: NUMERIC) -> tuple:
            return x < y, x <= y, x > y, x >= y

        # Test with int
        lt, le, gt, ge = perform_comparisons(5, 10)
        assert lt is True
        assert le is True
        assert gt is False
        assert ge is False

        # Test with float
        lt, le, gt, ge = perform_comparisons(5.0, 5.0)
        assert lt is False
        assert le is True
        assert gt is False
        assert ge is True

    def test_mixed_type_arithmetic(self):
        """Test that mixed type arithmetic works (int + float)."""

        def add_numbers(x: NUMERIC, y: NUMERIC) -> NUMERIC:
            return x + y

        # int + float should work
        result = add_numbers(5, 3.0)
        assert result == 8.0
        assert isinstance(result, float)

        # float + int should work
        result = add_numbers(5.0, 3)
        assert result == 8.0
        assert isinstance(result, float)

    def test_numpy_broadcasting(self):
        """Test that numpy broadcasting works with NUMERIC."""

        def scale_array(x: NUMERIC, scalar: NUMERIC) -> NUMERIC:
            return x * scalar

        arr = np.array([1, 2, 3])
        result = scale_array(arr, 2)
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))

        result = scale_array(arr, 2.5)
        np.testing.assert_array_equal(result, np.array([2.5, 5.0, 7.5]))


class TestTypeAliases:
    """Test the type aliases built on NUMERIC."""

    def test_real_is_numeric_alias(self):
        """Test that Real is an alias for NUMERIC with annotation."""
        # Real is just an Annotated[NUMERIC, "Real"]
        # It should work the same as NUMERIC
        x: Real = 5
        assert x == 5

        y: Real = 5.0
        assert y == 5.0

    def test_natural_is_numeric_alias(self):
        """Test that Natural is an alias for NUMERIC with annotation."""
        x: Natural = 5
        assert x == 5

    def test_positive_real_is_numeric_alias(self):
        """Test that PositiveReal is an alias for NUMERIC with annotations."""
        x: PositiveReal = 5.0
        assert x == 5.0

    def test_probability_is_numeric_alias(self):
        """Test that Probability is an alias for NUMERIC with annotation."""
        x: Probability = 0.5
        assert x == 0.5
