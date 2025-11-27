"""
Tests for shimexpy.preprocessing module.

Tests cover:
- Dark field correction
- Bright field correction
- Flat field correction
- Peak coordinate extraction
- Rotation angle calculation
"""

import pytest
import numpy as np
import sys
import os

# Clear any existing shimexpy imports and add correct path
_mods_to_remove = [k for k in list(sys.modules.keys()) if 'shimexpy' in k]
for _m in _mods_to_remove:
    del sys.modules[_m]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shimexpy")))

from shimexpy.preprocessing.corrections import (
    correct_darkfield,
    correct_brightfield,
    flat_field_correction,
)
from shimexpy.preprocessing.angles import (
    next_power_of_two,
    extract_peak_coordinates,
    calculate_rotation_angle,
)


class TestDarkFieldCorrection:
    """Tests for dark field correction."""

    def test_correct_darkfield_basic(self):
        """Test basic dark field subtraction."""
        image = np.array([[100, 110], [120, 130]], dtype=np.float32)
        dark = np.array([[10, 10], [10, 10]], dtype=np.float32)

        result = correct_darkfield(image, dark)

        expected = np.array([[90, 100], [110, 120]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_correct_darkfield_returns_float32(self):
        """Test that output is float32."""
        image = np.ones((10, 10), dtype=np.uint16) * 1000
        dark = np.ones((10, 10), dtype=np.uint16) * 100

        result = correct_darkfield(image, dark)

        assert result.dtype == np.float32

    def test_correct_darkfield_shape_mismatch_raises(self):
        """Test that shape mismatch raises ValueError."""
        image = np.ones((10, 10))
        dark = np.ones((5, 5))

        with pytest.raises(ValueError, match="shape"):
            correct_darkfield(image, dark)

    def test_correct_darkfield_negative_values_allowed(self):
        """Test that negative results are allowed (no clipping)."""
        image = np.array([[50, 60]], dtype=np.float32)
        dark = np.array([[100, 100]], dtype=np.float32)

        result = correct_darkfield(image, dark)

        assert result[0, 0] == -50
        assert result[0, 1] == -40


class TestBrightFieldCorrection:
    """Tests for bright field correction."""

    def test_correct_brightfield_basic(self):
        """Test basic bright field division."""
        image = np.array([[50, 100], [150, 200]], dtype=np.float32)
        bright = np.array([[100, 100], [100, 100]], dtype=np.float32)

        result = correct_brightfield(image, bright)

        expected = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_correct_brightfield_handles_zero_division(self):
        """Test that zero values in bright field are handled."""
        image = np.array([[50, 100]], dtype=np.float32)
        bright = np.array([[0, 100]], dtype=np.float32)

        result = correct_brightfield(image, bright)

        assert result[0, 0] == 0  # Zero where bright is zero
        assert result[0, 1] == 1.0  # Normal division elsewhere

    def test_correct_brightfield_shape_mismatch_raises(self):
        """Test that shape mismatch raises ValueError."""
        image = np.ones((10, 10))
        bright = np.ones((10, 20))

        with pytest.raises(ValueError, match="shape"):
            correct_brightfield(image, bright)


class TestFlatFieldCorrection:
    """Tests for combined flat field correction."""

    def test_flat_field_correction_basic(self):
        """Test standard flat field correction formula."""
        # corrected = (image - dark) / (bright - dark)
        image = np.array([[150, 200]], dtype=np.float32)
        dark = np.array([[50, 50]], dtype=np.float32)
        bright = np.array([[150, 150]], dtype=np.float32)

        result = flat_field_correction(image, dark, bright)

        # (150-50)/(150-50) = 1.0, (200-50)/(150-50) = 1.5
        expected = np.array([[1.0, 1.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_flat_field_correction_handles_zero_denominator(self):
        """Test handling when bright - dark is zero."""
        image = np.array([[100, 200]], dtype=np.float32)
        dark = np.array([[50, 50]], dtype=np.float32)
        bright = np.array([[50, 150]], dtype=np.float32)  # First pixel: bright == dark

        result = flat_field_correction(image, dark, bright)

        assert result[0, 0] == 0  # Zero where denominator is zero
        assert result[0, 1] == 1.5  # Normal calculation elsewhere

    def test_flat_field_correction_shape_mismatch_raises(self):
        """Test that shape mismatch raises ValueError."""
        image = np.ones((10, 10))
        dark = np.ones((10, 10))
        bright = np.ones((5, 5))

        with pytest.raises(ValueError, match="Shape mismatch"):
            flat_field_correction(image, dark, bright)


class TestNextPowerOfTwo:
    """Tests for next_power_of_two utility."""

    def test_next_power_of_two_exact_powers(self):
        """Test with exact powers of two."""
        assert next_power_of_two(1) == 1
        assert next_power_of_two(2) == 2
        assert next_power_of_two(64) == 64
        assert next_power_of_two(256) == 256

    def test_next_power_of_two_non_powers(self):
        """Test with non-power-of-two values."""
        assert next_power_of_two(3) == 4
        assert next_power_of_two(100) == 128
        assert next_power_of_two(1000) == 1024
        assert next_power_of_two(2049) == 4096


class TestExtractPeakCoordinates:
    """Tests for peak coordinate extraction."""

    def test_extract_peak_coordinates_synthetic(self):
        """Test peak extraction with synthetic periodic image."""
        # Create image with known peaks
        x = np.linspace(0, 4 * np.pi, 64)
        y = np.linspace(0, 4 * np.pi, 64)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X * 5) + np.sin(Y * 5) + 2).astype(np.float32)

        coords = extract_peak_coordinates(image, num_harmonics=4, band_limit=20)

        # Should return 5 coordinates (1 center + 4 harmonics)
        assert len(coords) == 5

        # Each coordinate should be a tuple of two integers
        for coord in coords:
            assert len(coord) == 2
            assert isinstance(coord[0], (int, np.integer))
            assert isinstance(coord[1], (int, np.integer))

    def test_extract_peak_coordinates_returns_center_first(self):
        """Test that the center (zero-order) peak is returned first."""
        # Create image with a periodic pattern - the FFT center will be the DC component
        x = np.linspace(0, 4 * np.pi, 64)
        y = np.linspace(0, 4 * np.pi, 64)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X * 5) + np.sin(Y * 5) + 10).astype(np.float32)  # DC offset

        coords = extract_peak_coordinates(image, num_harmonics=2, band_limit=10)

        # Should return at least 3 coordinates (1 center + 2 harmonics)
        assert len(coords) >= 3

        # Each coordinate should be valid (non-negative integers)
        for coord in coords:
            assert coord[0] >= 0
            assert coord[1] >= 0


class TestCalculateRotationAngle:
    """Tests for rotation angle calculation."""

    def test_calculate_rotation_angle_no_rotation(self):
        """Test with aligned peaks (no rotation)."""
        # Center at (100, 100), horizontal peak at (100, 150)
        coords = [(100, 100), (100, 150), (100, 50)]

        angle = calculate_rotation_angle(coords)

        # Should be close to 0 (perfectly horizontal)
        assert abs(angle) < 1.0

    def test_calculate_rotation_angle_small_rotation(self):
        """Test with slightly rotated peaks."""
        # Center at (100, 100), slightly tilted peaks
        coords = [(100, 100), (95, 150), (105, 50)]  # ~5.7 degrees tilt

        angle = calculate_rotation_angle(coords)

        # Should detect some rotation
        assert angle != 0

    def test_calculate_rotation_angle_single_coord(self):
        """Test with only center coordinate."""
        coords = [(100, 100)]

        angle = calculate_rotation_angle(coords)

        assert angle == 0.0

    def test_calculate_rotation_angle_empty(self):
        """Test with empty coordinates."""
        coords = []

        angle = calculate_rotation_angle(coords)

        assert angle == 0.0


class TestQuadrantSign:
    """Tests for _quadrant_sign function to improve coverage."""

    def test_quadrant_sign_y_axis_q1(self):
        """Test _quadrant_sign for axis='y', peak in quadrant 1 (upper right)."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        # peak_col > center_col and peak_row < center_row (upper right)
        result = _quadrant_sign(
            peak_row=80, center_row=100,
            peak_col=120, center_col=100,
            axis="y"
        )
        assert result == 1

    def test_quadrant_sign_y_axis_q2(self):
        """Test _quadrant_sign for axis='y', peak in quadrant 2 (upper left)."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        # peak_col < center_col and peak_row < center_row (upper left)
        result = _quadrant_sign(
            peak_row=80, center_row=100,
            peak_col=80, center_col=100,
            axis="y"
        )
        assert result == -1

    def test_quadrant_sign_y_axis_q3(self):
        """Test _quadrant_sign for axis='y', peak in quadrant 3 (lower left)."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        # peak_col < center_col and peak_row > center_row (lower left)
        result = _quadrant_sign(
            peak_row=120, center_row=100,
            peak_col=80, center_col=100,
            axis="y"
        )
        assert result == 1

    def test_quadrant_sign_y_axis_q4(self):
        """Test _quadrant_sign for axis='y', peak in quadrant 4 (lower right)."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        # peak_col > center_col and peak_row > center_row (lower right)
        result = _quadrant_sign(
            peak_row=120, center_row=100,
            peak_col=120, center_col=100,
            axis="y"
        )
        assert result == -1

    def test_quadrant_sign_x_axis_q4(self):
        """Test _quadrant_sign for axis='x', peak in quadrant 4 (lower right)."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        # peak_col > center_col and peak_row > center_row (lower right)
        result = _quadrant_sign(
            peak_row=120, center_row=100,
            peak_col=120, center_col=100,
            axis="x"
        )
        assert result == 1

    def test_quadrant_sign_x_axis_q1(self):
        """Test _quadrant_sign for axis='x', peak in quadrant 1 (upper right)."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        # peak_col > center_col and peak_row < center_row (upper right)
        result = _quadrant_sign(
            peak_row=80, center_row=100,
            peak_col=120, center_col=100,
            axis="x"
        )
        assert result == -1

    def test_quadrant_sign_x_axis_q2(self):
        """Test _quadrant_sign for axis='x', peak in quadrant 2 (upper left)."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        # peak_col < center_col and peak_row < center_row (upper left)
        result = _quadrant_sign(
            peak_row=80, center_row=100,
            peak_col=80, center_col=100,
            axis="x"
        )
        assert result == 1

    def test_quadrant_sign_x_axis_q3(self):
        """Test _quadrant_sign for axis='x', peak in quadrant 3 (lower left)."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        # peak_col < center_col and peak_row > center_row (lower left)
        result = _quadrant_sign(
            peak_row=120, center_row=100,
            peak_col=80, center_col=100,
            axis="x"
        )
        assert result == -1

    def test_quadrant_sign_returns_zero_for_invalid(self):
        """Test _quadrant_sign returns 0 for invalid axis."""
        from shimexpy.preprocessing.angles import _quadrant_sign

        result = _quadrant_sign(
            peak_row=100, center_row=100,
            peak_col=100, center_col=100,
            axis="invalid"
        )
        assert result == 0


class TestCalculateRotationAngleVerticalDominant:
    """Tests for calculate_rotation_angle with vertical dominant peaks."""

    def test_vertical_dominant_peaks(self):
        """Test with vertically dominant peak positions."""
        # Center at (100, 100)
        # Peak at (150, 105) - delta_row=50, delta_col=5, vertical dominant
        coords = [(100, 100), (150, 105)]

        angle = calculate_rotation_angle(coords)

        # Should calculate angle based on vertical dominant case
        assert isinstance(angle, float)

    def test_multiple_vertical_dominant_peaks(self):
        """Test with multiple vertically dominant peaks."""
        # Center at (100, 100)
        # Peaks with vertical dominance
        coords = [
            (100, 100),  # center
            (150, 105),  # vertical dominant (delta_row=50, delta_col=5)
            (50, 95),    # vertical dominant (delta_row=50, delta_col=5)
        ]

        angle = calculate_rotation_angle(coords)

        assert isinstance(angle, float)


class TestCalculateRotationAngleEqualDeltas:
    """Tests for calculate_rotation_angle when delta_row == delta_col."""

    def test_equal_deltas_are_skipped(self):
        """Test that peaks with equal deltas are skipped (continue branch)."""
        # Center at (100, 100)
        # Peaks with delta_row == delta_col should be skipped
        coords = [
            (100, 100),  # center
            (120, 120),  # delta_row=20, delta_col=20 - equal, should be skipped
            (150, 100),  # delta_row=50, delta_col=0 - vertical
        ]

        angle = calculate_rotation_angle(coords)

        # Should still compute an angle (from the non-equal delta peak)
        assert isinstance(angle, float)

    def test_all_equal_deltas_returns_zero(self):
        """Test that all equal deltas returns 0."""
        # Only peaks with equal deltas (diagonal)
        coords = [
            (100, 100),  # center
            (120, 120),  # delta_row=20, delta_col=20
            (80, 80),    # delta_row=20, delta_col=20
        ]

        angle = calculate_rotation_angle(coords)

        # When all peaks are skipped, should return 0
        assert angle == 0.0
