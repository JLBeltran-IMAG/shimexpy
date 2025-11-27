"""
Preprocessing utilities for X-ray image correction.

This module provides functions for common preprocessing steps in X-ray imaging:
- Dark field correction
- Bright field (flat field) correction
- Image rotation and cropping
- Angle correction based on harmonic peak detection
"""

from shimexpy.preprocessing.corrections import (
    correct_darkfield,
    correct_brightfield,
    flat_field_correction,
)

from shimexpy.preprocessing.angles import (
    extract_peak_coordinates,
    calculate_rotation_angle,
    next_power_of_two,
)

__all__ = [
    # Corrections
    "correct_darkfield",
    "correct_brightfield",
    "flat_field_correction",
    # Angle detection
    "extract_peak_coordinates",
    "calculate_rotation_angle",
    "next_power_of_two",
]
