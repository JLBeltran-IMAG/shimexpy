"""
ShimExPy: Spatial Harmonics Imaging for X-ray Physics in Python
===============================================================

This package provides tools for spatial harmonics X-ray imaging analysis.
"""

# Import main functionality
from shimexpy.core.spatial_harmonics import (
    shi_fft_gpu,
    shi_fft_cpu,
    shi_fft,
    spatial_harmonics_of_fourier_spectrum
)

from shimexpy.core.contrast import (
    contrast_retrieval,
    get_harmonics,
    get_contrast,
    get_contrasts,
    get_all_contrasts,
    get_all_harmonic_contrasts
)

from shimexpy.core.unwrapping import (
    skimage_unwrap,
    ls_unwrap
)

from shimexpy.io.file_io import (
    load_image,
    save_image,
    save_block_grid,
    load_block_grid,
    save_results,
    load_results,
    cli_export
)

from shimexpy.utils.parallelization import apply_harmonic_chunking

from shimexpy.preprocessing import (
    correct_darkfield,
    correct_brightfield,
    flat_field_correction,
    extract_peak_coordinates,
    calculate_rotation_angle,
)

__all__ = [
    # Spatial Harmonics
    "shi_fft_gpu",
    "shi_fft_cpu",
    "shi_fft",
    "spatial_harmonics_of_fourier_spectrum",
    "contrast_retrieval",
    "get_harmonics",
    "get_contrast",
    "get_contrasts",
    "get_all_contrasts",
    "get_all_harmonic_contrasts",

    # Unwrapping Phase
    "skimage_unwrap",
    "ls_unwrap",

    # Preprocessing
    "correct_darkfield",
    "correct_brightfield",
    "flat_field_correction",
    "extract_peak_coordinates",
    "calculate_rotation_angle",

    # File I/O
    "load_image",
    "save_image",
    "save_block_grid",
    "load_block_grid",
    "save_results",
    "load_results",
    "cli_export",

    # Utilities
    "apply_harmonic_chunking"
]

# Version information
__version__ = '0.1.0'


