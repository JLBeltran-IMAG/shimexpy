"""
ShimExPy: Spatial Harmonics Imaging for X-ray Physics in Python
===============================================================

This package provides tools for spatial harmonics X-ray imaging analysis.
"""

# Import main functionality
from shimexpy.core.spatial_harmonics import (
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
    load_results
)

from shimexpy.utils.ffc import ffc, FFCQualityAssessment
from shimexpy.utils.crop import cropImage as crop
from shimexpy.utils.parallelization import apply_harmonic_chunking


__all__ = [
    # Spatial Harmonics
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

    # File I/O
    "load_image",
    "save_image",
    "save_block_grid",
    "load_block_grid",
    "save_results",
    "load_results",

    # Utilities
    "ffc",
    "FFCQualityAssessment",
    "crop",
    "apply_harmonic_chunking"
]

# Version information
__version__ = '0.1.0'


