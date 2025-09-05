"""
ShimExPy: Spatial Harmonics Imaging for X-ray Physics in Python
===============================================================

This package provides tools for spatial harmonics X-ray imaging analysis.
"""

# Import main functionality
from shimexpy.core.spatial_harmonics import (
    shi_fft_gpu,
    shi_fft_cpu,
    spatial_harmonics_of_fourier_spectrum
)

from shimexpy.core.contrast import (
    contrast_retrieval,
    get_harmonics,
    get_contrast,
    get_contrasts,
    get_all_contrasts
)

from shimexpy.core.unwrapping import (
    skimage_unwrap,
    branch_cut_unwrap,
    ls_unwrap,
    quality_guided_unwrap
)

from shimexpy.utils.crop import set_crop


__all__ = [
    # Spatial Harmonics
    "shi_fft_gpu",
    "shi_fft_cpu",
    "spatial_harmonics_of_fourier_spectrum",
    "contrast_retrieval",
    "set_crop",
    "get_harmonics",
    "get_contrast",
    "get_contrasts",
    "get_all_contrasts",

    # Unwrapping Phase
    "skimage_unwrap",
    "branch_cut_unwrap",
    "ls_unwrap",
    "quality_guided_unwrap"
]

# Version information
__version__ = '0.1.0'


