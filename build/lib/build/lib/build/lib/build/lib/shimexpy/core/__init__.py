"""
Core initialization file to import sub-modules.
"""

from shimexpy.core.spatial_harmonics import (
    shi_fft_gpu,
    shi_fft_cpu,
    spatial_harmonics_of_fourier_spectrum,
    FFTResult
)

from shimexpy.core.unwrapping import (
    skimage_unwrap,
    branch_cut_unwrap,
    ls_unwrap,
    quality_guided_unwrap
)

from shimexpy.core.contrast import (
    contrast_retrieval,
    get_harmonics,
    get_contrast,
    get_contrasts,
    get_all_contrasts
)

# Re-export at the core level
__all__ = [
    # Fourier transforms
    "shi_fft_gpu",
    "shi_fft_cpu",
    "spatial_harmonics_of_fourier_spectrum",
    "FFTResult",
    
    # Unwrapping
    "skimage_unwrap",
    "branch_cut_unwrap",
    "ls_unwrap",
    "quality_guided_unwrap",
    
    # Contrast
    "contrast_retrieval",
    "get_harmonics",
    "get_contrast",
    "get_contrasts",
    "get_all_contrasts"
]
