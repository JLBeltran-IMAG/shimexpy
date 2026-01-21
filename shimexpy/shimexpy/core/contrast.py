"""
Contrast computation module.

This module contains functions for computing various types of contrast.
"""

import numpy as np
import xarray as xr
from dask.base import compute

from shimexpy.core import unwrapping as uphase
from shimexpy.core.spatial_harmonics import (
    shi_fft,
    spatial_harmonics_of_fourier_spectrum
)

from shimexpy.utils.parallelization import apply_harmonic_chunking


# Contrast retrieval specifications
# These are the types of contrast that can be retrieved for "get_contrast(...)" function
CONTRASTS = {
    "horizontal": [
        "harmonic_horizontal_positive",
        "harmonic_horizontal_negative",
        "harmonic_diagonal_p1_p1",
        "harmonic_diagonal_n1_n1",
        "harmonic_diagonal_p1_n1",
        "harmonic_diagonal_n1_p1"
    ],
    "vertical": [
        "harmonic_vertical_positive",
        "harmonic_vertical_negative",
        "harmonic_diagonal_p1_p1",
        "harmonic_diagonal_n1_n1",
        "harmonic_diagonal_p1_n1",
        "harmonic_diagonal_n1_p1"
    ],
    "bidirectional": [
        "harmonic_horizontal_positive",
        "harmonic_horizontal_negative",
        "harmonic_vertical_positive",
        "harmonic_vertical_negative",
        "harmonic_diagonal_p1_p1",
        "harmonic_diagonal_n1_n1",
        "harmonic_diagonal_p1_n1",
        "harmonic_diagonal_n1_p1"
    ]
}

CONTRASTS_CLI = {
    "horizontal": [
        "harmonic_horizontal_positive",
        "harmonic_horizontal_negative"
    ],
    "vertical": [
        "harmonic_vertical_positive",
        "harmonic_vertical_negative"
    ],
    "bidirectional": [
        "harmonic_horizontal_positive",
        "harmonic_horizontal_negative",
        "harmonic_vertical_positive",
        "harmonic_vertical_negative"
    ]
}


# -------------------- contrast retrieval
def _compute_phase_map(
    ifft_harmonics: xr.DataArray,
    main_harmonic: xr.DataArray,
    unwrap: str | None = None,
    eps: float = 1e-12
):
    """
    Computes the unwrapped phase map from the inverse Fourier transform
    and the main harmonic.

    Parameters:
    -----------
    ifft_harmonics : np.ndarray
        Array containing the inverse Fourier transform of the data.
    main_harmonic : np.ndarray
        Array containing the main harmonic in the Fourier domain.
    unwrap : str, optional
        The unwrapping algorithm to use. Default is None (uses skimage_unwrap).
    eps : float, optional
        Small value added to the denominator to avoid
        division by zero (default is 1e-12).

    Returns:
    --------
    unwrapped_phase_map : np.ndarray
        The unwrapped phase map.
    """
    # Compute the ratio, avoiding division by zero by adding a small eps
    ratio = ifft_harmonics / (main_harmonic + eps)

    # Unwrap the phase using the skimage algorithm
    if unwrap is None:
        unwrapped_phase_map = xr.apply_ufunc(
            uphase.skimage_unwrap,
            ratio,
            input_core_dims = [["y", "x"]],
            output_core_dims = [["y", "x"]],
            dask="parallelized",
            output_dtypes=[ratio.dtype]
        )

    elif unwrap == "least_squares":
        unwrapped_phase_map = xr.apply_ufunc(
            uphase.ls_unwrap,
            ratio,
            input_core_dims = [["y", "x"]],
            output_core_dims = [["y", "x"]],
            dask="parallelized",
            output_dtypes=[ratio.dtype]
        )

    else:
        raise ValueError("Unknown phase unwrapping algorithm")

    return unwrapped_phase_map


def _compute_scattering(ifft_harmonics, main_harmonic, eps=1e-12):
    """
    Computes the scattering value from the inverse Fourier transform
    and the main harmonic.

    Parameters:
    -----------
    ifft_harmonics : np.ndarray
        Array containing the inverse Fourier transform of the data.
    main_harmonic : np.ndarray
        Array containing the main harmonic in the Fourier domain.
    eps : float, optional
        Small value added to the denominator to avoid
        division by zero (default is 1e-12).

    Returns:
    --------
    scattering_value : np.ndarray
        The computed scattering value.
    """
    # Compute the ratio and avoid division by zero by adding eps
    ratio = ifft_harmonics / (main_harmonic + eps)

    # Get the absolute value of the ratio
    abs_ratio = np.abs(ratio)

    # Clip the absolute ratio to avoid taking the logarithm of values too close to zero
    abs_ratio = abs_ratio.clip(min=eps)

    # Compute the scattering as the natural logarithm of (1 / abs_ratio)
    scattering_value = np.log(1 / abs_ratio)

    return scattering_value


def contrast_retrieval(
    harmonics: xr.DataArray,
    type_of_contrast: str,
    unwrap: str | None = None,
    eps: float = 1e-12
) -> xr.DataArray:
    """
    Retrieves individual contrast members from a harmonic component.

    This function processes harmonic components to retrieve different types of contrast
    (absorption, scattering, or phase map) from the inverse Fourier transform of the input.

    Parameters
    ----------
    harmonic : ndarray
        The harmonic component in Fourier space to be processed.
    type_of_contrast : str
        The type of contrast to retrieve. Must be one of:
        - 'absorption': Computes absorption contrast
        - 'scattering': Computes scattering contrast
        - 'phasemap': Computes phase map contrast
    unwrap : str, optional
        The unwrapping algorithm to use when type_of_contrast is 'phasemap'.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-12.

    Returns
    -------
    ndarray
        The retrieved contrast map according to the specified type_of_contrast.

    Raises
    ------
    ValueError
        If type_of_contrast is not one of 'absorption', 'scattering', or 'phasemap'.
    """
    # Compute the inverse Fourier transform of harmonics.
    ifft_harmonic = xr.apply_ufunc(
        np.fft.ifft2,
        harmonics,
        input_core_dims = [["ky", "kx"]],
        output_core_dims = [["y", "x"]],
        dask = "parallelized",
        output_dtypes = [harmonics.dtype],
        dask_gufunc_kwargs={
            "output_sizes": {
                "y": harmonics.sizes["ky"],
                "x": harmonics.sizes["kx"]
            }
        }
    )

    main_harmonic = ifft_harmonic.sel(harmonic="harmonic_00")
    ifft_harmonic = ifft_harmonic.drop_sel(harmonic="harmonic_00")

    if type_of_contrast == "absorption":
        # Avoid division by zero by adding a small constant to the magnitude
        abs_ifft = np.abs(main_harmonic) + eps
        return np.log(1 / abs_ifft)

    elif type_of_contrast == "scattering":
        return _compute_scattering(ifft_harmonic, main_harmonic)

    elif type_of_contrast == "phasemap":
        return _compute_phase_map(ifft_harmonic, main_harmonic, unwrap)

    else:
        # Raise an error if the provided type_of_contrast is not recognized.
        raise ValueError(f"Unknown type_of_contrast: {type_of_contrast}")


# --------------- Dark-field estimation
def _harmonic_direction_weights(
    block_grid: dict, labels: list, direction: str
) -> xr.DataArray:
    """
    Calculates weights based on the spatial direction of harmonic blocks relative to a reference block.

    This function computes the directional weight for each harmonic block defined in `labels`,
    relative to the 'harmonic_00' reference block found in `block_grid`. It calculates
    the centroids of the blocks and determines weights based on the specified `direction`.

    Parameters
    ----------
    block_grid : dict
        A dictionary where keys are harmonic labels (str) and values are tuples or lists
        containing boundary coordinates in the order (top, bottom, left, right).
        Must contain the key "harmonic_00" to serve as the reference origin.
    labels : list
        A list of strings representing the harmonic labels for which weights should be calculated.
    direction : str
        The direction to emphasize. Options are:
        - "horizontal": Weights are based on the horizontal distance ratio (wx / hypotenuse).
        - "vertical": Weights are based on the vertical distance ratio (wy / hypotenuse).
        - "bidirectional": All weights are set to 1.0.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the calculated weights.
        - Dims: ["harmonic"]
        - Coords: {"harmonic": labels}
        - Dtype: np.float32

    Raises
    ------
    ValueError
        If the provided `direction` is not one of "horizontal", "vertical", or "bidirectional".
    """
    top0, bottom0, left0, right0 = block_grid["harmonic_00"]
    cy0 = 0.5 * (top0 + bottom0)
    cx0 = 0.5 * (left0 + right0)

    w = []

    for label in labels:
        top, bottom, left, right = block_grid[label]
        cy = 0.5 * (top + bottom)
        cx = 0.5 * (left + right)

        wx = abs(cx - cx0)
        wy = abs(cy - cy0)
        hyp = np.hypot(wx, wy)

        if direction == "horizontal": w.append(wx / hyp)
        elif direction == "vertical": w.append(wy / hyp)
        elif direction == "bidirectional": w.append(1.0)
        else: raise ValueError(f"Unknown direction: {direction}")

    weights = xr.DataArray(
        np.asarray(w, dtype=np.float32),
        dims=["harmonic"],
        coords={"harmonic": labels}
    )

    return weights


# -------------------- main functions
def get_harmonics(image, projected_grid, block_grid=None, unwrap = None):
    """
    Set reference image for spatial harmonics analysis.
    This function performs spatial harmonics analysis on a given image
    and returns the absorption, scattering, and differential phase maps,
    along with the block grid for harmonics.

    Parameters
    ----------
    image : np.ndarray
        The input image to analyze.
    projected_grid : float
        The projected grid period (in real-space units) used to compute
        the spatial frequency axes.
    block_grid : dict, optional
        A dictionary containing the limits for each harmonic in the reference image.
    unwrap : str, optional
        The unwrapping algorithm to use for phase map retrieval. Default is None.

    Returns
    -------
    absorption : xr.DataArray
        The computed absorption contrast for the input image.
    scattering : xr.DataArray
        The computed scattering contrast for the input image.
    diff_phase : xr.DataArray
        The computed differential phase map for the input image.
    ref_block_grid : dict
        A dictionary containing the limits for each harmonic in the reference image.
    """
    # La imagen ya debe venir recortada si es necesario

    # Perform the Fourier transform and extract harmonics
    result = shi_fft(image, projected_grid)
    kx, ky, fft_img = result.kx, result.ky, result.fft

    # Extract spatial harmonics from the Fourier spectrum
    if block_grid:
        harmonics, _ = spatial_harmonics_of_fourier_spectrum(fft_img, ky, kx, reference=False, reference_block_grid=block_grid)
    else:
        harmonics, block_grid = spatial_harmonics_of_fourier_spectrum(fft_img, ky, kx, reference=True)

    # Chunk the harmonics for parallel processing
    harmonics_chunked = harmonics.chunk({"harmonic": 1, "ky": -1, "kx": -1})

    # Compute the contrasts from the harmonics
    absolute_absorption = contrast_retrieval(harmonics_chunked, type_of_contrast="absorption")
    absolute_scattering = contrast_retrieval(harmonics_chunked, type_of_contrast="scattering")
    absolute_diff_phase = contrast_retrieval(harmonics_chunked, type_of_contrast="phasemap", unwrap=unwrap)

    return absolute_absorption, absolute_scattering, absolute_diff_phase, block_grid


def get_contrast(
    sample_img, reference, ref_block_grid, type_of_contrast, unwrap = None
):
    """
    Execute spatial harmonics analysis on a sample image and retrieve
    the specified contrast.
    This function performs spatial harmonics analysis on a sample image
    and computes the contrast with respect to a reference image.

    Parameters
    ----------
    sample_img : np.ndarray
        The sample image to analyze.
    reference : xr.DataArray
        The reference image containing the pre-computed contrasts
        (absorption, scattering, phase map).
    ref_block_grid : dict
        A dictionary containing the limits for each harmonic in the reference image.
    type_of_contrast : str
        The type of contrast to retrieve. Must be one of:
        - 'absorption': Computes absorption contrast
        - '(direction)_scattering': Computes (direction) scattering contrast
        - '(direction)_phasemap': Computes (direction) phase map contrast
    unwrap : str, optional
        The unwrapping algorithm to use for phase map retrieval. Default is None.
    
    Returns
    -------
    xr.DataArray
        The computed contrast for the sample image, relative to the reference image.
    """
    sample_result = shi_fft(sample_img)
    sample_harmonics, _ = spatial_harmonics_of_fourier_spectrum(
        sample_result.fft, None, None, reference=False, reference_block_grid=ref_block_grid
    )
    sample_harmonics_chunked = sample_harmonics.chunk({"harmonic": 1, "ky": -1, "kx": -1})

    if type_of_contrast == "absorption":
        sample_contrast = contrast_retrieval(sample_harmonics_chunked, "absorption")
        output = sample_contrast - reference

    elif "_scattering" in type_of_contrast or "_phasemap" in type_of_contrast:
        direction, contrasts = type_of_contrast.split('_')
        sample_contrast = contrast_retrieval(sample_harmonics_chunked, contrasts, unwrap=unwrap)

        harmonics = CONTRASTS[direction]
        weights = _harmonic_direction_weights(ref_block_grid, harmonics, direction)
        result = sample_contrast.sel(harmonic=harmonics) - reference.sel(harmonic=harmonics)

        if contrasts == "scattering":
            output = abs(result * weights).sum("harmonic")
        else:
            positive, negative = harmonics[0], harmonics[1]
            output = result.sel(harmonic=positive) - result.sel(harmonic=negative)

            if direction == "bidirectional":
                positive2, negative2 = harmonics[2], harmonics[3]
                output = output + result.sel(harmonic=positive2) - result.sel(harmonic=negative2)

    else:
        raise ValueError(f"Unknown type_of_contrast: {type_of_contrast}")

    contrast = output.compute()
    return contrast


def get_contrasts(sample_img, reference, ref_block_grid, unwrap = None):
    """
    Compute all contrast types for a sample image against reference images.
    
    Parameters
    ----------
    sample_img : np.ndarray
        The sample image to analyze.
    reference : tuple
        A tuple containing (reference_absorption, reference_scattering, 
        reference_diff_phase).
    ref_block_grid : dict
        A dictionary containing the limits for each harmonic in the reference image.
    unwrap : str, optional
        The unwrapping algorithm to use for phase map retrieval. Default is None.
        
    Returns
    -------
    tuple
        A tuple containing (absorption_contrast, scattering_contrast, 
        diff_phase_contrast).
    """
    # Sample
    # Contrast retrieval of sample image
    sample_result = shi_fft(sample_img)
    sample_fft_img = sample_result.fft
    sample_harmonics, _ = spatial_harmonics_of_fourier_spectrum(
        sample_fft_img,
        None,
        None,
        reference=False,
        reference_block_grid=ref_block_grid
    )
    sample_harmonics_chunked = sample_harmonics.chunk(
        {"harmonic": 1, "ky": -1, "kx": -1}
    )

    # Contrast retrieval reference and sample images
    sample_absorption = contrast_retrieval(
        sample_harmonics_chunked, "absorption"
    )
    sample_scattering = contrast_retrieval(
        sample_harmonics_chunked, "scattering"
    )
    sample_diff_phase = contrast_retrieval(
        sample_harmonics_chunked, "phasemap", unwrap=unwrap
    )

    harmonics = CONTRASTS["bidirectional"]
    weights = _harmonic_direction_weights(ref_block_grid, harmonics, "bidirectional")
    # result = sample_contrast.sel(harmonic=harmonics) - reference.sel(harmonic=harmonics)

    absorption = sample_absorption - reference[0]
    scattering = (sample_scattering.sel(harmonic=harmonics) - 
                  reference[1].sel(harmonic=harmonics))
    diff_phase = (sample_diff_phase.sel(harmonic=harmonics) - 
                  reference[2].sel(harmonic=harmonics))

    scattering = abs(scattering * weights).sum("harmonic")
    diff_phase = (
        diff_phase.sel(harmonic="harmonic_horizontal_positive")
        - diff_phase.sel(harmonic="harmonic_horizontal_negative")
        + diff_phase.sel(harmonic="harmonic_vertical_positive")
        - diff_phase.sel(harmonic="harmonic_vertical_negative")
    )

    # ---------
    #  Compute
    # ---------
    (
        absorption_contrast,
        scattering_contrast,
        diff_phase_contrast
    ) = compute(absorption, scattering, diff_phase)

    return absorption_contrast, scattering_contrast, diff_phase_contrast


def get_all_contrasts(sample_img, reference_img, projected_grid, unwrap = None):
    """
    Compute all contrast types between a sample and reference image.
    
    This is a convenience function that performs all the steps needed to compute
    absorption, scattering, and differential phase contrasts between two images.
    
    Parameters
    ----------
    sample_img : np.ndarray
        The sample image to analyze.
    reference_img : np.ndarray
        The reference image to compare against.
    projected_grid : float
        The projected grid period (in real-space units) used to compute the spatial 
        frequency axes.
    unwrap : str, optional
        The unwrapping algorithm to use for phase map retrieval. Default is None.
        
    Returns
    -------
    tuple
        A tuple containing (absorption_contrast, scattering_contrast, 
        diff_phase_contrast).
    """
    # Reference
    # Contrast retrieval of reference image
    (
        ref_absorption,
        ref_scattering,
        ref_diff_phase,
        ref_block_grid
    ) = get_harmonics(reference_img, projected_grid, unwrap=unwrap)

    (
        sample_absorption,
        sample_scattering,
        sample_diff_phase,
        _
    ) = get_harmonics(sample_img, projected_grid, ref_block_grid, unwrap=unwrap)

    # ---------
    #  Compute
    # ---------
    absorption = sample_absorption - ref_absorption
    scattering = sample_scattering - ref_scattering
    diff_phase = sample_diff_phase - ref_diff_phase

    (
        absorption_contrast,
        scattering_contrast,
        diff_phase_contrast
    ) = compute(absorption, scattering, diff_phase)

    return abs(absorption_contrast), abs(scattering_contrast), diff_phase_contrast


def get_all_harmonic_contrasts(
    sample_img, reference, ref_block_grid, unwrap=None
):
    """
    Compute harmonic-level contrasts for a sample image against reference harmonics.
    Unlike get_contrasts(), this function returns the contrasts for each individual 
    harmonic, not the combined absorption/scattering/phase contrasts.

    Parameters
    ----------
    sample_img : np.ndarray
        The sample image to analyze.
    reference : tuple
        A tuple containing (reference_absorption, reference_scattering, 
        reference_diff_phase).
    ref_block_grid : dict
        The harmonic block grid from the reference image.
    unwrap : str, optional
        Phase unwrapping algorithm. Default is None.
    crop : tuple, optional
        Crop region for the sample image. Default is None.

    Returns
    -------
    xr.Dataset
        A dataset containing harmonic-wise contrasts:
        {harmonic_name: xr.DataArray}
    """
    # --- Step 1: FFT and harmonic extraction
    sample_fft_img = shi_fft(sample_img).fft
    sample_harmonics, _ = spatial_harmonics_of_fourier_spectrum(
        sample_fft_img,
        None,
        None,
        reference=False,
        reference_block_grid=ref_block_grid,
    )

    # Apply Dask chunking (lazy operation, no compute)
    sample_harmonics = apply_harmonic_chunking(sample_harmonics)

    # --- Step 2: Compute contrasts lazily (no .compute() yet)
    sample_absorption = contrast_retrieval(sample_harmonics, "absorption")
    sample_scattering = contrast_retrieval(sample_harmonics, "scattering")
    sample_diff_phase = contrast_retrieval(sample_harmonics, "phasemap", unwrap=unwrap)

    # --- Step 3: Calculate deltas lazily
    delta_abs = (sample_absorption - reference[0]).reset_coords("harmonic", drop=True)
    delta_scat = sample_scattering - reference[1]
    delta_phase = sample_diff_phase - reference[2]

    # --- Step 4: Combine contrasts per direction (no compute inside loop)
    contrast_list = []
    contrast_label = []

    # “main” direction → absorption only
    contrast_list.append(delta_abs)
    contrast_label.append("absorption")

    for direction, harmonics in CONTRASTS_CLI.items():
        # Select harmonics lazily (xarray handles Dask indexing)
        scat_sel = delta_scat.sel(harmonic=harmonics)
        phase_sel = delta_phase.sel(harmonic=harmonics)

        # scattering = promedio de intensidades
        scat_combined = scat_sel.sum("harmonic") / len(harmonics)
        contrast_list.append(
            scat_combined
        )
        contrast_label.append(f"scattering_{direction}")

        # phase combination
        if direction in ("horizontal", "vertical"):
            pos, neg = harmonics
            phase_combined = phase_sel.sel(harmonic=pos) - phase_sel.sel(harmonic=neg)
        elif direction == "bidirectional":
            pos_h, neg_h, pos_v, neg_v = harmonics
            phase_combined = (
                phase_sel.sel(harmonic=pos_h) - phase_sel.sel(harmonic=neg_h)
                + phase_sel.sel(harmonic=pos_v) - phase_sel.sel(harmonic=neg_v)
            )
        else:
            raise ValueError(f"Unknown direction '{direction}' in CONTRASTS")

        contrast_list.append(
            phase_combined
        )
        contrast_label.append(f"diff_phase_{direction}")

    # --- Step 5: Concatenate all lazily computed datasets
    result = xr.concat(contrast_list, dim="contrast")
    result = result.assign_coords(contrast=contrast_label)

    # --- Step 6: Compute everything once (most efficient)
    return result



