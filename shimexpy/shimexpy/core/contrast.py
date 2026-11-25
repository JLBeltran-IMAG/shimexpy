"""
This module provides functions for retrieving and computing X-ray multicontrast images using Spatial Harmonic Imaging (SHI). It extracts harmonic information from the Fourier spectrum to reconstruct absorption, dark-field scattering, and differential phase contrast images.

The module supports individual and combined contrast retrieval, directional analysis of scattering and phase signals, harmonic-specific contrast computation, and multiple phase-unwrapping algorithms.

Note:
-----
This module is still under development, so the improvements are ongoing.
"""

import numpy as np
import xarray as xr
from dask.base import compute

from ..core import unwrapping as uphase
from .fourier_harmonics import (
    shi_fft,
    get_fourier_harmonics
)

from ..utils.parallelization import apply_harmonic_chunking


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
    
    elif unwrap == "snaphu":
        unwrapped_phase_map = xr.apply_ufunc(
            uphase.snaphu_unwrap,
            ratio,
            main_harmonic,
            input_core_dims=[["y", "x"], ["y", "x"]],
            output_core_dims=[["y", "x"]],
            dask="parallelized",
            output_dtypes=[np.float32],
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
    eps: np.floating = np.finfo(np.float64).eps
) -> xr.DataArray:
    """
    Retrieves individual contrast members from a harmonic component.

    This function processes harmonic components to retrieve different types of contrast
    (absorption, scattering, or phase map) from the inverse Fourier transform of the input.

    The contrast retrieval is performed by computing the inverse Fourier transform of the harmonics and then applying the appropriate mathematical operations according to the specified contrast type. At this stage, the corresponding normalization is applied without reference-image normalization, so the output represents the contrast relative to the main harmonic.

    Parameters
    ----------
    harmonics : xr.DataArray
        The harmonic component in Fourier space to be processed. This parameter is obtained directly from the `get_fourier_harmonics` function.
    type_of_contrast : str
        The type of contrast to retrieve. Must be one of:
        - 'absorption': Computes absorption contrast
        - 'scattering': Computes scattering contrast
        - 'phasemap': Computes phase map contrast
    unwrap : str, optional
        The unwrapping algorithm to use when type_of_contrast is 'phasemap'.
        > **Note:** This parameter supports several phase-unwrapping algorithms implemented in the `unwrapping` submodule. These algorithms can also be used independently, as they are designed to work with general NumPy `ndarray` structures. In this function, phase unwrapping is integrated into `contrast_retrieval` as part of the contrast reconstruction process.

    eps : float, optional
        Small constant to avoid division by zero. Default is 'np.finfo(np.float64).eps'.

    Returns
    -------
    ndarray
        The retrieved contrast map normalized with respect to the main harmonic (zero-order component), without reference-image normalization.

    Raises
    ------
    ValueError
        If type_of_contrast is not one of 'absorption', 'scattering', or 'phasemap'.
    """
    # Compute the inverse Fourier transform of harmonics
    ifft_harmonic = xr.apply_ufunc(
        np.fft.ifft2,
        harmonics,
        input_core_dims = [["ky", "kx"]],
        output_core_dims = [["y", "x"]],
        kwargs={"norm": "ortho"},
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


# --------------- Dark-field geometrical and statistical weights
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
def get_spatial_harmonics(
    image: np.ndarray,
    projected_grid: float,
    block_grid: dict | None = None,
    unwrap: str | None = None
) -> tuple:
    """
    Retrieve harmonic-based contrast maps from a Spatial Harmonic Imaging (SHI) image.

    This function extracts spatial harmonics from the Fourier spectrum and applies contrast-specific operations to retrieve absorption, scattering, and differential phase maps.

    The resulting contrasts are computed from a single image, using the main (zero-order) harmonic as the basis for contrast retrieval, without normalization against an external reference image.

    Parameters
    ----------
    image : np.ndarray
        The input image to analyze.
    projected_grid : float
        The projected grid period used to compute the spatial frequency axes. This parameter can be determined in two ways:

        1. By zooming into the image and counting the number of pixels within one projected grid period. The resulting integer value is internally converted to a float.
        2. By calculating the projected grid period from the geometry of the experimental setup (see Tutorials for more information).
    block_grid : dict, optional
        A dictionary containing the boundaries of the windows surrounding each spatial harmonic in the Fourier spectrum, used to isolate the corresponding harmonic components.
    unwrap : str, optional
        The unwrapping algorithm to use when type_of_contrast is 'phasemap'.
        > **Note:** This parameter supports several phase-unwrapping algorithms implemented in the `unwrapping` submodule. These algorithms can also be used independently, as they are designed to work with general NumPy `ndarray` structures. In this function, phase unwrapping is integrated into `contrast_retrieval` as part of the contrast reconstruction process.

    Returns
    -------
    (absorption, scattering, diff_phase, block_grid) : tuple
        A tuple containing the following elements:

        1. absorption : xr.DataArray
            Absorption map computed exclusively from the main (zero-order) harmonic, without normalization against other harmonics or an external reference image.

        2. scattering : xr.DataArray
            Scattering maps computed from the spatial harmonics, normalized with respect to the main harmonic.

        3. diff_phase : xr.DataArray
            Differential phase maps obtained by applying the corresponding mathematical operations to the spatial harmonics (see Tutorials for more information).

        4. block_grid : dict
            The Fourier-space window boundaries used to isolate each spatial harmonic.
    """
    # Perform the Fourier transform and extract harmonics
    result = shi_fft(image, projected_grid)
    kx, ky, fft_img = result.kx, result.ky, result.fft

    # Extract spatial harmonics from the Fourier spectrum
    if block_grid:
        harmonics, _ = get_fourier_harmonics(
            fft_img, ky, kx, reference=False, reference_block_grid=block_grid
        )
    else:
        harmonics, block_grid = get_fourier_harmonics(
            fft_img, ky, kx, reference=True
        )

    # Chunk the harmonics for parallel processing
    harmonics_chunked = harmonics.chunk({"harmonic": 1, "ky": -1, "kx": -1})

    # Compute the contrasts from the harmonics
    absolute_absorption = contrast_retrieval(
        harmonics_chunked, type_of_contrast="absorption"
    )
    absolute_scattering = contrast_retrieval(
        harmonics_chunked, type_of_contrast="scattering"
    )
    absolute_diff_phase = contrast_retrieval(
        harmonics_chunked, type_of_contrast="phasemap", unwrap=unwrap
    )

    return absolute_absorption, absolute_scattering, absolute_diff_phase, block_grid


def get_contrast(
    sample_img: np.ndarray,
    reference: xr.DataArray,
    ref_block_grid: dict,
    type_of_contrast: str,
    unwrap: str | None = None
):
    """
    Retrieve a specified multicontrast image from a Spatial Harmonic Imaging (SHI) measurement.

    This function performs the corresponding analysis on a sample image, extracts the relevant harmonic components from its Fourier spectrum, and applies contrast-specific operations to reconstruct the requested contrast.

    Unlike `get_harmonics`, this function computes the contrast relative to a reference image previously processed by `get_harmonics`. For directional scattering and differential phase contrast, the corresponding harmonic components are selected and combined according to the specified direction.

    Parameters
    ----------
    sample_img : np.ndarray
        The sample image to analyze.
    reference : xr.DataArray
        The precomputed reference contrast corresponding to the selected `type_of_contrast`. For directional contrasts, the reference must contain the corresponding harmonic components.
    ref_block_grid : dict
        A dictionary containing the boundaries of the windows surrounding each spatial harmonic in the reference Fourier spectrum. These boundaries are reused to extract the corresponding harmonics from the sample image.
    type_of_contrast : str
        The type of contrast to retrieve. Supported options are:

        1. "absorption":
            Computes absorption contrast relative to the reference image.

        2. "{direction}_scattering":
            Computes directional dark-field scattering contrast by combining the corresponding harmonic components using geometric weights.

        3. "{direction}_phasemap":
            Computes directional differential phase contrast from the corresponding harmonic phase maps.

        For scattering and phase contrast, `{direction}` must be "horizontal", "vertical", or "bidirectional".

    unwrap : str, optional
        The phase-unwrapping algorithm to use when retrieving differential phase contrast. Default is None.

        > **Note:** The available phase-unwrapping algorithms are implemented in the `unwrapping` submodule and can also be used independently with NumPy `ndarray` inputs. In this function, phase unwrapping is performed through `contrast_retrieval`.

    Returns
    -------
    contrast : xr.DataArray
        The reconstructed contrast map for the sample image, computed relative to the corresponding reference contrast. The output is a two-dimensional contrast image obtained after applying the mathematical operations associated with the selected contrast type (see Tutorials for more information).

    Raises
    ------
    ValueError
        If the specified contrast type or direction is not supported.
    """
    sample_result = shi_fft(sample_img)
    sample_harmonics, _ = get_fourier_harmonics(
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
        result = sample_contrast.sel(harmonic=harmonics) - reference.sel(harmonic=harmonics)

        # geometric weights
        weights = _harmonic_direction_weights(ref_block_grid, harmonics, direction)

        if contrasts == "scattering":
            output = abs(result * weights).sum("harmonic") / weights.sum("harmonic")
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
    sample_harmonics, _ = get_fourier_harmonics(
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

    absorption = sample_absorption - reference[0]
    scattering = (sample_scattering.sel(harmonic=harmonics) - 
                  reference[1].sel(harmonic=harmonics))
    diff_phase = (sample_diff_phase.sel(harmonic=harmonics) - 
                  reference[2].sel(harmonic=harmonics))

    scattering = scattering.sum("harmonic")
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
    ) = get_spatial_harmonics(reference_img, projected_grid, unwrap=unwrap)

    (
        sample_absorption,
        sample_scattering,
        sample_diff_phase,
        _
    ) = get_spatial_harmonics(sample_img, projected_grid, ref_block_grid, unwrap=unwrap)

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

    return absorption_contrast, scattering_contrast, diff_phase_contrast


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

    Returns
    -------
    xr.Dataset
        A dataset containing harmonic-wise contrasts:
        {harmonic_name: xr.DataArray}
    """
    # --- Step 1: FFT and harmonic extraction
    sample_fft_img = shi_fft(sample_img).fft
    sample_harmonics, _ = get_fourier_harmonics(
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



