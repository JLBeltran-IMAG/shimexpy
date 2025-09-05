# Scientific imports
import numpy as np
import xarray as xr
import cupy as cp
import cupyx.scipy.fft as cufft
from dask.base import compute

# Standard imports
import math
from dataclasses import dataclass

# Custom imports
from shimexpy import unwrapping_phase as uphase


# Contrast retrieval especifications
# These are the types of contrast that can be retrieved for "get_contrast(...)" function
CONTRASTS = {
    "horizontal": ["harmonic_horizontal_positive", "harmonic_horizontal_negative"],
    "vertical": ["harmonic_vertical_positive",   "harmonic_vertical_negative"],
    "bidirectional": [
        "harmonic_horizontal_positive", "harmonic_horizontal_negative",
        "harmonic_vertical_positive", "harmonic_vertical_negative"
    ]
}


@dataclass
class FFTResult:
    kx: np.ndarray | None
    ky: np.ndarray | None
    fft: np.ndarray


# ---------- math
def shi_fft_gpu(
    image: np.ndarray,
    projected_grid: float | None = None,
    logspect: bool = False
) -> FFTResult:
    """
    Fast Fourier Transform calculation using GPU acceleration via CuPy.
    This function performs a 2D FFT on an input image using GPU acceleration through CuPy and cuFFT.
    The process involves transferring data to GPU, computing FFT, optional log-spectrum calculation,
    and transferring results back to CPU.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image array to transform
    projected_grid : float, optional
        Grid spacing for frequency axis calculation. If None, frequency axes are not computed
    logspect : bool, default=False
        If True, computes log10(1 + abs(FFT)) of the spectrum on GPU

    Returns
    -------
    FFTResult
        Named tuple containing:
        - kx: Frequency axis for x dimension (None if projected_grid not provided)
        - ky: Frequency axis for y dimension (None if projected_grid not provided) 
        - img_fft: 2D array with FFT results

    Notes
    -----
    The function uses CuPy's FFT implementation which internally uses NVIDIA's cuFFT library.
    The computation is done on GPU for improved performance compared to CPU implementations.
    """
    # 1) Transfer image to GPU (CuPy)
    img_gpu = cp.asarray(image, dtype=cp.float32, order="C")

    # 2) Compute the 2D FFT and shift the zero frequency component to the center
    fft_gpu = cufft.fft2(img_gpu)
    fft_gpu = cufft.fftshift(fft_gpu)

    # 3) Log-spectrum
    if logspect:
        # esto se ejecuta de forma paralela en la GPU
        fft_gpu = cp.log10(1 + cp.abs(fft_gpu))

    # 4) Transfer the FFT result back to CPU
    img_fft = cp.asnumpy(fft_gpu)

    # 5) If a projected grid is specified, compute the spatial frequency axes
    if projected_grid is not None:
        h, w = image.shape
        kx = np.fft.fftfreq(w, d=1 / projected_grid)
        ky = np.fft.fftfreq(h, d=1 / projected_grid)
        return FFTResult(kx, ky, img_fft)

    return FFTResult(None, None, img_fft)


def shi_fft_cpu(
    image: np.ndarray,
    projected_grid: float | None = None,
    logspect: bool = False
) -> FFTResult:
    """
    Computes the 2D Fast Fourier Transform (FFT) of an input image and returns either the linear
    or logarithmic spectrum. If a projected grid period is provided, the corresponding spatial
    frequency axes are also returned.

    Parameters
    ----------
    image : np.ndarray
        A 2D array representing the input image. Must be a real-valued array.
    
    projected_grid : float or None, optional
        The projected grid period (in real-space units) used to compute the spatial frequency axes.
        If None, only the transformed image is returned. Default is None.
    
    logspect : bool, optional
        If True, returns the logarithmic amplitude spectrum: log10(1 + |FFT|).
        If False, returns the complex FFT result directly. Default is False.

    Returns
    -------
    fft_image : np.ndarray
        If `projected_grid` is None, returns either the raw FFT result or its log spectrum,
        depending on the value of `logspect`.

    wavevector_kx, wavevector_ky, fft_image : tuple of np.ndarray
        If `projected_grid` is specified, returns the spatial frequencies in the x and y directions,
        along with the FFT result (or its log spectrum).

    Notes
    -----
    The FFT result is centered with `np.fft.fftshift`.
    The log spectrum is computed as `log10(1 + abs(FFT))` to avoid issues with log(0).
    The spatial frequency axes are computed using `np.fft.fftfreq` with spacing `1 / projected_grid`.
    The image is internally cast to `np.float32` before applying the FFT.
    Use projected_grid to limit the harmonics for reference images
    """
    # Check if the input image is a 2D array
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")

    # Image height and width
    img_height, img_width = image.shape

    # Calculate Fourier transform
    img_fft = np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))

    # If logspect is True, we compute the logarithmic spectrum
    if logspect:
        # Compute the logarithmic spectrum
        img_fft = np.log10(1 + np.abs(img_fft))

    # If a projected grid is specified, we will use it to limit the harmonics
    # This is only useful for reference images
    if projected_grid:
        # Spatial frequencies (Fourier space) for limiting the selected harmonics
        kx = np.fft.fftfreq(img_width, d=1 / projected_grid)
        ky = np.fft.fftfreq(img_height, d=1 / projected_grid)
    
        return FFTResult(kx, ky, img_fft)
    
    return FFTResult(None, None, img_fft)


# -------------------- extraction separation and labelling
def _zero_fft_region(
    array2d: np.ndarray,
    top: float | np.integer,
    bottom: float | np.integer,
    left: float | np.integer,
    right: float | np.integer
) -> np.ndarray:
    """
    Sets a specific rectangular region of a 2D complex array to zero.

    This function is useful for filtering out certain frequency components
    in the Fourier domain of an image.

    Parameters
    ----------
    array2d : np.ndarray
        A 2D NumPy array representing the Fourier transform of an image.
        It must be a complex-valued array.
    top : int
        The starting row index of the region to be zeroed.
    bottom : int
        The ending row index of the region to be zeroed (exclusive).
    left : int
        The starting column index of the region to be zeroed.
    right : int
        The ending column index of the region to be zeroed (exclusive).

    Returns
    -------
    np.ndarray
        The modified 2D array with the specified region set to zero.
    """
    array2d[top:bottom, left:right] = np.complex128(0)
    return array2d


def _extracting_harmonic(
    fourier_transform: np.ndarray,
    ky_band_limit: np.integer,
    kx_band_limit: np.integer
) -> tuple[int, int, int, int, np.intp, np.intp]:
    """
    Extracts a rectangular region around the maximum harmonic component in a Fourier transform.

    This function locates the point with the highest magnitude in the Fourier transform and
    defines a rectangular region centered on that point, using the provided vertical and
    horizontal band limits.

    Parameters
    ----------
    fourier_transform : np.ndarray
        A 2D NumPy array representing the Fourier transform of an image (complex values).
    ky_band_limit : int
        The vertical band limit (number of rows) to extract around the maximum component.
    kx_band_limit : int
        The horizontal band limit (number of columns) to extract around the maximum component.

    Returns
    -------
    tuple
        A tuple containing:
            top_limit (int): The top boundary of the extracted region.
            bottom_limit (int): The bottom boundary of the extracted region.
            left_limit (int): The left boundary of the extracted region.
            right_limit (int): The right boundary of the extracted region.
            max_row_index (int): The row index of the maximum magnitude component.
            max_col_index (int): The column index of the maximum magnitude component.
    """
    # Compute the absolute value of the Fourier transform to find the magnitude
    # This is necessary to locate the maximum component in the Fourier domain.
    abs_fft = np.abs(fourier_transform)

    # Find the index of the maximum magnitude in the Fourier transform
    max_row_index, max_col_index = np.unravel_index(np.argmax(abs_fft), abs_fft.shape)

    # Calculate boundaries and ensure they stay within the array dimensions
    top_limit = max(0, max_row_index - ky_band_limit)
    bottom_limit = min(fourier_transform.shape[0], max_row_index + ky_band_limit)
    left_limit = max(0, max_col_index - kx_band_limit)
    right_limit = min(fourier_transform.shape[1], max_col_index + kx_band_limit)

    return top_limit, bottom_limit, left_limit, right_limit, max_row_index, max_col_index


def _identifying_harmonics_x1y1_higher_orders(x, y):
    """
    Identifies the harmonic diagonal based on the signs of x and y.

    Parameters:
    -----------
    x : numeric
        The x-coordinate.
    y : numeric
        The y-coordinate.

    Returns:
    --------
    str
        A string representing the harmonic diagonal:
          - "harmonic_diagonal_p1_p1" if x > 0 and y > 0.
          - "harmonic_diagonal_n1_p1" if x < 0 and y > 0.
          - "harmonic_diagonal_n1_n1" if x < 0 and y < 0.
          - "harmonic_diagonal_p1_n1" if x > 0 and y < 0.

    Raises:
    -------
    ValueError:
        If either x or y is zero, as the harmonic diagonal is undefined in that case.
    """
    if x == 0 and y == 0:
        raise ValueError("Invalid input: x and y must be non-zero to determine a harmonic diagonal.")

    if x > 0 and y > 0:
        return "harmonic_diagonal_p1_p1"
    elif x < 0 and y > 0:
        return "harmonic_diagonal_n1_p1"
    elif x < 0 and y < 0:
        return "harmonic_diagonal_n1_n1"
    elif x > 0 and y < 0:
        return "harmonic_diagonal_p1_n1"


def _identifying_harmonic(
    main_harmonic_height: np.integer,
    main_harmonic_width: np.integer,
    harmonic_height: np.integer,
    harmonic_width: np.integer,
    angle_threshold: float = 15
):
    """
    Identifies the type of harmonic based on its position relative to a main harmonic.

    This function determines whether a harmonic peak is vertical, horizontal, or of a higher
    order by comparing its position to a main harmonic's position and analyzing the deviation angle.
    The angle_threshold parameter sets the threshold (in degrees) to determine if the deviation
    is predominantly vertical or horizontal.

    Parameters
    ----------
    main_harmonic_height : np.integer (integer)
        The y-coordinate of the main harmonic peak.
    main_harmonic_width : np.integer (integer)
        The x-coordinate of the main harmonic peak.
    harmonic_height : np.integer (integer)
        The y-coordinate of the harmonic peak being analyzed.
    harmonic_width : np.integer (integer)
        The x-coordinate of the harmonic peak being analyzed.
    angle_threshold : float, optional
        The threshold angle (in degrees) used to decide if a harmonic is primarily vertical or horizontal.
        Default is 15.

    Returns
    -------
    str
        The type of harmonic identified:
          - "harmonic_vertical_positive": Vertical harmonic above the main peak.
          - "harmonic_vertical_negative": Vertical harmonic below the main peak.
          - "harmonic_horizontal_positive": Horizontal harmonic to the right of the main peak.
          - "harmonic_horizontal_negative": Horizontal harmonic to the left of the main peak.
          - In other cases, the result of _identifying_harmonics_x1y1_higher_orders(dx, dy).

    Notes
    -----
    If the deviation angle exceeds the angle_threshold, the function delegates the analysis to
    _identifying_harmonics_x1y1_higher_orders(), which is assumed to handle higher order cases.

    For info about type np.integer, read Nupy docs
    """
    # Calculate differences between the harmonic and the main harmonic coordinates.
    dy = harmonic_height - main_harmonic_height
    dx = harmonic_width - main_harmonic_width
    abs_dy = abs(dy)
    abs_dx = abs(dx)

    # Case: Dominant vertical deviation.
    if abs_dy > abs_dx:
        # Calculate the deviation angle with respect to the vertical axis.
        deviation_angle = math.degrees(math.atan2(abs_dx, abs_dy))
        if deviation_angle < angle_threshold:
            return "harmonic_vertical_positive" if dy > 0 else "harmonic_vertical_negative"
        else:
            return _identifying_harmonics_x1y1_higher_orders(dx, dy)

    # Case: Dominant horizontal deviation.
    elif abs_dx > abs_dy:
        # Calculate the deviation angle with respect to the horizontal axis.
        deviation_angle = math.degrees(math.atan2(abs_dy, abs_dx))
        if deviation_angle < angle_threshold:
            return "harmonic_horizontal_positive" if dx > 0 else "harmonic_horizontal_negative"
        else:
            return _identifying_harmonics_x1y1_higher_orders(dx, dy)

    # Case: When vertical and horizontal deviations are equal.
    else:
        return _identifying_harmonics_x1y1_higher_orders(dx, dy)


def spatial_harmonics_of_fourier_spectrum(
    fourier_transform: np.ndarray,
    ky: np.ndarray | None,
    kx: np.ndarray | None,
    reference: bool = True,
    reference_block_grid: dict | None = None,
    limit_band: float = 0.5
) -> tuple[xr.DataArray, dict[str, list[np.integer]]]:
    """
    Extracts and labels spatial harmonics from a 2D Fourier spectrum, returning them as a structured xarray DataArray.

    This function can operate in two modes:
      - Reference mode (`reference=True`): Automatically identifies and extracts harmonics from the given Fourier spectrum.
      - Non-reference mode (`reference=False`): Extracts harmonics using pre-defined spatial limits from a reference block grid.

    Parameters
    ----------
    fourier_transform : np.ndarray
        2D array (complex-valued) representing the Fourier transform of an image.
    ky : np.ndarray
        1D array of vertical wavevector components corresponding to the Fourier domain's y-axis.
    kx : np.ndarray
        1D array of horizontal wavevector components corresponding to the Fourier domain's x-axis.
    reference : bool, optional
        If True, harmonics are extracted by automatically locating and masking the dominant harmonic components.
        If False, the harmonic regions are extracted using `reference_block_grid`. Default is False.
    reference_block_grid : dict or None, optional
        Dictionary mapping harmonic labels to coordinate limits [top, bottom, left, right].
        Required if `reference=False`. Ignored if `reference=True`.
    limit_band : float, optional
        Frequency distance from the center (in wavevector units) to define the harmonic extraction window. Default is 0.5.

    Returns
    -------
    harmonics_da : xr.DataArray
        DataArray of shape (n_harmonics, y, x) containing the extracted harmonic regions.
        Coordinates include:
          - "harmonic" (label for each extracted region),
          - "y" and "x" (spatial indices of the full Fourier image).
    labels : list of str
        List of string labels corresponding to each harmonic.
    block_grid : dict
        Dictionary mapping harmonic labels to their extraction limits: [top, bottom, left, right].

    Raises
    ------
    ValueError
        If `reference` is False and `reference_block_grid` is not provided.

    Notes
    -----
    - Reference is when you are processing a reference image

    Examples
    --------
    >>> da, labels, grid = spatial_harmonics_of_fourier_spectrum(
    ...     fourier_transform=my_fft,
    ...     ky=ky_array,
    ...     kx=kx_array,
    ...     reference=True,
    ...     limit_band=0.3
    ... )
    >>> da.sel(harmonic="harmonic_horizontal_positive").plot()
    """
    # Create a copy of the Fourier transform to avoid modifying the original.
    copy_of_fourier_transform = np.copy(fourier_transform)

    if reference and ky is not None and kx is not None:
        # Identify the main maximum harmonic (assumed to be near the center)
        abs_fft = np.abs(fourier_transform)
        max_index = np.argmax(abs_fft)
        main_max_h, main_max_w = np.unravel_index(max_index, abs_fft.shape)

        # Determine band limits based on the wavevector arrays.
        ky_band_limit = np.argmin(np.abs(ky - limit_band))
        kx_band_limit = np.argmin(np.abs(kx - limit_band))

        harmonics = []
        block_grid = {}

        # Extract the 0-order harmonic.
        top = main_max_h - ky_band_limit
        bottom = main_max_h + ky_band_limit
        left = main_max_w - kx_band_limit
        right = main_max_w + kx_band_limit

        harmonics.append(fourier_transform[top:bottom, left:right])
        label = "harmonic_00"
        block_grid[label] = [top, bottom, left, right]

        # Zero out the extracted region in the copy to avoid re-detection.
        _zero_fft_region(copy_of_fourier_transform, top, bottom, left, right)

        # Extract higher-order harmonics (by default, 4 additional harmonics).
        for i in range(8):
            top, bottom, left, right, harmonic_h, harmonic_w = _extracting_harmonic(
                copy_of_fourier_transform, ky_band_limit, kx_band_limit
            )

            harmonics.append(fourier_transform[top:bottom, left:right])
            label = _identifying_harmonic(main_max_h, main_max_w, harmonic_h, harmonic_w)
            block_grid[label] = [top, bottom, left, right]
            _zero_fft_region(copy_of_fourier_transform, top, bottom, left, right)

        # Create a DataArray to hold the harmonics.
        da = xr.DataArray(
            harmonics,
            dims=["harmonic", "ky", "kx"],
            coords={
                "harmonic": list(block_grid.keys()),
                "ky": np.arange(bottom - top),
                "kx": np.arange(right - left)
            }
        )

        return da, block_grid

    else:
        harmonics = []
        top, bottom, left, right = 0, 0, 0, 0

        # Reconstruct harmonic regions using the stored limits.
        if reference_block_grid is None:
            raise ValueError("Reference block grid (parameter -> reference_block_grid) must be provided when reference is False.")
        else:
            for label, limits in reference_block_grid.items():
                top, bottom, left, right = limits
                harmonics.append(fourier_transform[top:bottom, left:right])

        da = xr.DataArray(
            harmonics,
            dims=["harmonic", "ky", "kx"],
            coords={
                "harmonic": list(reference_block_grid.keys()),
                "ky": np.arange(bottom - top),
                "kx": np.arange(right - left)
            }
        )

        return da, reference_block_grid


# -------------------- contrast retrieval
def _compute_phase_map(
    iftt_harmonics: xr.DataArray,
    main_harmonic: xr.DataArray,
    unwrap: str | None = None,
    eps: float = 1e-12
):
    """
    Computes the unwrapped phase map from the inverse Fourier transform and the main harmonic.

    Parameters:
    -----------
    iftt_harmonics : np.ndarray
        Array containing the inverse Fourier transform of the data.
    main_harmonic : np.ndarray
        Array containing the main harmonic in the Fourier domain.
    eps : float, optional
        Small value added to the denominator to avoid division by zero (default is 1e-12).

    Returns:
    --------
    unwrapped_phase_map : np.ndarray
        The unwrapped phase map.
    """
    # Compute the ratio, avoiding division by zero by adding a small eps
    ratio = iftt_harmonics / (main_harmonic + eps)

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

    elif unwrap == "branch_cut":
        unwrapped_phase_map = xr.apply_ufunc(
            uphase.branch_cut_unwrap,
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

    elif unwrap == "quality_guided":
        unwrapped_phase_map = xr.apply_ufunc(
            uphase.quality_guided_unwrap,
            ratio,
            input_core_dims = [["y", "x"]],
            output_core_dims = [["y", "x"]],
            dask="parallelized",
            output_dtypes = [ratio.dtype],
        )

    elif unwrap == "numpy":
        unwrapped_phase_map = xr.apply_ufunc(
            uphase.sequential_np_unwrap,
            ratio,
            input_core_dims = [["y", "x"]],
            output_core_dims = [["y", "x"]],
            dask = "parallelized",
            output_dtypes = [np.float32],
        )

    else:
        raise ValueError("Unknown phase unwrapping algorithm")

    return unwrapped_phase_map


def _compute_scattering(iftt_harmonics, main_harmonic, eps=1e-12):
    """
    Computes the scattering value from the inverse Fourier transform and the main harmonic.

    Parameters:
    -----------
    iftt_harmonics : np.ndarray
        Array containing the inverse Fourier transform of the data.
    main_harmonic : np.ndarray
        Array containing the main harmonic in the Fourier domain.
    eps : float, optional
        Small value added to the denominator to avoid division by zero (default is 1e-12).

    Returns:
    --------
    scattering_value : np.ndarray
        The computed scattering value.
    """
    # Compute the ratio and avoid division by zero by adding eps
    ratio = iftt_harmonics / (main_harmonic + eps)

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
    label : any, optional
        Label used for phase unwrapping when type_of_contrast is 'phasemap'.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-12.
    main_harmonic : any, optional
        Reserved for potential future use with main harmonic reference.

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


# -------------------- main functions REAL TIME
def set_crop():
    pass


def get_harmonics(image, projected_grid, block_grid=None, unwrap = None, crop = None):
    """
    Set reference image for spatial harmonics analysis.
    This function performs spatial harmonics analysis on a given image and returns the
    absorption, scattering, and differential phase maps, along with the block grid for harmonics.

    Parameters
    ----------
    image : np.ndarray
        The input image to analyze.
    projected_grid : float
        The projected grid period (in real-space units) used to compute the spatial frequency axes.
    unwrap : str, optional
        The unwrapping algorithm to use for phase map retrieval. Default is None.
    crop : tuple, optional
        A tuple specifying the crop region (slice(top, bottom), slice(left, right)) to apply to the image.
        If None, no cropping is applied.

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
    # Crop the image if specified
    image = image[crop] if crop else image

    # Perform the Fourier transform and extract harmonics
    result = shi_fft_gpu(image, projected_grid)
    kx, ky, fft_img = result.kx, result.ky, result.fft

    # Extract spatial harmonics from the Fourier spectrum
    if block_grid:
        harmonics, _ = spatial_harmonics_of_fourier_spectrum(fft_img, ky, kx)
    else:
        harmonics, block_grid = spatial_harmonics_of_fourier_spectrum(fft_img, ky, kx)

    # Chunk the harmonics for parallel processing
    harmonics_chunked = harmonics.chunk({"harmonic": 1, "ky": "auto", "kx": "auto"})

    # Compute the contrasts from the harmonics
    absorption = contrast_retrieval(harmonics_chunked, type_of_contrast="absorption")
    scattering = contrast_retrieval(harmonics_chunked, type_of_contrast="scattering")
    diff_phase = contrast_retrieval(harmonics_chunked, type_of_contrast="phasemap")

    return absorption, scattering, diff_phase, block_grid


def get_contrast(sample_img, reference, ref_block_grid, type_of_contrast, unwrap = None, crop = None):
    """
    Execute spatial harmonics analysis on a sample image and retrieve the specified contrast.
    This function performs spatial harmonics analysis on a sample image and computes the contrast
    with respect to a reference image.

    Parameters
    ----------
    sample_img : np.ndarray
        The sample image to analyze.
    reference : xr.DataArray
        The reference image containing the pre-computed contrasts (absorption, scattering, phase map).
    ref_block_grid : dict
        A dictionary containing the limits for each harmonic in the reference image.
    type_of_contrast : str
        The type of contrast to retrieve. Must be one of:
        - 'absorption': Computes absorption contrast
        - '(direction)_scattering': Computes (direction) scattering contrast
        - '(direction)_phasemap': Computes (direction) phase map contrast
    unwrap : str, optional
        The unwrapping algorithm to use for phase map retrieval. Default is None.
    crop : tuple, optional
        A tuple specifying the crop region (slice(top, bottom), slice(left, right)) to apply to the sample image.
        If None, no cropping is applied.
    
    Returns
    -------
    xr.DataArray
        The computed contrast for the sample image, relative to the reference image.
    """
    sample_result = shi_fft_gpu(sample_img)
    sample_harmonics, _ = spatial_harmonics_of_fourier_spectrum(
        sample_result.fft, None, None, False, ref_block_grid
    )
    sample_harmonics_chunked = sample_harmonics.chunk({"harmonic": 1, "ky": "auto", "kx": "auto"})

    if type_of_contrast == "absorption":
        sample_contrast = contrast_retrieval(sample_harmonics_chunked, "absorption")
        output = sample_contrast - reference

    elif "_scattering" in type_of_contrast or "_phasemap" in type_of_contrast:
        direction, contrasts = type_of_contrast.split('_')
        harmonics = CONTRASTS[direction]
        sample_contrast = contrast_retrieval(sample_harmonics_chunked, contrasts)
        result = sample_contrast.sel(harmonic=harmonics) - reference.sel(harmonic=harmonics)

        if contrasts == "scattering":
            output = result.sum("harmonic")
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


def get_contrasts(sample_img, references, ref_block_grid, unwrap = None, crop = None):
    """
    """
    # Sample
    # Contrast retrieval of sample image
    sample_result = shi_fft_gpu(sample_img)
    sample_fft_img = sample_result.fft
    sample_harmonics, _ = spatial_harmonics_of_fourier_spectrum(
        sample_fft_img,
        None,
        None,
        False,
        ref_block_grid
    )
    sample_harmonics_chunked = sample_harmonics.chunk({"harmonic": 1, "ky": "auto", "kx": "auto"})

    # Contrast retrieval reference and sample images
    sample_absorption = contrast_retrieval(sample_harmonics_chunked, "absorption")
    sample_scattering = contrast_retrieval(sample_harmonics_chunked, "scattering")
    sample_diff_phase = contrast_retrieval(sample_harmonics_chunked, "phasemap")

    harmonics = CONTRASTS["bidirectional"]

    absorption = sample_absorption - references[0]
    scattering = sample_scattering.sel(harmonic=harmonics) - references[1].sel(harmonic=harmonics)
    diff_phase = sample_diff_phase.sel(harmonic=harmonics) - references[2].sel(harmonic=harmonics)

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
    (absorption_contrast,
    scattering_contrast,
    diff_phase_contrast) = compute(absorption, scattering, diff_phase)

    return absorption_contrast, scattering_contrast, diff_phase_contrast


def get_all_contrasts(sample_img, reference_img, projected_grid, unwrap = None, crop = None):
    """
    """
    # Reference
    # Contrast retrieval of reference image
    (
        ref_absorption,
        ref_scattering,
        ref_diff_phase,
        ref_block_grid
    ) = get_harmonics(reference_img, projected_grid)

    (
        sample_absorption,
        sample_scattering,
        sample_diff_phase,
        _
    ) = get_harmonics(sample_img, projected_grid, ref_block_grid)

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


# def shiexecute_multiple_tests(sample_img, image, projected_grid, unwrap = None, crop = None):
#     """
#     Execute spatial harmonics analysis on a set of images.

#     This function performs spatial harmonics analysis on a set of images and exports the results to the specified directory.

#     Parameters
#     ----------
#     path_to_images : list of str
#         A list of paths to the images for analysis.
#     path_to_result : str
#         The path to the directory where the results will be exported.
#     projected_grid : int
#         The period of the mask used in the analysis.

#     """
#     # Reference
#     reference_absorption, reference_scattering, reference_diff, ref_block_grid = get_harmonics(image, projected_grid)


#     for sample_img in sample_img:
#         # Sample
#         # FFT and harmonic extraction
#         sample_result = shi_fft_gpu(sample_img)
#         sample_fft_img = sample_result.fft
#         sample_harmonics, _, _ = spatial_harmonics_of_fourier_spectrum(
#             sample_fft_img,
#             None,
#             None,
#             False,
#             ref_block_grid
#         )
#         sample_harmonics_chunked = sample_harmonics.chunk({"harmonic": 1, "ky": "auto", "kx": "auto"})

#         # Contrast retrieval reference and sample images
#         sample_absorption = contrast_retrieval(sample_harmonics_chunked, "absorption")
#         sample_scattering = contrast_retrieval(sample_harmonics_chunked, "scattering")
#         # sample_diff_phase = contrast_retrieval(sample_harmonics_chunked, "phasemap")

#         # ---------
#         #  Compute
#         # ---------
#         absorption = sample_absorption - reference_absorption
#         scattering = sample_scattering - reference_scattering
#         # diff_phase = sample_diff_phase - reference_diff

#         (absorption_contrast,
#         scattering_contrast) = compute(absorption, scattering)


