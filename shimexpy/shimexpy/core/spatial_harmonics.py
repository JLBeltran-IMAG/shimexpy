"""
Core module for spatial harmonics transformations.
"""

# Scientific imports
import numpy as np
import xarray as xr
import math
from dataclasses import dataclass


try:
    import cupy as cp
    import cupyx.scipy.fft as cufft
    _USE_CUDA = cp.is_available()
except Exception:
    _USE_CUDA = False


@dataclass
class FFTResult:
    kx: np.ndarray | None
    ky: np.ndarray | None
    fft: np.ndarray


def _shi_fft_gpu(
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
    img_gpu = cp.asarray(image, dtype=cp.float32, order="C") # type: ignore

    # 2) Compute the 2D FFT and shift the zero frequency component to the center
    fft_gpu = cufft.fft2(img_gpu, norm="ortho") # type: ignore
    fft_gpu = cufft.fftshift(fft_gpu) # type: ignore

    # 3) Log-spectrum
    if logspect:
        # esto se ejecuta de forma paralela en la GPU
        fft_gpu = cp.log10(1 + cp.abs(fft_gpu)) # type: ignore

    # 4) Transfer the FFT result back to CPU
    img_fft = cp.asnumpy(fft_gpu) # type: ignore

    # 5) If a projected grid is specified, compute the spatial frequency axes
    if projected_grid is not None:
        h, w = image.shape
        kx = np.fft.fftfreq(w, d=1 / projected_grid)
        ky = np.fft.fftfreq(h, d=1 / projected_grid)
        return FFTResult(kx, ky, img_fft)

    return FFTResult(None, None, img_fft)


def _shi_fft_cpu(
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
    img_fft = np.fft.fftshift(np.fft.fft2(image.astype(np.float32), norm="ortho"))

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


def shi_fft(
    image: np.ndarray,
    projected_grid: float | None = None,
    logspect: bool = False
) -> FFTResult:
    """
    Automatically select CPU or GPU FFT depending on CuPy availability.
    """
    if _USE_CUDA:
        return _shi_fft_gpu(image, projected_grid, logspect)
    else:
        return _shi_fft_cpu(image, projected_grid, logspect)


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
    reference: bool = False,
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
    if reference and ky is not None and kx is not None:
        # Create a copy of the Fourier transform to avoid modifying the original.
        copy_of_fourier_transform = np.copy(fourier_transform)

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

            # Save the limits of the extracted harmonic region.
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
            raise ValueError("Reference block grid (parameter -> reference_block_grid)" \
            "must be provided when reference is False.")
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
