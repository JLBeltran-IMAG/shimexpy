"""
Angle detection and correction utilities for X-ray imaging.

These functions detect the rotation angle of spatial harmonics patterns
by analyzing the FFT peak positions.
"""

import numpy as np
from typing import List, Tuple


def next_power_of_two(n: int) -> int:
    """
    Calculate the next power of two greater than or equal to n.

    Parameters
    ----------
    n : int
        Input number.

    Returns
    -------
    int
        The smallest power of two >= n.

    Examples
    --------
    >>> next_power_of_two(100)
    128
    >>> next_power_of_two(256)
    256
    """
    return int(2 ** np.ceil(np.log2(n)))


def _pad_to_square_power_of_two(image: np.ndarray) -> np.ndarray:
    """
    Pad image to square dimensions that are a power of two.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.

    Returns
    -------
    np.ndarray
        Zero-padded image with square power-of-two dimensions.
    """
    height, width = image.shape
    new_dim = next_power_of_two(max(height, width))

    pad_height = new_dim - height
    pad_width = new_dim - width

    padded = np.pad(
        image,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )
    return padded


def _compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute the centered FFT magnitude spectrum of an image.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.

    Returns
    -------
    np.ndarray
        Magnitude of the centered FFT.
    """
    padded = _pad_to_square_power_of_two(image.astype(np.float32))
    fft = np.fft.fftshift(np.fft.fft2(padded))
    return np.abs(fft)


def _zero_region(array: np.ndarray, top: int, bottom: int, left: int, right: int) -> None:
    """
    Zero out a rectangular region in an array (in-place).

    Parameters
    ----------
    array : np.ndarray
        Input array to modify.
    top, bottom, left, right : int
        Region boundaries.
    """
    array[top:bottom, left:right] = 0


def _find_peak_and_region(
    fft_magnitude: np.ndarray,
    band_limit: int,
) -> Tuple[int, int, int, int, int, int]:
    """
    Find the maximum peak and its surrounding region.

    Parameters
    ----------
    fft_magnitude : np.ndarray
        FFT magnitude array.
    band_limit : int
        Half-width of the region around the peak.

    Returns
    -------
    tuple
        (top, bottom, left, right, peak_row, peak_col)
    """
    peak_row, peak_col = np.unravel_index(
        np.argmax(fft_magnitude), fft_magnitude.shape
    )

    top = max(0, peak_row - band_limit)
    bottom = min(fft_magnitude.shape[0], peak_row + band_limit)
    left = max(0, peak_col - band_limit)
    right = min(fft_magnitude.shape[1], peak_col + band_limit)

    return top, bottom, left, right, peak_row, peak_col


def extract_peak_coordinates(
    image: np.ndarray,
    num_harmonics: int = 4,
    band_limit: int = 500,
) -> List[Tuple[int, int]]:
    """
    Extract coordinates of harmonic peaks from an image's FFT.

    This function identifies the main (zero-order) harmonic and up to
    `num_harmonics` higher-order harmonics by iteratively finding and
    masking peaks in the FFT magnitude spectrum.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image (typically a reference image with grid pattern).
    num_harmonics : int, optional
        Number of higher-order harmonics to detect. Default is 4.
    band_limit : int, optional
        Half-width of the masking region around each peak. Default is 500.

    Returns
    -------
    list of tuple
        List of (row, col) coordinates for each detected peak.
        First element is the zero-order (center) peak.

    Examples
    --------
    >>> coords = extract_peak_coordinates(reference_image)
    >>> center_peak = coords[0]
    >>> higher_order_peaks = coords[1:]
    """
    fft_magnitude = _compute_fft_magnitude(image)
    fft_copy = fft_magnitude.copy()

    coordinates = []

    # Extract zero-order (main) harmonic
    top, bottom, left, right, peak_row, peak_col = _find_peak_and_region(
        fft_copy, band_limit
    )
    coordinates.append((peak_row, peak_col))
    _zero_region(fft_copy, top, bottom, left, right)

    # Extract higher-order harmonics
    for _ in range(num_harmonics):
        top, bottom, left, right, peak_row, peak_col = _find_peak_and_region(
            fft_copy, band_limit
        )
        coordinates.append((peak_row, peak_col))
        _zero_region(fft_copy, top, bottom, left, right)

    return coordinates


def _quadrant_sign(
    peak_row: int,
    center_row: int,
    peak_col: int,
    center_col: int,
    axis: str,
) -> int:
    """
    Determine the sign contribution based on quadrant location.

    Parameters
    ----------
    peak_row, peak_col : int
        Peak position.
    center_row, center_col : int
        Center (zero-order) position.
    axis : str
        Either "y" or "x" to determine which axis dominates.

    Returns
    -------
    int
        Sign value (-1, 0, or 1).
    """
    if axis == "y":
        if peak_col > center_col and peak_row < center_row:
            return 1
        elif peak_col < center_col and peak_row < center_row:
            return -1
        elif peak_col < center_col and peak_row > center_row:
            return 1
        elif peak_col > center_col and peak_row > center_row:
            return -1
    elif axis == "x":
        if peak_col > center_col and peak_row > center_row:
            return 1
        elif peak_col > center_col and peak_row < center_row:
            return -1
        elif peak_col < center_col and peak_row < center_row:
            return 1
        elif peak_col < center_col and peak_row > center_row:
            return -1
    return 0


def calculate_rotation_angle(
    coordinates: List[Tuple[int, int]],
) -> float:
    """
    Calculate the average rotation angle from harmonic peak coordinates.

    This function computes the rotation angle of the grid pattern by
    analyzing the positions of higher-order harmonics relative to the
    zero-order (center) peak.

    Parameters
    ----------
    coordinates : list of tuple
        List of (row, col) coordinates from extract_peak_coordinates.
        First element should be the zero-order peak.

    Returns
    -------
    float
        Average rotation angle in degrees.

    Examples
    --------
    >>> coords = extract_peak_coordinates(reference_image)
    >>> angle = calculate_rotation_angle(coords)
    >>> rotated = skimage.transform.rotate(image, angle)
    """
    if len(coordinates) < 2:
        return 0.0

    center_row, center_col = coordinates[0]
    angles = []
    signs = []

    for peak_row, peak_col in coordinates[1:]:
        delta_row = abs(peak_row - center_row)
        delta_col = abs(peak_col - center_col)

        if delta_row > delta_col:
            # Vertical dominant
            sign = _quadrant_sign(peak_row, center_row, peak_col, center_col, "y")
            angle = np.rad2deg(np.arctan2(delta_col, delta_row))
        elif delta_col > delta_row:
            # Horizontal dominant
            sign = _quadrant_sign(peak_row, center_row, peak_col, center_col, "x")
            angle = np.rad2deg(np.arctan2(delta_row, delta_col))
        else:
            continue

        angles.append(angle)
        signs.append(sign)

    if not angles:
        return 0.0

    return float(np.mean(np.array(angles) * np.array(signs)))
