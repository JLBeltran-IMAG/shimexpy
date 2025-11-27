"""
Image correction functions for X-ray preprocessing.

These functions implement standard corrections for X-ray imaging:
- Dark field subtraction
- Bright field normalization (flat field correction)
"""

import numpy as np
from typing import Union
from pathlib import Path


def correct_darkfield(
    image: np.ndarray,
    dark_field: np.ndarray,
) -> np.ndarray:
    """
    Apply dark field correction to an image.

    Dark field correction removes the detector's dark current noise by
    subtracting the dark field image from the sample image.

    Parameters
    ----------
    image : np.ndarray
        The input image to correct (2D array).
    dark_field : np.ndarray
        The dark field image (2D array, same shape as image).
        If multiple dark images are available, pass their mean.

    Returns
    -------
    np.ndarray
        The dark-corrected image (float32).

    Examples
    --------
    >>> corrected = correct_darkfield(sample_image, dark_image)

    Notes
    -----
    For multiple dark field images, compute the mean first:
    >>> dark_mean = np.mean(dark_images, axis=0)
    >>> corrected = correct_darkfield(sample, dark_mean)
    """
    if image.shape != dark_field.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match dark field shape {dark_field.shape}"
        )

    corrected = image.astype(np.float32) - dark_field.astype(np.float32)
    return corrected


def correct_brightfield(
    image: np.ndarray,
    bright_field: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Apply bright field (flat field) correction to an image.

    Bright field correction normalizes the image by the detector's response
    to uniform illumination, correcting for pixel-to-pixel sensitivity variations.

    Parameters
    ----------
    image : np.ndarray
        The input image to correct (2D array).
    bright_field : np.ndarray
        The bright field image (2D array, same shape as image).
        If multiple bright images are available, pass their mean.
    epsilon : float, optional
        Small value to prevent division by zero. Default is 1e-10.

    Returns
    -------
    np.ndarray
        The bright-corrected image (float32).

    Examples
    --------
    >>> corrected = correct_brightfield(dark_corrected_image, bright_image)

    Notes
    -----
    The standard preprocessing pipeline is:
    1. Dark field correction: sample - dark
    2. Bright field correction: (sample - dark) / (bright - dark)
    """
    if image.shape != bright_field.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match bright field shape {bright_field.shape}"
        )

    image_float = image.astype(np.float32)
    bright_float = bright_field.astype(np.float32)

    # Avoid division by zero
    result = np.zeros_like(image_float)
    mask = np.abs(bright_float) > epsilon
    result[mask] = image_float[mask] / bright_float[mask]

    return result


def flat_field_correction(
    image: np.ndarray,
    dark_field: np.ndarray,
    bright_field: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Apply full flat field correction (dark and bright field) to an image.

    This combines dark field subtraction and bright field normalization
    in a single operation:

        corrected = (image - dark) / (bright - dark)

    Parameters
    ----------
    image : np.ndarray
        The input image to correct (2D array).
    dark_field : np.ndarray
        The dark field image (2D array).
    bright_field : np.ndarray
        The bright field image (2D array).
    epsilon : float, optional
        Small value to prevent division by zero. Default is 1e-10.

    Returns
    -------
    np.ndarray
        The fully corrected image (float32).

    Examples
    --------
    >>> corrected = flat_field_correction(sample, dark, bright)
    """
    if not (image.shape == dark_field.shape == bright_field.shape):
        raise ValueError(
            f"Shape mismatch: image {image.shape}, dark {dark_field.shape}, bright {bright_field.shape}"
        )

    image_float = image.astype(np.float32)
    dark_float = dark_field.astype(np.float32)
    bright_float = bright_field.astype(np.float32)

    numerator = image_float - dark_float
    denominator = bright_float - dark_float

    # Avoid division by zero
    result = np.zeros_like(image_float)
    mask = np.abs(denominator) > epsilon
    result[mask] = numerator[mask] / denominator[mask]

    return result
