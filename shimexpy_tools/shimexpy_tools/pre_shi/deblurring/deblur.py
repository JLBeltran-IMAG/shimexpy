from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from skimage.restoration import richardson_lucy, unsupervised_wiener


def _gaussian_psf(
    size: int = 21,
    sigma: float = 3.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Create a 2D Gaussian PSF.

    Parameters
    ----------
    size:
        Kernel size. Must be odd.
    sigma:
        Standard deviation of the Gaussian PSF in pixels.
    normalize:
        If True, normalize the PSF so that its sum is 1.

    Returns
    -------
    np.ndarray
        2D Gaussian PSF.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd.")

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    if normalize:
        psf /= psf.sum()

    return psf


def _motion_psf(
    size: int = 21,
    angle: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Create a simple linear motion PSF.

    Parameters
    ----------
    size:
        Kernel size. Must be odd.
    angle:
        Motion angle in degrees.
    normalize:
        If True, normalize the PSF so that its sum is 1.

    Returns
    -------
    np.ndarray
        2D motion PSF.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd.")

    psf = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    psf[center, :] = 1.0

    if angle != 0:
        from scipy.ndimage import rotate

        psf = rotate(psf, angle, reshape=False, order=1)

    if normalize:
        psf /= psf.sum()

    return psf


def _defocus_psf(
    size: int = 21,
    radius: float = 5.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Create a circular defocus PSF.

    Parameters
    ----------
    size:
        Kernel size. Must be odd.
    radius:
        Radius of the defocus disk in pixels.
    normalize:
        If True, normalize the PSF so that its sum is 1.

    Returns
    -------
    np.ndarray
        2D circular defocus PSF.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd.")

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    psf = ((xx**2 + yy**2) <= radius**2).astype(np.float64)

    if normalize:
        psf /= psf.sum()

    return psf


def _wiener_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    balance: float = 0.01,
    clip: bool = False,
) -> np.ndarray:
    """
    Deblur an image using Wiener deconvolution with an assumed PSF.

    Parameters
    ----------
    image:
        Input blurred image.
    psf:
        Assumed point spread function.
    balance:
        Regularization parameter. Larger values suppress noise amplification.
    clip:
        If True, clip output to the input image range.

    Returns
    -------
    np.ndarray
        Deblurred image.
    """
    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)

    psf_padded = np.zeros_like(image)
    psf_shape = psf.shape

    psf_padded[: psf_shape[0], : psf_shape[1]] = psf
    psf_padded = np.fft.ifftshift(psf_padded)

    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf_padded)

    result_fft = (
        np.conj(psf_fft)
        / (np.abs(psf_fft) ** 2 + balance)
        * image_fft
    )

    result = np.real(np.fft.ifft2(result_fft))

    if clip:
        result = np.clip(result, image.min(), image.max())

    return result


def _inverse_filter_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    epsilon: float = 1e-3,
    clip: bool = False,
) -> np.ndarray:
    """
    Deblur an image using regularized inverse filtering.

    Parameters
    ----------
    image:
        Input blurred image.
    psf:
        Assumed point spread function.
    epsilon:
        Small regularization value to avoid division by zero.
    clip:
        If True, clip output to the input image range.

    Returns
    -------
    np.ndarray
        Deblurred image.
    """
    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)

    psf_padded = np.zeros_like(image)
    psf_shape = psf.shape

    psf_padded[: psf_shape[0], : psf_shape[1]] = psf
    psf_padded = np.fft.ifftshift(psf_padded)

    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf_padded)

    result_fft = image_fft / (psf_fft + epsilon)
    result = np.real(np.fft.ifft2(result_fft))

    if clip:
        result = np.clip(result, image.min(), image.max())

    return result


def _richardson_lucy_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    num_iter: int = 20,
    clip: bool = False,
) -> np.ndarray:
    """
    Deblur an image using Richardson-Lucy deconvolution.

    Parameters
    ----------
    image:
        Input blurred image.
    psf:
        Assumed point spread function.
    num_iter:
        Number of Richardson-Lucy iterations.
    clip:
        If True, clip output to the input image range.

    Returns
    -------
    np.ndarray
        Deblurred image.
    """
    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)

    result = richardson_lucy(
        image,
        psf,
        num_iter=num_iter,
        clip=clip,
    )

    return result


def _unsupervised_wiener_deblur(
    image: np.ndarray,
    psf: np.ndarray,
    clip: bool = False,
) -> np.ndarray:
    """
    Deblur an image using unsupervised Wiener deconvolution.

    Parameters
    ----------
    image:
        Input blurred image.
    psf:
        Assumed point spread function.
    clip:
        If True, clip output to the input image range.

    Returns
    -------
    np.ndarray
        Deblurred image.
    """
    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)

    result, _ = unsupervised_wiener(image, psf)

    if clip:
        result = np.clip(result, image.min(), image.max())

    return result


def _blur_image(
    image: np.ndarray,
    psf: np.ndarray,
) -> np.ndarray:
    """
    Blur an image using a given PSF.

    Useful for testing deblurring methods.

    Parameters
    ----------
    image:
        Input image.
    psf:
        Point spread function.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)

    return fftconvolve(image, psf, mode="same")





