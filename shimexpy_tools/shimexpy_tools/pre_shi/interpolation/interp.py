"""
Local 2D Interpolation for Detector Defects

This module provides local two-dimensional interpolation methods for correcting missing or defective detector pixels in X-ray images. The methods aim to reconstruct missing information while preserving spatial structures and periodic intensity modulations characteristic of Spatial Harmonic Imaging (SHI).

Four interpolation methods are available: linear, polynomial, spline, and periodic interpolation.

All methods follow the same conventions. Missing or defective pixels are identified by a Boolean mask, where True indicates pixels requiring interpolation. Neighborhoods are constructed exclusively from original, unmodified image values, excluding all masked pixels to prevent error propagation.

For periodic interpolation, spatial frequencies must be expressed in cycles per image pixel, following the coordinate order [fx, fy].

The periodic interpolation method is particularly relevant to SHI, where preserving spatial harmonic information is essential for accurate multicontrast reconstruction.
"""

from functools import lru_cache, partial
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.special import xlogy
from shimexpy import shi_fft, get_fourier_harmonics


RealArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]
ComplexArray = NDArray[np.complexfloating[Any, Any]]
BlockGrid = Mapping[str, Sequence[int | np.integer[Any]]]

Prediction = Callable[[RealArray], np.floating[Any]]

ModelFactory = Callable[
    [RealArray,
     RealArray,
     BoolArray],
     Prediction
]


def _validate_inputs(
    image: RealArray,
    mask: BoolArray,
    projected_period: float,
    window_periods: float,
) -> tuple[RealArray, BoolArray, int]:
    """Validate shared inputs before any fitting or spectral processing."""
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("image and mask must be 2D arrays of the same shape")

    if not np.isfinite(image[~mask]).all():
        raise ValueError("unmasked pixels must be finite")

    if not np.isfinite(projected_period) or projected_period <= 0:
        raise ValueError("projected_period must be positive and finite")

    if not np.isfinite(window_periods) or window_periods <= 0:
        raise ValueError("window_periods must be positive and finite")

    half_width = window_periods * projected_period / 2

    if not np.isfinite(half_width) or half_width > min(image.shape) / 2:
        raise ValueError("interpolation window is larger than the image")

    radius = int(np.ceil(half_width))
    size = 2 * radius + 1
    height, width = image.shape

    if height < size or width < size:
        raise ValueError("interpolation window is larger than the image")

    return image, mask, size


def _interpolate_local(
    image: RealArray,
    mask: BoolArray,
    projected_period: float,
    window_periods: float,
    make_model: ModelFactory,
) -> RealArray:
    """Apply local prediction models to masked pixels using original data.

    ``make_model`` receives relative x/y grids and a valid-neighbor mask,
    and returns a callable predicting the origin from valid pixel values.
    Models are cached by relative window position and mask pattern within
    this call. Edge windows shift inward to keep their full size.

    Returns an independent array with the input dtype. Raises ValueError for mismatched
    dimensions, non-finite valid pixels, invalid window parameters, or a
    window larger than the image. Fit errors include the defective pixel
    and neighborhood context.
    """
    image, mask, size = _validate_inputs(image, mask, projected_period, window_periods)

    radius = size // 2
    height, width = image.shape

    result = image.copy()

    @lru_cache(maxsize=4096)
    def prepare(
        start_u: int,
        start_v: int,
        valid_bytes: bytes
    ) -> Prediction:
        """Build or retrieve the model for one relative neighborhood."""

        valid = np.frombuffer(valid_bytes, dtype=np.bool_).reshape(size, size)

        v, u = np.meshgrid(
            np.arange(start_v, start_v + size),
            np.arange(start_u, start_u + size),
            indexing="ij"
        )

        return make_model(u.astype(np.float32), v.astype(np.float32), valid)

    for y, x in np.argwhere(mask):
        start_y = min(max(y - radius, 0), height - size)
        start_x = min(max(x - radius, 0), width - size)

        rows = slice(start_y, start_y + size)
        cols = slice(start_x, start_x + size)

        valid = ~mask[rows, cols]

        try:
            model = prepare(start_x - x, start_y - y, valid.tobytes())
            prediction = model(image[rows, cols][valid])

            if not np.isfinite(prediction) or abs(prediction) > np.finfo(np.float32).max:
                raise ValueError("prediction is not finite or exceeds float32 range")

            result[y, x] = prediction

        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(
                f"interpolation failed at (y={y}, x={x}); window={size}x{size}, "
                f"valid neighbors={np.count_nonzero(valid)}: {exc}"
            ) from exc

    return result


def _validate_design(design: RealArray | ComplexArray, model: str) -> None:
    """Require enough independent observations for the model coefficients."""

    if len(design) < design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(f"insufficient independent neighbors for {model} model")


def _polynomial_model(
    u: RealArray,
    v: RealArray,
    valid: BoolArray,
    degree: int,
) -> Prediction:
    """Build origin-prediction weights for a total-degree polynomial fit."""

    coefficients = (degree + 1) * (degree + 2) // 2

    if np.count_nonzero(valid) < coefficients:
        raise ValueError(f"polynomial degree {degree} requires {coefficients} neighbors")

    scale = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), 1.0)
    u, v = u / scale, v / scale

    # Total polynomial degree: 1, u, v, u², uv, v², ...
    powers = [
        (total - j, j)
        for total in range(degree + 1)
        for j in range(total + 1)
    ]

    design = np.stack([(u ** i) * (v ** j) for i, j in powers], axis=-1)
    fit = design[valid]

    _validate_design(fit, "polynomial")

    target = np.zeros(len(powers))
    target[0] = 1

    weights = target @ np.linalg.pinv(fit)

    return lambda values: cast(np.floating[Any], weights @ values)


def linear_2d(
    image: RealArray,
    mask: BoolArray,
    projected_period: float,
    window_periods: float = 3
) -> RealArray:
    """Fit a local plane to valid neighbors.

    Parameters
    ----------
    image : RealArray
        Two-dimensional detector image. Unmasked pixels must be finite.
    mask : BoolArray
        Same shape as image; True marks defective pixels to replace.
    projected_period : float
        Positive finite period in pixels, used only to choose window size.
    window_periods : float, default 3
        Positive finite window width in periods. The square window has radius
        ceil(window_periods * projected_period / 2) and must fit in the image.

    Returns
    -------
    RealArray
        Independent image with the input dtype. Unmasked values are preserved.
        Predictions exclude all masked pixels and use original
        input values, including when correcting adjacent defective pixels.

    Raises
    ------
    ValueError
        For invalid inputs, an oversized window, or insufficient independent
        neighbors. At least three independent observations are needed for the plane.
    """
    return _interpolate_local(
        image,
        mask,
        projected_period,
        window_periods,
        partial(_polynomial_model, degree=1)
    )


def polynomial_2d(
    image: RealArray,
    mask: BoolArray,
    projected_period: float,
    degree: int = 2,
    window_periods: float = 3
) -> RealArray:
    """Fit a local polynomial of total degree ``degree``.

    Parameters
    ----------
    image : RealArray
        Two-dimensional detector image. Unmasked pixels must be finite.
    mask : BoolArray
        Same shape as image; True marks defective pixels to replace.
    projected_period : float
        Positive finite period in pixels, used only to choose window size.
    window_periods : float, default 3
        Positive finite window width in periods. The square window has radius
        ceil(window_periods * projected_period / 2) and must fit in the image.
    degree : int, default 2
        Non-negative total polynomial degree.

    Returns
    -------
    RealArray
        Independent image with the input dtype. Unmasked values are preserved.
        Predictions exclude all masked pixels and use original
        input values, including when correcting adjacent defective pixels.

    Raises
    ------
    ValueError
        For invalid inputs, an oversized window, or insufficient independent
        neighbors. The polynomial design matrix must have full column rank.
    """
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a non-negative integer")

    return _interpolate_local(
        image,
        mask,
        projected_period,
        window_periods,
        partial(_polynomial_model, degree=degree)
    )


def spline_2d(
    image: RealArray,
    mask: BoolArray,
    projected_period: float,
    window_periods: float = 3,
    smoothing: float = 0.0
) -> RealArray:
    """Fit a local thin-plate spline RBF to valid neighbors.

    Parameters
    ----------
    image : RealArray
        Two-dimensional detector image. Unmasked pixels must be finite.
    mask : BoolArray
        Same shape as image; True marks defective pixels to replace.
    projected_period : float
        Positive finite period in pixels, used only to choose window size.
    window_periods : float, default 3
        Positive finite window width in periods. The square window has radius
        ceil(window_periods * projected_period / 2) and must fit in the image.
    smoothing : float, default 0.0
        Finite non-negative regularization. Zero interpolates valid samples
        exactly; positive values smooth the fit.

    Returns
    -------
    RealArray
        Independent image with the input dtype. Unmasked values are preserved.
        Predictions exclude all masked pixels and use original
        input values, including when correcting adjacent defective pixels.

    Raises
    ------
    ValueError
        For invalid inputs, an oversized window, or insufficient independent
        neighbors. Each window needs at least three non-collinear valid neighbors.
    """
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and non-negative")

    def make_model(
        u: RealArray,
        v: RealArray,
        valid: BoolArray
    ) -> Prediction:
        points = np.column_stack((u[valid], v[valid]))

        if len(points) < 3 or np.linalg.matrix_rank(
            np.column_stack((np.ones(len(points)), points))
        ) < 3:
            raise ValueError("spline requires at least three non-collinear neighbors")

        # Solve the transposed RBF system once to predict only the origin.
        # phi(r) = r² log(r), with phi(0) = 0.
        distances = cdist(points, points)
        kernel = xlogy(distances ** 2, distances)
        kernel.flat[::len(points) + 1] += smoothing

        affine = np.column_stack((np.ones(len(points)), points))

        system = np.block([
            [kernel, affine],
            [affine.T, np.zeros((3, 3))]
        ])

        origin_distances = np.linalg.norm(points, axis=1)

        target = np.r_[
            xlogy(origin_distances ** 2, origin_distances),
            1., 0., 0.
        ]

        weights = np.linalg.solve(system.T, target)[:len(points)]

        return lambda values: cast(np.floating[Any], weights @ values)

    return _interpolate_local(image, mask, projected_period, window_periods, make_model)


def periodic_2d(
    image: RealArray,
    mask: BoolArray,
    projected_period: float,
    window_periods: float = 3,
    *,
    carriers: RealArray | None = None,
) -> RealArray:
    """Fit a local periodic model with affine envelopes.

    Optional keyword-only ``carriers`` supplies frequencies with shape (n, 2)
    in cycles/pixel, ordered [fx, fy], preferably from a clean reference.
    Otherwise frequencies are estimated after nearest-neighbor filling of
    defective pixels. This provisional fill is used only for frequency detection.
    Conjugate frequency pairs are completed automatically.

    Parameters
    ----------
    image : RealArray
        Two-dimensional detector image. Unmasked pixels must be finite.
    mask : BoolArray
        Same shape as image; True marks defective pixels to replace.
    projected_period : float
        Positive finite period in pixels, used for window size and automatic
        frequency detection.
    window_periods : float, default 3
        Positive finite window width in periods. The square window has radius
        ceil(window_periods * projected_period / 2) and must fit in the image.
    carriers : ndarray of floating-point values, optional
        Frequencies of shape (n_harmonics, 2), ordered [fx, fy], in cycles/pixel.

    Returns
    -------
    RealArray
        Independent image with the input dtype. Unmasked values are preserved.
        Predictions exclude all masked pixels and use original
        input values, including when correcting adjacent defective pixels.

    Raises
    ------
    ValueError
        For invalid inputs, an oversized window, or insufficient independent
        neighbors. The periodic design must have full column rank and real prediction weights.
    """
    image, mask, _ = _validate_inputs(image, mask, projected_period, window_periods)

    if not mask.any():
        return image.copy()
    if mask.all():
        raise ValueError("periodic interpolation requires unmasked pixels")

    if carriers is None:
        # Fill only the image used for frequency detection. Local fits always
        # use original, unmasked observations.
        nearest = distance_transform_edt(
            mask,
            return_distances=False,
            return_indices=True
        )

        detection_image = image[tuple(nearest)]

        fft_result = shi_fft(
            detection_image,
            projected_grid=projected_period,
            logspect=False
        )

        _, block_grid = get_fourier_harmonics(
            fft_result.fft,
            ky=fft_result.ky,
            kx=fft_result.kx,
            reference=True
        )

        carriers = _estimate_carriers(block_grid, fft_result.fft)

    if (
        carriers.ndim != 2
        or carriers.shape[1] != 2
        or not len(carriers)
        or not np.isfinite(carriers).all()
    ):
        raise ValueError("carriers must have shape (n_harmonics, 2) and be finite")

    # Frequencies differing by integer cycles/pixel are equivalent.
    # Complete conjugate pairs explicitly so the prediction weights are real.
    carriers = (carriers + 0.5) % 1.0 - 0.5
    carriers = np.unique(
        np.round(np.vstack((carriers, (-carriers + 0.5) % 1.0 - 0.5)), 14),
        axis=0
    )

    def make_model(
        u: RealArray,
        v: RealArray,
        valid: BoolArray
    ) -> Prediction:

        positions = np.stack((u, v), axis=-1)
        waves = np.exp(2j * np.pi * (positions @ carriers.T))

        design = np.concatenate(
            (waves, u[..., None] * waves, v[..., None] * waves),
            axis=-1
        )

        fit = design[valid]

        _validate_design(fit, "periodic")

        target = np.r_[np.ones(len(carriers)), np.zeros(2 * len(carriers))]
        weights = target @ np.linalg.pinv(fit)

        if np.max(np.abs(weights.imag)) > 1e-8:
            raise ValueError(
                "periodic prediction weights are not approximately real;" \
                "check frequency pairs"
            )

        weights = weights.real

        return lambda values: cast(np.floating[Any], weights @ values)

    return _interpolate_local(image, mask, projected_period, window_periods, make_model)


# Utils
def _estimate_carriers(
    block_grid: BlockGrid,
    spectrum: ComplexArray
) -> RealArray:
    """
    """

    height, width = spectrum.shape

    kx = np.fft.fftshift(np.fft.fftfreq(width))
    ky = np.fft.fftshift(np.fft.fftfreq(height))

    carriers = []

    for top, bottom, left, right in block_grid.values():
        top, bottom = max(0, int(top)), min(height, int(bottom))
        left, right = max(0, int(left)), min(width, int(right))

        if top >= bottom or left >= right:
            raise ValueError("harmonic block is empty or outside the spectrum")

        dy, dx = np.unravel_index(
            np.argmax(np.abs(spectrum[top:bottom, left:right])),
            (bottom - top, right - left),
        )

        iy, ix = top + dy, left + dx
        carriers.append([kx[ix], ky[iy]])

    return np.asarray(carriers, dtype=np.float32)


