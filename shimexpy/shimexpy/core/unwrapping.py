"""
Phase unwrapping algorithms.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from skimage.restoration import unwrap_phase


# -------------------------------------------------
# 0. Skimage unwrapping phase algorithm (Best)
# -------------------------------------------------
def skimage_unwrap(
    block: np.ndarray,
    wrap_around: bool = True
) -> np.ndarray:
    """
    Unwrap a wrapped phase map using scikit-image's unwrap_phase and return a (1, M, N) array.

    This function accepts either a 2D wrapped phase array of shape (M, N) or a 3D array
    with a leading singleton channel (1, M, N). If a 3D array is provided, the first
    slice (index 0) is used as the wrapped phase. The wrapped phase may be a complex
    ratio (complex-valued array) or a real-valued array; the function converts it to
    an angular wrapped-phase via numpy.angle before unwrapping.

    Parameters
    ----------
    block : numpy.ndarray
        Input wrapped phase. Expected shapes:
          - (M, N) : 2D wrapped phase matrix.
          - (1, M, N) : 3D array where the first slice contains the wrapped phase.
        Elements may be complex (e.g., a complex ratio) or real; the phase angle
        (in radians) is computed internally.
    wrap_around : bool, optional
        Passed through to skimage.restoration.unwrap_phase to control wrap-around
        handling at the array boundaries. Default is True.

    Returns
    -------
    numpy.ndarray
        Unwrapped phase map with shape (1, M, N) and dtype float. The unwrapped
        values are in radians.

    Raises
    ------
    ValueError
        If `block` does not have 2 or 3 dimensions (i.e., is not (M, N) or (1, M, N)).

    Notes
    -----
    - The function uses numpy.angle to extract the wrapped phase from possibly complex
      inputs, then calls skimage.restoration.unwrap_phase on the full 2D phase map.
    - The returned array always has a leading singleton dimension to preserve a
      consistent (1, M, N) output shape.

    Examples
    --------
    # For a 2D wrapped phase:
    >>> wrapped = np.exp(1j * np.linspace(-np.pi, np.pi, 100).reshape(10,10))
    >>> result = skimage_unwrap(wrapped)
    >>> result.shape
    (1, 10, 10)

    # For an input already shaped (1, M, N):
    >>> wrapped3 = np.expand_dims(np.angle(wrapped), 0)
    >>> result = skimage_unwrap(wrapped3, wrap_around=False)
    """
    # Normalize shape
    if block.ndim == 3:
        wrapped_phase = block[0]   # shape → (M, N)
    elif block.ndim == 2:
        wrapped_phase = block
    else:
        raise ValueError("Input block must be 2D or 3D with shape (1, M, N) or (M, N).")

    # Convert complex ratio → wrapped phase
    angle = np.angle(wrapped_phase)

    # Apply skimage unwrap *on the full 2D matrix*
    unwrap_phase_result = unwrap_phase(angle, wrap_around=wrap_around)

    # Return with shape (1, M, N)
    return unwrap_phase_result[np.newaxis, ...]


def ls_unwrap(block: np.ndarray) -> np.ndarray:
    """Least-squares (LS) phase unwrapping using a Poisson solver in the Fourier domain.

    This function computes an unwrapped phase estimate from a wrapped phase or
    complex-valued field using the least-squares formulation. It is a drop-in,
    optimized replacement for an ls_unwrap_poisson implementation.

    Parameters
    ----------
    block : numpy.ndarray
        Input array containing either:
          - a 2-D array of samples (M, N), interpreted as complex-valued data
            (the phase is taken with numpy.angle) or as already-wrapped phase
            values in radians, or
          - a 3-D array (C, M, N), in which case only the first slice along
            axis 0 (block[0]) is used.
        The function expects finite values (no NaNs or infinities). For complex
        inputs the wrapped phase is computed as angle(block).

    Returns
    -------
    numpy.ndarray
        A 3-D array with shape (1, M, N) containing the unwrapped phase in
        radians. The returned phase has zero mean (the global constant offset
        is removed as part of the Fourier-domain Poisson solution).

    Notes
    -----
    - Method: computes forward wrapped differences, forms their divergence,
      and solves the Poisson equation in the Fourier domain. Wrapped differences
      are computed robustly via angle(exp(1j * delta_phi)) to handle 2*pi jumps.
    - Boundary handling: the implementation assembles a discrete divergence
      consistent with forward differences and enforces a zero-mean solution by
      setting the DC component of the Fourier-domain solution to zero.
    - Complexity: O(M * N * log(M * N)) dominated by the 2-D FFTs.
    - Input requirements: M and N should be >= 2 for sensible differencing;
      inputs must be finite. The algorithm assumes a dense regular grid and
      non-masked data (no explicit handling of invalid/masked pixels).

    References
    ----------
    - Ghiglia, D. C., & Pritt, M. D. (1998). Two-Dimensional Phase Unwrapping:
      Theory, Algorithms, and Software. (for background on LS unwrapping and
      Poisson-based formulations).
    """
    # Normalize dimensions
    z = block[0] if block.ndim == 3 else block
    phiw = np.angle(z)
    M, N = phiw.shape

    # Compute wrapped differences (dx, dy) — vectorized and minimal temporaries
    dx = np.zeros_like(phiw)
    dy = np.zeros_like(phiw)

    # forward differences
    diff_x = phiw[:, 1:] - phiw[:, :-1]
    diff_y = phiw[1:, :] - phiw[:-1, :]

    # wrapped differences (vectorized)
    dx[:, :-1] = np.angle(np.exp(1j * diff_x))
    dy[:-1, :] = np.angle(np.exp(1j * diff_y))

    # Divergence (fully vectorized)
    div = np.zeros_like(phiw)
    div[:, 0] = dx[:, 0]
    div[:, 1:-1] = dx[:, 1:-1] - dx[:, :-2]
    div[:, -1] = -dx[:, -2]

    div[0, :] += dy[0, :]
    div[1:-1, :] += dy[1:-1, :] - dy[:-2, :]
    div[-1, :] += -dy[-2, :]

    # Poisson solver in Fourier domain
    ky = fftfreq(M)[:, None]
    kx = fftfreq(N)[None, :]

    denom = (2 * np.pi)**2 * (kx**2 + ky**2)
    denom[0, 0] = 1.0

    div_fft = fft2(div)
    phi_fft = div_fft / denom
    phi_fft[0, 0] = 0.0

    phi = np.real(ifft2(phi_fft))
    return phi[np.newaxis, ...]


# # -------------------------------------------------
# # New implementations
# # -------------------------------------------------
