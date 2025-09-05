import numpy as np
import heapq
from numpy.fft import fft2, ifft2, fftfreq
from skimage.restoration import unwrap_phase

# -------------------------------------------------
# 0. Skimage unwrapping phase algorithm
# -------------------------------------------------
def skimage_unwrap(
    block: np.ndarray,
    wrap_around: bool = True
) -> np.ndarray:
    angle = np.angle(block)

    unwrap_phase_result = unwrap_phase(angle[0], wrap_around=wrap_around)
    unwrap_phase_result = unwrap_phase_result[np.newaxis, ...]

    return unwrap_phase_result


# -------------------------------------------------
# 1. Branch-Cut Unwrapping (Goldstein's method)
# -------------------------------------------------
def branch_cut_unwrap(
    block: np.ndarray,
    residue_threshold: float = 0.5
) -> np.ndarray:
    """
    Simplified Goldstein Branch-Cut Phase Unwrapping.

    This implementation:
      - Computes a residue map from the wrapped phase via 2x2 loops.
      - Marks pixels with |residue| > residue_threshold as branch cuts.
      - Uses a stack-based flood-fill to propagate unwrapping while avoiding cuts.

    Parameters
    ----------
    wrapped_phase : np.ndarray
        2D array of wrapped phase values in radians.
    residue_threshold : float, optional
        Threshold (in cycles) for marking branch cuts; defaults to 0.5.

    Returns
    -------
    unwrapped : np.ndarray
        2D array of the unwrapped phase.
    """
    if block.ndim == 3:
        wrapped_phase = block[0]
    elif block.ndim == 2:
        wrapped_phase = block
    else:
        raise ValueError("Input block must be 2D or 3D with shape (1, M, N) or (M, N).")

    rows, cols = wrapped_phase.shape

    # 1) Compute the residue map
    # Phase differences around each 2×2 cell
    d1 = np.angle(np.exp(1j * (np.roll(wrapped_phase, -1, axis=1) - wrapped_phase)))
    d2 = np.angle(np.exp(1j * (np.roll(wrapped_phase, (-1, -1), axis=(0,1)) - np.roll(wrapped_phase, -1, axis=1))))
    d3 = np.angle(np.exp(1j * (np.roll(wrapped_phase, -1, axis=0) - np.roll(wrapped_phase, (-1, -1), axis=(0,1)))))
    d4 = np.angle(np.exp(1j * (wrapped_phase - np.roll(wrapped_phase, -1, axis=0))))

    # Residue in cycles
    residues = (d1 + d2 + d3 + d4) / (2 * np.pi)

    # 2) Branch cuts mask
    branch_cut = np.abs(residues) > residue_threshold

    # 3) Prepare unwrapped array and visited mask
    unwrapped = wrapped_phase.copy()
    visited = np.zeros_like(wrapped_phase, dtype=bool)

    # 4) Find a valid start point (not on a branch cut)
    starts = np.argwhere(~branch_cut)
    if starts.size == 0:
        raise RuntimeError("All pixels are branch cuts; cannot start unwrapping.")
    start_i, start_j = starts[0]
    visited[start_i, start_j] = True

    # 5) Flood-fill stack
    stack = [(start_i, start_j)]
    neighbor_offsets = [(-1,0), (1,0), (0,-1), (0,1)]

    while stack:
        i, j = stack.pop()
        base_val = unwrapped[i, j]
        for di, dj in neighbor_offsets:
            ni, nj = i + di, j + dj
            if (
                0 <= ni < rows and 0 <= nj < cols
                and not visited[ni, nj]
                and not branch_cut[ni, nj]
            ):
                # wrap difference into [-π, π]
                diff = wrapped_phase[ni, nj] - wrapped_phase[i, j]
                diff = np.angle(np.exp(1j * diff))
                unwrapped[ni, nj] = base_val + diff
                visited[ni, nj] = True
                stack.append((ni, nj))

    unwrap_phase_result = unwrapped[np.newaxis, ...]

    return unwrap_phase_result


# -----------------------------------------------
# 2. Least-Squares Phase Unwrapping using FFT
# -----------------------------------------------
def ls_unwrap(block: np.ndarray) -> np.ndarray:
    """
    Least-Squares Phase Unwrapping using an FFT-based Poisson solver.
    
    The method computes finite differences in x and y, forms a divergence,
    and then solves the Poisson equation in the Fourier domain.
    
    Parameters:
        wrapped_phase (np.ndarray): 2D array of wrapped phase (radians).
    
    Returns:
        np.ndarray: Unwrapped phase.
    """
    if block.ndim == 3:
        wrapped_phase = block[0]
    elif block.ndim == 2:
        wrapped_phase = block
    else:
        raise ValueError("Input block must be 2D or 3D with shape (1, M, N) or (M, N).")

    M, N = wrapped_phase.shape

    # Compute wrapped finite differences along x and y directions.
    dx = np.zeros_like(wrapped_phase)
    dy = np.zeros_like(wrapped_phase)
    dx[:, :-1] = np.angle(np.exp(1j * (wrapped_phase[:, 1:] - wrapped_phase[:, :-1])))
    dy[:-1, :] = np.angle(np.exp(1j * (wrapped_phase[1:, :] - wrapped_phase[:-1, :])))

    # Compute divergence of the gradient differences.
    div = np.zeros_like(wrapped_phase)

    # For x-direction:
    div[:, 0] = dx[:, 0]
    div[:, 1:-1] = dx[:, 1:-1] - dx[:, :-2]
    div[:, -1] = -dx[:, -2]

    # For y-direction:
    div[0, :] += dy[0, :]
    div[1:-1, :] += dy[1:-1, :] - dy[:-2, :]
    div[-1, :] += -dy[-2, :]

    # Solve the Poisson equation using FFT.
    k1 = fftfreq(M).reshape(-1, 1)
    k2 = fftfreq(N).reshape(1, -1)

    # Discrete Laplacian eigenvalues (using cosine formulation)
    laplacian = (2 * np.cos(2 * np.pi * k1) - 2) + (2 * np.cos(2 * np.pi * k2) - 2)
    laplacian[0, 0] = 1  # Avoid division by zero for the DC term
    div_fft = fft2(div)
    unwrapped_fft = div_fft / laplacian
    unwrapped = np.real(ifft2(unwrapped_fft))

    # Remove arbitrary constant offset
    unwrapped -= unwrapped[0, 0]

    unwrap_phase_result = unwrapped[np.newaxis, ...]

    return unwrap_phase_result


# -----------------------------------------------
# 3. Quality-Guided Phase Unwrapping
# -----------------------------------------------
def quality_guided_unwrap(block: np.ndarray) -> np.ndarray:
    """
    Quality-Guided Phase Unwrapping.
    
    A simple implementation that computes a quality map based on the local gradient
    (the lower the gradient, the higher the quality), then unwraps the phase starting
    from the pixel with the highest quality, propagating to neighbors in a greedy fashion.
    
    Parameters:
        wrapped_phase (np.ndarray): 2D array of wrapped phase (radians).
    
    Returns:
        np.ndarray: Unwrapped phase.
    """
    if block.ndim == 3:
        wrapped_phase = block[0]
    elif block.ndim == 2:
        wrapped_phase = block
    else:
        raise ValueError("Input block must be 2D or 3D with shape (1, M, N) or (M, N).")

    rows, cols = wrapped_phase.shape

    # Compute a simple quality map: higher quality for lower gradient magnitude.
    grad_x = np.gradient(wrapped_phase, axis=1)
    grad_y = np.gradient(wrapped_phase, axis=0)
    quality = 1.0 / (np.abs(grad_x) + np.abs(grad_y) + 1e-6)
    
    unwrapped = np.full_like(wrapped_phase, np.nan)

    # Priority queue: use negative quality to simulate a max-heap.
    heap = []
    start_idx = np.unravel_index(np.argmax(quality), wrapped_phase.shape)
    unwrapped[start_idx] = wrapped_phase[start_idx]
    visited = np.zeros_like(wrapped_phase, dtype=bool)
    visited[start_idx] = True
    heapq.heappush(heap, (-quality[start_idx], start_idx))
    
    while heap:
        neg_q, (i, j) = heapq.heappop(heap)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                continue
            if visited[ni, nj]:
                continue
            diff = wrapped_phase[ni, nj] - wrapped_phase[i, j]
            diff = np.angle(np.exp(1j * diff))
            unwrapped[ni, nj] = unwrapped[i, j] + diff
            visited[ni, nj] = True
            heapq.heappush(heap, (-quality[ni, nj], (ni, nj)))

    unwrap_phase_result = unwrapped[np.newaxis, ...]

    return unwrap_phase_result


# -----------------------------------------------
# 4. Sequential Phase Unwrapping
# -----------------------------------------------
def sequential_np_unwrap(block: np.ndarray) -> np.ndarray:
    """
    Unwraps a 2D phase map sequentially using numpy's unwrap function.
    This method unwraps the phase map first along the rows and then along the columns.

    Parameters
    ----------
    block : np.ndarray
        A 2D numpy array containing the wrapped phase map.

    Returns
    -------
    np.ndarray
        A 2D numpy array containing the unwrapped phase map.
    """
    if block.ndim == 3:
        wrapped_phase = block[0]
    elif block.ndim == 2:
        wrapped_phase = block
    else:
        raise ValueError("Input block must be 2D or 3D with shape (1, M, N) or (M, N).")

    unwrapped = np.unwrap(wrapped_phase, axis=0)
    unwrapped = np.unwrap(unwrapped, axis=1)

    unwrap_phase_result = unwrapped[np.newaxis, ...]

    return unwrap_phase_result

