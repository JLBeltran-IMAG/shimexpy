# stack.py
# GPU-accelerated stack extraction for ShimexPy



import cupy as cp


def extract_stack(fft: cp.ndarray, bands_arr: cp.ndarray) -> cp.ndarray:
    """
    bands_arr: (n, 4) -> (top, bottom, left, right)
    """
    n = bands_arr.shape[0]
    t0, b0, l0, r0 = bands_arr[0]
    bh = int(b0 - t0)
    bw = int(r0 - l0)

    stack = cp.empty((n, bh, bw), dtype=fft.dtype)

    for i in range(n):
        t, b, l, r = bands_arr[i]
        stack[i] = fft[t:b, l:r]

    return stack


def extract_stack_unshifted(
    fft: cp.ndarray,
    y_indices: cp.ndarray,
    x_indices: cp.ndarray
) -> cp.ndarray:
    """
    Extract crops from unshifted FFT using precomputed indices.

    fft: (H, W) unshifted complex array
    y_indices: (N, H_crop) int32 array of y coordinates
    x_indices: (N, W_crop) int32 array of x coordinates
    """
    n, h_crop = y_indices.shape
    _, w_crop = x_indices.shape

    stack = cp.empty((n, h_crop, w_crop), dtype=fft.dtype)

    for i in range(n):
        stack[i] = fft[cp.ix_(y_indices[i], x_indices[i])]

    return stack
