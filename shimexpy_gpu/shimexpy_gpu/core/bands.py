import cupy as cp


def find_reference_bands(fft, kx, ky, limit_band=0.5):
    abs_fft = cp.abs(fft)

    max_idx = cp.argmax(abs_fft)
    h0, w0 = cp.unravel_index(max_idx, abs_fft.shape)
    h0, w0 = int(h0), int(w0)

    ky_lim = int(cp.argmin(cp.abs(ky - limit_band)))
    kx_lim = int(cp.argmin(cp.abs(kx - limit_band)))

    bands = {}

    def zero_region(arr, t, b, l, r):
        arr[t:b, l:r].fill(0)

    # main harmonic
    t = h0 - ky_lim
    b = h0 + ky_lim
    l = w0 - kx_lim
    r = w0 + kx_lim

    bands["harmonic_00"] = (t, b, l, r)
    zero_region(abs_fft, t, b, l, r)

    for _ in range(8):
        idx = cp.argmax(abs_fft)
        h, w = cp.unravel_index(idx, abs_fft.shape)
        h, w = int(h), int(w)

        t = h - ky_lim
        b = h + ky_lim
        l = w - kx_lim
        r = w + kx_lim

        dy = h - h0
        dx = w - w0

        if abs(dy) > abs(dx):
            label = (
                "harmonic_vertical_positive"
                if dy > 0 else
                "harmonic_vertical_negative"
            )
        elif abs(dx) > abs(dy):
            label = (
                "harmonic_horizontal_positive"
                if dx > 0 else
                "harmonic_horizontal_negative"
            )
        else:
            label = "harmonic_diagonal"

        bands[label] = (t, b, l, r)
        zero_region(abs_fft, t, b, l, r)

    return bands


def bands_to_array(bands: dict, order: list[str]) -> cp.ndarray:
    arr = cp.zeros((len(order), 4), dtype=cp.int32)
    for i, k in enumerate(order):
        arr[i] = cp.asarray(bands[k], dtype=cp.int32)
    return arr
