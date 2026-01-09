import cupy as cp
import cupyx.scipy.fft as cufft
from functools import lru_cache


@lru_cache(maxsize=32)
def _get_freqs(h, w, projected_grid):
    with cp.cuda.Device():
        kx = cp.fft.fftfreq(w, d=1 / projected_grid)
        ky = cp.fft.fftfreq(h, d=1 / projected_grid)
    return kx, ky


def shi_fft_gpu(image, projected_grid=None, shift=True):
    if not isinstance(image, cp.ndarray):
        img = cp.asarray(image, dtype=cp.float32)
    else:
        img = image

    fft = cufft.fft2(img, norm="ortho")
    
    if shift:
        fft = cufft.fftshift(fft)

    if projected_grid is None:
        return fft, None, None

    h, w = image.shape
    kx, ky = _get_freqs(h, w, projected_grid)
    return fft, kx, ky
