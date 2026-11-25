# contrast.py
# GPU-accelerated contrast computation for ShimexPy

import cupy as cp
import cupyx.scipy.fft as cufft


# Fused kernel for scattering + DPC
_contrast_kernel = cp.ElementwiseKernel(
    in_params='complex64 harmonic, complex64 dc, float32 eps',
    out_params='float32 scat, float32 phase',
    operation=r'''
    complex<float> ratio = harmonic / (dc + eps);
    float abs_ratio = abs(ratio);
    scat = -log(max(abs_ratio, eps));
    phase = arg(ratio);
    ''',
    name='contrast_kernel',
)

_dpc_kernel = cp.ElementwiseKernel(
    in_params='complex64 harmonic, complex64 dc, float32 eps',
    out_params='float32 phase',
    operation=r'''
    complex<float> ratio = harmonic / (dc + eps);
    phase = arg(ratio);
    ''',
    name='dpc_kernel',
)

_scattering_kernel = cp.ElementwiseKernel(
    in_params='complex64 harmonic, complex64 dc, float32 eps',
    out_params='float32 scat',
    operation=r'''
    complex<float> ratio = harmonic / (dc + eps);
    float abs_ratio = abs(ratio);
    scat = -log(max(abs_ratio, eps));
    ''',
    name='scattering_kernel',
)


def compute_contrasts_batch(
    stack: cp.ndarray,
    idx_main: int,
    eps: float = 1e-12,
    only_phase: bool = False,
    only_scattering: bool = False,
) -> dict[str, cp.ndarray | None]:
    """
    Compute absorption, scattering, and wrapped DPC from a batch of
    Fourier-domain harmonic bands.

    Parameters
    ----------
    stack : cp.ndarray
        Complex array of shape (N, H, W) containing harmonic bands.
    idx_main : int
        Index of the DC/main harmonic along axis 0.
    eps : float, optional
        Numerical stabilization constant.
    only_phase : bool, optional
        If True, compute only wrapped DPC.
    only_scattering : bool, optional
        If True, compute only scattering.

    Returns
    -------
    dict
        Dictionary with:
        - "absorption": cp.ndarray of shape (H, W), dtype float32
        - "scattering": cp.ndarray of shape (N, H, W), dtype float32, or None
        - "dpc_wrapped": cp.ndarray of shape (N, H, W), dtype float32, or None
    """
    if stack.ndim != 3:
        raise ValueError(
            f"`stack` must have shape (N, H, W), but got {stack.shape}."
        )

    if not cp.iscomplexobj(stack):
        raise TypeError("`stack` must be a complex-valued CuPy array.")

    n_harmonics = stack.shape[0]
    if not (-n_harmonics <= idx_main < n_harmonics):
        raise IndexError(
            f"`idx_main`={idx_main} is out of bounds for stack with "
            f"{n_harmonics} harmonics."
        )

    if only_phase and only_scattering:
        raise ValueError(
            "`only_phase` and `only_scattering` cannot both be True."
        )

    if stack.dtype != cp.complex64:
        stack = stack.astype(cp.complex64, copy=False)

    eps = cp.float32(eps)

    # Batched inverse FFT in spatial domain
    ifft_stack = cufft.ifft2(stack, axes=(-2, -1))
    dc = ifft_stack[idx_main]

    # Absorption from the DC/main harmonic
    absorption = cp.abs(dc).astype(cp.float32, copy=False)
    absorption += eps
    cp.log(absorption, out=absorption)
    absorption *= -1.0

    scattering = None
    dpc_wrapped = None

    if only_phase:
        dpc_wrapped = cp.empty(ifft_stack.shape, dtype=cp.float32)
        _dpc_kernel(ifft_stack, dc, eps, dpc_wrapped)

    elif only_scattering:
        scattering = cp.empty(ifft_stack.shape, dtype=cp.float32)
        _scattering_kernel(ifft_stack, dc, eps, scattering)

    else:
        scattering = cp.empty(ifft_stack.shape, dtype=cp.float32)
        dpc_wrapped = cp.empty(ifft_stack.shape, dtype=cp.float32)
        _contrast_kernel(ifft_stack, dc, eps, scattering, dpc_wrapped)

    return {
        "absorption": absorption,
        "scattering": scattering,
        "dpc_wrapped": dpc_wrapped,
    }


