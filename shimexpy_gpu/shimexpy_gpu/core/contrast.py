# contrast.py
# GPU-accelerated contrast computation for ShimexPy


import cupy as cp
import cupyx.scipy.fft as cufft


# Fused kernel for contrast computation
# Computes scattering and phase in one go
_contrast_kernel = cp.ElementwiseKernel(
    'complex64 val, complex64 main, float32 eps',
    'float32 scat, float32 phase',
    '''
    complex<float> ratio = val / (main + eps);
    float abs_ratio = abs(ratio);
    scat = -log(max(abs_ratio, eps));
    phase = arg(ratio);
    ''',
    'contrast_kernel'
)
_scattering_kernel = cp.ElementwiseKernel(
    'complex64 val, complex64 main, float32 eps',
    'float32 scat',
    '''
    complex<float> ratio = val / (main + eps);
    float abs_ratio = abs(ratio);
    scat = -log(max(abs_ratio, eps));
    ''',
    'scattering_kernel'
)


def compute_contrasts_batch(stack: cp.ndarray, idx_main: int, eps=1e-12, compute_phase: bool = False):
    eps = cp.float32(eps)

    # Batch IFFT
    ifft_stack = cufft.ifft2(stack, axes=(-2, -1))
    main = ifft_stack[idx_main]

    # main is (H, W), stack is (N, H, W)
    # We don't need to manually broadcast main_b if we rely on ElementwiseKernel broadcasting,
    # but we need to ensure shapes are compatible.
    # ElementwiseKernel broadcasts automatically.

    # Compute absorption separately (it's just on main)
    # absorption = -log(abs(main) + eps)
    absorption = cp.abs(main)
    absorption += eps
    cp.log(absorption, out=absorption)
    absorption *= -1

    # Compute scattering and phase using fused kernel
    # We need to pass 'main' broadcasted or let cupy handle it.
    # ifft_stack: (N, H, W)
    # main: (H, W) -> broadcasts to (N, H, W)

    scattering = cp.empty_like(ifft_stack, dtype=cp.float32)

    if compute_phase:
        phase_wrapped = cp.empty_like(ifft_stack, dtype=cp.float32)
        _contrast_kernel(ifft_stack, main, eps, scattering, phase_wrapped)
    else:
        _scattering_kernel(ifft_stack, main, eps, scattering)
        phase_wrapped = None

    return {
        "absorption": absorption,
        "scattering": scattering,
        "phase_wrapped": phase_wrapped,
    }