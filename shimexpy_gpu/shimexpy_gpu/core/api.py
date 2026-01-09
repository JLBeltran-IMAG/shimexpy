import cupy as cp
from dataclasses import dataclass
from .fft import shi_fft_gpu
from .bands import find_reference_bands, bands_to_array
from .stack import extract_stack, extract_stack_unshifted
from .contrast import compute_contrasts_batch
from .reduce import reduce_outputs


@dataclass
class ReferenceGPU:
    bands_arr: cp.ndarray        # (n, 4)
    y_indices: cp.ndarray        # (n, h_crop)
    x_indices: cp.ndarray        # (n, w_crop)
    order_labels: list[str]
    idx_main: int
    reduce_maps: dict            # indices for directions
    ref_reduced: dict | None     # pre-reduced reference contrasts


def prepare_reference(
    reference_img,
    projected_grid=None,
    limit_band=0.5,
    store_reference_contrasts=True,
    compute_phase: bool = False
):
    # FFT (shifted for band finding)
    fft, kx, ky = shi_fft_gpu(reference_img, projected_grid, shift=True)

    # Find harmonic bands (CPU-like logic, once)
    bands = find_reference_bands(fft, kx, ky, limit_band)

    # Fix harmonic order ONCE
    order_labels = list(bands.keys())
    bands_arr = bands_to_array(bands, order_labels)

    # Identify main harmonic
    idx_main = order_labels.index("harmonic_00")

    # Precompute unshifted indices for fast extraction
    h, w = reference_img.shape
    n = bands_arr.shape[0]
    # Assuming all bands have same size (enforced by find_reference_bands logic)
    t0, b0, l0, r0 = bands_arr[0]
    h_crop = int(b0 - t0)
    w_crop = int(r0 - l0)
    
    y_indices = cp.empty((n, h_crop), dtype=cp.int32)
    x_indices = cp.empty((n, w_crop), dtype=cp.int32)
    
    for i in range(n):
        t, b, l, r = bands_arr[i]
        t, b, l, r = int(t), int(b), int(l), int(r)
        # Shifted indices
        ys = cp.arange(t, b, dtype=cp.int32)
        xs = cp.arange(l, r, dtype=cp.int32)
        
        # Unshifted indices mapping: u = (s + N//2) % N
        yu = (ys + h // 2) % h
        xu = (xs + w // 2) % w
        
        y_indices[i] = yu
        x_indices[i] = xu

    # Precompute reduction maps
    reduce_maps = {
        "horizontal": [
            order_labels.index("harmonic_horizontal_positive"),
            order_labels.index("harmonic_horizontal_negative"),
        ],
        "vertical": [
            order_labels.index("harmonic_vertical_positive"),
            order_labels.index("harmonic_vertical_negative"),
        ],
        "bidirectional": [
            order_labels.index("harmonic_horizontal_positive"),
            order_labels.index("harmonic_horizontal_negative"),
            order_labels.index("harmonic_vertical_positive"),
            order_labels.index("harmonic_vertical_negative"),
        ],
    }

    ref_reduced = None
    if store_reference_contrasts:
        # Use the shifted extraction for reference as we already have shifted FFT
        stack = extract_stack(fft, bands_arr)
        ref_contrasts = compute_contrasts_batch(stack, idx_main, compute_phase=compute_phase)
        # Pre-reduce reference contrasts
        ref_reduced = reduce_outputs(ref_contrasts, reduce_maps, ref_reduced=None)

    return ReferenceGPU(
        bands_arr=bands_arr,
        y_indices=y_indices,
        x_indices=x_indices,
        order_labels=order_labels,
        idx_main=idx_main,
        reduce_maps=reduce_maps,
        ref_reduced=ref_reduced,
    )


def process_frame(sample_img, ref: ReferenceGPU, compute_phase: bool = False):
    # FFT without shift (faster)
    fft, _, _ = shi_fft_gpu(sample_img, projected_grid=None, shift=False)

    # Extract using unshifted indices
    stack = extract_stack_unshifted(fft, ref.y_indices, ref.x_indices)

    contrasts = compute_contrasts_batch(stack, ref.idx_main, compute_phase=compute_phase)

    outputs = reduce_outputs(
        contrasts,
        ref.reduce_maps,
        ref.ref_reduced,
    )

    return outputs



