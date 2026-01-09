import numpy as np
import cupy as cp
import imageio.v3 as iio

# ===== CPU backend (the trusted one) =====
from shimexpy import get_harmonics, get_contrasts

# ===== GPU-native backend (new) =====
from shimexpy_gpu import prepare_reference, process_frame

# ---------------- CONFIG ----------------
REF_IMG_PATH = "tests/example_data/flat_roi2.tif"
SAMPLE_IMG_PATH = "tests/example_data/smp_roi2.tif"
PROJECTED_GRID = 5.0
UNWRAP = None
RTOL = 5e-4
ATOL = 2e-4
# ----------------------------------------


def main():
    ref_img = iio.imread(REF_IMG_PATH).astype(np.float32)
    sample_img = iio.imread(SAMPLE_IMG_PATH).astype(np.float32)

    # ---------- CPU (gold) ----------
    # get_harmonics returns: (ref_abs, ref_scat_dict, ref_phase_dict, ref_block_grid)
    ref_abs, ref_scat, ref_phase, ref_block = get_harmonics(
        ref_img, PROJECTED_GRID, unwrap=UNWRAP
    )

    # get_contrasts returns: (absorption_contrast, scattering_contrast, diff_phase_contrast)
    abs_cpu, scat_cpu, _ = get_contrasts(
        sample_img, (ref_abs, ref_scat, ref_phase), ref_block, unwrap=UNWRAP
    )

    # Ensure CPU outputs are cupy arrays for comparison
    abs_cpu = cp.asarray(abs_cpu)
    scat_cpu = cp.asarray(scat_cpu)

    # ---------- GPU-native (new) ----------
    ref = prepare_reference(ref_img, PROJECTED_GRID, store_reference_contrasts=True)
    out_new = process_frame(sample_img, ref)

    # ---------- Compare ----------
    diff = out_new["absorption"] - abs_cpu

    print("max |Δ|:", float(cp.max(cp.abs(diff))))
    print("mean |Δ|:", float(cp.mean(cp.abs(diff))))
    print("L2 relative:",
        float(cp.linalg.norm(diff) / cp.linalg.norm(abs_cpu)))



if __name__ == "__main__":
    main()
