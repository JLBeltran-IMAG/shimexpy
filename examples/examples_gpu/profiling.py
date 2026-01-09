import time
from shimexpy import load_image
from shimexpy_gpu import prepare_reference, process_frame

REF_IMG_PATH = "tests/example_data/flat_roi2.tif"
SAMPLE_IMG_PATH = "tests/example_data/smp_roi2.tif"
PROJECTED_GRID = 5.0


def main():
    ref_img = load_image(REF_IMG_PATH)
    sample_img = load_image(SAMPLE_IMG_PATH)

    t0 = time.time()
    ref = prepare_reference(ref_img, PROJECTED_GRID, store_reference_contrasts=True)
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.3f} seconds")

    t0 = time.time()
    out = process_frame(sample_img, ref)
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.3f} seconds")

if __name__ == "__main__":
    main()
