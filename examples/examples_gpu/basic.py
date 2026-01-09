import cupy as cp
from shimexpy_gpu import prepare_reference, process_frame
from shimexpy import load_image


# Imagen fake o real pequeña
img = load_image("tests/example_data/smp_roi2.tif")
ref_img = load_image("tests/example_data/flat_roi2.tif")

ref = prepare_reference(
    reference_img=ref_img,
    projected_grid=5.0,
)

out = process_frame(img, ref)

import matplotlib.pyplot as plt

plt.imshow(cp.asnumpy(out["absorption"]), cmap='gray')
plt.colorbar()
plt.title("Absorption Contrast")
plt.show()
