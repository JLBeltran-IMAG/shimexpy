from tifffile import imread
from cupyx.profiler import benchmark


from shimexpy import (
    get_harmonics,
    get_contrast,
    get_contrasts,
    get_all_contrasts
)


# -----------------------------------------------------------------
reference_img=imread("tests/test_reference.tif")
sample_img = imread("tests/test_sample.tif")


# Set reference image
ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(reference_img, projected_grid=5)


print(
    benchmark(
        get_contrast,
        (sample_img, ref_absorption, ref_block_grid, "absorption"),
        n_repeat=5
    )
)

# print(
#     benchmark(
#         get_contrast,
#         (sample_img, ref_diff_phase, ref_block_grid, "horizontal_phasemap"),
#         n_repeat=5
#     )
# )

# print(
#     benchmark(
#         get_contrast,
#         (sample_img, ref_scattering, ref_block_grid, "horizontal_scattering"),
#         n_repeat=5
#     )
# )

# print(
#     benchmark(
#         get_contrasts,
#         (sample_img, (ref_absorption, ref_scattering, ref_diff_phase), ref_block_grid),
#         n_repeat=5
#     )
# )


# # -------------------------------------------------------
# print(
#     benchmark(
#         shiexecute,
#         (sample_img[0], reference_img, 5),
#         n_repeat=5
#     )
# )
# # -------------------------------------------------------


# contra = get_contrast(
#     sample_img,
#     ref_scattering,
#     ref_block_grid,
#     "horizontal_phasemap"
# )

# # Get contrast
# contra = get_contrast(sample_img, ref_diff_phase, ref_block_grid, "vertical_phasemap")

# import numpy as np
# plt.imshow(np.abs(contra), cmap='gray')
# plt.show()


# r = get_contrasts(sample_img, (ref_absorption, ref_scattering, ref_diff_phase), ref_block_grid)

