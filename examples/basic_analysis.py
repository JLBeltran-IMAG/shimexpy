"""
Example script demonstrating how to use the shimexpy package.
"""
import matplotlib.pyplot as plt
from shimexpy import (
    get_harmonics,
    get_contrasts,
    load_image
)


def main():
    # Load test images
    reference_img = load_image("tests/example_data/flat_roi1.tif")
    sample_img = load_image("tests/example_data/smp_roi1.tif")

    # Set reference image and compute harmonics
    ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
        reference_img, projected_grid=5
    )

    # Calculate all contrasts at once
    absorption_contrast, scattering_contrast, diff_phase_contrast = get_contrasts(
        sample_img,
        (ref_absorption, ref_scattering, ref_diff_phase),
        ref_block_grid
    )

    # Create a single figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot the horizontal phase map
    im0 = axes[0].imshow(absorption_contrast.data, cmap='gray')
    axes[0].set_title('Absorption Contrast')

    # Plot the absorption contrast
    im1 = axes[1].imshow(scattering_contrast.data, cmap='gray')
    axes[1].set_title('Scattering Contrast')

    # Plot the differential phase contrast
    im2 = axes[2].imshow(diff_phase_contrast.data, cmap='gray')
    axes[2].set_title('Differential Phase Contrast')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
