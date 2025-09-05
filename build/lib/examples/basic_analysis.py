#!/usr/bin/env python3
"""
Example script demonstrating how to use the shimexpy package for spatial harmonics analysis.
"""

from tifffile import imread
from cupyx.profiler import benchmark
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shimexpy import (
    get_harmonics,
    get_contrast,
    get_contrasts,
    get_all_contrasts
)


def main():
    # Load test images
    reference_img = imread("tests/test_reference.tif")
    sample_img = imread("tests/test_sample.tif")

    # Set reference image and compute harmonics
    ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(reference_img, projected_grid=5)

    # Benchmark contrast computation
    print("Benchmarking absorption contrast computation:")
    print(
        benchmark(
            get_contrast,
            (sample_img, ref_absorption, ref_block_grid, "absorption"),
            n_repeat=5
        )
    )

    # Calculate horizontal phase map contrast
    horizontal_phase = get_contrast(
        sample_img, 
        ref_diff_phase, 
        ref_block_grid, 
        "horizontal_phasemap"
    )

    # Calculate all contrasts at once
    print("\nCalculating all contrasts at once...")
    absorption_contrast, scattering_contrast, diff_phase_contrast = get_all_contrasts(
        sample_img, reference_img, projected_grid=5
    )

    # Print shape information to debug
    print("Absorption contrast shape:", absorption_contrast.shape)
    print("Scattering contrast shape:", scattering_contrast.shape)
    print("Differential phase contrast shape:", diff_phase_contrast.shape)
    
    # Create a single figure with multiple subplots
    print("\nGenerando visualización...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot the horizontal phase map
    im0 = axes[0].imshow(horizontal_phase, cmap='gray')
    axes[0].set_title('Horizontal Phase Contrast')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot the absorption contrast
    im1 = axes[1].imshow(absorption_contrast, cmap='gray')
    axes[1].set_title('Absorption Contrast')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    # Guardar la figura en un archivo
    output_path = "contrast_visualization.png"
    plt.savefig(output_path)
    print(f"Figura guardada en: {output_path}")
    plt.show()  # Mantener esta línea para mostrar la figura si hay un entorno gráfico


if __name__ == "__main__":
    main()
