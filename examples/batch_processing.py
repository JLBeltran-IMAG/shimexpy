#!/usr/bin/env python3
"""
Script to demonstrate batch processing of multiple images using shimexpy.
"""

import os
from pathlib import Path
import sys
from tifffile import imread
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from shimexpy import get_harmonics, get_contrast
from shimexpy.io.file_io import save_block_grid
from tifffile import imwrite  # Usaremos directamente imwrite en lugar de save_image


def process_sample(
    sample_path,
    reference,
    ref_block_grid,
    output_dir,
    contrast_type="absorption"
) -> tuple[str, bool]:
    """Process a single sample image and save the result."""
    try:
        # Load the sample image
        sample_img = imread(sample_path)

        # Get the contrast
        contrast = get_contrast(sample_img, reference, ref_block_grid, contrast_type)

        # Save the result
        sample_name = Path(sample_path).stem
        output_path = Path(output_dir) / f"{sample_name}_{contrast_type}_contrast.tif"

        # Crear el directorio de salida si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Guardar con imwrite directamente
        # Si contrast es un xarray, convertirlo a numpy
        if hasattr(contrast, 'values'):
            contrast = contrast.values
        imwrite(str(output_path), contrast)

        return sample_path, True
    except Exception as e:
        print(f"Error processing {sample_path}: {str(e)}")
        return sample_path, False


def batch_process_samples(
    reference_path,
    sample_dir,
    output_dir,
    contrast_type="absorption",
    n_workers=None
) -> None:
    """Process multiple sample images sequentially to avoid CUDA initialization issues."""
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process the reference image
    reference_img = imread(reference_path)
    ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(reference_img, projected_grid=5)

    # Save the reference block grid for future use
    save_block_grid(ref_block_grid, output_dir / "reference_block_grid.json")

    # Determine which reference to use based on contrast type
    if contrast_type == "absorption":
        reference = ref_absorption
    elif "scattering" in contrast_type:
        reference = ref_scattering
    elif "phasemap" in contrast_type:
        reference = ref_diff_phase
    else:
        raise ValueError(f"Unknown contrast type: {contrast_type}")

    # Find all sample images in the directory
    sample_dir = Path(sample_dir)
    sample_paths = list(sample_dir.glob("*.tif"))

    if not sample_paths:
        print(f"No .tif files found in {sample_dir}")
        return

    print(f"Processing {len(sample_paths)} sample images...")

    # Process all samples sequentially to avoid CUDA initialization issues
    results = []
    for path in sample_paths:
        result = process_sample(str(path), reference, ref_block_grid, output_dir, contrast_type)
        results.append(result)

    # Report results
    success = sum(1 for _, success in results if success)
    print(f"Successfully processed {success} out of {len(sample_paths)} images")

    if success > 0:
        print(f"Results saved to {output_dir}")


def main():
    # Configuration
    reference_path = "tests/test_reference.tif"  # Path to reference image
    sample_dir = "tests"                         # Directory containing sample images
    output_dir = "results"                       # Directory to save results
    contrast_type = "horizontal_phasemap"        # Type of contrast to compute

    # Process the samples
    batch_process_samples(reference_path, sample_dir, output_dir, contrast_type)


if __name__ == "__main__":
    main()
