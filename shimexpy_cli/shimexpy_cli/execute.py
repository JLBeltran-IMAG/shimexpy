import numpy as np
import xarray as xr
import tifffile
from pathlib import Path

from .logging import logger
from shimexpy import get_harmonics, get_all_harmonic_contrasts, load_image, save_image


def cli_export(
    result: xr.DataArray,
    output_path: Path
) -> Path:
    """
    Export all contrasts as one multipage TIFF per contrast.

    result dims:
        ('image', 'contrast', 'y', 'x')

    Output:
        output_path/
            absorption.tif
            scattering_horizontal.tif
            diff_phase_horizontal.tif
            scattering_vertical.tif
            diff_phase_vertical.tif
            scattering_bidirectional.tif
            diff_phase_bidirectional.tif
    """
    if not isinstance(result, xr.DataArray):
        raise TypeError("cli_export expects an xarray.DataArray")

    output_path.mkdir(parents=True, exist_ok=True)

    contrast_labels = list(result.coords["contrast"].values)

    logger.info(f"Exporting {len(contrast_labels)} contrasts as multipage TIFFs")

    for contrast in contrast_labels:
        out_file = output_path / f"{contrast}.tif"
        logger.info(f"Exporting {out_file.name}")

        # ('image', 'y', 'x')
        data = result.sel(contrast=contrast)

        save_image(data, out_file)

    return output_path


def execute_SHI(
    path_to_images: Path,
    path_to_reference: Path,
    path_to_result: Path,
    projected_grid: int,
    unwrap: str | None = None
) -> None:
    """
    Execute spatial harmonics analysis on a set of images.

    This function performs spatial harmonics analysis on a set of images and exports the results to the specified directory.
    Uses shimexpy.core functionality for processing.

    Parameters
    ----------
    path_to_images : list of str or str
        A list of paths to the images for analysis or a directory path.
    path_to_result : str
        The path to the directory where the results will be exported.
    mask_period : int
        The period of the mask used in the analysis.
    unwrap : str or None
        The unwrapping method to use for phase maps.
    flat : bool, default=True
        Whether to use the first image as a reference image. 
        Default is True since we always assume reference image is available.
    """

    image_paths = list(path_to_images.glob("*.tif"))
    if not image_paths:
        logger.error("No .tif files found in the specified path")
        return

    reference_paths = list(path_to_reference.glob("*.tif"))
    if not reference_paths:
        logger.error("No .tif files found in the specified reference path")
        return

    # reference_data = (ref_abs, ref_scat, ref_phase, ref_block_grid)
    reference_images = load_image(reference_paths[0])

    ref_abs, ref_scat, ref_phase, ref_block_grid = get_harmonics(
        reference_images,
        projected_grid=projected_grid,
        unwrap=unwrap
    )

    # --- Step 1: accumulate lazy Datasets
    results = []
    labels = []

    for image in image_paths:
        img = tifffile.imread(image)
        result_lazy = get_all_harmonic_contrasts(
            img,
            (ref_abs, ref_scat, ref_phase),
            ref_block_grid,
            unwrap=unwrap
        )
        results.append(result_lazy)
        labels.append(image.stem)

    # --- Step 2: combine all lazy results into one global Dataset
    combined_lazy = xr.concat(results, dim="image")
    combined_lazy = combined_lazy.assign_coords(image=labels)

    # --- Step 4: compute once at the end (optional)
    # Uncomment the next two lines if you want to run it here
    final_result = combined_lazy.compute()

    # --- Export (una sola vez, como querías)
    cli_export(
        final_result,
        path_to_result
    )

