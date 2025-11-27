import xarray as xr
import tifffile
from pathlib import Path

from .logging import logger
from shimexpy import get_all_harmonic_contrasts, cli_export


def execute_SHI(
    path_to_images: Path,
    path_to_result: Path,
    reference_data: tuple[xr.DataArray, xr.DataArray, xr.DataArray, dict],
    unwrap: str | None
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

    reference = reference_data[0:3]
    ref_block_grid = reference_data[3]

    # --- Step 1: accumulate lazy Datasets
    results = []
    labels = []

    for image in image_paths:
        img = tifffile.imread(image)
        result_lazy = get_all_harmonic_contrasts(
            img,
            reference,
            ref_block_grid,
            unwrap=unwrap
        )
        results.append(result_lazy)
        labels.append(image.name)

    # --- Step 2: combine all lazy results into one global Dataset
    combined_lazy = xr.concat(results, dim="image")
    combined_lazy = combined_lazy.assign_coords(image=labels)

    # --- Step 4: compute once at the end (optional)
    # Uncomment the next two lines if you want to run it here
    final_result = combined_lazy.compute()

    cli_export(
        final_result,
        path_to_result,
        fmt="tif"
    )

