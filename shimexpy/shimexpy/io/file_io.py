"""
I/O utilities for loading and saving images and results.
"""
import numpy as np
import xarray as xr
from pathlib import Path
import tifffile
import json
import pickle

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

from shimexpy.utils.parallelization import move_to_cpu


def load_image(file_path):
    """
    Load an image file as a numpy array.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the image file to load.
        
    Returns
    -------
    np.ndarray
        The loaded image as a numpy array.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() in ['.tif', '.tiff']:
        return tifffile.imread(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_image(image, file_path):
    """
    Save an image array to a file. When saving as TIFF, the file is saved with
    ImageJ compatibility enabled and the data is converted to float32 format.
    
    Parameters
    ----------
    image : np.ndarray or xr.DataArray
        The image to save. Will be converted to float32 format.
    file_path : str or Path
        Path where the image will be saved. For TIFF files (.tif, .tiff),
        the file will be saved with ImageJ compatibility.
    """
    file_path = Path(file_path)

    # Convert xarray to numpy if needed
    if isinstance(image, xr.DataArray):
        image = image.values

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Asegurar que la imagen sea float32
    image = np.asarray(image).astype(np.float32)

    if file_path.suffix.lower() in ['.tif', '.tiff']:
        tifffile.imwrite(file_path, image, imagej=True)
    else:
        raise ValueError(f"Unsupported file format for saving: {file_path.suffix}")


def save_block_grid(block_grid, file_path):
    """
    Save a block grid dictionary to a JSON file.
    
    Parameters
    ----------
    block_grid : dict
        The block grid dictionary to save.
    file_path : str or Path
        Path where the JSON file will be saved.
    """
    file_path = Path(file_path)
    
    # Convert numpy integers to Python integers for JSON serialization
    serializable_grid = {}
    for key, value in block_grid.items():
        serializable_grid[key] = [int(v) for v in value]

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(serializable_grid, f, indent=2)


def load_block_grid(file_path):
    """
    Load a block grid dictionary from a JSON file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the JSON file to load.
        
    Returns
    -------
    dict
        The loaded block grid dictionary.
    """
    file_path = Path(file_path)
    with open(file_path, 'r') as f:
        return json.load(f)


def save_results(results, file_path):
    """
    Save analysis results to a pickle file.
    
    Parameters
    ----------
    results : object
        The results to save (can be any pickle-serializable object).
    file_path : str or Path
        Path where the pickle file will be saved.
    """
    file_path = Path(file_path)

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def load_results(file_path):
    """
    Load analysis results from a pickle file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the pickle file to load.
        
    Returns
    -------
    object
        The loaded results object.
    """
    file_path = Path(file_path)
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def cli_export(
    data: xr.DataArray,
    output_path: Path,
    name: str = "result",
    fmt: str = "netcdf"
) -> Path:
    """
    Export SHI final result (DataArray format) to disk.

    Parameters
    ----------
    data : xr.DataArray
        Result with dimensions (image, contrast, y, x),
        as returned by get_all_harmonic_contrasts().
    output_path : Path
        Directory where files will be saved.
    name : str, optional
        Base name for output files (default: 'result').
    fmt : str, optional
        Output format: 'netcdf', 'zarr', 'tiff' or 'tif'.

    Notes
    -----
    - Moves data to CPU if on GPU (CuPy backend).
    - For TIFF export:
        * Each contrast is saved as a separate .tif file.
        * Each file contains the full image stack (axis 'image').
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Ensure data is on CPU
    data = move_to_cpu(data)
    fmt = fmt.lower()
    file_path = output_path / f"{name}.{fmt}"

    # ---------- EXPORT ----------
    if fmt in ("netcdf", "nc"):
        # Export as single NetCDF file
        data.to_netcdf(file_path)

    elif fmt in ("zarr",):
        data.to_zarr(output_path / f"{name}.zarr", mode="w")

    elif fmt in ("tiff", "tif"):
        # Each contrast label → one TIFF stack (image, y, x)
        for label in data.contrast.values:
            img = data.sel(contrast=label)
            save_image(img, output_path / f"{label}.tif")

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return output_path

