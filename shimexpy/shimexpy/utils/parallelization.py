import xarray as xr
import cupy as cp


def apply_harmonic_chunking(
    dataset_or_array: xr.DataArray,
    chunks: dict | None = None
) -> xr.DataArray:
    """
    Apply Dask chunking strategy to harmonic-based data.

    Parameters
    ----------
    dataset_or_array : xr.Dataset or xr.DataArray
        Input data to chunk.
    chunks : dict or None
        Chunking strategy. Default is {"harmonic": 1, "ky": -1, "kx": -1}.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The chunked data.
    """
    if chunks is None:
        chunks = {"harmonic": 1, "ky": -1, "kx": -1}

    return dataset_or_array.chunk(chunks)


def move_to_cpu(xobj: xr.DataArray):
    """
    Transfer xarray data from GPU (CuPy) to CPU (NumPy) in-place.
    Preserves all labels, coords, dims, and attributes.

    Parameters
    ----------
    xobj : xr.DataArray or xr.Dataset
        Object whose data resides in GPU memory (CuPy).

    Returns
    -------
    xr.DataArray or xr.Dataset
        Same object, now with NumPy-backed arrays.
    """

    if isinstance(xobj, xr.DataArray):
        if isinstance(xobj.data, cp.ndarray):
            xobj.data = cp.asnumpy(xobj.data)
        return xobj

    else:
        raise TypeError("Expected xarray.DataArray or xarray.Dataset")
