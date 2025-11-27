"""
Tests for shimexpy.io.file_io module.

Tests cover:
- Image loading (TIFF format)
- Image saving (TIFF format)
- Block grid save/load (JSON)
- Results save/load (pickle)
- CLI export functionality
"""

import pytest
import numpy as np
import xarray as xr
import sys
import os
from pathlib import Path

# Clear any existing shimexpy imports and add correct path
_mods_to_remove = [k for k in list(sys.modules.keys()) if 'shimexpy' in k]
for _m in _mods_to_remove:
    del sys.modules[_m]



from shimexpy.io.file_io import (
    load_image,
    save_image,
    save_block_grid,
    load_block_grid,
    save_results,
    load_results,
    cli_export,
)


def _has_zarr():
    """Check if zarr is installed."""
    try:
        import zarr
        return True
    except ImportError:
        return False


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_tiff_image(self, tmp_path):
        """Test loading a TIFF image."""
        import tifffile

        # Create a test TIFF image
        test_image = np.random.rand(64, 64).astype(np.float32)
        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(tiff_path, test_image)

        # Load and verify
        loaded = load_image(tiff_path)

        np.testing.assert_array_almost_equal(loaded, test_image)

    def test_load_tiff_with_tiff_extension(self, tmp_path):
        """Test loading with .tiff extension."""
        import tifffile

        test_image = np.ones((32, 32), dtype=np.float32) * 100
        tiff_path = tmp_path / "test.tiff"
        tifffile.imwrite(tiff_path, test_image)

        loaded = load_image(tiff_path)

        np.testing.assert_array_almost_equal(loaded, test_image)

    def test_load_string_path(self, tmp_path):
        """Test loading with string path instead of Path object."""
        import tifffile

        test_image = np.zeros((10, 10), dtype=np.float32)
        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(tiff_path, test_image)

        # Pass string path
        loaded = load_image(str(tiff_path))

        np.testing.assert_array_almost_equal(loaded, test_image)

    def test_load_unsupported_format_raises(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        png_path = tmp_path / "test.png"
        png_path.touch()  # Create empty file

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_image(png_path)


class TestSaveImage:
    """Tests for save_image function."""

    def test_save_numpy_array(self, tmp_path):
        """Test saving a numpy array as TIFF."""
        test_image = np.random.rand(50, 50).astype(np.float32)
        save_path = tmp_path / "output.tif"

        save_image(test_image, save_path)

        assert save_path.exists()
        loaded = load_image(save_path)
        np.testing.assert_array_almost_equal(loaded, test_image)

    def test_save_xarray_dataarray(self, tmp_path):
        """Test saving an xarray DataArray as TIFF."""
        data = np.random.rand(30, 30).astype(np.float32)
        da = xr.DataArray(data, dims=['y', 'x'])
        save_path = tmp_path / "xarray_output.tif"

        save_image(da, save_path)

        assert save_path.exists()
        loaded = load_image(save_path)
        np.testing.assert_array_almost_equal(loaded, data)

    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save_image creates parent directories."""
        test_image = np.ones((10, 10), dtype=np.float32)
        nested_path = tmp_path / "nested" / "dirs" / "output.tif"

        save_image(test_image, nested_path)

        assert nested_path.exists()

    def test_save_converts_to_float32(self, tmp_path):
        """Test that save_image converts to float32."""
        # Save uint16 image
        test_image = np.ones((10, 10), dtype=np.uint16) * 1000
        save_path = tmp_path / "converted.tif"

        save_image(test_image, save_path)

        loaded = load_image(save_path)
        assert loaded.dtype == np.float32

    def test_save_unsupported_format_raises(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        test_image = np.ones((10, 10))
        png_path = tmp_path / "output.png"

        with pytest.raises(ValueError, match="Unsupported file format"):
            save_image(test_image, png_path)

    def test_save_with_tiff_extension(self, tmp_path):
        """Test saving with .tiff extension."""
        test_image = np.random.rand(20, 20).astype(np.float32)
        save_path = tmp_path / "output.tiff"

        save_image(test_image, save_path)

        assert save_path.exists()


class TestBlockGrid:
    """Tests for block grid save/load functions."""

    def test_save_and_load_block_grid(self, tmp_path):
        """Test saving and loading a block grid."""
        block_grid = {
            "block_0": [0, 0, 100, 100],
            "block_1": [100, 0, 200, 100],
        }
        json_path = tmp_path / "block_grid.json"

        save_block_grid(block_grid, json_path)
        loaded = load_block_grid(json_path)

        assert loaded == block_grid

    def test_save_block_grid_with_numpy_integers(self, tmp_path):
        """Test that numpy integers are converted for JSON serialization."""
        block_grid = {
            "block_0": [np.int64(0), np.int64(0), np.int64(100), np.int64(100)],
        }
        json_path = tmp_path / "numpy_grid.json"

        save_block_grid(block_grid, json_path)
        loaded = load_block_grid(json_path)

        assert loaded["block_0"] == [0, 0, 100, 100]

    def test_save_block_grid_creates_directories(self, tmp_path):
        """Test that save_block_grid creates parent directories."""
        block_grid = {"test": [1, 2, 3, 4]}
        nested_path = tmp_path / "deep" / "nested" / "grid.json"

        save_block_grid(block_grid, nested_path)

        assert nested_path.exists()


class TestResults:
    """Tests for results save/load functions (pickle)."""

    def test_save_and_load_dict_results(self, tmp_path):
        """Test saving and loading dictionary results."""
        results = {
            "angles": [0.0, 45.0, 90.0],
            "contrasts": np.array([1.0, 2.0, 3.0]),
            "metadata": {"version": "1.0"}
        }
        pkl_path = tmp_path / "results.pkl"

        save_results(results, pkl_path)
        loaded = load_results(pkl_path)

        assert loaded["angles"] == results["angles"]
        np.testing.assert_array_equal(loaded["contrasts"], results["contrasts"])
        assert loaded["metadata"] == results["metadata"]

    def test_save_and_load_numpy_array(self, tmp_path):
        """Test saving and loading a numpy array."""
        results = np.random.rand(100, 100).astype(np.float32)
        pkl_path = tmp_path / "array.pkl"

        save_results(results, pkl_path)
        loaded = load_results(pkl_path)

        np.testing.assert_array_equal(loaded, results)

    def test_save_and_load_xarray(self, tmp_path):
        """Test saving and loading an xarray DataArray."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        da = xr.DataArray(data, dims=['z', 'y', 'x'])
        pkl_path = tmp_path / "xarray.pkl"

        save_results(da, pkl_path)
        loaded = load_results(pkl_path)

        xr.testing.assert_equal(loaded, da)

    def test_save_results_creates_directories(self, tmp_path):
        """Test that save_results creates parent directories."""
        results = {"key": "value"}
        nested_path = tmp_path / "a" / "b" / "c" / "results.pkl"

        save_results(results, nested_path)

        assert nested_path.exists()


class TestCliExport:
    """Tests for cli_export function."""

    @pytest.fixture
    def sample_dataarray(self):
        """Create a sample DataArray with SHI result structure."""
        # Structure: (image, contrast, y, x)
        data = np.random.rand(2, 3, 64, 64).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=['image', 'contrast', 'y', 'x'],
            coords={
                'image': [0, 1],
                'contrast': ['absorption', 'dpcx', 'dpcy'],
            }
        )
        return da

    def test_export_netcdf(self, tmp_path, sample_dataarray):
        """Test exporting to NetCDF format."""
        output_path = cli_export(
            sample_dataarray,
            tmp_path,
            name="test_result",
            fmt="netcdf"
        )

        assert output_path == tmp_path
        assert (tmp_path / "test_result.netcdf").exists()

    def test_export_nc_format(self, tmp_path, sample_dataarray):
        """Test exporting with 'nc' format string."""
        cli_export(
            sample_dataarray,
            tmp_path,
            name="nc_result",
            fmt="nc"
        )

        # Check file exists (format normalized to nc)
        assert (tmp_path / "nc_result.nc").exists()

    def test_export_tiff(self, tmp_path, sample_dataarray):
        """Test exporting to TIFF format (one file per contrast)."""
        cli_export(
            sample_dataarray,
            tmp_path,
            name="tiff_result",
            fmt="tiff"
        )

        # Each contrast should have its own file
        assert (tmp_path / "absorption.tif").exists()
        assert (tmp_path / "dpcx.tif").exists()
        assert (tmp_path / "dpcy.tif").exists()

    def test_export_tif_format(self, tmp_path, sample_dataarray):
        """Test exporting with 'tif' format string."""
        cli_export(
            sample_dataarray,
            tmp_path,
            name="tif_result",
            fmt="tif"
        )

        assert (tmp_path / "absorption.tif").exists()

    @pytest.mark.skipif(
        not _has_zarr(),
        reason="zarr not installed"
    )
    def test_export_zarr(self, tmp_path, sample_dataarray):
        """Test exporting to Zarr format."""
        cli_export(
            sample_dataarray,
            tmp_path,
            name="zarr_result",
            fmt="zarr"
        )

        assert (tmp_path / "zarr_result.zarr").exists()

    def test_export_creates_output_directory(self, tmp_path, sample_dataarray):
        """Test that cli_export creates the output directory."""
        new_dir = tmp_path / "new_output_dir"

        cli_export(sample_dataarray, new_dir, fmt="netcdf")

        assert new_dir.exists()

    def test_export_unsupported_format_raises(self, tmp_path, sample_dataarray):
        """Test that unsupported formats raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            cli_export(sample_dataarray, tmp_path, fmt="csv")

    def test_export_case_insensitive_format(self, tmp_path, sample_dataarray):
        """Test that format string is case insensitive."""
        cli_export(
            sample_dataarray,
            tmp_path,
            name="upper_test",
            fmt="NETCDF"
        )

        assert (tmp_path / "upper_test.netcdf").exists()


class TestRoundTrip:
    """Integration tests for complete save/load cycles."""

    def test_image_roundtrip_preserves_data(self, tmp_path):
        """Test that image data is preserved through save/load cycle."""
        original = np.random.rand(128, 128).astype(np.float32)
        path = tmp_path / "roundtrip.tif"

        save_image(original, path)
        loaded = load_image(path)

        np.testing.assert_array_almost_equal(loaded, original, decimal=6)

    def test_block_grid_roundtrip(self, tmp_path):
        """Test block grid data preservation through save/load."""
        original = {
            f"block_{i}": [i*100, i*100, (i+1)*100, (i+1)*100]
            for i in range(5)
        }
        path = tmp_path / "grid.json"

        save_block_grid(original, path)
        loaded = load_block_grid(path)

        assert loaded == original

    def test_complex_results_roundtrip(self, tmp_path):
        """Test complex nested results through save/load."""
        original = {
            "images": [np.random.rand(10, 10) for _ in range(3)],
            "metadata": {
                "version": "1.0",
                "params": {"threshold": 0.5, "iterations": 100}
            },
            "dataarray": xr.DataArray(np.random.rand(5, 5), dims=['y', 'x'])
        }
        path = tmp_path / "complex.pkl"

        save_results(original, path)
        loaded = load_results(path)

        for i, img in enumerate(loaded["images"]):
            np.testing.assert_array_equal(img, original["images"][i])
        assert loaded["metadata"] == original["metadata"]
        xr.testing.assert_equal(loaded["dataarray"], original["dataarray"])
