"""
Tests for shimexpy.utils module.

Tests cover:
- apply_harmonic_chunking function
- move_to_cpu function
"""

import pytest
import numpy as np
import xarray as xr
import sys
import os

# Clear any existing shimexpy imports and add correct path
_mods_to_remove = [k for k in list(sys.modules.keys()) if 'shimexpy' in k]
for _m in _mods_to_remove:
    del sys.modules[_m]



from shimexpy.utils.parallelization import (
    apply_harmonic_chunking,
    move_to_cpu,
    _HAS_CUPY,
)


class TestApplyHarmonicChunking:
    """Tests for apply_harmonic_chunking function."""

    def test_apply_harmonic_chunking_default_chunks(self):
        """Test chunking with default chunk sizes."""
        data = np.random.rand(9, 64, 64).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=['harmonic', 'ky', 'kx'],
            coords={
                'harmonic': [f'harmonic_{i}' for i in range(9)],
                'ky': np.linspace(-0.5, 0.5, 64),
                'kx': np.linspace(-0.5, 0.5, 64)
            }
        )

        result = apply_harmonic_chunking(da)

        # Should be chunked (have dask backing)
        assert result.chunks is not None
        # Harmonic dimension (index 0) should be chunked by 1
        assert result.chunks[0] == (1,) * 9

    def test_apply_harmonic_chunking_custom_chunks(self):
        """Test chunking with custom chunk sizes."""
        data = np.random.rand(4, 32, 32).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=['harmonic', 'ky', 'kx'],
            coords={
                'harmonic': [f'h_{i}' for i in range(4)],
                'ky': np.linspace(-0.5, 0.5, 32),
                'kx': np.linspace(-0.5, 0.5, 32)
            }
        )

        custom_chunks = {"harmonic": 2, "ky": 16, "kx": 16}
        result = apply_harmonic_chunking(da, chunks=custom_chunks)

        assert result.chunks is not None
        # Harmonic (index 0) should be chunked by 2
        assert result.chunks[0] == (2, 2)

    def test_apply_harmonic_chunking_preserves_coords(self):
        """Test that chunking preserves coordinates."""
        harmonics = ['harmonic_00', 'harmonic_hp']
        ky = np.linspace(-0.5, 0.5, 16)
        kx = np.linspace(-0.5, 0.5, 16)

        data = np.random.rand(2, 16, 16).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=['harmonic', 'ky', 'kx'],
            coords={'harmonic': harmonics, 'ky': ky, 'kx': kx}
        )

        result = apply_harmonic_chunking(da)

        # Check coordinates are preserved
        assert list(result.coords['harmonic'].values) == harmonics
        np.testing.assert_array_equal(result.coords['ky'].values, ky)
        np.testing.assert_array_equal(result.coords['kx'].values, kx)

    def test_apply_harmonic_chunking_preserves_dtype(self):
        """Test that chunking preserves data type."""
        data = np.random.rand(2, 8, 8).astype(np.complex64)
        da = xr.DataArray(data, dims=['harmonic', 'ky', 'kx'])

        result = apply_harmonic_chunking(da)

        assert result.dtype == np.complex64

    def test_apply_harmonic_chunking_returns_dataarray(self):
        """Test that chunking returns a DataArray."""
        data = np.random.rand(3, 10, 10).astype(np.float32)
        da = xr.DataArray(data, dims=['harmonic', 'ky', 'kx'])

        result = apply_harmonic_chunking(da)

        assert isinstance(result, xr.DataArray)


class TestMoveToCpu:
    """Tests for move_to_cpu function."""

    def test_move_to_cpu_numpy_array(self):
        """Test move_to_cpu with already-CPU (numpy) data."""
        data = np.random.rand(10, 10).astype(np.float32)
        da = xr.DataArray(data, dims=['y', 'x'])

        result = move_to_cpu(da)

        # Should return the same DataArray (already on CPU)
        assert result is da
        np.testing.assert_array_equal(result.values, data)

    def test_move_to_cpu_preserves_coords(self):
        """Test that move_to_cpu preserves coordinates."""
        data = np.random.rand(5, 5).astype(np.float32)
        coords = {'y': np.arange(5), 'x': np.arange(5)}
        da = xr.DataArray(data, dims=['y', 'x'], coords=coords)

        result = move_to_cpu(da)

        assert list(result.dims) == ['y', 'x']
        np.testing.assert_array_equal(result.coords['y'].values, coords['y'])

    def test_move_to_cpu_preserves_attrs(self):
        """Test that move_to_cpu preserves attributes."""
        data = np.random.rand(5, 5).astype(np.float32)
        da = xr.DataArray(data, dims=['y', 'x'], attrs={'units': 'meters'})

        result = move_to_cpu(da)

        assert result.attrs == {'units': 'meters'}

    def test_move_to_cpu_invalid_type_raises(self):
        """Test that move_to_cpu raises on invalid types."""
        with pytest.raises(TypeError, match="Expected xarray.DataArray"):
            move_to_cpu(np.array([1, 2, 3]))

    def test_move_to_cpu_dict_raises(self):
        """Test that move_to_cpu raises on dict input."""
        with pytest.raises(TypeError, match="Expected xarray.DataArray"):
            move_to_cpu({"key": "value"})

    def test_move_to_cpu_list_raises(self):
        """Test that move_to_cpu raises on list input."""
        with pytest.raises(TypeError, match="Expected xarray.DataArray"):
            move_to_cpu([1, 2, 3])


class TestCuPyAvailability:
    """Tests for CuPy availability flag."""

    def test_has_cupy_is_boolean(self):
        """Test that _HAS_CUPY is a boolean."""
        assert isinstance(_HAS_CUPY, bool)

    def test_module_loads_without_cupy(self):
        """Test that module loads regardless of CuPy availability."""
        # If we got here, the module loaded successfully
        from shimexpy.utils.parallelization import apply_harmonic_chunking, move_to_cpu
        assert callable(apply_harmonic_chunking)
        assert callable(move_to_cpu)


class TestUtilsModuleImports:
    """Tests for module imports."""

    def test_can_import_from_utils(self):
        """Test that functions can be imported from utils module."""
        from shimexpy.utils.parallelization import (
            apply_harmonic_chunking,
            move_to_cpu
        )

        assert callable(apply_harmonic_chunking)
        assert callable(move_to_cpu)

    def test_can_import_from_main_package(self):
        """Test that apply_harmonic_chunking is exported from main package."""
        from shimexpy import apply_harmonic_chunking

        assert callable(apply_harmonic_chunking)


class TestMoveToCpuWithCuPy:
    """Tests for move_to_cpu with CuPy if available."""

    def test_move_to_cpu_with_cupy_data_if_available(self):
        """Test move_to_cpu transfers CuPy array to NumPy if CuPy is available."""
        if not _HAS_CUPY:
            pytest.skip("CuPy not available")

        import cupy as cp

        # Create a CuPy array
        gpu_data = cp.random.rand(10, 10).astype(cp.float32)
        da = xr.DataArray(gpu_data, dims=['y', 'x'])

        result = move_to_cpu(da)

        # Data should now be a NumPy array
        assert isinstance(result.data, np.ndarray)
        assert not isinstance(result.data, cp.ndarray)

    def test_move_to_cpu_preserves_values_from_cupy(self):
        """Test move_to_cpu preserves values when transferring from CuPy."""
        if not _HAS_CUPY:
            pytest.skip("CuPy not available")

        import cupy as cp

        # Create data first on CPU, then transfer to GPU
        cpu_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gpu_data = cp.asarray(cpu_data)
        da = xr.DataArray(gpu_data, dims=['y', 'x'])

        result = move_to_cpu(da)

        # Values should be preserved
        np.testing.assert_array_almost_equal(result.data, cpu_data)

    def test_cupy_import_sets_has_cupy_true(self):
        """Test that _HAS_CUPY is True when CuPy is importable."""
        if not _HAS_CUPY:
            pytest.skip("CuPy not available")

        # If CuPy is available, _HAS_CUPY should be True
        assert _HAS_CUPY is True

        # And cp should be defined (not None)
        from shimexpy.utils.parallelization import cp
        assert cp is not None


class TestCuPyNotAvailable:
    """Tests for behavior when CuPy is not available."""

    def test_has_cupy_false_when_no_cupy(self):
        """Test _HAS_CUPY is False when CuPy cannot be imported."""
        if _HAS_CUPY:
            pytest.skip("CuPy is available, cannot test unavailable case")

        assert _HAS_CUPY is False

    def test_cp_is_none_when_no_cupy(self):
        """Test cp module reference is None when CuPy not available."""
        if _HAS_CUPY:
            pytest.skip("CuPy is available, cannot test unavailable case")

        from shimexpy.utils.parallelization import cp
        assert cp is None
