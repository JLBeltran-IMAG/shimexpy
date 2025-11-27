"""
Tests for shimexpy.core.unwrapping module.

Tests cover:
- skimage_unwrap function
- ls_unwrap function (least-squares Poisson solver)
- Input shape handling (2D and 3D)
- Error handling
"""

import pytest
import numpy as np
import sys
import os

# Clear any existing shimexpy imports and add correct path
_mods_to_remove = [k for k in list(sys.modules.keys()) if 'shimexpy' in k]
for _m in _mods_to_remove:
    del sys.modules[_m]



from shimexpy.core.unwrapping import skimage_unwrap, ls_unwrap


class TestSkimageUnwrap:
    """Tests for skimage_unwrap function."""

    def test_skimage_unwrap_2d_input(self):
        """Test skimage_unwrap with 2D input array."""
        # Create a simple wrapped phase
        x = np.linspace(-np.pi, np.pi, 100).reshape(10, 10)
        wrapped = np.exp(1j * x)

        result = skimage_unwrap(wrapped)

        # Should return 3D array with shape (1, M, N)
        assert result.ndim == 3
        assert result.shape == (1, 10, 10)

    def test_skimage_unwrap_3d_input(self):
        """Test skimage_unwrap with 3D input array (1, M, N)."""
        # Create wrapped phase and add leading dimension
        x = np.linspace(-np.pi, np.pi, 64).reshape(8, 8)
        wrapped = np.exp(1j * x)
        wrapped_3d = wrapped[np.newaxis, ...]

        result = skimage_unwrap(wrapped_3d)

        assert result.ndim == 3
        assert result.shape == (1, 8, 8)

    def test_skimage_unwrap_returns_float(self):
        """Test that result is float dtype."""
        wrapped = np.exp(1j * np.linspace(0, 2*np.pi, 25).reshape(5, 5))

        result = skimage_unwrap(wrapped)

        assert np.issubdtype(result.dtype, np.floating)

    def test_skimage_unwrap_with_real_input(self):
        """Test skimage_unwrap with real-valued (already wrapped) input."""
        # Real values representing phase angles
        wrapped = np.linspace(-np.pi, np.pi, 36).reshape(6, 6).astype(np.float32)

        result = skimage_unwrap(wrapped)

        assert result.shape == (1, 6, 6)
        assert np.isfinite(result).all()

    def test_skimage_unwrap_wrap_around_false(self):
        """Test skimage_unwrap with wrap_around=False."""
        wrapped = np.exp(1j * np.linspace(-np.pi, np.pi, 49).reshape(7, 7))

        result = skimage_unwrap(wrapped, wrap_around=False)

        assert result.shape == (1, 7, 7)

    def test_skimage_unwrap_invalid_dims_raises(self):
        """Test that invalid dimensions raise ValueError."""
        invalid_1d = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="must be 2D or 3D"):
            skimage_unwrap(invalid_1d)

    def test_skimage_unwrap_invalid_4d_raises(self):
        """Test that 4D input raises ValueError."""
        invalid_4d = np.ones((2, 3, 4, 5))

        with pytest.raises(ValueError, match="must be 2D or 3D"):
            skimage_unwrap(invalid_4d)


class TestLsUnwrap:
    """Tests for ls_unwrap function (least-squares Poisson solver)."""

    def test_ls_unwrap_2d_input(self):
        """Test ls_unwrap with 2D input array."""
        # Create wrapped phase
        x = np.linspace(-np.pi, np.pi, 64).reshape(8, 8)
        wrapped = np.exp(1j * x)

        result = ls_unwrap(wrapped)

        # Should return 3D array with shape (1, M, N)
        assert result.ndim == 3
        assert result.shape == (1, 8, 8)

    def test_ls_unwrap_3d_input(self):
        """Test ls_unwrap with 3D input array (1, M, N)."""
        x = np.linspace(-np.pi, np.pi, 100).reshape(10, 10)
        wrapped = np.exp(1j * x)
        wrapped_3d = wrapped[np.newaxis, ...]

        result = ls_unwrap(wrapped_3d)

        assert result.ndim == 3
        assert result.shape == (1, 10, 10)

    def test_ls_unwrap_returns_float(self):
        """Test that result is float dtype."""
        wrapped = np.exp(1j * np.linspace(0, 2*np.pi, 36).reshape(6, 6))

        result = ls_unwrap(wrapped)

        assert np.issubdtype(result.dtype, np.floating)

    def test_ls_unwrap_with_real_input(self):
        """Test ls_unwrap with real-valued input."""
        wrapped = np.linspace(-np.pi, np.pi, 49).reshape(7, 7).astype(np.float64)

        result = ls_unwrap(wrapped)

        assert result.shape == (1, 7, 7)
        assert np.isfinite(result).all()

    def test_ls_unwrap_zero_mean(self):
        """Test that ls_unwrap result has approximately zero mean."""
        wrapped = np.exp(1j * np.linspace(-np.pi, np.pi, 100).reshape(10, 10))

        result = ls_unwrap(wrapped)

        # LS solver removes DC component, so mean should be close to zero
        assert np.abs(result.mean()) < 1e-10

    def test_ls_unwrap_produces_finite_values(self):
        """Test that ls_unwrap produces finite values."""
        wrapped = np.exp(1j * np.random.uniform(-np.pi, np.pi, (16, 16)))

        result = ls_unwrap(wrapped)

        assert np.isfinite(result).all()


class TestUnwrapComparison:
    """Tests comparing skimage_unwrap and ls_unwrap behavior."""

    def test_both_handle_same_input_shape(self):
        """Test that both functions handle the same input and output shapes."""
        wrapped = np.exp(1j * np.linspace(-np.pi, np.pi, 64).reshape(8, 8))

        result_sk = skimage_unwrap(wrapped)
        result_ls = ls_unwrap(wrapped)

        assert result_sk.shape == result_ls.shape
        assert result_sk.shape == (1, 8, 8)

    def test_both_return_real_values(self):
        """Test that both functions return real-valued results."""
        wrapped = np.exp(1j * np.linspace(-np.pi, np.pi, 49).reshape(7, 7))

        result_sk = skimage_unwrap(wrapped)
        result_ls = ls_unwrap(wrapped)

        assert np.isrealobj(result_sk)
        assert np.isrealobj(result_ls)


class TestUnwrapEdgeCases:
    """Edge case tests for unwrapping functions."""

    def test_skimage_unwrap_uniform_phase(self):
        """Test unwrapping a uniform (constant) phase."""
        # Uniform phase should remain constant after unwrapping
        uniform = np.ones((10, 10), dtype=np.complex64) * np.exp(1j * 0.5)

        result = skimage_unwrap(uniform)

        # All values should be approximately equal
        assert np.allclose(result, result[0, 0, 0], atol=1e-6)

    def test_ls_unwrap_uniform_phase(self):
        """Test ls_unwrap with uniform phase."""
        uniform = np.ones((10, 10), dtype=np.complex64) * np.exp(1j * 0.5)

        result = ls_unwrap(uniform)

        # LS solver should produce near-constant result with zero mean
        assert np.std(result) < 1e-10

    def test_skimage_unwrap_small_image(self):
        """Test unwrapping a small image (2x2)."""
        wrapped = np.exp(1j * np.array([[0, np.pi/2], [np.pi, -np.pi/2]]))

        result = skimage_unwrap(wrapped)

        assert result.shape == (1, 2, 2)
        assert np.isfinite(result).all()

    def test_ls_unwrap_small_image(self):
        """Test ls_unwrap with small image (2x2)."""
        wrapped = np.exp(1j * np.array([[0, np.pi/2], [np.pi, -np.pi/2]]))

        result = ls_unwrap(wrapped)

        assert result.shape == (1, 2, 2)
        assert np.isfinite(result).all()


class TestUnwrapModuleImports:
    """Tests for module structure and imports."""

    def test_can_import_unwrap_functions(self):
        """Test that unwrapping functions can be imported from module."""
        from shimexpy.core.unwrapping import skimage_unwrap, ls_unwrap

        assert callable(skimage_unwrap)
        assert callable(ls_unwrap)

    def test_can_import_from_main_package(self):
        """Test that unwrapping functions are exported from main package."""
        from shimexpy import skimage_unwrap, ls_unwrap

        assert callable(skimage_unwrap)
        assert callable(ls_unwrap)
