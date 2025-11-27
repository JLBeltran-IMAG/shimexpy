"""
Tests for shimexpy.core.contrast module.

Tests cover:
- Contrast retrieval (absorption, scattering, phasemap)
- Module constants
- Error handling
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



from shimexpy.core.contrast import (
    contrast_retrieval,
    _compute_scattering,
    CONTRASTS,
    HARMONICS,
)


class TestContrastConstants:
    """Tests for contrast module constants."""

    def test_contrasts_dict_keys(self):
        """Test CONTRASTS dictionary has expected keys."""
        assert "horizontal" in CONTRASTS
        assert "vertical" in CONTRASTS
        assert "bidirectional" in CONTRASTS

    def test_contrasts_horizontal_harmonics(self):
        """Test horizontal contrasts have correct harmonics."""
        assert CONTRASTS["horizontal"] == [
            "harmonic_horizontal_positive",
            "harmonic_horizontal_negative"
        ]

    def test_contrasts_vertical_harmonics(self):
        """Test vertical contrasts have correct harmonics."""
        assert CONTRASTS["vertical"] == [
            "harmonic_vertical_positive",
            "harmonic_vertical_negative"
        ]

    def test_contrasts_bidirectional_harmonics(self):
        """Test bidirectional contrasts combine both directions."""
        assert len(CONTRASTS["bidirectional"]) == 4

    def test_harmonics_list_contains_all_types(self):
        """Test HARMONICS list contains all harmonic types."""
        assert len(HARMONICS) == 8
        assert "harmonic_horizontal_positive" in HARMONICS
        assert "harmonic_vertical_negative" in HARMONICS
        assert "harmonic_diagonal_p1_p1" in HARMONICS


class TestComputeScattering:
    """Tests for _compute_scattering function."""

    def test_compute_scattering_basic(self):
        """Test basic scattering computation."""
        ifft = np.array([[0.5, 0.25], [0.1, 0.05]], dtype=np.complex64)
        main = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex64)

        result = _compute_scattering(ifft, main)

        # scattering = log(1 / |ifft/main|)
        expected = np.log(1 / np.abs(ifft / main))
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_scattering_avoids_zero_division(self):
        """Test scattering handles zero main harmonic."""
        ifft = np.array([[0.5, 0.5]], dtype=np.complex64)
        main = np.array([[0.0, 1.0]], dtype=np.complex64)

        result = _compute_scattering(ifft, main, eps=1e-12)

        # Should not raise or produce NaN/inf
        assert np.isfinite(result).all()

    def test_compute_scattering_clips_small_values(self):
        """Test scattering clips values to avoid log(0)."""
        ifft = np.array([[0.0, 0.001]], dtype=np.complex64)
        main = np.array([[1.0, 1.0]], dtype=np.complex64)

        result = _compute_scattering(ifft, main)

        # Should not produce -inf
        assert np.isfinite(result).all()

    def test_compute_scattering_with_xarray(self):
        """Test scattering computation with xarray input."""
        ifft = xr.DataArray(
            np.array([[0.5, 0.25], [0.1, 0.05]], dtype=np.complex64),
            dims=['y', 'x']
        )
        main = xr.DataArray(
            np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex64),
            dims=['y', 'x']
        )

        result = _compute_scattering(ifft, main)

        assert isinstance(result, xr.DataArray)
        assert np.isfinite(result.values).all()


class TestContrastRetrieval:
    """Tests for contrast_retrieval function."""

    @pytest.fixture
    def mock_harmonics(self):
        """Create mock harmonics DataArray for testing."""
        # Create harmonics in frequency space
        ky = np.linspace(-0.5, 0.5, 64)
        kx = np.linspace(-0.5, 0.5, 64)

        harmonics_list = ["harmonic_00", "harmonic_horizontal_positive"]
        # Create random complex data
        data = (np.random.rand(len(harmonics_list), 64, 64) +
                1j * np.random.rand(len(harmonics_list), 64, 64)).astype(np.complex64)

        da = xr.DataArray(
            data,
            dims=['harmonic', 'ky', 'kx'],
            coords={
                'harmonic': harmonics_list,
                'ky': ky,
                'kx': kx
            }
        )
        return da

    def test_contrast_retrieval_absorption(self, mock_harmonics):
        """Test absorption contrast retrieval."""
        result = contrast_retrieval(mock_harmonics, "absorption")

        # Absorption should be a single value per pixel
        assert 'y' in result.dims
        assert 'x' in result.dims
        assert result.dtype in [np.float32, np.float64]

    def test_contrast_retrieval_scattering(self, mock_harmonics):
        """Test scattering contrast retrieval."""
        result = contrast_retrieval(mock_harmonics, "scattering")

        assert 'y' in result.dims
        assert 'x' in result.dims
        # Result should have harmonic dimension (one per non-zero harmonic)
        assert 'harmonic' in result.dims

    def test_contrast_retrieval_unknown_type_raises(self, mock_harmonics):
        """Test that unknown contrast type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown type_of_contrast"):
            contrast_retrieval(mock_harmonics, "invalid_contrast_type")

    def test_contrast_retrieval_absorption_finite_values(self, mock_harmonics):
        """Test that absorption contrast has finite values."""
        result = contrast_retrieval(mock_harmonics, "absorption")

        # Compute if lazy
        if hasattr(result, 'compute'):
            result = result.compute()

        # Check all values are finite
        assert np.isfinite(result.values).all()

    def test_contrast_retrieval_with_chunked_input(self, mock_harmonics):
        """Test contrast retrieval with chunked (dask) input."""
        chunked = mock_harmonics.chunk({"harmonic": 1, "ky": "auto", "kx": "auto"})

        result = contrast_retrieval(chunked, "absorption")

        # Should return a lazy result
        assert result is not None
        # Compute to verify it works
        computed = result.compute() if hasattr(result, 'compute') else result
        assert np.isfinite(computed.values).all()


class TestContrastRetrievalEdgeCases:
    """Edge case tests for contrast retrieval."""

    def test_absorption_with_small_values(self):
        """Test absorption handles very small harmonic values."""
        ky = np.linspace(-0.5, 0.5, 32)
        kx = np.linspace(-0.5, 0.5, 32)

        # Create harmonics with very small values
        data = np.ones((2, 32, 32), dtype=np.complex64) * 1e-10
        data[0] = 1.0  # harmonic_00 should have reasonable values

        da = xr.DataArray(
            data,
            dims=['harmonic', 'ky', 'kx'],
            coords={
                'harmonic': ["harmonic_00", "harmonic_horizontal_positive"],
                'ky': ky,
                'kx': kx
            }
        )

        result = contrast_retrieval(da, "absorption")

        # Should not have NaN or inf
        computed = result.compute() if hasattr(result, 'compute') else result
        assert np.isfinite(computed.values).all()

    def test_scattering_with_zero_ifft(self):
        """Test scattering handles zero values in ifft."""
        ky = np.linspace(-0.5, 0.5, 16)
        kx = np.linspace(-0.5, 0.5, 16)

        # Create harmonics
        data = np.zeros((2, 16, 16), dtype=np.complex64)
        data[0] = 1.0  # harmonic_00

        da = xr.DataArray(
            data,
            dims=['harmonic', 'ky', 'kx'],
            coords={
                'harmonic': ["harmonic_00", "harmonic_horizontal_positive"],
                'ky': ky,
                'kx': kx
            }
        )

        result = contrast_retrieval(da, "scattering")

        # Should handle zero values without errors
        computed = result.compute() if hasattr(result, 'compute') else result
        assert computed is not None


class TestContrastModuleImports:
    """Tests for module imports and structure."""

    def test_can_import_contrast_functions(self):
        """Test all main functions can be imported."""
        from shimexpy.core.contrast import (
            contrast_retrieval,
            get_harmonics,
            get_contrast,
            get_contrasts,
            get_all_contrasts,
            get_all_harmonic_contrasts,
        )
        assert callable(contrast_retrieval)
        assert callable(get_harmonics)
        assert callable(get_contrast)
        assert callable(get_contrasts)
        assert callable(get_all_contrasts)
        assert callable(get_all_harmonic_contrasts)

    def test_can_import_internal_functions(self):
        """Test internal helper functions can be imported."""
        from shimexpy.core.contrast import (
            _compute_phase_map,
            _compute_scattering,
        )
        assert callable(_compute_phase_map)
        assert callable(_compute_scattering)


class TestComputePhaseMapViaContrastRetrieval:
    """Tests for phasemap contrast retrieval with different unwrap options."""

    def test_contrast_retrieval_phasemap_least_squares(self):
        """Test phasemap contrast retrieval with least_squares unwrap."""
        # Create mock harmonics in frequency space
        ky = np.linspace(-0.5, 0.5, 32)
        kx = np.linspace(-0.5, 0.5, 32)

        harmonics_list = ["harmonic_00", "harmonic_horizontal_positive"]
        data = (np.random.rand(len(harmonics_list), 32, 32) +
                1j * np.random.rand(len(harmonics_list), 32, 32)).astype(np.complex64)
        data[0] = data[0] + 1.0  # Ensure non-zero main harmonic

        da = xr.DataArray(
            data,
            dims=['harmonic', 'ky', 'kx'],
            coords={
                'harmonic': harmonics_list,
                'ky': ky,
                'kx': kx
            }
        )

        result = contrast_retrieval(da, "phasemap", unwrap="least_squares")

        assert result is not None
        assert 'y' in result.dims
        assert 'x' in result.dims

    def test_contrast_retrieval_phasemap_invalid_unwrap_raises(self):
        """Test that invalid unwrap algorithm raises ValueError."""
        ky = np.linspace(-0.5, 0.5, 16)
        kx = np.linspace(-0.5, 0.5, 16)

        harmonics_list = ["harmonic_00", "harmonic_horizontal_positive"]
        data = (np.random.rand(len(harmonics_list), 16, 16) +
                1j * np.random.rand(len(harmonics_list), 16, 16)).astype(np.complex64)
        data[0] = data[0] + 1.0

        da = xr.DataArray(
            data,
            dims=['harmonic', 'ky', 'kx'],
            coords={
                'harmonic': harmonics_list,
                'ky': ky,
                'kx': kx
            }
        )

        with pytest.raises(ValueError, match="Unknown phase unwrapping algorithm"):
            contrast_retrieval(da, "phasemap", unwrap="invalid_algorithm")


class TestGetHarmonicsWithBlockGrid:
    """Tests for get_harmonics with pre-existing block_grid."""

    def test_get_harmonics_with_existing_block_grid(self, reference_image):
        """Test get_harmonics uses existing block_grid when provided."""
        from shimexpy.core.contrast import get_harmonics

        # First call without block_grid to get one
        _, _, _, ref_block_grid = get_harmonics(reference_image, projected_grid=5.0)

        # Second call with existing block_grid (covers line 244-245)
        absorption, scattering, diff_phase, returned_grid = get_harmonics(
            reference_image, projected_grid=5.0, block_grid=ref_block_grid
        )

        assert isinstance(absorption, xr.DataArray)
        assert isinstance(scattering, xr.DataArray)
        # The returned grid should match the one we passed in
        assert returned_grid == ref_block_grid


class TestGetContrastDirections:
    """Tests for get_contrast with different directions and contrast types."""

    def test_get_contrast_vertical_scattering(self, reference_image, sample_image):
        """Test get_contrast with vertical_scattering."""
        from shimexpy.core.contrast import get_contrast, get_harmonics

        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        contrast = get_contrast(
            sample_image,
            ref_scattering,
            ref_block_grid,
            "vertical_scattering"
        )

        assert isinstance(contrast, (np.ndarray, xr.DataArray))

    def test_get_contrast_bidirectional_scattering(self, reference_image, sample_image):
        """Test get_contrast with bidirectional_scattering."""
        from shimexpy.core.contrast import get_contrast, get_harmonics

        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        contrast = get_contrast(
            sample_image,
            ref_scattering,
            ref_block_grid,
            "bidirectional_scattering"
        )

        assert isinstance(contrast, (np.ndarray, xr.DataArray))

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_get_contrast_vertical_phasemap(self, reference_image, sample_image):
        """Test get_contrast with vertical_phasemap (covers lines 306-308)."""
        from shimexpy.core.contrast import get_contrast, get_harmonics

        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        contrast = get_contrast(
            sample_image,
            ref_diff_phase,
            ref_block_grid,
            "vertical_phasemap"
        )

        assert isinstance(contrast, (np.ndarray, xr.DataArray))

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_get_contrast_bidirectional_phasemap(self, reference_image, sample_image):
        """Test get_contrast with bidirectional_phasemap (covers lines 309-311)."""
        from shimexpy.core.contrast import get_contrast, get_harmonics

        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        contrast = get_contrast(
            sample_image,
            ref_diff_phase,
            ref_block_grid,
            "bidirectional_phasemap"
        )

        assert isinstance(contrast, (np.ndarray, xr.DataArray))

    def test_get_contrast_invalid_type_raises(self, reference_image, sample_image):
        """Test get_contrast raises for invalid contrast type."""
        from shimexpy.core.contrast import get_contrast, get_harmonics

        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        with pytest.raises(ValueError, match="Unknown type_of_contrast"):
            get_contrast(
                sample_image,
                ref_absorption,
                ref_block_grid,
                "invalid_type"
            )


class TestGetContrastsFunction:
    """Tests for get_contrasts function (lines 345-382)."""

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_get_contrasts_returns_three_elements(self, reference_image, sample_image):
        """Test get_contrasts returns (absorption, scattering, phase)."""
        from shimexpy.core.contrast import get_harmonics, get_contrasts

        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        reference = (ref_absorption, ref_scattering, ref_diff_phase)

        absorption, scattering, diff_phase = get_contrasts(
            sample_image,
            reference,
            ref_block_grid
        )

        assert isinstance(absorption, (np.ndarray, xr.DataArray))
        assert isinstance(scattering, (np.ndarray, xr.DataArray))
        assert isinstance(diff_phase, (np.ndarray, xr.DataArray))


class TestGetAllContrastsFunction:
    """Tests for get_all_contrasts function (lines 410-437)."""

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_get_all_contrasts_basic(self, reference_image, sample_image):
        """Test get_all_contrasts computes all contrast types."""
        from shimexpy.core.contrast import get_all_contrasts

        absorption, scattering, diff_phase = get_all_contrasts(
            sample_image,
            reference_image,
            projected_grid=5.0
        )

        assert isinstance(absorption, (np.ndarray, xr.DataArray))
        assert isinstance(scattering, (np.ndarray, xr.DataArray))
        assert isinstance(diff_phase, (np.ndarray, xr.DataArray))


class TestGetAllHarmonicContrastsFunction:
    """Tests for get_all_harmonic_contrasts function (lines 466-531)."""

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_get_all_harmonic_contrasts_basic(self, reference_image, sample_image):
        """Test get_all_harmonic_contrasts returns expected structure."""
        from shimexpy.core.contrast import get_harmonics, get_all_harmonic_contrasts

        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        reference = (ref_absorption, ref_scattering, ref_diff_phase)

        result = get_all_harmonic_contrasts(
            sample_image,
            reference,
            ref_block_grid
        )

        assert isinstance(result, xr.DataArray)
        assert 'contrast' in result.dims
        # Should have absorption + (scattering + phase) for each direction
        assert len(result.coords['contrast']) > 0

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_get_all_harmonic_contrasts_contrast_labels(self, reference_image, sample_image):
        """Test get_all_harmonic_contrasts has expected contrast labels."""
        from shimexpy.core.contrast import get_harmonics, get_all_harmonic_contrasts

        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        reference = (ref_absorption, ref_scattering, ref_diff_phase)

        result = get_all_harmonic_contrasts(
            sample_image,
            reference,
            ref_block_grid
        )

        # Check expected contrast labels are present
        labels = list(result.coords['contrast'].values)
        assert 'absorption' in labels
        assert 'scattering_horizontal' in labels
        assert 'scattering_vertical' in labels
        assert 'diff_phase_horizontal' in labels
        assert 'diff_phase_vertical' in labels
