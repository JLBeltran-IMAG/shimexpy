"""
Integration baseline tests for shimexpy.

These tests capture the current behavior of the main API functions to ensure
refactoring doesn't break existing functionality. Run these BEFORE making
changes to establish a baseline, then run again AFTER to verify no regressions.

Usage:
    pytest tests/test_integration_baseline.py -v
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

# Add the shimexpy package to path (handles nested structure)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shimexpy")))

from shimexpy import (
    shi_fft,
    shi_fft_cpu,
    spatial_harmonics_of_fourier_spectrum,
    get_harmonics,
    get_contrast,
    get_contrasts,
    get_all_contrasts,
    contrast_retrieval,
    skimage_unwrap,
    ls_unwrap,
    load_image,
    save_image,
)
from shimexpy.core.spatial_harmonics import FFTResult

from conftest import EXPECTED_HARMONIC_LABELS


class TestFFTBaseline:
    """Baseline tests for FFT functions."""

    def test_shi_fft_cpu_returns_fft_result(self, sample_image_small):
        """Verify shi_fft_cpu returns FFTResult dataclass."""
        result = shi_fft_cpu(sample_image_small)
        assert isinstance(result, FFTResult)
        assert result.fft is not None
        assert result.fft.shape == sample_image_small.shape
        # Without projected_grid, kx and ky should be None
        assert result.kx is None
        assert result.ky is None

    def test_shi_fft_cpu_with_grid(self, sample_image_small):
        """Verify shi_fft_cpu with projected_grid returns frequency axes."""
        projected_grid = 5.0
        result = shi_fft_cpu(sample_image_small, projected_grid=projected_grid)
        assert result.kx is not None
        assert result.ky is not None
        assert len(result.kx) == sample_image_small.shape[1]
        assert len(result.ky) == sample_image_small.shape[0]

    def test_shi_fft_cpu_logspect(self, sample_image_small):
        """Verify logspect option produces non-negative values."""
        result = shi_fft_cpu(sample_image_small, logspect=True)
        assert np.all(result.fft >= 0), "Log spectrum should be non-negative"

    def test_shi_fft_auto_selects_cpu_without_cuda(self, sample_image_small):
        """Verify shi_fft works (falls back to CPU without CUDA)."""
        result = shi_fft(sample_image_small)
        assert isinstance(result, FFTResult)
        assert result.fft.shape == sample_image_small.shape


class TestSpatialHarmonicsBaseline:
    """Baseline tests for spatial harmonics extraction."""

    def test_spatial_harmonics_reference_mode_labels(self, reference_image):
        """Verify spatial_harmonics returns expected harmonic labels in reference mode."""
        result = shi_fft_cpu(reference_image, projected_grid=5.0)
        da, block_grid = spatial_harmonics_of_fourier_spectrum(
            result.fft, result.ky, result.kx, reference=True, limit_band=0.5
        )

        # Check return types
        assert isinstance(da, xr.DataArray)
        assert isinstance(block_grid, dict)

        # Check expected labels
        assert set(block_grid.keys()) == EXPECTED_HARMONIC_LABELS

        # Check block_grid structure
        for label, limits in block_grid.items():
            assert len(limits) == 4, f"{label} should have 4 limits [top, bottom, left, right]"
            assert all(isinstance(x, (int, np.integer)) for x in limits)

    def test_spatial_harmonics_non_reference_mode(self, reference_image):
        """Verify non-reference mode uses provided block_grid."""
        result = shi_fft_cpu(reference_image, projected_grid=5.0)

        # First, get block_grid from reference mode
        _, ref_block_grid = spatial_harmonics_of_fourier_spectrum(
            result.fft, result.ky, result.kx, reference=True, limit_band=0.5
        )

        # Then use it in non-reference mode
        da_nonref, block_grid_nonref = spatial_harmonics_of_fourier_spectrum(
            result.fft, None, None, reference=False, reference_block_grid=ref_block_grid
        )

        # Block grids should match
        assert block_grid_nonref == ref_block_grid

    def test_spatial_harmonics_raises_without_block_grid(self, sample_image_small):
        """Verify error is raised when reference=False without block_grid."""
        result = shi_fft_cpu(sample_image_small)

        with pytest.raises(ValueError, match="Reference block grid.*must be provided"):
            spatial_harmonics_of_fourier_spectrum(
                result.fft, None, None, reference=False, reference_block_grid=None
            )


class TestGetHarmonicsBaseline:
    """Baseline tests for get_harmonics function."""

    def test_get_harmonics_returns_four_elements(self, reference_image):
        """Verify get_harmonics returns (absorption, scattering, phase, block_grid)."""
        absorption, scattering, diff_phase, block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        # Check types
        assert isinstance(absorption, xr.DataArray)
        assert isinstance(scattering, xr.DataArray)
        assert isinstance(diff_phase, xr.DataArray)
        assert isinstance(block_grid, dict)

    def test_get_harmonics_output_shapes(self, reference_image):
        """Verify get_harmonics output shapes are consistent."""
        absorption, scattering, diff_phase, block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        # Absorption should be scalar-like (single harmonic processed)
        # Scattering and phase should have harmonic dimension
        assert "harmonic" in scattering.dims or scattering.ndim >= 2
        assert "harmonic" in diff_phase.dims or diff_phase.ndim >= 2

    def test_get_harmonics_block_grid_structure(self, reference_image):
        """Verify block_grid has expected structure."""
        _, _, _, block_grid = get_harmonics(reference_image, projected_grid=5.0)

        assert set(block_grid.keys()) == EXPECTED_HARMONIC_LABELS


class TestGetContrastBaseline:
    """Baseline tests for get_contrast function."""

    def test_get_contrast_absorption(self, reference_image, sample_image):
        """Verify get_contrast works for absorption."""
        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        contrast = get_contrast(
            sample_image, ref_absorption, ref_block_grid, "absorption"
        )

        assert isinstance(contrast, (np.ndarray, xr.DataArray))

    def test_get_contrast_horizontal_scattering(self, reference_image, sample_image):
        """Verify get_contrast works for horizontal_scattering."""
        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        contrast = get_contrast(
            sample_image, ref_scattering, ref_block_grid, "horizontal_scattering"
        )

        assert isinstance(contrast, (np.ndarray, xr.DataArray))

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_get_contrast_horizontal_phasemap(self, reference_image, sample_image):
        """Verify get_contrast works for horizontal_phasemap."""
        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        contrast = get_contrast(
            sample_image, ref_diff_phase, ref_block_grid, "horizontal_phasemap"
        )

        assert isinstance(contrast, (np.ndarray, xr.DataArray))


class TestGetAllContrastsBaseline:
    """Baseline tests for get_all_contrasts function."""

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_get_all_contrasts_returns_three_elements(self, reference_image, sample_image):
        """Verify get_all_contrasts returns (absorption, scattering, phase)."""
        absorption, scattering, diff_phase = get_all_contrasts(
            sample_image, reference_image, projected_grid=5.0
        )

        assert isinstance(absorption, (np.ndarray, xr.DataArray))
        assert isinstance(scattering, (np.ndarray, xr.DataArray))
        assert isinstance(diff_phase, (np.ndarray, xr.DataArray))


class TestUnwrappingBaseline:
    """Baseline tests for phase unwrapping functions."""

    def test_skimage_unwrap_2d_input(self, wrapped_phase):
        """Verify skimage_unwrap works with 2D input."""
        result = skimage_unwrap(wrapped_phase)

        # Output should be 3D with shape (1, M, N)
        assert result.ndim == 3
        assert result.shape[0] == 1
        assert result.shape[1:] == wrapped_phase.shape

    def test_skimage_unwrap_3d_input(self, wrapped_phase):
        """Verify skimage_unwrap works with 3D input (1, M, N)."""
        input_3d = wrapped_phase[np.newaxis, ...]
        result = skimage_unwrap(input_3d)

        assert result.ndim == 3
        assert result.shape == input_3d.shape

    def test_ls_unwrap_2d_input(self, wrapped_phase):
        """Verify ls_unwrap works with 2D input."""
        result = ls_unwrap(wrapped_phase)

        # Output should be 3D with shape (1, M, N)
        assert result.ndim == 3
        assert result.shape[0] == 1
        assert result.shape[1:] == wrapped_phase.shape

    def test_ls_unwrap_output_is_real(self, wrapped_phase):
        """Verify ls_unwrap output is real-valued."""
        result = ls_unwrap(wrapped_phase)
        assert np.isrealobj(result)


class TestIOBaseline:
    """Baseline tests for I/O functions."""

    def test_load_image_tiff(self, tmp_tiff_file):
        """Verify load_image can load TIFF files."""
        image = load_image(tmp_tiff_file)
        assert isinstance(image, np.ndarray)
        assert image.ndim == 2

    def test_save_image_tiff(self, sample_image_small, tmp_output_dir):
        """Verify save_image can save TIFF files."""
        output_path = tmp_output_dir / "output.tif"
        save_image(sample_image_small, output_path)

        assert output_path.exists()

        # Verify we can load it back
        loaded = load_image(output_path)
        assert loaded.shape == sample_image_small.shape

    def test_save_load_roundtrip(self, sample_image_small, tmp_output_dir):
        """Verify save/load roundtrip preserves data."""
        output_path = tmp_output_dir / "roundtrip.tif"
        save_image(sample_image_small, output_path)
        loaded = load_image(output_path)

        # Should be close (float32 precision)
        np.testing.assert_allclose(loaded, sample_image_small, rtol=1e-5)


class TestEndToEndBaseline:
    """End-to-end baseline tests for full pipeline."""

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_full_pipeline_reference_to_contrast(self, reference_image, sample_image):
        """Test complete pipeline from reference to contrast computation."""
        # Step 1: Process reference
        ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
            reference_image, projected_grid=5.0
        )

        # Step 2: Compute contrasts
        absorption, scattering, diff_phase = get_all_contrasts(
            sample_image, reference_image, projected_grid=5.0
        )

        # Verify outputs are finite
        assert np.all(np.isfinite(np.asarray(absorption)))
        assert np.all(np.isfinite(np.asarray(scattering)))
        # Phase might have some edge artifacts, but center should be finite
        phase_arr = np.asarray(diff_phase)
        center_region = phase_arr[
            phase_arr.shape[0]//4:3*phase_arr.shape[0]//4,
            phase_arr.shape[1]//4:3*phase_arr.shape[1]//4
        ]
        assert np.all(np.isfinite(center_region))

    @pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
    def test_full_pipeline_produces_reasonable_values(self, reference_image, sample_image):
        """Verify pipeline produces values in reasonable ranges."""
        absorption, scattering, diff_phase = get_all_contrasts(
            sample_image, reference_image, projected_grid=5.0
        )

        # Absorption contrast shouldn't be extremely large
        abs_arr = np.asarray(absorption)
        assert np.abs(abs_arr).max() < 100, "Absorption values seem unreasonably large"

        # Scattering should be bounded
        scat_arr = np.asarray(scattering)
        assert np.abs(scat_arr).max() < 100, "Scattering values seem unreasonably large"


# Store baseline values for regression testing
# These will be populated after running the baseline tests
BASELINE_VALUES = {}


@pytest.mark.skip(reason="skimage unwrap_phase segfaults with dask parallelization")
def test_capture_baseline_values(reference_image, sample_image):
    """
    Capture baseline values for future regression testing.

    This test always passes but prints values that should be stored
    for regression testing after refactoring.
    """
    absorption, scattering, diff_phase = get_all_contrasts(
        sample_image, reference_image, projected_grid=5.0
    )

    # Capture statistics
    abs_arr = np.asarray(absorption)
    scat_arr = np.asarray(scattering)
    phase_arr = np.asarray(diff_phase)

    baseline = {
        "absorption_mean": float(np.mean(abs_arr)),
        "absorption_std": float(np.std(abs_arr)),
        "absorption_min": float(np.min(abs_arr)),
        "absorption_max": float(np.max(abs_arr)),
        "scattering_mean": float(np.mean(scat_arr)),
        "scattering_std": float(np.std(scat_arr)),
        "diff_phase_mean": float(np.nanmean(phase_arr)),
        "diff_phase_std": float(np.nanstd(phase_arr)),
    }

    print("\n=== BASELINE VALUES (save these for regression testing) ===")
    for key, value in baseline.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Store for potential use
    BASELINE_VALUES.update(baseline)

    # This test always passes - it's for capturing values
    assert True
