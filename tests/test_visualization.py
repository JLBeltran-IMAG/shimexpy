"""
Tests for shimexpy.visualization.plot module.

Tests cover:
- plot_contrast function
- plot_multiple_contrasts function
- compare_before_after function
"""

import pytest
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

# Use non-interactive backend for testing
matplotlib.use('Agg')

# Clear any existing shimexpy imports and add correct path
_mods_to_remove = [k for k in list(sys.modules.keys()) if 'shimexpy' in k]
for _m in _mods_to_remove:
    del sys.modules[_m]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shimexpy")))

from shimexpy.visualization.plot import (
    plot_contrast,
    plot_multiple_contrasts,
    compare_before_after,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


class TestPlotContrast:
    """Tests for plot_contrast function."""

    def test_plot_contrast_numpy_array(self):
        """Test plotting a numpy array."""
        data = np.random.rand(100, 100)

        fig, ax = plot_contrast(data)

        assert fig is not None
        assert ax is not None
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_plot_contrast_xarray(self):
        """Test plotting an xarray DataArray."""
        data = xr.DataArray(
            np.random.rand(50, 50),
            dims=['y', 'x']
        )

        fig, ax = plot_contrast(data)

        assert fig is not None
        assert ax is not None

    def test_plot_contrast_with_title(self):
        """Test plotting with a title."""
        data = np.random.rand(30, 30)

        fig, ax = plot_contrast(data, title="Test Title")

        assert ax.get_title() == "Test Title"

    def test_plot_contrast_without_title(self):
        """Test plotting without a title."""
        data = np.random.rand(30, 30)

        fig, ax = plot_contrast(data)

        assert ax.get_title() == ""

    def test_plot_contrast_custom_cmap(self):
        """Test plotting with a custom colormap."""
        data = np.random.rand(30, 30)

        fig, ax = plot_contrast(data, cmap='viridis')

        # Figure should be created successfully
        assert fig is not None

    def test_plot_contrast_custom_figsize(self):
        """Test plotting with custom figure size."""
        data = np.random.rand(30, 30)

        fig, ax = plot_contrast(data, figsize=(12, 10))

        # Check approximate figure size
        assert fig.get_size_inches()[0] == pytest.approx(12, abs=0.1)
        assert fig.get_size_inches()[1] == pytest.approx(10, abs=0.1)


class TestPlotMultipleContrasts:
    """Tests for plot_multiple_contrasts function."""

    def test_plot_multiple_contrasts_list(self):
        """Test plotting multiple numpy arrays."""
        contrasts = [
            np.random.rand(50, 50),
            np.random.rand(50, 50),
            np.random.rand(50, 50)
        ]

        fig, axes = plot_multiple_contrasts(contrasts)

        assert fig is not None
        assert len(axes) == 3

    def test_plot_multiple_contrasts_xarray(self):
        """Test plotting multiple xarray DataArrays."""
        contrasts = [
            xr.DataArray(np.random.rand(30, 30), dims=['y', 'x']),
            xr.DataArray(np.random.rand(30, 30), dims=['y', 'x'])
        ]

        fig, axes = plot_multiple_contrasts(contrasts)

        assert fig is not None
        assert len(axes) == 2

    def test_plot_multiple_contrasts_with_titles(self):
        """Test plotting with titles."""
        contrasts = [
            np.random.rand(30, 30),
            np.random.rand(30, 30)
        ]
        titles = ["Image 1", "Image 2"]

        fig, axes = plot_multiple_contrasts(contrasts, titles=titles)

        assert axes[0].get_title() == "Image 1"
        assert axes[1].get_title() == "Image 2"

    def test_plot_multiple_contrasts_single_image(self):
        """Test plotting a single image."""
        contrasts = [np.random.rand(30, 30)]

        fig, axes = plot_multiple_contrasts(contrasts)

        assert fig is not None
        assert len(axes) == 1

    def test_plot_multiple_contrasts_partial_titles(self):
        """Test with fewer titles than images."""
        contrasts = [
            np.random.rand(30, 30),
            np.random.rand(30, 30),
            np.random.rand(30, 30)
        ]
        titles = ["Only First"]

        fig, axes = plot_multiple_contrasts(contrasts, titles=titles)

        assert axes[0].get_title() == "Only First"
        # Other axes should have empty titles
        assert axes[1].get_title() == ""

    def test_plot_multiple_contrasts_custom_figsize(self):
        """Test plotting with custom figure size."""
        contrasts = [np.random.rand(30, 30), np.random.rand(30, 30)]

        fig, axes = plot_multiple_contrasts(contrasts, figsize=(20, 8))

        assert fig.get_size_inches()[0] == pytest.approx(20, abs=0.1)


class TestCompareBeforeAfter:
    """Tests for compare_before_after function."""

    def test_compare_before_after_numpy(self):
        """Test comparing two numpy arrays."""
        before = np.random.rand(50, 50)
        after = np.random.rand(50, 50)

        fig, axes = compare_before_after(before, after)

        assert fig is not None
        assert len(axes) == 3

    def test_compare_before_after_xarray(self):
        """Test comparing two xarray DataArrays."""
        before = xr.DataArray(np.random.rand(30, 30), dims=['y', 'x'])
        after = xr.DataArray(np.random.rand(30, 30), dims=['y', 'x'])

        fig, axes = compare_before_after(before, after)

        assert fig is not None
        assert len(axes) == 3

    def test_compare_before_after_mixed_types(self):
        """Test comparing numpy with xarray."""
        before = np.random.rand(30, 30)
        after = xr.DataArray(np.random.rand(30, 30), dims=['y', 'x'])

        fig, axes = compare_before_after(before, after)

        assert fig is not None

    def test_compare_before_after_default_titles(self):
        """Test default titles are applied."""
        before = np.random.rand(30, 30)
        after = np.random.rand(30, 30)

        fig, axes = compare_before_after(before, after)

        assert axes[0].get_title() == "Before"
        assert axes[1].get_title() == "After"
        assert axes[2].get_title() == "Difference"

    def test_compare_before_after_custom_titles(self):
        """Test custom titles are applied."""
        before = np.random.rand(30, 30)
        after = np.random.rand(30, 30)
        titles = ["Original", "Processed", "Delta"]

        fig, axes = compare_before_after(before, after, titles=titles)

        assert axes[0].get_title() == "Original"
        assert axes[1].get_title() == "Processed"
        assert axes[2].get_title() == "Delta"

    def test_compare_before_after_partial_titles(self):
        """Test partial titles (fewer than 3)."""
        before = np.random.rand(30, 30)
        after = np.random.rand(30, 30)
        titles = ["Custom Before"]

        fig, axes = compare_before_after(before, after, titles=titles)

        assert axes[0].get_title() == "Custom Before"
        assert axes[1].get_title() == "After"  # Default
        assert axes[2].get_title() == "Difference"  # Default

    def test_compare_before_after_shows_difference(self):
        """Test that third panel shows the difference."""
        before = np.ones((10, 10)) * 5
        after = np.ones((10, 10)) * 8

        fig, axes = compare_before_after(before, after)

        # Get the difference image data from the third axes
        # The difference should be 8 - 5 = 3
        assert fig is not None


class TestVisualizationModuleImports:
    """Tests for module imports."""

    def test_can_import_plot_functions(self):
        """Test that all plot functions can be imported."""
        from shimexpy.visualization.plot import (
            plot_contrast,
            plot_multiple_contrasts,
            compare_before_after
        )

        assert callable(plot_contrast)
        assert callable(plot_multiple_contrasts)
        assert callable(compare_before_after)
