"""
Shared pytest fixtures for shimexpy tests.

These fixtures provide test data and utilities used across multiple test modules.
"""

import pytest
import numpy as np
from pathlib import Path
import tifffile
import sys
import os

# Clear any existing shimexpy imports and add correct path
_mods_to_remove = [k for k in list(sys.modules.keys()) if 'shimexpy' in k]
for _m in _mods_to_remove:
    del sys.modules[_m]

# Add the shimexpy package to path (handles nested structure)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shimexpy")))


# Path to test data directory
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR


@pytest.fixture
def sample_image_small():
    """Generate a small synthetic test image with harmonic pattern (64x64)."""
    x = np.linspace(0, 4 * np.pi, 64)
    y = np.linspace(0, 4 * np.pi, 64)
    X, Y = np.meshgrid(x, y)
    # Create image with periodic pattern (simulates grid pattern)
    image = (np.sin(X * 5) + np.sin(Y * 5) + 2).astype(np.float32)
    return image


@pytest.fixture
def sample_image_medium():
    """Generate a medium synthetic test image with harmonic pattern (256x256)."""
    x = np.linspace(0, 4 * np.pi, 256)
    y = np.linspace(0, 4 * np.pi, 256)
    X, Y = np.meshgrid(x, y)
    # Create image with periodic pattern
    image = (np.sin(X * 5) + np.sin(Y * 5) + 2).astype(np.float32)
    return image


@pytest.fixture
def reference_image():
    """Load the actual test reference image if available."""
    ref_path = TEST_DATA_DIR / "test_reference.tif"
    if ref_path.exists():
        return tifffile.imread(ref_path).astype(np.float32)
    else:
        pytest.skip("test_reference.tif not found")


@pytest.fixture
def sample_image():
    """Load the actual test sample image if available."""
    sample_path = TEST_DATA_DIR / "test_sample.tif"
    if sample_path.exists():
        return tifffile.imread(sample_path).astype(np.float32)
    else:
        pytest.skip("test_sample.tif not found")


@pytest.fixture
def complex_fft_array():
    """Generate complex FFT array for testing FFT-related functions."""
    np.random.seed(42)  # For reproducibility
    arr = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    return arr.astype(np.complex128)


@pytest.fixture
def wrapped_phase():
    """Generate a wrapped phase array for testing unwrapping."""
    x = np.linspace(-3 * np.pi, 3 * np.pi, 100)
    y = np.linspace(-3 * np.pi, 3 * np.pi, 100)
    X, Y = np.meshgrid(x, y)
    # Create a phase that wraps around multiple times
    phase = X + Y
    wrapped = np.angle(np.exp(1j * phase))
    return wrapped.astype(np.float32)


@pytest.fixture
def tmp_tiff_file(tmp_path, sample_image_small):
    """Create a temporary TIFF file for testing I/O."""
    path = tmp_path / "test_image.tif"
    tifffile.imwrite(path, sample_image_small)
    return path


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# Constants for expected values (to be filled after baseline tests)
EXPECTED_HARMONIC_LABELS = {
    "harmonic_00",
    "harmonic_horizontal_positive",
    "harmonic_horizontal_negative",
    "harmonic_vertical_positive",
    "harmonic_vertical_negative",
    "harmonic_diagonal_p1_p1",
    "harmonic_diagonal_n1_p1",
    "harmonic_diagonal_n1_n1",
    "harmonic_diagonal_p1_n1"
}
