# ShimExPy Refactoring Plan

## Executive Summary

This plan addresses architectural issues, code quality improvements, and test coverage expansion to achieve 90% coverage. The work is divided into 6 phases with clear subtasks.

---

## Phase 1: Fix Critical Bugs and Hard Dependencies

### 1.1 Fix hard CuPy imports (breaks on systems without CUDA)

**Files affected:**
- `shimexpy/shimexpy/io/file_io.py:6` - `import cupy as cp` (unconditional)
- `shimexpy/shimexpy/utils/parallelization.py:2` - `import cupy as cp` (unconditional)

**Changes:**
```python
# Before
import cupy as cp

# After
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False
```

### 1.2 Remove GUI dependencies from core library

**File:** `shimexpy/shimexpy/utils/crop.py`
- Lines 6-7 import PySide6 which shouldn't be in core
- This widget belongs in `shimexpy_gui`, not core

**Action:** Delete `crop.py` from core or rewrite without GUI dependencies

### 1.3 Fix placeholder module

**File:** `shimexpy/shimexpy/utils/ffc.py`
- Currently just returns "Hello from ffc"
- Either implement flat-field correction or remove from exports

---

## Phase 2: Restructure Core Package

### 2.1 Create preprocessing module in core

Move scientific preprocessing from CLI to core:

**New file:** `shimexpy/shimexpy/preprocessing/__init__.py`
**New file:** `shimexpy/shimexpy/preprocessing/corrections.py`

Functions to move from `shimexpy_cli/corrections.py`:
- `correct_darkfield()` - dark field subtraction
- `correct_brightfield()` - bright field normalization
- `process_flat_correction()` - flat field correction

Functions to move from `shimexpy_cli/angles_correction.py`:
- `extracting_coordinates_of_peaks()`
- `calculating_angles_of_peaks_average()`
- `squared_fft()`

### 2.2 Simplify io module

**Current:** `shimexpy/shimexpy/io/file_io.py` with many thin wrappers

**Action:**
- Keep only `load_image()`, `save_image()` (with optional format detection)
- Move `cli_export()` to `shimexpy_cli`
- Remove `save_results()`/`load_results()` (users can use pickle directly)
- Remove `save_block_grid()`/`load_block_grid()` (users can use json directly)

### 2.3 Update __init__.py exports

After restructuring, update `shimexpy/shimexpy/__init__.py`:
- Add preprocessing functions
- Remove deprecated functions
- Fix ffc export

---

## Phase 3: Fix CLI Package

### 3.1 Update CLI to use core preprocessing

**File:** `shimexpy_cli/shimexpy_cli/corrections.py`
- Change to import from `shimexpy.preprocessing`
- Keep only CLI-specific orchestration logic

### 3.2 Move cli_export to CLI

**From:** `shimexpy/shimexpy/io/file_io.py`
**To:** `shimexpy_cli/shimexpy_cli/export.py`

### 3.3 Remove duplicate code

**File:** `shimexpy_cli/shimexpy_cli/angles_correction.py`
- Has duplicate functions: `zero_fft_region()`, `extracting_harmonic()`
- These duplicate `shimexpy/core/spatial_harmonics.py`
- Import from core instead

---

## Phase 4: Fix Tools Package

### 4.1 Fix broken imports

**File:** `shimexpy_tools/post_shi/morphostructural.py:23-25`
```python
# Broken
from src.shi_core.cleaner import Cleaner
from src.shi_core.exceptions import SHIError
from src.shi_core.logging import logger

# Fixed - use shimexpy_cli or create in shimexpy
from shimexpy_cli.exceptions import SHIError
from shimexpy_cli.logging import logger
```

### 4.2 Add pyproject.toml for tools

**New file:** `shimexpy_tools/pyproject.toml`
- Define dependencies on shimexpy and shimexpy_cli
- Add entry point for morphos command

---

## Phase 5: Comprehensive Test Suite (Target: 90% Coverage)

### 5.1 Fix existing tests

**File:** `tests/test_spatial_harmonics.py`
- Update imports (function names changed)
- Current: `from spatial_harmonics import shi_fft_linear_and_log`
- Should be: `from shimexpy.core.spatial_harmonics import shi_fft_cpu`

### 5.2 New test files needed

```
tests/
├── test_spatial_harmonics.py  # EXISTS - needs update
├── test_contrast.py           # NEW
├── test_unwrapping.py         # NEW
├── test_io.py                 # NEW
├── test_preprocessing.py      # NEW
├── test_visualization.py      # NEW
├── test_utils.py              # NEW
├── conftest.py                # NEW - shared fixtures
└── data/
    └── test_reference.tif     # EXISTS
```

### 5.3 Test coverage targets per module

| Module | Current | Target | Test File |
|--------|---------|--------|-----------|
| core/spatial_harmonics.py | ~40% | 90% | test_spatial_harmonics.py |
| core/contrast.py | 0% | 90% | test_contrast.py |
| core/unwrapping.py | 0% | 90% | test_unwrapping.py |
| io/file_io.py | 0% | 90% | test_io.py |
| preprocessing/corrections.py | 0% | 90% | test_preprocessing.py |
| visualization/plot.py | 0% | 85% | test_visualization.py |
| utils/parallelization.py | 0% | 90% | test_utils.py |

### 5.4 Detailed test cases

#### test_contrast.py
```python
# Functions to test:
- test_compute_phase_map_basic()
- test_compute_phase_map_with_unwrap()
- test_compute_scattering_basic()
- test_compute_scattering_edge_cases()
- test_contrast_retrieval_absorption()
- test_contrast_retrieval_scattering()
- test_contrast_retrieval_phasemap()
- test_contrast_retrieval_invalid_type()
- test_get_harmonics_basic()
- test_get_harmonics_with_block_grid()
- test_get_contrast_absorption()
- test_get_contrast_scattering()
- test_get_contrast_phasemap()
- test_get_contrasts_all()
- test_get_all_contrasts()
- test_get_all_harmonic_contrasts()
```

#### test_unwrapping.py
```python
# Functions to test:
- test_skimage_unwrap_2d_input()
- test_skimage_unwrap_3d_input()
- test_skimage_unwrap_complex_input()
- test_skimage_unwrap_invalid_dims()
- test_ls_unwrap_2d_input()
- test_ls_unwrap_3d_input()
- test_ls_unwrap_complex_input()
- test_ls_unwrap_output_shape()
- test_ls_unwrap_zero_mean()
```

#### test_io.py
```python
# Functions to test:
- test_load_image_tiff()
- test_load_image_unsupported_format()
- test_load_image_nonexistent()
- test_save_image_tiff()
- test_save_image_creates_dirs()
- test_save_image_xarray_input()
- test_save_image_unsupported_format()
- test_save_block_grid()
- test_load_block_grid()
- test_save_load_results_roundtrip()
```

#### test_preprocessing.py
```python
# Functions to test:
- test_correct_darkfield_basic()
- test_correct_darkfield_with_crop()
- test_correct_darkfield_with_rotation()
- test_correct_brightfield_basic()
- test_process_flat_correction()
- test_extracting_coordinates_of_peaks()
- test_calculating_angles_of_peaks_average()
```

#### test_visualization.py
```python
# Functions to test:
- test_plot_contrast_numpy()
- test_plot_contrast_xarray()
- test_plot_multiple_contrasts()
- test_compare_before_after()
```

#### conftest.py (shared fixtures)
```python
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_image():
    """Generate a synthetic test image with harmonic pattern."""
    x = np.linspace(0, 4*np.pi, 256)
    y = np.linspace(0, 4*np.pi, 256)
    X, Y = np.meshgrid(x, y)
    return (np.sin(X) + np.sin(Y) + 2).astype(np.float32)

@pytest.fixture
def reference_image(tmp_path):
    """Load actual test reference image."""
    test_dir = Path(__file__).parent
    return tifffile.imread(test_dir / "data" / "test_reference.tif")

@pytest.fixture
def complex_fft_array():
    """Generate complex FFT array for testing."""
    arr = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    return arr.astype(np.complex128)

@pytest.fixture
def tmp_tiff(tmp_path, sample_image):
    """Create temporary TIFF file for testing."""
    path = tmp_path / "test.tif"
    tifffile.imwrite(path, sample_image)
    return path
```

---

## Phase 6: Documentation and Cleanup

### 6.1 Update CLAUDE.md

After all changes, update CLAUDE.md with:
- New module structure
- Updated test commands
- New preprocessing module documentation

### 6.2 Update README.md

- Fix outdated project structure
- Update installation instructions
- Add preprocessing examples

### 6.3 Remove unused code

- `shimexpy/shimexpy/data/loader.py` - empty file
- Duplicate functions in CLI

---

## Implementation Order

### Week 1: Critical Fixes
1. [ ] 1.1 Fix hard CuPy imports
2. [ ] 1.2 Remove GUI deps from core
3. [ ] 1.3 Fix/remove ffc placeholder
4. [ ] 5.1 Fix existing tests

### Week 2: Core Restructuring
5. [ ] 2.1 Create preprocessing module
6. [ ] 2.2 Simplify io module
7. [ ] 2.3 Update __init__.py
8. [ ] 3.1-3.3 Update CLI

### Week 3: Tests Part 1
9. [ ] Create conftest.py with fixtures
10. [ ] test_spatial_harmonics.py updates
11. [ ] test_contrast.py
12. [ ] test_unwrapping.py

### Week 4: Tests Part 2
13. [ ] test_io.py
14. [ ] test_preprocessing.py
15. [ ] test_visualization.py
16. [ ] test_utils.py

### Week 5: Tools and Cleanup
17. [ ] 4.1-4.2 Fix tools package
18. [ ] 6.1-6.3 Documentation updates
19. [ ] Final coverage verification
20. [ ] Integration testing

---

## Success Criteria

- [ ] All imports work without CUDA installed
- [ ] Core package has no GUI dependencies
- [ ] No duplicate code between packages
- [ ] Test coverage >= 90%
- [ ] All tests pass
- [ ] CLI works end-to-end
- [ ] GUI works end-to-end
- [ ] Tools package works with correct imports
