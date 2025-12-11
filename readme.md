# Shimexpy: A python package for Spatial Harmonics Imaging and mesh-based X-ray Imaging

ShimExPy is a Python package for spatial harmonics X-ray imaging analysis. It provides tools for performing spatial harmonics analysis on X-ray images, extracting absorption, scattering, and phase contrast.

## Installation

### Prerequisites

- Python 3.7+
- CUDA (for GPU acceleration) - optional, but recomandable
- 

### Basic Installation

```bash
pip install shimexpy
```

### Development Installation

```bash
git clone https://github.com/yourusername/shimexpy.git
cd shimexpy
pip install -e ".[dev]"
```

El proyecto utiliza ahora `pyproject.toml` en lugar de `setup.py` para configuración y empaquetado, siguiendo los estándares modernos de Python.

### GUI Installation

```bash
pip install "shimexpy[gui]"
```

## Usage

### Basic Example

```python
from tifffile import imread
import matplotlib.pyplot as plt
from shimexpy import get_harmonics, get_contrast

# Load images
reference_img = imread("reference.tif")
sample_img = imread("sample.tif")

# Process reference image
ref_absorption, ref_scattering, ref_diff_phase, ref_block_grid = get_harmonics(
    reference_img, projected_grid=5
)

# Compute contrast
contrast = get_contrast(
    sample_img, ref_diff_phase, ref_block_grid, "horizontal_phasemap"
)

# Display result
plt.imshow(contrast, cmap='gray')
plt.colorbar()
plt.title('Horizontal Phase Contrast')
plt.show()
```

### Running the GUI

```bash
# If installed with the GUI option
shimexpy
```

## Project Structure

```
shimexpy/
├── core/
│   ├── spatial_harmonics.py  - Core FFT and spatial harmonics functions
│   ├── unwrapping.py        - Phase unwrapping algorithms
│   └── contrast.py          - Contrast computation functions
├── io/
│   └── file_io.py           - File input/output utilities
├── utils/
│   └── crop.py              - Image cropping utilities
├── gui/
│   ├── image_widget.py     - Widget for image display and ROI selection
│   ├── image_processor.py  - Processing logic controller
│   ├── shimexpy_gui.py     - Main refactored GUI interface
│   └── app.py             - Application entry point
└── visualization/
    └── plot.py              - Plotting utilities
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation

If you use ShimExPy in your research, please cite:

```
@software{shimexpy,
  author       = {Jorge Luis Beltran Diaz},
  title        = {ShiMeXpy: Spatial Harmonics Imaging for X-ray Physics},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/shimexpy}}
}
```