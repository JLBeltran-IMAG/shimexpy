![Shimexpy logo](logo.png)

# Shimexpy: A Python package for Spatial Harmonic Imaging and mesh-based X-ray imaging

Shimexpy is an open-source Python package for **Spatial Harmonic Imaging (SHI)**, also known as **mesh-based X-ray multicontrast imaging**. It provides a reproducible and cross-platform computational pipeline for Fourier-domain harmonic extraction and the reconstruction of attenuation, differential phase, and scattering contrasts from single-shot X-ray measurements.

---

## Requirements

- Python ≥ 3.7
- CUDA (optional, for GPU acceleration via CuPy)

---

## Installation

### Install full ecosystem (recommended)

```bash
pip install shimexpy[all]
```

### Install selected components

```bash
pip install shimexpy[core]
pip install shimexpy[gui]
pip install shimexpy[cli]
```

### Install components individually

```bash
pip install shimexpy-core
pip install shimexpy_gui
pip install shimexpy_cli
```

### GPU support (optional)

Select the CUDA version explicitly:

```bash
pip install shimexpy-core[cuda12x]
pip install shimexpy-core[cuda11x]
```

---

## Usage

### Python API (core)

```python
from shimexpy import load_image, ffc, get_all_contrasts

reference = load_image("reference.tif")
sample = load_image("sample.tif")
bright = load_image("bright.tif")
dark = load_image("dark.tif")

reference_ffc = ffc(reference, dark, bright)
sample_ffc = ffc(sample, dark, bright)

absorption, scattering, dpc = get_all_contrasts(
    sample_ffc, reference_ffc, projected_grid=5
)
```

### Graphical user interface (GUI)

```bash
shimexpy
```

The GUI provides interactive access to the full SHI workflow, including image loading,
Fourier visualization, harmonic extraction, ROI selection, and contrast reconstruction.

### Command-line interface (CLI)

```bash
shi
```

The CLI enables scripted and batch processing of SHI datasets.

---

## Development installation

```bash
git clone https://github.com/JLBeltran-IMAG/shimexpy.git
cd shimexpy
pip install -e .
```

This project uses `pyproject.toml` (PEP 517/518). No `setup.py` is required.

---

## Repository layout

```
shimexpy/
├── shimexpy/          # Core SHI algorithms and processing pipeline
├── shimexpy_gui/      # Graphical user interface (PySide6)
├── shimexpy_cli/      # Command-line interface
├── shimexpy_gpu/      # Optional GPU backend utilities
├── pyproject.toml     # Installer / meta-package
└── README.md
```

Each submodule is versioned and distributed independently via PyPI.

---

## License

Apache License 2.0

---

## Citation

```bibtex
@article{BeltranDiaz_Shimexpy,
  title   = {Shimexpy: A Python package for Spatial Harmonic Imaging},
  author  = {Beltran Diaz, Jorge Luis and Kunka, Danays},
  year    = {2026}
}
```