# Shimexpy

Shimexpy is a Python package for **Spatial Harmonic Imaging (SHI)** and **mesh-based X-ray multicontrast imaging**. It provides tools for Fourier-domain harmonic analysis and the reconstruction of absorption, dark-field (scattering), and differential phase contrast from X-ray images.

---

## Installation

```bash
pip install shimexpy[all]
```

Individual components:

```bash
pip install shimexpy[core]
pip install shimexpy[gui]
pip install shimexpy[cli]
```

GPU support (optional):

```bash
pip install shimexpy-core[cuda12x]
pip install shimexpy-core[cuda11x]
```

---

## Usage

```python
import shimexpy
```

```bash
shimexpy
```

```bash
shi
```

---

## License

Apache License 2.0
