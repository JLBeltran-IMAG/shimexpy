
# Getting Started

Shimexpy provides Python tools for spatial harmonic and mesh-based X-ray imaging. The project is organized into six components: **CORE, GUI, CLI, GPU, TOOLS, and SIM**.

Each component serves a distinct purpose, but all share the same objective: reconstructing multicontrast X-ray images from experimental Spatial Harmonic Imaging (SHI) data through the analysis of spatial harmonic frequencies in Fourier space.

> **Note:** Some components are still under development and have not yet been publicly released.

## Publication and Citation

For more information about Shimexpy, please refer (and cite :smile:) to our publication in *SoftwareX*:

[*Shimexpy: A Python package for spatial harmonic imaging*, **Beltran, J. L., & Kunka, D. (2026).**](https://doi.org/10.1016/j.softx.2026.102700)


## Install from the repository

Use Python 3.12 for the commands below. Clone the repository and install the core
library in a virtual environment:

```bash
git clone https://github.com/JLBeltran-IMAG/shimexpy.git
cd shimexpy
python -m venv .venv
source .venv/bin/activate
python -m pip install ./shimexpy
```

## Load an image

```python
from shimexpy.io.file_io import load_image

image = load_image("sample.tif")
print(image.shape)
```

## Choose an interface

- [Python API](../api/api.md)
- [Command-line Interface](../cli/cli.md)
- [Graphical User Interface](../gui/gui.md)

[Back to Home](../index.md)
