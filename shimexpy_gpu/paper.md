---
title: "Shimexpy: an open-source Python package for Spatial Harmonic Imaging"
tags:
  - Python
  - Spatial Harmonic Imaging
  - Mesh-based X-ray Imaging
  - Multicontrast X-ray imaging
  - Fourier analysis
authors:
  - name: Jorge Luis Beltran Diaz
    orcid: 0000-0001-6657-5999
    affiliation: "1"
  - name: Danays Kunka
    orcid: 0000-0001-6500-7201
    affiliation: "1"

affiliations:
  - name: Institute of Microstructure Technology, Karlsruhe Institute of Technology
    index: 1

date:
bibliography: paper.bib
---

# Summary

Spatial Harmonic Imaging (SHI) is a single-shot multicontrast X-ray imaging technique [see Fig.\autoref{fig:setup_in_out_data}] capable of retrieving absorption, differential phase, and scattering information from a single exposure [@Wen_2008:2008;@Wen_2009:2009;@Wen_2010:2010;@He_2018:2018;@Lee_2018:2018;@He_2019:2019;@Sun_2019_1:2019;@Sun_2019_2:2019;@Sun_2022:2022;@Mikhaylov_2024:2024]. Although the modality is conceptually simple, SHI requires a well-defined computational workflow involving Fourier-domain analysis, spatial harmonic extraction, and multicontrast reconstruction [@Beltran_2025:2025].

Recently, Shimexpy, a Python package implementing the complete SHI analysis workflow in a portable, modular, and reproducible manner, was introduced in SoftwareX [cite]. The framework provides flat-field correction, harmonic extraction, phase unwrapping [@GhigliaPritt1998:1998;@Herraez_2002:2002], and multicontrast reconstruction through a unified Python API, graphical user interface (GUI), and command-line interface (CLI).

While single-shot acquisition makes SHI inherently suitable for real-time applications, achieving low-latency processing is not straightforward. The general-purpose Shimexpy implementation is designed for flexibility and reproducibility rather than strict latency constraints, and its CPU-based execution model limits its suitability for real-time processing during acquisition [cite].

In this work, we introduce Shimexpy-GPU, a GPU-native execution engine specifically designed for real-time SHI processing. Unlike conventional approaches based on direct CPU-to-GPU porting, the proposed implementation restructures the computational pipeline by separating reference-dependent operations from per-frame processing, precomputing harmonic indices, and employing batch Fourier reconstruction together with fused elementwise kernels for contrast retrieval. This design minimizes per-frame overhead and enables low-latency execution suitable for real-time acquisition scenarios.



![Spatial harmonic imaging technique. (a) Experimental setup; (b) Input data to Shimexpy workflow corresponding to dark-field, bright-field, reference, and sample images; (c) Main output corresponding to absorption, bidirectional scattering, and bidirectional differential phase contrasts.\label{fig:setup_in_out_data}](Setup.png)

# Statement of Need

Researchers working with Spatial Harmonic Imaging (SHI) require computational tools capable of processing SHI data under different experimental conditions, particularly in scenarios involving dynamic processes. While the SHI methodology is compatible with single-shot acquisition, practical use in time-resolved experiments depends on the availability of efficient processing pipelines.  

Shimexpy provides a general-purpose, reproducible implementation of the SHI workflow and supports offline and near real-time analysis. However, its design prioritizes flexibility over strict latency constraints, which limits its use in experiments requiring millisecond-scale temporal resolution.  

Shimexpy GPU addresses this limitation by introducing a GPU-native execution pipeline designed for low-latency processing. The implementation restructures the computational workflow by separating reference-dependent operations from per-frame processing and reducing per-frame overhead, enabling consistent performance in real-time acquisition scenarios.  

To the best of our knowledge, no open-source Python software currently provides a GPU-native, real-time processing pipeline specifically designed for SHI.

# Overview of Shimexpy



# Module structure



# Availability, Reproducibility, and Licensing

- **Source code:**  
  https://github.com/JLBeltran-IMAG/shimexpy

- **Dataset (Zenodo):**  
  https://doi.org/10.5281/zenodo.17347347

# Acknowledgements

The authors acknowledge support from the Karlsruhe School of Optics and Photonics (KSOP) and the Ministry of Science, Research and Arts of Baden-Württemberg.

# References
