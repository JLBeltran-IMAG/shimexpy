"""
GPU-native backend for Shimexpy.

This package implements a batch/stack-based GPU pipeline
for real-time SHI contrast retrieval.
"""
from shimexpy_gpu.core.api import (
    prepare_reference,
    process_frame
)

__all__ = [
    "prepare_reference",
    "process_frame"
]