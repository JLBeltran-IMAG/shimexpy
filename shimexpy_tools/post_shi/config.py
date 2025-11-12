"""Configuration settings for the application."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WindowConfig:
    title: str = "Two Image Containers — Mirroring Demo"
    width: int = 1400
    height: int = 800
    splitter_sizes: tuple[int, int] = (700, 700)

@dataclass
class ImageConfig:
    default_contrast: str = "linear"
    supported_formats: tuple[str, ...] = (".tif", ".png", ".jpg", ".jpeg")

class Config:
    window = WindowConfig()
    image = ImageConfig()
    
    @staticmethod
    def get_default_image_path() -> Path:
        return Path(__file__).parent.parent.parent / "docs" / "acq_scheme.png"
