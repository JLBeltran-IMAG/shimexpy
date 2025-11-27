"""Image loading and validation utilities."""
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from skimage import io
from ..config import Config

class ImageLoadError(Exception):
    """Raised when there's an error loading an image."""
    pass

class ImageLoader:
    @staticmethod
    def validate_image_path(path: str | Path) -> Path:
        """Validate that the image file exists and has a supported format."""
        path = Path(path)
        if not path.exists():
            raise ImageLoadError(f"Image file not found: {path}")
        if path.suffix.lower() not in Config.image.supported_formats:
            raise ImageLoadError(f"Unsupported image format: {path.suffix}")
        return path

    @staticmethod
    def load_image_pair(path_left: str | Path, path_right: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a pair of images for comparison."""
        try:
            path_left = ImageLoader.validate_image_path(path_left)
            path_right = ImageLoader.validate_image_path(path_right)
            
            img_left = io.imread(str(path_left))
            img_right = io.imread(str(path_right))
            
            return img_left, img_right
        except Exception as e:
            raise ImageLoadError(f"Error loading images: {str(e)}")

    @staticmethod
    def load_demo_images() -> Tuple[np.ndarray, np.ndarray]:
        """Load demo images for testing."""
        demo_path = Config.get_default_image_path()
        img = io.imread(str(demo_path))
        return img, img.copy()
