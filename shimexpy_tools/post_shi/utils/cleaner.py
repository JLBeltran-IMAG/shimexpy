"""Module for cleanup operations in SHI package."""
import shutil
from pathlib import Path
from typing import Union, List

from .config import config
from .exceptions import CleanupError
from .logging import logger

class Cleaner:
    """Class for handling cleanup operations."""
    
    @staticmethod
    def clean_cache() -> None:
        """Clean the cache directory."""
        if config.CACHE_DIR.exists():
            try:
                logger.info(f"Removing cache folder: {config.CACHE_DIR}")
                shutil.rmtree(config.CACHE_DIR)
            except Exception as e:
                raise CleanupError(f"Failed to remove cache folder: {e}")
        else:
            logger.warning(f"Cache directory {config.CACHE_DIR} does not exist.")
    
    @staticmethod
    def clean_extra(measurement_directory: Union[str, Path]) -> None:
        """Clean extra files from a measurement directory."""
        measurement_directory = Path(measurement_directory)
        
        # Clean flat directory
        flat_dir = measurement_directory / "flat"
        Cleaner._remove_directory(flat_dir)
        
        # Clean contrast directories
        for contrast in config.CONTRAST_TYPES:
            contrast_dir = measurement_directory / contrast
            if contrast_dir.exists():
                try:
                    files_to_remove = list(contrast_dir.glob("*.tif"))
                    for file in files_to_remove:
                        logger.info(f"Removing file: {file}")
                        file.unlink()
                except Exception as e:
                    raise CleanupError(f"Error removing files in {contrast_dir}: {e}")
    
    @staticmethod
    def _remove_directory(directory: Path) -> None:
        """Remove a directory if it exists."""
        if directory.exists():
            try:
                logger.info(f"Removing directory: {directory}")
                shutil.rmtree(directory)
            except Exception as e:
                raise CleanupError(f"Error removing {directory}: {e}")
        else:
            logger.warning(f"Directory {directory} does not exist.")
