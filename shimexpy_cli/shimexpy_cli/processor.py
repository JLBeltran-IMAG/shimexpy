"""Main processor class for SHI operations."""
import numpy as np
import xarray as xr
import skimage.io as io
from typing import Optional, List, Union
from pathlib import Path

from shimexpy import get_harmonics
from shimexpy import load_image

from shimexpy_cli.config import config
from shimexpy_cli.exceptions import ImageNotFoundError, ProcessingError
from shimexpy_cli.logging import logger
import shimexpy_cli.execute as execute
import shimexpy_cli.directories as directories
import shimexpy_cli.corrections as corrections
import shimexpy_cli.crop as crop
import shimexpy_cli.angles_correction as angles_correction


class SHIProcessor:
    """Main class for handling SHI processing operations."""
    
    def __init__(
        self,
        mask_period: int,
        unwrap_method: Optional[str] = None,
        allow_crop: bool = False
    ) -> None:
        """
        Initialize SHI processor.
        
        Args:
            mask_period: Number of projected pixels in the mask
            unwrap_method: Phase unwrapping method to use
        """
        self.mask_period = mask_period
        if unwrap_method and not config.validate_unwrap_method(unwrap_method):
            raise ValueError(f"Invalid unwrap method: {unwrap_method}")

        self.unwrap_method = unwrap_method
        self.allow_crop = allow_crop

    def mask_period_definition(self):
        """Get the mask period. Por ahora no se usa."""
        pass


    def process_directory(
        self,
        images_path: Union[str, Path],
        reference_path: Union[str, Path],
        dark_path: Optional[Union[str, Path]] = None,
        bright_path: Optional[Union[str, Path]] = None,
        angle_after: bool = False
    ) -> None:
        """
        Process all .tif files in a directory.

        Args:
            images_path: Path to directory containing sample images
            dark_path: Optional path to dark images
            reference_path: Optional path to flat images
            bright_path: Optional path to bright images
            mode: Processing mode ('2d' or '3d')
            angle_after: Whether to apply angle correction after measurements
            average: Whether to apply averaging
            export: Whether to export results
        """
        images_path = Path(images_path)
        
        # Convert other paths to Path objects if they exist
        reference_path = Path(reference_path)
        dark_path = Path(dark_path) if dark_path else None
        bright_path = Path(bright_path) if bright_path else None

        # Find all .tif files in the directory
        image_files = list(images_path.glob("*.tif"))
        if not image_files:
            raise ImageNotFoundError(f"No .tif files found in {images_path}")

        logger.info(f"Processing {len(image_files)} images in {images_path}")

        # Get angle correction if needed
        deg = self._get_angle_correction(
            image_files[0],
            reference_path
        ) if angle_after else np.float32(0)

        # Important: Only do crop ONCE per directory, not per image
        # Get crop from first image in the directory
        # and make sure to use an actual image file
        first_image = image_files[0]  # Use the first .tif file found
        logger.info(f"Using image {first_image} for crop reference")

        if self.allow_crop:
            crop_from_tmptxt = crop.cropImage(first_image)
        else:
            crop_from_tmptxt = (0, -1, 0, -1)

        # Apply corrections based on the crop settings
        if dark_path:
            self._apply_dark_bright_corrections(
                dark_path,
                reference_path,
                images_path,
                bright_path,
                crop_from_tmptxt,
                self.allow_crop,
                deg
            )
            foldername_to = "corrected_images"
        else:
            self._apply_crop_only(
                images_path,
                reference_path,
                crop_from_tmptxt,
                self.allow_crop,
                deg
            )
            foldername_to = "crop_without_correction"

        # Create result directories and process
        corrected_dir = images_path / foldername_to
        if not corrected_dir.exists():
            corrected_dir.mkdir(parents=True, exist_ok=True)

        _, path_to_result = directories.create_result_subfolders(
            file_dir=str(corrected_dir),
            result_folder=images_path.name,
            sample_folder=""
        )

        # Execute SHI processing
        self._process(
            corrected_path=corrected_dir,
            reference_path=reference_path,
            result_path=path_to_result
        )


    def process_single_image(
        self,
        image_path: Path,
        reference_path: Path,
        dark_path: Optional[Path],
        bright_path: Optional[Path],
        angle_after: bool
    ) -> None:
        """Process a single image file."""
        logger.info(f"Processing measurement: {image_path}")

        # Get angle correction if needed
        deg = self._get_angle_correction(
            image_path,
            reference_path
        ) if angle_after else np.float32(0)

        # Crop image
        crop_from_tmptxt = crop.cropImage(image_path)

        # Apply corrections
        if dark_path:
            self._apply_dark_bright_corrections(
                dark_path,
                reference_path,
                image_path,
                bright_path,
                crop_from_tmptxt,
                self.allow_crop,
                deg
            )
            foldername_to = "corrected_images"
        else:
            self._apply_crop_only(
                image_path,
                reference_path,
                crop_from_tmptxt,
                self.allow_crop,
                deg
            )
            foldername_to = "crop_without_correction"

        # Create result directories and process. Instead of creating corrected_images
        # as a child of the .tif file (which is wrong), create it as a sibling
        # directory to the parent folder containing the .tif
        parent_dir = image_path.parent
        corrected_dir = parent_dir / foldername_to

        # Ensure the directory exists
        if not corrected_dir.exists():
            corrected_dir.mkdir(parents=True, exist_ok=True)

        # Use the image stem as a subfolder name within corrected_images
        _, path_to_result = directories.create_result_subfolders(
            file_dir=str(corrected_dir),
            result_folder=image_path.stem,
            sample_folder=""
        )

        # Execute SHI processing
        self._process(
            corrected_path=corrected_dir,
            reference_path=reference_path,
            result_path=path_to_result
        )

    def _process(
        self,
        corrected_path: Path,
        reference_path: Path,
        result_path: Path,
    ) -> None:
        """
        Execute the complete Spatial Harmonic Imaging (SHI) processing pipeline.

        This method performs flat-field correction, runs SHI processing for both
        flat and sample images, applies the flat correction to all contrasts, and
        finally handles 2D/3D specific post-processing.

        Parameters
        ----------
        corrected_path : Path
            Path to the corrected sample directory to be processed.
        reference_path : Path
            Path to the reference directory where the corrected flats will be stored.
        result_path : Path
            Path where the final processing results will be saved.
        mode : str
            Either "2d" or "3d" depending on the acquisition type.
        average : bool
            If True and mode is "2d", average all resulting contrast maps.
        export : bool
            If True, export the processed results after completion.
        """
        # --- Validate input paths ---
        if not corrected_path.exists() or not corrected_path.is_dir():
            logger.error(f"Invalid corrected path: {corrected_path}")
            return

        if not reference_path.exists() or not reference_path.is_dir():
            logger.error(f"Invalid reference path: {reference_path}")
            return

        # --- Create directory for corrected flat ---
        path_to_corrected_flat = reference_path / corrected_path.name

        execute.execute_SHI(
            path_to_images=corrected_path,
            path_to_reference=path_to_corrected_flat,
            path_to_result=result_path,
            projected_grid=self.mask_period,
            unwrap=self.unwrap_method
        )


    def _get_angle_correction(
        self,
        image_path: Path,
        reference_path: Optional[Path]
    ) -> np.float32:
        """Calculate angle correction."""
        path_to_ang = reference_path if reference_path else image_path
        tif_files = list(path_to_ang.glob("*.tif"))

        if not tif_files:
            logger.warning(f"No .tif files found for angle correction in {path_to_ang}")
            return np.float32(0)

        path_to_angle_correction = tif_files[0]
        image_angle = io.imread(str(path_to_angle_correction))
        cords = angles_correction.extracting_coordinates_of_peaks(image_angle)

        return angles_correction.calculating_angles_of_peaks_average(cords)


    def _apply_dark_bright_corrections(
        self,
        dark_path: Path,
        reference_path: Path,
        images_path: Path,
        bright_path: Optional[Path],
        crop: tuple,
        allow_crop: bool,
        angle: np.float32
    ) -> None:
        """Apply dark and bright field corrections to all images in a directory."""
        # Apply corrections to the flat path
        corrections.correct_darkfield(
            path_to_dark=str(dark_path),
            path_to_images=str(reference_path),
            crop=crop,
            allow_crop=allow_crop,
            angle=angle
        )
        
        # Apply corrections to the sample images directory
        corrections.correct_darkfield(
            path_to_dark=str(dark_path),
            path_to_images=str(images_path),
            crop=crop,
            allow_crop=allow_crop,
            angle=angle
        )
        
        if bright_path:
            corrections.correct_darkfield(
                path_to_dark=str(dark_path),
                path_to_images=str(bright_path),
                crop=crop,
                allow_crop=allow_crop,
                angle=angle
            )
            corrections.correct_brightfield(
                path_to_bright=str(bright_path),
                path_to_images=str(reference_path)
            )
            corrections.correct_brightfield(
                path_to_bright=str(bright_path),
                path_to_images=str(images_path)
            )


    def _apply_crop_only(
        self,
        images_path: Path,
        reference_path: Path,
        crop: tuple,
        allow_crop: bool,
        angle: np.float32
    ) -> None:
        """Apply cropping without corrections to all images in a directory."""
        corrections.crop_without_corrections(
            path_to_images=str(images_path),
            crop=crop,
            allow_crop=allow_crop,
            angle=angle
        )
        corrections.crop_without_corrections(
            path_to_images=str(reference_path),
            crop=crop,
            allow_crop=allow_crop,
            angle=angle
        )


    def _handle_2d_averaging(self, result_path: Path) -> None:
        """Handle 2D averaging operations."""
        for contrast in config.CONTRAST_TYPES:
            logger.info(f"Averaging contrast: {contrast}")


    def _handle_3d_organization(self, result_path: Path) -> None:
        """Handle 3D organization operations."""
        for contrast in config.CONTRAST_TYPES:
            logger.info(f"Organizing contrast: {contrast}")


