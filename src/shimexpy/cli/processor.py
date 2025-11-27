"""Main processor class for SHI operations."""
import numpy as np
import xarray as xr
import skimage.io as io
from typing import Optional, List, Union
from pathlib import Path

from shimexpy import get_harmonics
from shimexpy import load_image

from shimexpy.cli.config import config
from shimexpy.cli.exceptions import ImageNotFoundError, ProcessingError
from shimexpy.cli.logging import logger
import shimexpy.cli.execute as execute
import shimexpy.cli.directories as directories
import shimexpy.cli.corrections as corrections
from shimexpy.preprocessing import extract_peak_coordinates, calculate_rotation_angle


class SHIProcessor:
    """Main class for handling SHI processing operations."""

    def __init__(
        self,
        mask_period: int,
        unwrap_method: Optional[str] = None,
        crop: Optional[tuple] = None
    ) -> None:
        """
        Initialize SHI processor.

        Args:
            mask_period: Number of projected pixels in the mask
            unwrap_method: Phase unwrapping method to use
            crop: Optional crop coordinates as (top, bottom, left, right)
        """
        self.mask_period = mask_period
        if unwrap_method and not config.validate_unwrap_method(unwrap_method):
            raise ValueError(f"Invalid unwrap method: {unwrap_method}")

        self.unwrap_method = unwrap_method
        self.crop = crop if crop else (0, -1, 0, -1)  # Default: no cropping

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

        # Use crop coordinates from initialization
        crop_coords = self.crop
        if crop_coords != (0, -1, 0, -1):
            logger.info(f"Using crop region: top={crop_coords[0]}, bottom={crop_coords[1]}, left={crop_coords[2]}, right={crop_coords[3]}")

        # Apply corrections based on the crop settings
        allow_crop = crop_coords != (0, -1, 0, -1)
        if dark_path:
            self._apply_dark_bright_corrections(
                dark_path,
                reference_path,
                images_path,
                bright_path,
                crop_coords,
                allow_crop,
                deg
            )
            foldername_to = "corrected_images"
        else:
            self._apply_crop_only(
                images_path,
                reference_path,
                crop_coords,
                allow_crop,
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

        # Use crop coordinates from initialization
        crop_coords = self.crop
        allow_crop = crop_coords != (0, -1, 0, -1)
        if allow_crop:
            logger.info(f"Using crop region: top={crop_coords[0]}, bottom={crop_coords[1]}, left={crop_coords[2]}, right={crop_coords[3]}")

        # Apply corrections
        if dark_path:
            self._apply_dark_bright_corrections(
                dark_path,
                reference_path,
                image_path,
                bright_path,
                crop_coords,
                allow_crop,
                deg
            )
            foldername_to = "corrected_images"
        else:
            self._apply_crop_only(
                image_path,
                reference_path,
                crop_coords,
                allow_crop,
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

        # --- Create directory for corrected flat ---
        path_to_corrected_flat = reference_path / corrected_path.name
        path_to_corrected_flat.mkdir(parents=True, exist_ok=True)

        # --- Create subfolders for results ---
        path_to_flat, _ = directories.create_result_subfolders(
            file_dir=str(path_to_corrected_flat),
            result_folder=result_path.name,
            sample_folder="flat",
        )

        reference_data = []
        for refs in path_to_flat:
            logger.info(f"Processing flat image: {refs}")
            reference_images = load_image(refs)

            reference_data.append(
                get_harmonics(
                    reference_images,
                    projected_grid=self.mask_period,
                    unwrap=self.unwrap_method
                )
            )

        execute.execute_SHI(
            path_to_images=corrected_path,
            path_to_result=result_path,
            reference_data=reference_data[0],
            unwrap=self.unwrap_method
        )

        # # --- Apply flat-field corrections ---
        # result_corrections = directories.create_corrections_folder(result_path)

        # # --- Handle 2D or 3D post-processing ---
        # if mode == "2d" and average:
        #     self._handle_2d_averaging(result_corrections)
        # elif mode == "3d":
        #     self._handle_3d_organization(result_corrections)

        # # --- Optional export ---
        # if export:
        #     logger.info(f"Exporting results to {result_path}")
        #     directories.export_results(result_path)


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
        coords = extract_peak_coordinates(image_angle)

        return np.float32(calculate_rotation_angle(coords))


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


