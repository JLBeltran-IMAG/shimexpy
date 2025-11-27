import numpy as np
from skimage.transform import rotate
from tifffile import imread, imwrite
from pathlib import Path
import logging

# Import core preprocessing functions from shimexpy
from shimexpy.preprocessing import (
    correct_darkfield as core_correct_darkfield,
    correct_brightfield as core_correct_brightfield,
    extract_peak_coordinates,
    calculate_rotation_angle,
)


def crop_without_corrections(
    path_to_images: Path | str,
    crop: tuple[int | None, int | None, int | None, int | None],
    allow_crop: bool = False,
    angle: np.float32 = np.float32(0.0)
) -> None:
    # Convert string to Path if needed
    path_to_images = Path(path_to_images)

    # Find all .tif files in the directory
    images = list(path_to_images.glob("*.tif"))

    # Create the output directory
    path_to_cropped_images = path_to_images / "crop_without_correction"

    if not path_to_cropped_images.exists():
        path_to_cropped_images.mkdir()

    if allow_crop:
        y0, y1, x0, x1 = crop
    else:
        y0, y1, x0, x1 = 0, -1, 0, -1

    for imgs in images:
        corrected_images = imread(imgs)

        # Use core preprocessing functions for angle detection
        coords = extract_peak_coordinates(corrected_images)
        deg = calculate_rotation_angle(coords)

        imwrite(
            path_to_cropped_images.joinpath("{}".format(imgs.name)),
            rotate(corrected_images, angle)[y0:y1, x0:x1].astype(np.float32),
            imagej=True,
        )


def correct_darkfield(
    path_to_dark: Path | str,
    path_to_images: Path | str,
    crop: tuple[int | None, int | None, int | None, int | None],
    allow_crop: bool = False,
    angle: np.float32 = np.float32(0.0)
) -> None:
    """
    Correct images using dark field correction.

    This function corrects images using dark field correction based on provided dark field images.

    Parameters
    ----------
    path_to_dark : str
        The path to the directory containing dark field images.
    path_to_images : str
        The path to the directory containing the images to be corrected.
    crop : tuple, optional
        A tuple specifying the crop region as (y0, y1, x0, x1), by default None.
    allow_crop : bool, optional
        A boolean indicating whether to allow cropping, by default False.

    """
    # Convert string to Path if needed
    path_to_images = Path(path_to_images)
    path_to_dark = Path(path_to_dark)

    # Find all .tif files in the directory
    images = list(path_to_images.glob("*.tif"))

    # Create the output directory
    path_to_corrected_images = path_to_images / "corrected_images"

    if not path_to_corrected_images.exists():
        path_to_corrected_images.mkdir()

    if allow_crop:
        y0, y1, x0, x1 = crop
    else:
        y0, y1, x0, x1 = 0, -1, 0, -1

    dark = imread([dark for dark in Path(path_to_dark).glob("*.tif")])

    # checking if there is only one element in the list
    if dark.ndim == 2:
        dark = dark[None, :, :]

    dark_image_average = np.mean(dark, axis=0)
    for imgs in images:
        image = imread(imgs)
        # Use core preprocessing function for dark field correction
        corrected_images = core_correct_darkfield(image, dark_image_average)
        imwrite(
            path_to_corrected_images.joinpath("{}".format(imgs.name)),
            rotate(corrected_images, angle)[y0:y1, x0:x1].astype(np.float32),
            imagej=True,
        )


def process_flat_correction(image_path, flat_path, output_dir):
    """
    Reads an image and its corresponding flat field image, performs flat correction by subtraction,
    and writes the corrected image to the output directory.

    Parameters:
    -----------
    image_path : Path
        Path to the image file.
    flat_path : Path
        Path to the flat field image file.
    output_dir : Path
        Directory where the corrected image will be saved.
    """
    try:
        image = imread(image_path)
        flat = imread(flat_path)
        corrected = image - flat

        # Ensure the output directory exists
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{image_path.stem}_flatcorrected.tif"

        imwrite(output_file, corrected.astype(np.float32), imagej=True)

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")


def correct_brightfield(path_to_bright, path_to_images):
    """
    Correct bright field in images.

    This function corrects the bright field in images based on the provided bright field images.

    Parameters
    ----------
    path_to_bright : Path
        The path to the directory containing the bright field images.
    path_to_images : Path
        The path to the directory containing the images to be corrected.

    """
    path_to_corrected_bright = Path(path_to_bright).joinpath("corrected_images")
    path_to_corrected_images = Path(path_to_images).joinpath("corrected_images")

    images = list(Path(path_to_corrected_images).glob("*.tif"))

    if not path_to_corrected_images.exists():
        path_to_corrected_images.mkdir()

    bright = imread([bright for bright in Path(path_to_corrected_bright).glob("*.tif")])

    # checking if there is only one element in the list
    if bright.ndim == 2:
        bright = bright[None, :, :]

    bright_image_average = np.mean(bright, axis=0)

    for imgs in images:
        img_array = imread(imgs).astype(np.float32)
        # Use core preprocessing function for bright field correction
        corrected_images = core_correct_brightfield(img_array, bright_image_average)

        imwrite(
            path_to_corrected_images.joinpath("{}".format(imgs.name)),
            corrected_images.astype(np.float32),
            imagej=True
        )

