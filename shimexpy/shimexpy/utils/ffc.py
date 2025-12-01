import numpy as np
from skimage.transform import rotate


def ffc(
    image: np.ndarray,
    dark: np.ndarray,
    bright: np.ndarray,
    crop: tuple[int | None, int | None, int | None, int | None] | None = None,
    angle: float = 0.0,
    allow_crop: bool = False
) -> np.ndarray:
    """
    Perform flat-field correction on a single image.
    Returns the corrected image as a NumPy array.

    Correction steps:
        1. dark-corrected
        2. bright-corrected
        3. optional rotation
        4. optional cropping

    Parameters
    ----------
    image : np.ndarray
        The raw input image (2D).
    dark_path : str or Path
        Directory containing dark-field TIFF images.
    bright_path : str or Path
        Directory containing bright-field TIFF images.
    crop : tuple or None
        (y0, y1, x0, x1) crop region. Ignored if allow_crop=False.
    angle : float
        Rotation angle in degrees.
    allow_crop : bool
        If False, no cropping is applied even if crop is provided.

    Returns
    -------
    corrected : np.ndarray (float32)
        The flat-field corrected image.
    """
    if dark.ndim == 3:
        dark = np.mean(dark, axis=0, dtype=np.float32)

    if bright.ndim == 3:
        bright = np.mean(bright, axis=0, dtype=np.float32)

    # ---------------------------------------------
    # Dark correction: (I - D)
    # ---------------------------------------------
    img = image.astype(np.float32)
    image_darkcorrected = img - dark

    # ---------------------------------------------
    # Bright correction: (I - D) / (F - D)
    # avoiding divide-by-zero
    # ---------------------------------------------
    bright_darkcorrected = bright - dark
    bright_darkcorrected[bright_darkcorrected == 0] = 1

    image_ffcnorm = image_darkcorrected / bright_darkcorrected * np.mean(bright_darkcorrected)

    # ---------------------------------------------
    # Optional rotation
    # ---------------------------------------------
    if angle != 0:
        image_ffcnorm = rotate(image_ffcnorm, angle, preserve_range=True)

    # ---------------------------------------------
    # Optional cropping
    # ---------------------------------------------
    if allow_crop and crop is not None:
        y0, y1, x0, x1 = crop
        image_ffcnorm = image_ffcnorm[y0:y1, x0:x1]

    return image_ffcnorm

