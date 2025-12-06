import numpy as np
import matplotlib.pyplot as plt
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


class FFCQualityAssessment:
    """
    Quality assessment tools for Flat-Field Correction (FFC).

    Provides:
        - 2D statistics
        - 1D averaged line profiles
        - Publication-ready plots
        - Comparison metrics between RAW and FFC images
    """

    def __init__(self, raw: np.ndarray, ffc: np.ndarray):
        if raw.shape != ffc.shape:
            raise ValueError("RAW and FFC images must have the same shape.")

        self.raw = raw.astype(np.float32)
        self.ffc = ffc.astype(np.float32)
        self.h, self.w = raw.shape

    # -------------------------------------------------------
    # 2D STATISTICS
    # -------------------------------------------------------
    def compute_stats_2d(self):
        raw = self.raw
        ffc = self.ffc

        metrics = {
            "mean_raw": float(np.mean(raw)),
            "mean_ffc": float(np.mean(ffc)),
            "std_raw": float(np.std(raw)),
            "std_ffc": float(np.std(ffc)),
            "ptp_raw": float(np.ptp(raw)),
            "ptp_ffc": float(np.ptp(ffc)),
        }

        metrics["std_reduction_%"] = 100 * (metrics["std_raw"] - metrics["std_ffc"]) / metrics["std_raw"]
        metrics["ptp_reduction_%"] = 100 * (metrics["ptp_raw"] - metrics["ptp_ffc"]) / metrics["ptp_raw"]

        return metrics

    def compute_stats_1d(self):
        profiles = self.compute_profiles()

        raw_row = profiles["row_raw"]
        ffc_row = profiles["row_ffc"]
        raw_col = profiles["col_raw"]
        ffc_col = profiles["col_ffc"]

        def _compute_1d_metrics(raw, ffc):
            metrics = {
                "mean_raw": float(np.mean(raw)),
                "mean_ffc": float(np.mean(ffc)),
                "std_raw": float(np.std(raw)),
                "std_ffc": float(np.std(ffc)),
                "ptp_raw": float(np.ptp(raw)),
                "ptp_ffc": float(np.ptp(ffc)),
            }

            metrics["std_reduction_%"] = 100 * (metrics["std_raw"] - metrics["std_ffc"]) / metrics["std_raw"]
            metrics["ptp_reduction_%"] = 100 * (metrics["ptp_raw"] - metrics["ptp_ffc"]) / metrics["ptp_raw"]

            return metrics

        metrics_row = _compute_1d_metrics(raw_row, ffc_row)
        metrics_col = _compute_1d_metrics(raw_col, ffc_col)

        return profiles, metrics_row, metrics_col

    # -------------------------------------------------------
    # AVERAGED PROFILES (1D)
    # -------------------------------------------------------
    def compute_profiles(self):
        """Compute averaged row profile (default) and column profile."""
        profile_row_raw = self.raw.mean(axis=0)
        profile_row_ffc = self.ffc.mean(axis=0)

        profile_col_raw = self.raw.mean(axis=1)
        profile_col_ffc = self.ffc.mean(axis=1)

        return {
            "row_raw": profile_row_raw,
            "row_ffc": profile_row_ffc,
            "col_raw": profile_col_raw,
            "col_ffc": profile_col_ffc,
        }

    # -------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------
    def plot_profiles(self):
        profiles, statistics_row, statistics_col = self.compute_stats_1d()

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        def _set_axes(ax, title, raw_line, ffc_line, statistics):
            ax.plot(raw_line, label="RAW", alpha=0.6)
            ax.plot(ffc_line, label="FFC", alpha=0.6)
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.2)

            text = (
                f"Std raw: {statistics['std_raw']:.2f}\n"
                f"Std FFC: {statistics['std_ffc']:.2f}\n"
                f"Std Reduction: {statistics['std_reduction_%']:.1f}%\n"
                f"PTP raw: {statistics['ptp_raw']:.1f}\n"
                f"PTP FFC: {statistics['ptp_ffc']:.1f}"
            )
            ax.text(
                0.5,
                0.3,
                text,
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6)
            )

        # Row profile
        _set_axes(
            axes[0],
            "Row-Averaged Profile",
            profiles["row_raw"],
            profiles["row_ffc"],
            statistics_row
        )

        # Column profile
        _set_axes(
            axes[1],
            "Column-Averaged Profile",
            profiles["col_raw"],
            profiles["col_ffc"],
            statistics_col
        )

        plt.tight_layout()
        return fig


    def plot_images(self):
        statistics = self.compute_stats_2d()

        fig, axes = plt.subplots(1, 3, figsize=(8, 3))

        axes[0].imshow(self.raw, cmap="gray")
        axes[0].set_title("RAW")
        axes[0].axis("off")

        axes[1].imshow(self.ffc, cmap="gray")
        axes[1].set_title("FFC")
        axes[1].axis("off")

        diff = self.raw - self.ffc
        axes[2].imshow(diff, cmap="bwr")
        axes[2].set_title("RAW - FFC")
        axes[2].axis("off")

        text = (
            f"Std Difference: {diff.std():.2f}\n"
            f"Std Reduction: {statistics['std_reduction_%']:.1f}%\n"
            f"Ptp Reduction: {statistics['ptp_reduction_%']:.1f}%"
        )
        axes[2].text(
            0.02,
            0.04,
            text,
            transform=axes[2].transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6)
        )

        plt.tight_layout()
        return fig

    # -------------------------------------------------------
    # FULL REPORT
    # -------------------------------------------------------
    def report(self):
        return {
            "1D_stats": self.compute_stats_1d(),
            "2D_stats": self.compute_stats_2d(),
        }


