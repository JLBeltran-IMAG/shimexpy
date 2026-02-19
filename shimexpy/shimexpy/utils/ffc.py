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


    def _slope(self, profile: np.ndarray) -> float:
        x = np.arange(profile.size, dtype=np.float32)
        # slope of best-fit line via polyfit
        m, _ = np.polyfit(x, profile.astype(np.float32), 1)
        return float(m)

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
        metrics["nu_raw"] = metrics["std_raw"] / metrics["mean_raw"] if metrics["mean_raw"] != 0 else 0.0
        metrics["nu_ffc"] = metrics["std_ffc"] / metrics["mean_ffc"] if metrics["mean_ffc"] != 0 else 0.0

        return metrics


    def compute_stats_1d(self):
        profiles = self.compute_profiles()

        raw_row = profiles["row_raw"]
        ffc_row = profiles["row_ffc"]
        raw_col = profiles["col_raw"]
        ffc_col = profiles["col_ffc"]

        def _compute_1d_metrics(raw, ffc):
            mean_raw = float(np.mean(raw))
            mean_ffc = float(np.mean(ffc))

            std_raw = float(np.std(raw))
            std_ffc = float(np.std(ffc))

            ptp_raw = float(np.ptp(raw))
            ptp_ffc = float(np.ptp(ffc))

            raw_slope = self._slope(raw)
            ffc_slope = self._slope(ffc)

            metrics = {
                "mean_raw": mean_raw,
                "mean_ffc": mean_ffc,
                "std_raw": std_raw,
                "std_ffc": std_ffc,
                "ptp_raw": ptp_raw,
                "ptp_ffc": ptp_ffc,
                "nu_raw": std_raw / mean_raw if mean_raw != 0 else 0.0,
                "nu_ffc": std_ffc / mean_ffc if mean_ffc != 0 else 0.0,
                "slope_raw": raw_slope,
                "slope_ffc": ffc_slope,
                "abs_slope_raw": abs(raw_slope),
                "abs_slope_ffc": abs(ffc_slope),
            }

            metrics["std_reduction_%"] = (
                100.0 * (std_raw - std_ffc) / (std_raw if std_raw > 0 else 1.0)
            )

            metrics["ptp_reduction_%"] = (
                100.0 * (ptp_raw - ptp_ffc) / (ptp_raw if ptp_raw > 0 else 1.0)
            )

            metrics["slope_reduction_%"] = (
                100.0
                * (abs(raw_slope) - abs(ffc_slope))
                / (abs(raw_slope) if abs(raw_slope) > 0 else 1.0)
            )

            metrics["nu_reduction_%"] = (
                100.0
                * (metrics["nu_raw"] - metrics["nu_ffc"])
                / (metrics["nu_raw"] if metrics["nu_raw"] > 0 else 1.0)
            )

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


    def non_uniformity_map(self, mode="relative"):
        """
        mode="relative": (I - mean(I)) / mean(I)
        mode="zscore":   (I - mean(I)) / std(I)
        """
        def _nu(img):
            mu = float(np.mean(img))
            if mode == "relative":
                return (img - mu) / (mu if mu != 0 else 1.0)
            elif mode == "zscore":
                sigma = float(np.std(img))
                return (img - mu) / (sigma if sigma != 0 else 1.0)
            else:
                raise ValueError("mode must be 'relative' or 'zscore'")

        return _nu(self.raw), _nu(self.ffc)

    # -------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------
    def plot_profiles(self):
        profiles, statistics_row, statistics_col = self.compute_stats_1d()

        def _set_axes(ax, title, raw_line, ffc_line, statistics):
            ax.plot(raw_line, label="RAW", alpha=0.6)
            ax.plot(ffc_line, label="FFC", alpha=0.6)
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.2)

            # text = (
            #     # f"Std raw: {statistics['std_raw']:.2f}\n"
            #     # f"Std FFC: {statistics['std_ffc']:.2f}\n"
            #     f"Std Reduction: {statistics['std_reduction_%']:.1f}%"
            #     # f"PTP raw: {statistics['ptp_raw']:.1f}\n"
            #     # f"PTP FFC: {statistics['ptp_ffc']:.1f}"
            # )
            # ax.text(
            #     0.5,
            #     0.3,
            #     text,
            #     transform=ax.transAxes,
            #     fontsize=9,
            #     bbox=dict(facecolor="white", alpha=0.6)
            # )

        # Row profile
        fig_row, ax1 = plt.subplots(figsize=(4, 3))
        _set_axes(
            ax1,
            "Row-Averaged Profile",
            profiles["row_raw"],
            profiles["row_ffc"],
            statistics_row
        )

        # Column profile
        fig_col, ax2 = plt.subplots(figsize=(4, 3))
        _set_axes(
            ax2,
            "Column-Averaged Profile",
            profiles["col_raw"],
            profiles["col_ffc"],
            statistics_col
        )

        fig_row.tight_layout()
        fig_col.tight_layout()

        return fig_row, fig_col


    def plot_images(self):
        # statistics = self.compute_stats_2d()

        def get_diagonal_profile(img, linewidth=3):
            from skimage.measure import profile_line
            from matplotlib.lines import Line2D
            h, w = img.shape

            # Esquina superior izquierda → inferior derecha
            src = (0, 0)
            dst = (h-1, w-1)

            profile = profile_line(
                img,
                src,
                dst,
                linewidth=linewidth,   # promedio transversal
                order=1,               # interpolación bilinear
                mode='reflect'
            )
            line = Line2D([0, w-1], [0, h-1], color='red', linewidth=1)

            return profile, line

        fig_profile, ax_profile = plt.subplots(figsize=(6, 4))

        diff = self.raw - self.ffc
        raw_profile, raw_line = get_diagonal_profile(self.raw)
        ffc_profile, ffc_line = get_diagonal_profile(self.ffc)
        diff_profile, diff_line = get_diagonal_profile(diff)

        ax_profile.plot(raw_profile, label="RAW", alpha=0.6)
        ax_profile.plot(ffc_profile, label="FFC", alpha=0.6)
        ax_profile.plot(diff_profile, label="RAW - FFC", alpha=0.6)
        ax_profile.set_title("Diagonal Profile")
        ax_profile.legend()
        ax_profile.grid(alpha=0.2)


        fig, axes = plt.subplots(1, 3, figsize=(8, 4))

        axes[0].imshow(self.raw, cmap="gray")
        axes[0].set_title("RAW")
        axes[0].axis("off")
        axes[0].add_line(raw_line)

        axes[1].imshow(self.ffc, cmap="gray")
        axes[1].set_title("FFC")
        axes[1].axis("off")
        axes[1].add_line(ffc_line)

        axes[2].imshow(diff, cmap="bwr")
        axes[2].set_title("RAW - FFC")
        axes[2].axis("off")
        axes[2].add_line(diff_line)

        # text = (
        #     # f"Std Difference: {diff.std():.2f}\n"
        #     f"Std Reduction: {statistics['std_reduction_%']:.1f}%"
        #     # f"Ptp Reduction: {statistics['ptp_reduction_%']:.1f}%"
        # )
        # axes[2].text(
        #     0.02,
        #     0.04,
        #     text,
        #     transform=axes[2].transAxes,
        #     fontsize=9,
        #     bbox=dict(facecolor="white", alpha=0.6)
        # )

        fig.tight_layout()
        fig_profile.tight_layout()
        return fig, fig_profile


    def plot_histograms(
        self,
        bins=200,
        density=False,
        logy=False,
        use_percentile=True,
        p_low=0.5,
        p_high=99.5
    ):
        raw = self.raw.ravel()
        ffc = self.ffc.ravel()

        # --- Determine histogram range ---
        if use_percentile:
            combined = np.hstack((raw, ffc))
            vmin, vmax = np.percentile(combined, [p_low, p_high])
        else:
            vmin = min(raw.min(), ffc.min())
            vmax = max(raw.max(), ffc.max())

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(raw, bins=bins, range=(vmin, vmax), alpha=0.5, label="RAW", density=density)
        ax.hist(ffc, bins=bins, range=(vmin, vmax), alpha=0.5, label="FFC", density=density)

        if logy:
            ax.set_yscale("log")

        ax.set_xlim(vmin, vmax)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density" if density else "Counts")
        ax.legend()
        ax.grid(alpha=0.2)

        fig.tight_layout()
        return fig


    def plot_non_uniformity_maps(self, mode="relative"):
        nu_raw, nu_ffc = self.non_uniformity_map(mode=mode)
        vmax = max(np.abs(nu_raw).max(), np.abs(nu_ffc).max())

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(nu_raw, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"NU map RAW ({mode})")
        axes[0].axis("off")

        axes[1].imshow(nu_ffc, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"NU map FFC ({mode})")
        axes[1].axis("off")

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


