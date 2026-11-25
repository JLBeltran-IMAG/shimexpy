import numpy as np
import matplotlib.pyplot as plt


class StripeCorrectionQualityAssessment:
    """
    Quality assessment tools for detector stripe correction.

    Provides:
        - Difference image
        - Statistics restricted to interpolated pixels (mask=True)
        - Row/column averaged profiles
        - Comparison plots between RAW and corrected images
    """

    def __init__(
        self,
        raw: np.ndarray,
        corrected: np.ndarray,
        mask: np.ndarray,
        method: str | None = None,
    ):
        """Compare 2D images using a boolean mask of interpolated pixels.

        ``mask`` must have the same shape as the images. True selects pixels
        for local statistics. An all-False mask produces empty stripe stats.
        Global statistics and averaged profiles still use the full images.
        """
        raw = np.asarray(raw)
        corrected = np.asarray(corrected)
        mask = np.asarray(mask)

        if raw.ndim != 2 or corrected.ndim != 2:
            raise ValueError("RAW and corrected images must be 2D.")

        if raw.shape != corrected.shape:
            raise ValueError("RAW and corrected images must have the same shape.")

        if mask.shape != raw.shape:
            raise ValueError("Mask must have the same shape as the images.")

        if mask.dtype != np.bool_:
            raise TypeError("Mask must be a boolean array.")

        self.raw = raw.astype(np.float32)
        self.corrected = corrected.astype(np.float32)
        self.diff = self.corrected - self.raw

        self.mask = mask.copy()
        self.method = method or "unknown"

        self.h, self.w = raw.shape

    def _compute_profiles(self):
        return {
            "row_raw": self.raw.mean(axis=0),
            "row_corrected": self.corrected.mean(axis=0),
            "col_raw": self.raw.mean(axis=1),
            "col_corrected": self.corrected.mean(axis=1),
        }

    def _compute_global_stats(self):
        diff = self.diff

        return {
            "method": self.method,
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "std_diff": float(np.std(diff)),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }

    def _compute_stripe_stats(self):
        if not self.mask.any():
            return {}

        diff = self.diff[self.mask]

        return {
            "pixel_count": int(np.count_nonzero(self.mask)),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "std_diff": float(np.std(diff)),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }

    def plot_images(self, p_low=1, p_high=99):
        vmin, vmax = np.percentile(self.raw, [p_low, p_high])

        dmax = np.percentile(np.abs(self.diff), p_high)
        if dmax == 0:
            dmax = 1.0

        fig, axes = plt.subplots(1, 3, figsize=(9, 4))

        axes[0].imshow(self.raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("RAW")
        axes[0].axis("off")

        axes[1].imshow(self.corrected, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Corrected - {self.method}")
        axes[1].axis("off")

        im = axes[2].imshow(self.diff, cmap="bwr", vmin=-dmax, vmax=dmax)
        axes[2].set_title("Corrected - RAW")
        axes[2].axis("off")

        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        fig.tight_layout()

        return fig

    def plot_profiles(self):
        profiles = self._compute_profiles()

        fig_row, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(profiles["row_raw"], label="RAW", alpha=0.6)
        ax1.plot(profiles["row_corrected"], label="Corrected", alpha=0.6)

        for c in np.flatnonzero(self.mask.any(axis=0)):
            ax1.axvline(float(c), linestyle="--", alpha=0.3)

        ax1.set_title("Column-direction averaged profile")
        ax1.legend()
        ax1.grid(alpha=0.2)

        fig_col, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(profiles["col_raw"], label="RAW", alpha=0.6)
        ax2.plot(profiles["col_corrected"], label="Corrected", alpha=0.6)

        for r in np.flatnonzero(self.mask.any(axis=1)):
            ax2.axvline(float(r), linestyle="--", alpha=0.3)

        ax2.set_title("Row-direction averaged profile")
        ax2.legend()
        ax2.grid(alpha=0.2)

        fig_row.tight_layout()
        fig_col.tight_layout()

        return fig_row, fig_col

    def report(self):
        return {
            "global_stats": self._compute_global_stats(),
            "stripe_stats": self._compute_stripe_stats(),
        }

