import numpy as np
import matplotlib.pyplot as plt


class StripeCorrectionFourierQualityAssessment:
    """
    Fourier-domain quality assessment for stripe correction methods.

    This class compares the spectral content of a raw image and a
    stripe-corrected image without assuming any SHI harmonic model.
    """

    def __init__(self, raw, corrected, eps=1e-12):
        if raw.shape != corrected.shape:
            raise ValueError("RAW and corrected images must have the same shape.")

        self.raw = raw.astype(np.float32)
        self.corrected = corrected.astype(np.float32)
        self.diff = self.corrected - self.raw
        self.eps = eps

        self.fft_raw = np.fft.fftshift(np.fft.fft2(self.raw))
        self.fft_corrected = np.fft.fftshift(np.fft.fft2(self.corrected))
        self.fft_diff = self.fft_corrected - self.fft_raw

        self.mag_raw = np.abs(self.fft_raw)
        self.mag_corrected = np.abs(self.fft_corrected)
        self.mag_diff = np.abs(self.fft_diff)

        self.phase_raw = np.angle(self.fft_raw)
        self.phase_corrected = np.angle(self.fft_corrected)

    def compute_fft_energy_change(self):
        e_raw = np.sum(self.mag_raw**2)
        e_corrected = np.sum(self.mag_corrected**2)

        return {
            "fft_energy_raw": float(e_raw),
            "fft_energy_corrected": float(e_corrected),
            "fft_energy_absolute_change": float(e_corrected - e_raw),
            "fft_energy_relative_change": float((e_corrected - e_raw) / (e_raw + self.eps)),
        }

    def compute_fft_magnitude_rmse(self, log_scale=True):
        a = np.log1p(self.mag_raw) if log_scale else self.mag_raw
        b = np.log1p(self.mag_corrected) if log_scale else self.mag_corrected

        return float(np.sqrt(np.mean((b - a) ** 2)))

    def compute_fft_magnitude_mae(self, log_scale=True):
        a = np.log1p(self.mag_raw) if log_scale else self.mag_raw
        b = np.log1p(self.mag_corrected) if log_scale else self.mag_corrected

        return float(np.mean(np.abs(b - a)))

    def compute_fft_phase_rmse(self, magnitude_threshold=1e-3):
        mask = self.mag_raw > magnitude_threshold * np.max(self.mag_raw)

        phase_diff = np.angle(
            np.exp(1j * (self.phase_corrected - self.phase_raw))
        )

        return float(np.sqrt(np.mean(phase_diff[mask] ** 2)))

    def compute_high_frequency_energy_change(self, radius_fraction=0.35):
        h, w = self.raw.shape
        cy, cx = h // 2, w // 2

        yy, xx = np.indices((h, w))
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        rmax = np.sqrt(cy**2 + cx**2)
        mask = rr > radius_fraction * rmax

        e_raw = np.sum(self.mag_raw[mask] ** 2)
        e_corrected = np.sum(self.mag_corrected[mask] ** 2)

        return {
            "high_freq_energy_raw": float(e_raw),
            "high_freq_energy_corrected": float(e_corrected),
            "high_freq_energy_absolute_change": float(e_corrected - e_raw),
            "high_freq_energy_relative_change": float((e_corrected - e_raw) / (e_raw + self.eps)),
        }

    def compute_spectral_residual_energy(self):
        residual_energy = np.sum(self.mag_diff**2)
        raw_energy = np.sum(self.mag_raw**2)

        return {
            "spectral_residual_energy": float(residual_energy),
            "spectral_residual_relative_energy": float(residual_energy / (raw_energy + self.eps)),
        }

    def compute_all_metrics(self):
        return {
            **self.compute_fft_energy_change(),
            "fft_magnitude_rmse_log": self.compute_fft_magnitude_rmse(log_scale=True),
            "fft_magnitude_mae_log": self.compute_fft_magnitude_mae(log_scale=True),
            "fft_phase_rmse": self.compute_fft_phase_rmse(),
            **self.compute_high_frequency_energy_change(),
            **self.compute_spectral_residual_energy(),
        }

    def plot_fft_comparison(self, percentile=99.5):
        raw_log = np.log1p(self.mag_raw)
        corrected_log = np.log1p(self.mag_corrected)

        vmax = np.percentile(raw_log, percentile)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(raw_log, cmap="gray", vmax=vmax)
        axes[0].set_title("FFT RAW")
        axes[0].axis("off")

        axes[1].imshow(corrected_log, cmap="gray", vmax=vmax)
        axes[1].set_title("FFT Corrected")
        axes[1].axis("off")

        fig.tight_layout()
        return fig

    def plot_fft_difference(self, percentile=99.5):
        diff_log = np.log1p(self.mag_diff)
        vmax = np.percentile(diff_log, percentile)

        fig, ax = plt.subplots(figsize=(5, 4))

        im = ax.imshow(diff_log, cmap="gray", vmax=vmax) # type: ignore
        ax.set_title("FFT Difference Magnitude")
        ax.axis("off")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        return fig

