import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


GratingType = Literal[
    "line_1d",
    "checkerboard",
    "dot_array",
    "hartmann",
    "inverted_hartmann",
    "mesh_rectangular",
    "mesh_hexagonal",
]

SourceShape = Literal["gaussian", "uniform_disk", "uniform_rect"]
DetectorPSFType = Literal["gaussian", "none"]


@dataclass
class Geometry:
    # Distances in meters
    z_g: float  # source -> grating
    z_d: float  # source -> detector

    @property
    def magnification(self) -> float:
        return self.z_d / self.z_g

    @property
    def source_blur_factor(self) -> float:
        # detector-plane shift = (M - 1) * source_shift
        return self.magnification - 1.0


@dataclass
class Detector:
    # Detector pixel pitch in meters
    pixel_size_x: float
    pixel_size_y: float
    # Detector shape in pixels
    nx: int
    ny: int
    # Detector PSF
    psf_type: DetectorPSFType = "gaussian"
    psf_sigma_x_m: float = 0.0
    psf_sigma_y_m: float = 0.0
    # Mean counts scaling
    mean_open_counts: float = 1e4


@dataclass
class Oversampling:
    # detector-plane oversampling factor
    subpixel: int = 4


@dataclass
class Source:
    shape: SourceShape = "gaussian"

    # Gaussian sigmas in source plane [m]
    sigma_x_m: float = 0.0
    sigma_y_m: float = 0.0

    # Uniform disk radius in source plane [m]
    radius_m: float = 0.0

    # Uniform rectangle half-widths in source plane [m]
    half_width_x_m: float = 0.0
    half_width_y_m: float = 0.0


@dataclass
class GratingParams:
    grating_type: GratingType

    # Physical pitch parameters in grating plane [m]
    pitch_x_m: float
    pitch_y_m: Optional[float] = None

    # Open transmission and blocked transmission
    tau_open: float = 1.0
    tau_block: float = 0.0

    # Generic duty cycles
    duty_x: float = 0.5
    duty_y: float = 0.5

    # Dot / hole radius [m]
    radius_m: Optional[float] = None

    # Hartmann / mesh opening sizes [m]
    opening_x_m: Optional[float] = None
    opening_y_m: Optional[float] = None

    # Rotation in degrees for the entire grating
    rotation_deg: float = 0.0


@dataclass
class SimulationResult:
    raw_counts: np.ndarray
    expected_counts: np.ndarray
    grating_transmission_detector_plane: np.ndarray
    blurred_before_sampling: np.ndarray
    x_detector_m: np.ndarray
    y_detector_m: np.ndarray
    metadata: dict


class SingleGratingSimulator:
    """
    Forward intensity model for:
        source -> grating -> detector
    no sample, no Fresnel.

    Continuous model:
        I_pre(u) = I0 * [ T_g(u/M) * h_src * h_det ](u)

    Pixel expectation:
        Lambda_ij = integral over pixel of I_pre(u) du
    implemented numerically by oversampling + block averaging.
    """

    def __init__(
        self,
        geometry: Geometry,
        detector: Detector,
        oversampling: Oversampling = Oversampling(),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.geometry = geometry
        self.detector = detector
        self.oversampling = oversampling
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.detector.nx <= 0 or self.detector.ny <= 0:
            raise ValueError("Detector dimensions must be positive.")
        if self.oversampling.subpixel < 1:
            raise ValueError("Oversampling factor must be >= 1.")
        if self.geometry.z_d <= self.geometry.z_g:
            raise ValueError("Require z_d > z_g.")

    def simulate(
        self,
        grating: GratingParams,
        source: Source,
        add_poisson_noise: bool = True,
    ) -> SimulationResult:
        sub = self.oversampling.subpixel
        M = self.geometry.magnification

        # Oversampled detector-plane grid
        dx = self.detector.pixel_size_x / sub
        dy = self.detector.pixel_size_y / sub
        nx_hr = self.detector.nx * sub
        ny_hr = self.detector.ny * sub

        x_hr = (np.arange(nx_hr) - nx_hr / 2 + 0.5) * dx
        y_hr = (np.arange(ny_hr) - ny_hr / 2 + 0.5) * dy
        Xd, Yd = np.meshgrid(x_hr, y_hr)

        # Map detector-plane coordinates to grating plane via magnification
        Xg = Xd / M
        Yg = Yd / M

        # Real-space grating transmission projected to detector plane
        T_det = self._make_grating_transmission(Xg, Yg, grating)

        # Source blur kernel in detector plane
        h_src = self._make_source_kernel_detector_plane(source, dx, dy)

        # Detector PSF kernel in detector plane
        h_det = self._make_detector_psf_kernel(dx, dy)

        # Apply source blur
        if h_src is not None:
            T_blur = fftconvolve(T_det, h_src, mode="same")
        else:
            T_blur = T_det.copy()

        # Apply detector PSF
        if h_det is not None:
            T_blur = fftconvolve(T_blur, h_det, mode="same")

        # Normalize to mean open counts per detector pixel after sampling
        expected_hr = self.detector.mean_open_counts * T_blur

        # Pixel integration by block averaging at detector sampling
        expected_counts = self._block_average(expected_hr, sub)

        # Poisson noise
        if add_poisson_noise:
            raw_counts = self.rng.poisson(np.clip(expected_counts, 0.0, None))
        else:
            raw_counts = expected_counts.copy()

        x_det = (np.arange(self.detector.nx) - self.detector.nx / 2 + 0.5) * self.detector.pixel_size_x
        y_det = (np.arange(self.detector.ny) - self.detector.ny / 2 + 0.5) * self.detector.pixel_size_y

        metadata = {
            "magnification": M,
            "projected_pitch_x_m": grating.pitch_x_m * M,
            "projected_pitch_y_m": (
                (grating.pitch_y_m if grating.pitch_y_m is not None else grating.pitch_x_m) * M
            ),
            "detector_pixel_size_x_m": self.detector.pixel_size_x,
            "detector_pixel_size_y_m": self.detector.pixel_size_y,
            "oversampling": sub,
            "grating_type": grating.grating_type,
        }

        return SimulationResult(
            raw_counts=raw_counts,
            expected_counts=expected_counts,
            grating_transmission_detector_plane=self._block_average(T_det, sub),
            blurred_before_sampling=self._block_average(T_blur, sub),
            x_detector_m=x_det,
            y_detector_m=y_det,
            metadata=metadata,
        )

    def _make_grating_transmission(
        self,
        Xg: np.ndarray,
        Yg: np.ndarray,
        grating: GratingParams,
    ) -> np.ndarray:
        pitch_x = grating.pitch_x_m
        pitch_y = grating.pitch_y_m if grating.pitch_y_m is not None else grating.pitch_x_m

        if pitch_x <= 0 or pitch_y <= 0:
            raise ValueError("Pitch must be positive.")
        if not (0.0 <= grating.tau_block <= grating.tau_open <= 1.0):
            raise ValueError("Require 0 <= tau_block <= tau_open <= 1.")

        # Rotate coordinates in grating plane if requested
        theta = np.deg2rad(grating.rotation_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        Xr = cos_t * Xg + sin_t * Yg
        Yr = -sin_t * Xg + cos_t * Yg

        gtype = grating.grating_type

        if gtype == "line_1d":
            return self._grating_line_1d(Xr, grating)

        if gtype == "checkerboard":
            return self._grating_checkerboard(Xr, Yr, grating)

        if gtype == "dot_array":
            return self._grating_dot_array(Xr, Yr, grating)

        if gtype == "hartmann":
            return self._grating_hartmann(Xr, Yr, grating)

        if gtype == "inverted_hartmann":
            T_h = self._grating_hartmann(Xr, Yr, grating)
            return grating.tau_open + grating.tau_block - T_h

        if gtype == "mesh_rectangular":
            return self._grating_mesh_rectangular(Xr, Yr, grating)

        if gtype == "mesh_hexagonal":
            return self._grating_mesh_hexagonal(Xr, Yr, grating)

        raise ValueError(f"Unsupported grating type: {gtype}")

    def _grating_line_1d(self, X: np.ndarray, grating: GratingParams) -> np.ndarray:
        p = grating.pitch_x_m
        duty = grating.duty_x
        if not (0.0 < duty < 1.0):
            raise ValueError("duty_x must satisfy 0 < duty_x < 1.")

        phase = np.mod(X, p)
        open_region = phase < duty * p
        return np.where(open_region, grating.tau_open, grating.tau_block).astype(np.float64)

    def _grating_checkerboard(self, X: np.ndarray, Y: np.ndarray, grating: GratingParams) -> np.ndarray:
        px = grating.pitch_x_m
        py = grating.pitch_y_m if grating.pitch_y_m is not None else grating.pitch_x_m

        ix = np.floor(X / px).astype(np.int64)
        iy = np.floor(Y / py).astype(np.int64)
        parity = (ix + iy) % 2 == 0
        return np.where(parity, grating.tau_open, grating.tau_block).astype(np.float64)

    def _grating_dot_array(self, X: np.ndarray, Y: np.ndarray, grating: GratingParams) -> np.ndarray:
        px = grating.pitch_x_m
        py = grating.pitch_y_m if grating.pitch_y_m is not None else grating.pitch_x_m
        radius = grating.radius_m
        if radius is None or radius <= 0:
            raise ValueError("dot_array requires radius_m > 0.")

        x_local = ((X + 0.5 * px) % px) - 0.5 * px
        y_local = ((Y + 0.5 * py) % py) - 0.5 * py
        open_region = x_local**2 + y_local**2 <= radius**2
        return np.where(open_region, grating.tau_open, grating.tau_block).astype(np.float64)

    def _grating_hartmann(self, X: np.ndarray, Y: np.ndarray, grating: GratingParams) -> np.ndarray:
        px = grating.pitch_x_m
        py = grating.pitch_y_m if grating.pitch_y_m is not None else grating.pitch_x_m
        ox = grating.opening_x_m
        oy = grating.opening_y_m
        if ox is None or oy is None or ox <= 0 or oy <= 0:
            raise ValueError("hartmann requires opening_x_m > 0 and opening_y_m > 0.")

        x_local = ((X + 0.5 * px) % px) - 0.5 * px
        y_local = ((Y + 0.5 * py) % py) - 0.5 * py
        open_region = (np.abs(x_local) <= ox / 2.0) & (np.abs(y_local) <= oy / 2.0)
        return np.where(open_region, grating.tau_open, grating.tau_block).astype(np.float64)

    def _grating_mesh_rectangular(self, X: np.ndarray, Y: np.ndarray, grating: GratingParams) -> np.ndarray:
        # Interpreted as periodic rectangular holes within an absorbing grid
        return self._grating_hartmann(X, Y, grating)

    def _grating_mesh_hexagonal(self, X: np.ndarray, Y: np.ndarray, grating: GratingParams) -> np.ndarray:
        """
        Hexagonal lattice of circular holes.
        The user asked for mesh hexagonal; this implementation models a hexagonal Bravais lattice
        populated with circular openings.
        """
        p = grating.pitch_x_m
        radius = grating.radius_m
        if radius is None or radius <= 0:
            raise ValueError("mesh_hexagonal requires radius_m > 0.")

        # Lattice basis
        a1 = np.array([p, 0.0])
        a2 = np.array([0.5 * p, 0.5 * np.sqrt(3.0) * p])

        # Convert Cartesian -> lattice coordinates
        A = np.array([[a1[0], a2[0]], [a1[1], a2[1]]], dtype=np.float64)
        Ainv = np.linalg.inv(A)

        uv0 = Ainv[0, 0] * X + Ainv[0, 1] * Y
        uv1 = Ainv[1, 0] * X + Ainv[1, 1] * Y

        # Nearest lattice site
        m = np.round(uv0)
        n = np.round(uv1)

        Xc = m * a1[0] + n * a2[0]
        Yc = m * a1[1] + n * a2[1]

        open_region = (X - Xc) ** 2 + (Y - Yc) ** 2 <= radius**2
        return np.where(open_region, grating.tau_open, grating.tau_block).astype(np.float64)

    def _make_source_kernel_detector_plane(
        self,
        source: Source,
        dx: float,
        dy: float,
    ) -> Optional[np.ndarray]:
        factor = self.geometry.source_blur_factor

        if source.shape == "gaussian":
            sig_x = factor * source.sigma_x_m
            sig_y = factor * source.sigma_y_m
            if sig_x <= 0 and sig_y <= 0:
                return None
            return self._gaussian_kernel(sig_x, sig_y, dx, dy)

        if source.shape == "uniform_disk":
            radius = factor * source.radius_m
            if radius <= 0:
                return None
            return self._uniform_disk_kernel(radius, dx, dy)

        if source.shape == "uniform_rect":
            hx = factor * source.half_width_x_m
            hy = factor * source.half_width_y_m
            if hx <= 0 and hy <= 0:
                return None
            return self._uniform_rect_kernel(hx, hy, dx, dy)

        raise ValueError(f"Unsupported source shape: {source.shape}")

    def _make_detector_psf_kernel(self, dx: float, dy: float) -> Optional[np.ndarray]:
        if self.detector.psf_type == "none":
            return None

        if self.detector.psf_type == "gaussian":
            sig_x = self.detector.psf_sigma_x_m
            sig_y = self.detector.psf_sigma_y_m
            if sig_x <= 0 and sig_y <= 0:
                return None
            return self._gaussian_kernel(sig_x, sig_y, dx, dy)

        raise ValueError(f"Unsupported detector PSF type: {self.detector.psf_type}")

    @staticmethod
    def _gaussian_kernel(
        sigma_x_m: float,
        sigma_y_m: float,
        dx: float,
        dy: float,
        truncate: float = 4.0,
    ) -> np.ndarray:
        sigma_x_px = max(sigma_x_m / dx, 1e-12)
        sigma_y_px = max(sigma_y_m / dy, 1e-12)

        rx = int(math.ceil(truncate * sigma_x_px))
        ry = int(math.ceil(truncate * sigma_y_px))

        x = np.arange(-rx, rx + 1, dtype=np.float64)
        y = np.arange(-ry, ry + 1, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        K = np.exp(-0.5 * ((X / sigma_x_px) ** 2 + (Y / sigma_y_px) ** 2))
        K /= K.sum()
        return K

    @staticmethod
    def _uniform_disk_kernel(radius_m: float, dx: float, dy: float) -> np.ndarray:
        rx = max(1, int(math.ceil(radius_m / dx)))
        ry = max(1, int(math.ceil(radius_m / dy)))

        x = np.arange(-rx, rx + 1, dtype=np.float64) * dx
        y = np.arange(-ry, ry + 1, dtype=np.float64) * dy
        X, Y = np.meshgrid(x, y)

        K = (X**2 + Y**2 <= radius_m**2).astype(np.float64)
        K /= K.sum()
        return K

    @staticmethod
    def _uniform_rect_kernel(half_width_x_m: float, half_width_y_m: float, dx: float, dy: float) -> np.ndarray:
        rx = max(1, int(math.ceil(half_width_x_m / dx)))
        ry = max(1, int(math.ceil(half_width_y_m / dy)))

        x = np.arange(-rx, rx + 1, dtype=np.float64) * dx
        y = np.arange(-ry, ry + 1, dtype=np.float64) * dy
        X, Y = np.meshgrid(x, y)

        K = ((np.abs(X) <= half_width_x_m) & (np.abs(Y) <= half_width_y_m)).astype(np.float64)
        K /= K.sum()
        return K

    @staticmethod
    def _block_average(image_hr: np.ndarray, sub: int) -> np.ndarray:
        ny_hr, nx_hr = image_hr.shape
        if ny_hr % sub != 0 or nx_hr % sub != 0:
            raise ValueError("High-resolution image dimensions must be divisible by subpixel factor.")

        ny = ny_hr // sub
        nx = nx_hr // sub
        return image_hr.reshape(ny, sub, nx, sub).mean(axis=(1, 3))


def _fft2_phys(image: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return centered log-magnitude FFT and physical frequency axes.
    kx, ky are in rad/m.
    """
    image = np.asarray(image, dtype=np.float64)

    fft = np.fft.fftshift(np.fft.fft2(image))
    mag = np.log1p(np.abs(fft))

    ny, nx = image.shape
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))

    return mag, kx, ky


def _normalize_image(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Min-max normalize an image to [0, 1].
    """
    img = np.asarray(img, dtype=np.float64)
    mn = img.min()
    mx = img.max()
    if mx - mn < eps:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def show_result(result: SimulationResult, detector: Detector, title: str = "Simulation") -> None:
    images = [
        ("Projected grating", result.grating_transmission_detector_plane),
        ("After blur", result.blurred_before_sampling),
        ("Expected counts", result.expected_counts),
        ("Raw counts", result.raw_counts),
    ]

    dx = detector.pixel_size_x
    dy = detector.pixel_size_y

    x = result.x_detector_m
    y = result.y_detector_m

    n = len(images)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), constrained_layout=True)

    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    # -----------------------------
    # Top row: normalized images
    # -----------------------------
    top_ims = []
    for i, (name, img) in enumerate(images):
        img_norm = _normalize_image(img)

        ax = axes[0, i]
        im = ax.imshow(
            img_norm,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        top_ims.append(im)

    cbar1 = fig.colorbar(top_ims[-1], ax=axes[0, :], location="right", shrink=0.9)
    cbar1.set_label("Normalized intensity")

    # -----------------------------
    # Bottom row: normalized FFTs
    # -----------------------------
    bottom_ims = []
    for i, (name, img) in enumerate(images):
        # Subtract mean for counts/raw so DC does not dominate
        if "counts" in name.lower() or "raw" in name.lower():
            fft_mag, kx, ky = _fft2_phys(img - np.mean(img), dx, dy)
        else:
            fft_mag, kx, ky = _fft2_phys(img, dx, dy)

        fft_norm = _normalize_image(fft_mag)

        ax = axes[1, i]
        im = ax.imshow(
            fft_norm,
            extent=[kx.min(), kx.max(), ky.min(), ky.max()],
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_title(f"FFT - {name}")
        ax.set_xlabel(r"$k_x$ [rad/m]")
        ax.set_ylabel(r"$k_y$ [rad/m]")
        bottom_ims.append(im)

    cbar2 = fig.colorbar(bottom_ims[-1], ax=axes[1, :], location="right", shrink=0.9)
    cbar2.set_label("Normalized log |FFT|")

    fig.suptitle(title)
    plt.show()


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Example configuration
    # ------------------------------------------------------------
    geometry = Geometry(
        z_g=0.68,   # source -> grating [m]
        z_d=3.18,   # source -> detector [m]
    )

    detector = Detector(
        pixel_size_x=49.5e-6,
        pixel_size_y=49.5e-6,
        nx=1200,
        ny=900,
        psf_type="gaussian",
        psf_sigma_x_m=35e-6,
        psf_sigma_y_m=35e-6,
        mean_open_counts=2e4,
    )

    oversampling = Oversampling(subpixel=4)

    simulator = SingleGratingSimulator(
        geometry=geometry,
        detector=detector,
        oversampling=oversampling,
        rng=np.random.default_rng(42),
    )

    # ------------------------------------------------------------
    # Choose one grating type
    # ------------------------------------------------------------
    grating = GratingParams(
        grating_type="inverted_hartmann",
        pitch_x_m=50e-6,
        pitch_y_m=50e-6,
        tau_open=1.0,
        tau_block=0.02,
        opening_x_m=18e-6,
        opening_y_m=18e-6,
        rotation_deg=0.0,
    )

    # Examples for other types:
    #
    # line_1d
    # grating = GratingParams(
    #     grating_type="line_1d",
    #     pitch_x_m=20e-6,
    #     tau_open=1.0,
    #     tau_block=0.0,
    #     duty_x=0.5,
    #     rotation_deg=0.0,
    # )
    #
    # checkerboard
    # grating = GratingParams(
    #     grating_type="checkerboard",
    #     pitch_x_m=30e-6,
    #     pitch_y_m=30e-6,
    #     tau_open=1.0,
    #     tau_block=0.0,
    # )
    #
    # dot_array
    # grating = GratingParams(
    #     grating_type="dot_array",
    #     pitch_x_m=40e-6,
    #     pitch_y_m=40e-6,
    #     radius_m=8e-6,
    #     tau_open=1.0,
    #     tau_block=0.0,
    # )
    #
    # hartmann
    # grating = GratingParams(
    #     grating_type="hartmann",
    #     pitch_x_m=50e-6,
    #     pitch_y_m=50e-6,
    #     opening_x_m=15e-6,
    #     opening_y_m=15e-6,
    #     tau_open=1.0,
    #     tau_block=0.0,
    # )
    #
    # mesh_rectangular
    # grating = GratingParams(
    #     grating_type="mesh_rectangular",
    #     pitch_x_m=60e-6,
    #     pitch_y_m=60e-6,
    #     opening_x_m=30e-6,
    #     opening_y_m=30e-6,
    #     tau_open=1.0,
    #     tau_block=0.0,
    # )
    #
    # mesh_hexagonal
    # grating = GratingParams(
    #     grating_type="mesh_hexagonal",
    #     pitch_x_m=50e-6,
    #     radius_m=10e-6,
    #     tau_open=1.0,
    #     tau_block=0.0,
    # )

    # ------------------------------------------------------------
    # Source model
    # ------------------------------------------------------------
    source = Source(
        shape="gaussian",
        sigma_x_m=8e-6,
        sigma_y_m=8e-6,
    )

    # Other source examples:
    #
    # source = Source(
    #     shape="uniform_disk",
    #     radius_m=12e-6,
    # )
    #
    # source = Source(
    #     shape="uniform_rect",
    #     half_width_x_m=10e-6,
    #     half_width_y_m=5e-6,
    # )

    # ------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------
    result = simulator.simulate(
        grating=grating,
        source=source,
        add_poisson_noise=True,
    )

    print("Metadata:")
    for k, v in result.metadata.items():
        print(f"  {k}: {v}")

    show_result(result, detector, title=f"Grating type: {grating.grating_type}")