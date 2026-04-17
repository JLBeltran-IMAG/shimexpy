#------------------------------------------------------
# bands.py - GPU-accelerated band finding for ShimexPy
#------------------------------------------------------
import math
import cupy as cp
from dataclasses import dataclass
from fft import _fft


@dataclass(frozen=True)
class Harmonic:
    """
    Discrete harmonic index in the 2D Fourier domain.

    Represents the integer coordinates (nx, ny) of a spectral peak
    relative to the central (DC) component.

    Examples
    --------
    Harmonic(0, 0)   -> central component
    Harmonic(1, 0)   -> first order along positive x
    Harmonic(0, -1)  -> first order along negative y
    Harmonic(1, 1)   -> diagonal component
    """
    nx: int
    ny: int

    def __str__(self) -> str:
        return f"Harmonic(nx={self.nx}, ny={self.ny})"

    def __repr__(self) -> str:
        return f"Harmonic(nx={self.nx}, ny={self.ny})"

    @property
    def is_dc(self) -> bool:
        return self.nx == 0 and self.ny == 0

    @property
    def is_horizontal(self) -> bool:
        return self.ny == 0 and self.nx != 0

    @property
    def is_vertical(self) -> bool:
        return self.nx == 0 and self.ny != 0

    @property
    def is_diagonal(self) -> bool:
        return self.nx != 0 and self.ny != 0

    @property
    def order(self) -> int:
        return max(abs(self.nx), abs(self.ny))

    def opposite(self) -> "Harmonic":
        return Harmonic(-self.nx, -self.ny)

    def as_tuple(self) -> tuple[int, int]:
        return (self.nx, self.ny)


@dataclass(frozen=True)
class BandWindow:
    """
    Rectangular window in the Fourier domain corresponding to a detected harmonic.

    Defined by the top, bottom, left, and right indices that specify the
    region of the Fourier transform to extract for this band.

    Examples
    --------
    BandWindow(top=10, bottom=20, left=30, right=40) corresponds to
    fft[10:20, 30:40] in the Fourier domain.
    """
    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def center(self) -> tuple[float, float]:
        return (
            0.5 * (self.top + self.bottom),
            0.5 * (self.left + self.right)
        )

    def as_slice(self) -> tuple[slice, slice]:
        return (
            slice(self.top, self.bottom),
            slice(self.left, self.right)
        )
    
    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.top, self.bottom, self.left, self.right)

    def __repr__(self) -> str:
        return f"BandWindow(top={self.top}, bottom={self.bottom}, left={self.left}, right={self.right})"
    
    def __str__(self) -> str:
        return f"BandWindow(top={self.top}, bottom={self.bottom}, left={self.left}, right={self.right})"


@dataclass(frozen=True)
class Peak:
    """
    Represents a detected peak in the Fourier domain.

    Attributes
    ----------
    harmonic : Harmonic
        The harmonic index of the peak.

    row : int
        Row index of the peak in the Fourier array.

    col : int
        Column index of the peak in the Fourier array.

    magnitude : float
        Magnitude of the peak in the Fourier domain.
    """
    harmonic: Harmonic
    row: int
    col: int
    magnitude: float

    @property
    def position(self) -> tuple[int, int]:
        return (self.row, self.col)

    def as_tuple(self) -> tuple[Harmonic, int, int, float]:
        return (self.harmonic, self.row, self.col, self.magnitude)

    def __repr__(self) -> str:
        return f"Peak(Harmonic={self.harmonic}, row={self.row}, col={self.col}, value={self.magnitude:.3g})"

    def __str__(self) -> str:
        return f"Peak(Harmonic={self.harmonic}, row={self.row}, col={self.col}, value={self.magnitude:.3g})"


class SpatialHarmonics:
    """
    GPU-accelerated detection, representation, and calibration of
    spatial harmonics in the Fourier domain.

    This class extracts a discrete set of harmonic components from a
    2D Fourier transform and organizes them into a consistent spectral
    geometry that can be reused across measurements (e.g., reference → sample).

    The detected geometry includes:

    - DC component (main harmonic)
    - Characteristic spectral spacing
    - Discrete harmonic indices (axial and diagonal)
    - Rectangular extraction windows around each harmonic

    Attributes
    ----------
    _fft : cupy.ndarray
        Complex-valued Fourier transform of the input image (assumed fftshifted).

    _kx : cupy.ndarray
        1D array of frequency coordinates along the horizontal axis.

    _ky : cupy.ndarray
        1D array of frequency coordinates along the vertical axis.

    _k_radius : float
        Frequency radius used to determine the half-size of each extracted
        harmonic window.

    _n_peaks : int
        Number of harmonic peaks to detect (excluding the DC component).

    _abs_fft : cupy.ndarray
        Magnitude of the Fourier transform used for peak detection.

    _dc_row : int
        Row index of the detected DC component.

    _dc_col : int
        Column index of the detected DC component.

    _kx_lim : int
        Half-size of the harmonic window along the horizontal axis (in indices).

    _ky_lim : int
        Half-size of the harmonic window along the vertical axis (in indices).

    _delta_kx : int
        Estimated spectral spacing between DC and axial harmonics along x.

    _delta_ky : int
        Estimated spectral spacing between DC and axial harmonics along y.

    _delta_k : float
        Estimated diagonal spacing between DC and diagonal harmonics.

    harmonics : list[Harmonic]
        List of detected harmonic indices.

    windows : list[BandWindow]
        List of band windows corresponding to detected harmonics.

    peaks : list[Peak]
        List of detected peaks containing position and magnitude information.

    _geometry : dict[Harmonic, tuple[Peak, BandWindow]]
        Mapping from harmonic indices to their associated peak and window.

    Notes
    -----
    Two operating modes are supported:

    1. Reference mode (calibration=False):
    The spectral geometry is fully detected from the input FFT.

    2. Sample mode (calibration=True):
    The object is initialized without geometry and must be calibrated
    using a reference via ``calibrate()``.
    """
    def __init__(
        self,
        fft: cp.ndarray[cp.complex64],
        kx: cp.ndarray[cp.float32],
        ky: cp.ndarray[cp.float32],
        k_radius: float,
        n_peaks: int,
        calibration: bool = False,
        tolerance: float = 15.0,
    ) -> None:
        """
        Parameters
        ----------
        fft : cupy.ndarray
            2D complex-valued Fourier transform of the input image. It is assumed
            that the array is already centered (i.e., fftshift has been applied).

        kx : cupy.ndarray
            1D array of frequency coordinates along the horizontal axis.

        ky : cupy.ndarray
            1D array of frequency coordinates along the vertical axis.

        k_radius : float
            Frequency radius used to determine the half-size of each extracted
            band. The closest frequency index to this value is used to define the
            window extent in both directions.

        n_peaks : int
            Number of additional peaks to detect after the main harmonic.
        """
        self._fft = fft
        self._kx = kx
        self._ky = ky
        self._k_radius = k_radius
        self._n_peaks = n_peaks
        self._calibration = calibration
        self._tolerance = tolerance

        # Calculate internal state for detected peaks and windows
        self._abs_fft = cp.abs(self._fft)

        self.peaks: list[Peak] = []
        self.windows: list[BandWindow] = []
        self.harmonics: list[Harmonic] = []
        self._geometry: dict[Harmonic, tuple[Peak, BandWindow]] = {}

        if not calibration:
            self._w_half_size()
            self._find_dc()
            self._set_k_spacing()
            self._build_windows()


    def _zero_region(self, row, col):
        """
        Set to zero the rectangular region centered at a detected peak.

        This method suppresses the local spectral neighborhood of a detected
        harmonic in the magnitude spectrum in order to avoid repeated detection
        of the same lobe in subsequent peak searches.

        Parameters
        ----------
        row : int
            Row index of the peak center.

        col : int
            Column index of the peak center.

        Notes
        -----
        The suppressed region has the same half-size used for harmonic window
        construction, defined by ``_ky_lim`` and ``_kx_lim``.
        """
        top = row - self._ky_lim
        bottom = row + self._ky_lim
        left = col - self._kx_lim
        right = col + self._kx_lim
        self._abs_fft[top:bottom, left:right].fill(0)


    def _classify_harmonic(self, row, col) -> Harmonic | None:
        """
        Classify a detected peak into a discrete harmonic index.

        The classification is performed from the peak displacement with respect
        to the detected DC component. Peaks located closer to the axial directions
        than to the diagonal radius are classified as first-layer axial harmonics,
        while peaks whose radial distance is close to the characteristic diagonal
        spacing are classified as diagonal harmonics.

        Parameters
        ----------
        row : int
            Row index of the detected peak.

        col : int
            Column index of the detected peak.

        Returns
        -------
        Harmonic or None
            Discrete harmonic index associated with the detected peak. Returns
            ``None`` when the peak cannot be assigned to a supported harmonic
            family within the specified tolerance.

        Notes
        -----
        The current implementation distinguishes only two harmonic layers:

        - Layer 1: axial harmonics ``(±1, 0)`` and ``(0, ±1)``
        - Layer 2: diagonal harmonics ``(±1, ±1)``
        """
        dky = row - self._dc_row
        dkx = col - self._dc_col
        delta_k = math.hypot(dkx, dky)

        if delta_k < self._delta_k - self._tolerance:
            if abs(dky) > abs(dkx):
                return Harmonic(0, self._sign(dky))
            elif abs(dkx) > abs(dky):
                return Harmonic(self._sign(dkx), 0)
            else:
                return None # Unclassified

        elif abs(delta_k - self._delta_k) <= self._tolerance:
            return Harmonic(self._sign(dkx), self._sign(dky))

        else:
            return None # Unclassified


    def _detect_peak(self) -> tuple[int, int]:
        """
        Detect the current strongest peak in the magnitude spectrum.

        This method locates the global maximum of the working magnitude spectrum
        ``_abs_fft`` and returns its coordinates in array index space.

        Returns
        -------
        tuple[int, int]
            Row and column indices of the detected peak.

        Notes
        -----
        This method operates on the current internal working spectrum, which may
        already contain zeroed regions from previously detected peaks.
        """
        max_index = cp.argmax(self._abs_fft)
        row_index, col_index = cp.unravel_index(max_index, self._abs_fft.shape)
        return int(row_index), int(col_index)


    def _w_half_size(self) -> None:
        """
        Compute the half-size of the harmonic extraction window in index space.

        The half-size along each axis is obtained by finding the frequency index
        closest to the user-defined spectral radius ``_k_radius`` in the
        corresponding coordinate arrays ``_kx`` and ``_ky``.

        Notes
        -----
        The resulting values ``_kx_lim`` and ``_ky_lim`` are used both for:

        - defining the harmonic suppression region during peak detection
        - building the rectangular extraction windows around detected harmonics
        """
        self._ky_lim = int(cp.argmin(cp.abs(self._ky - self._k_radius)))
        self._kx_lim = int(cp.argmin(cp.abs(self._kx - self._k_radius)))


    def _find_dc(self) -> None:
        """
        Detect the DC component of the Fourier spectrum.

        The DC component is identified as the strongest peak in the current
        magnitude spectrum. Its row and column indices are stored internally as
        ``_dc_row`` and ``_dc_col``.

        Notes
        -----
        After detection, the DC position is suppressed in the working magnitude
        spectrum in order to prevent it from being detected again during the
        subsequent harmonic search.
        """
        self._dc_row, self._dc_col = self._detect_peak()

        # Excluding out DC for subsequent peak detection
        self._abs_fft[self._dc_row, self._dc_col] = -cp.inf


    def _set_k_spacing(self) -> None:
        """
        Estimate the characteristic spectral spacing between the DC component
        and the dominant harmonic families.

        The spacing along the horizontal and vertical directions is estimated
        from one-dimensional spectral profiles obtained by reducing the working
        magnitude spectrum along each axis. The main harmonic region around the
        DC is excluded before locating the strongest remaining maxima.

        Notes
        -----
        This method sets the following internal attributes:

        - ``_delta_kx`` : characteristic spacing along the horizontal direction
        - ``_delta_ky`` : characteristic spacing along the vertical direction
        - ``_delta_k``  : diagonal spacing computed from ``_delta_kx`` and ``_delta_ky``

        The diagonal spacing is defined as:

            sqrt(_delta_kx**2 + _delta_ky**2)
        """
        # Setting delta_k
        h, w = self._abs_fft.shape
        top = max(0, self._dc_row - self._ky_lim)
        bottom = min(h, self._dc_row + self._ky_lim + 1)
        left = max(0, self._dc_col - self._kx_lim)
        right = min(w, self._dc_col + self._kx_lim + 1)

        # Estimate characteristic harmonic spacing along x and y
        # by excluding the main harmonic region from the 1D spectral profiles.
        profile_x = cp.max(self._abs_fft, axis=0)
        profile_x[left:right] = -cp.inf
        col = int(cp.argmax(profile_x))
        self._delta_kx = abs(col - self._dc_col)

        profile_y = cp.max(self._abs_fft, axis=1)
        profile_y[top:bottom] = -cp.inf
        row = int(cp.argmax(profile_y))
        self._delta_ky = abs(row - self._dc_row)

        # Set delta_k
        self._delta_k = math.hypot(self._delta_kx, self._delta_ky)


    def _build_windows(self) -> None:
        """
        Construct the harmonic spectral geometry from the magnitude spectrum.

        This method iteratively detects the strongest remaining peaks in the
        working magnitude spectrum, classifies each peak into a discrete
        Harmonic index, and builds the corresponding BandWindow around it.

        For each valid (classified) peak:

        - A Peak object is created (position + magnitude)
        - A BandWindow is defined using the precomputed half-size limits
        - The pair (Peak, BandWindow) is stored in ``_geometry``
        - The harmonic index is recorded in ``harmonics``
        - The region around the peak is suppressed to avoid re-detection

        Unclassified peaks are ignored after suppressing their local region.

        Notes
        -----
        - The DC component must be detected and removed beforehand.
        - The characteristic spacing (``_delta_kx``, ``_delta_ky``, ``_delta_k``)
        must be initialized prior to calling this method.
        - The number of iterations is controlled by ``_n_peaks``.
        - This method defines the full spectral geometry used later for
        calibration and contrast computation.
        """
        for _ in range(self._n_peaks):
            row, col = self._detect_peak()
            harmonic = self._classify_harmonic(row, col)

            if harmonic is None:
                # If the peak cannot be classified,
                # we can choose to skip it or assign
                # a default label. For now, we will
                # skip unclassified peaks.
                self._zero_region(row, col)
                continue

            magnitude = float(self._abs_fft[row, col])

            top = row - self._ky_lim
            bottom = row + self._ky_lim + 1
            left = col - self._kx_lim
            right = col + self._kx_lim + 1

            bandwindow = BandWindow(
                top = top,
                bottom = bottom,
                left = left,
                right = right
            )

            peak = Peak(
                harmonic = harmonic,
                row = row,
                col = col,
                magnitude = magnitude
            )

            self.windows.append(bandwindow)
            self.peaks.append(peak)
            self.harmonics.append(harmonic)
            self._geometry[harmonic] = (peak, bandwindow)

            self._zero_region(row, col)


    def calibrate(self, reference: "SpatialHarmonics") -> None:
        """
        Calibrate the current object using the harmonic geometry of a reference.

        This method transfers the spectral geometry previously determined from a
        reference ``SpatialHarmonics`` instance and re-evaluates the peak
        magnitudes at the inherited harmonic positions using the current Fourier
        spectrum.

        Parameters
        ----------
        reference : "SpatialHarmonics"
            Reference harmonic object providing the calibrated spectral geometry.

        Notes
        -----
        The calibration transfers the geometric state, including:

        - DC position
        - harmonic window half-sizes
        - characteristic spectral spacings
        - detected harmonic identities
        - harmonic window definitions

        The Fourier transform itself is not transferred. Peak magnitudes are
        recomputed from the current object's spectrum at the inherited peak
        positions.
        """
        self._kx = reference._kx
        self._ky = reference._ky
        self._k_radius = reference._k_radius
        self._ky_lim = reference._ky_lim
        self._kx_lim = reference._kx_lim
        self._dc_row = reference._dc_row
        self._dc_col = reference._dc_col
        self._delta_kx = reference._delta_kx
        self._delta_ky = reference._delta_ky
        self._delta_k = reference._delta_k
        self._n_peaks = reference._n_peaks


        for harmonic, (peak_ref, window_ref) in reference._geometry.items():
            peak = Peak(
                harmonic=harmonic,
                row=peak_ref.row,
                col=peak_ref.col,
                magnitude=float(self._abs_fft[peak_ref.row, peak_ref.col])
            )
            self.harmonics.append(harmonic)
            self.windows.append(window_ref)
            self.peaks.append(peak)
            self._geometry[harmonic] = (peak, window_ref)


    @classmethod
    def from_image(
        cls,
        image: cp.ndarray,
        period: float,
        k_radius: float,
        n_peaks: int,
        calibration: bool = False,
        tolerance: float = 15.0,
    ) -> "SpatialHarmonics":
        """
        Create a SpatialHarmonics object directly from an input image.

        This class method computes the centered Fourier transform of the input
        image, generates the corresponding frequency coordinate arrays, and
        initializes a SpatialHarmonics instance from the resulting spectral data.

        Parameters
        ----------
        image : cupy.ndarray
            Input image in the spatial domain.

        period : float
            Characteristic spatial period used to generate the Fourier
            frequency coordinates.

        k_radius : float
            Frequency radius used to determine the half-size of each extracted
            harmonic window.

        n_peaks : int
            Number of harmonic peaks to detect after the DC component.

        calibration : bool, optional
            If True, the object is created without building its harmonic
            geometry and is expected to be calibrated later from a reference.
            Default is False.

        tolerance : float, optional
            Radial tolerance used during harmonic classification. Default is 15.0.

        Returns
        -------
        SpatialHarmonics
            A new SpatialHarmonics instance initialized from the input image.
        """
        fft, kx, ky = _fft(image, projected_grid=1/period, shift=True)

        return cls(
            fft=fft,
            kx=kx,
            ky=ky,
            k_radius=k_radius,
            n_peaks=n_peaks,
            calibration=calibration,
            tolerance=tolerance,
        )

    @staticmethod
    def _sign(x: int) -> int:
        """
        Return the sign of an integer as -1, 0, or +1.

        Parameters
        ----------
        x : int
            Input integer value.

        Returns
        -------
        int
            -1 if ``x < 0``, 0 if ``x == 0``, and +1 if ``x > 0``.
        """
        return 0 if x == 0 else (1 if x > 0 else -1)






def fft_bands(fft, kx, ky, k_radius, n_peaks):
    """
    Detect dominant spectral peaks in a 2D Fourier transform and extract
    rectangular frequency bands centered around them.

    This function identifies the strongest peaks in the magnitude of a
    Fourier-transformed image and constructs fixed-size rectangular regions
    ("bands") around each detected peak. The central (DC or main harmonic)
    component is detected first, followed by additional peaks found iteratively
    after masking previously detected regions.

    Parameters
    ----------
    fft : cupy.ndarray
        2D complex-valued Fourier transform of the input image. It is assumed
        that the array is already centered (i.e., fftshift has been applied).

    kx : cupy.ndarray
        1D array of frequency coordinates along the horizontal axis.

    ky : cupy.ndarray
        1D array of frequency coordinates along the vertical axis.

    k_radius : float
        Frequency radius used to determine the half-size of each extracted
        band. The closest frequency index to this value is used to define the
        window extent in both directions.

    n_peaks : int
        Number of additional peaks to detect after the main harmonic.

    Returns
    -------
    bands : dict[str, tuple[int, int, int, int]]
        Dictionary mapping band labels to rectangular regions in index space.
        Each region is defined as (top, bottom, left, right), corresponding to
        slices fft[top:bottom, left:right].

        The dictionary typically includes:
            - "harmonic_00": central component
            - "harmonic_horizontal_positive"
            - "harmonic_horizontal_negative"
            - "harmonic_vertical_positive"
            - "harmonic_vertical_negative"
            - "harmonic_diagonal" (optional)

    Notes
    -----
    - Peak detection is based on global maxima of the magnitude spectrum.
    - After each peak is detected, its surrounding region is zeroed out to
      avoid repeated detection.
    - The classification into horizontal, vertical, or diagonal components is
      based on relative displacement with respect to the main peak.
    - The number of detected peaks is fixed internally and may include
      low-energy components if no thresholding is applied.
    - No boundary clipping is performed; extracted regions may exceed array
      bounds if peaks are located near the edges.

    Limitations
    -----------
    - Assumes the strongest peak corresponds to the central component.
    - Sensitive to noise and spurious peaks in the spectrum.
    - Fixed window size may not adapt well to varying spectral structures.
    - Duplicate directional labels may overwrite previous entries.

    Examples
    --------
    >>> fft, kx, ky = compute_fft(image)
    >>> bands = fft_bands(fft, kx, ky, k_radius=0.5, n_peaks=4)
    >>> t, b, l, r = bands["harmonic_00"]
    >>> central_band = fft[t:b, l:r]
    """

    abs_fft = cp.abs(fft)

    max_idx = cp.argmax(abs_fft)
    h0, w0 = cp.unravel_index(max_idx, abs_fft.shape)
    h0, w0 = int(h0), int(w0)

    ky_lim = int(cp.argmin(cp.abs(ky - k_radius)))
    kx_lim = int(cp.argmin(cp.abs(kx - k_radius)))

    harmonics = {}

    def zero_region(arr, top, bottom, left, right):
        arr[top:bottom, left:right].fill(0)

    # main harmonic
    top    = h0 - ky_lim
    bottom = h0 + ky_lim
    left   = w0 - kx_lim
    right  = w0 + kx_lim

    harmonics["harmonic_00"] = (top, bottom, left, right)
    zero_region(abs_fft, top, bottom, left, right)

    for _ in range(n_peaks):
        idx = cp.argmax(abs_fft)
        h, w = cp.unravel_index(idx, abs_fft.shape)
        h, w = int(h), int(w)

        top    = h - ky_lim
        bottom = h + ky_lim
        left   = w - kx_lim
        right  = w + kx_lim

        dy = h - h0
        dx = w - w0

        if abs(dy) > abs(dx):
            label = (
                "harmonic_vertical_positive" # Harmonic(0, 1)
                if dy > 0 else
                "harmonic_vertical_negative" # Harmonic(0, -1)
            )
        elif abs(dx) > abs(dy):
            label = (
                "harmonic_horizontal_positive" # Harmonic(1, 0)
                if dx > 0 else
                "harmonic_horizontal_negative" # Harmonic(-1, 0)
            )
        else:
            if dy > 0 and dx > 0:
                label = "harmonic_diagonal_positive" # Harmonic(1, 1)
            elif dy > 0 and dx < 0:
                label = "harmonic_diagonal_negative" # Harmonic(-1, 1)
            elif dy < 0 and dx > 0:
                label = "harmonic_diagonal_negative" # Harmonic(1, -1)
            elif dy < 0 and dx < 0:
                label = "harmonic_diagonal_positive" # Harmonic(-1, -1)
            else:
                pass
            label = "harmonic_diagonal" # Harmonic(1, 1)

        harmonics[label] = (top, bottom, left, right)
        zero_region(abs_fft, top, bottom, left, right)

    return harmonics


def bands_to_array(bands: dict, order: list[str]) -> cp.ndarray:
    arr = cp.zeros((len(order), 4), dtype=cp.int32)
    for i, k in enumerate(order):
        arr[i] = cp.asarray(bands[k], dtype=cp.int32)
    return arr
