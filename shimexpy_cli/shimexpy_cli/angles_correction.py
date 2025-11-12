import numpy as np


def next_two_power_for_dimension_padding(image: np.ndarray) -> int:
    image_height, image_width = image.shape
    max_dimension = np.max([image_height, image_width])
    new_dimension_to_pad_image = np.power(2, np.ceil(np.log2(max_dimension)))
    return int(new_dimension_to_pad_image)


def squared_fft(image: np.ndarray) -> np.ndarray:
    image_height, image_width = image.shape

    new_dimension_to_pad_image = next_two_power_for_dimension_padding(image)
    amount_of_zeros_in_height = int(new_dimension_to_pad_image - image_height)
    amount_of_zeros_in_width = int(new_dimension_to_pad_image - image_width)
    new_image_padded_with_zeros = np.pad(
        image,
        ((0, amount_of_zeros_in_height), (0, amount_of_zeros_in_width)),
        mode="constant",
        constant_values=0,
    )

    fourier_transform_of_image = np.fft.fftshift(
        np.fft.fft2(new_image_padded_with_zeros.astype(np.float32))
    )

    return fourier_transform_of_image


def zero_fft_region(array2d, top, bottom, left, right):
    """
    Sets a specific rectangular region of a 2D complex array to zero.

    This function is useful for filtering out certain frequency components
    in the Fourier domain of an image.

    Parameters:
    -----------
    array2d : np.ndarray
        A 2D NumPy array representing the Fourier transform of an image.
        It must be a complex-valued array.
    top : int
        The starting row index of the region to be zeroed.
    bottom : int
        The ending row index of the region to be zeroed (exclusive).
    left : int
        The starting column index of the region to be zeroed.
    right : int
        The ending column index of the region to be zeroed (exclusive).

    Returns:
    --------
    np.ndarray
        The modified 2D array with the specified region set to zero.
    """
    array2d[top:bottom, left:right] = np.complex128(0)
    return array2d


def extracting_harmonic(fourier_transform, ky_band_limit, kx_band_limit):
    """
    Extracts a rectangular region around the maximum harmonic component in a Fourier transform.

    This function locates the point with the highest magnitude in the Fourier transform and
    defines a rectangular region centered on that point, using the provided vertical and
    horizontal band limits.

    Parameters:
    -----------
    fourier_transform : np.ndarray
        A 2D NumPy array representing the Fourier transform of an image (complex values).
    ky_band_limit : int
        The vertical band limit (number of rows) to extract around the maximum component.
    kx_band_limit : int
        The horizontal band limit (number of columns) to extract around the maximum component.

    Returns:
    --------
    tuple
        A tuple containing:
            top_limit (int): The top boundary of the extracted region.
            bottom_limit (int): The bottom boundary of the extracted region.
            left_limit (int): The left boundary of the extracted region.
            right_limit (int): The right boundary of the extracted region.
            max_row_index (int): The row index of the maximum magnitude component.
            max_col_index (int): The column index of the maximum magnitude component.
    """
    # Find the index of the maximum magnitude in the Fourier transform
    max_row_index, max_col_index = np.unravel_index(np.argmax(np.abs(fourier_transform)), fourier_transform.shape)

    # Calculate boundaries and ensure they stay within the array dimensions
    top_limit = max(0, max_row_index - ky_band_limit)
    bottom_limit = min(fourier_transform.shape[0], max_row_index + ky_band_limit)
    left_limit = max(0, max_col_index - kx_band_limit)
    right_limit = min(fourier_transform.shape[1], max_col_index + kx_band_limit)

    return top_limit, bottom_limit, left_limit, right_limit, max_row_index, max_col_index





def extracting_coordinates_of_peaks(image: np.ndarray) -> list:
    fourier_transform = squared_fft(image.astype(np.float32))
    copy_of_fourier_transform = np.copy(fourier_transform)

    index_of_main_maximun_height, index_of_main_maximun_width = np.unravel_index(
        np.argmax(np.abs(fourier_transform)), shape=fourier_transform.shape
    )
    ky_band_limit_of_harmonics = 500
    kx_band_limit_of_harmonics = 500

    coordenates = list()

    # Extracting 0-order harmonic
    top_limit = index_of_main_maximun_height - ky_band_limit_of_harmonics
    bottom_limit = index_of_main_maximun_height + ky_band_limit_of_harmonics
    left_limit = index_of_main_maximun_width - kx_band_limit_of_harmonics
    right_limit = index_of_main_maximun_width + kx_band_limit_of_harmonics

    coordenates.append([index_of_main_maximun_height, index_of_main_maximun_width])

    zero_fft_region(
        copy_of_fourier_transform, top_limit, bottom_limit, left_limit, right_limit
    )

    # plt.imshow(np.log(1 + np.abs(fourier_transform)), cmap = "gray")
    # plt.show()

    # Extracting higher-order harmonics (dafault 1-order)
    for i in range(4):
        (
            top_limit,
            bottom_limit,
            left_limit,
            right_limit,
            index_of_harmonic_height,
            index_of_harmonic_width,
        ) = extracting_harmonic(
            copy_of_fourier_transform,
            ky_band_limit_of_harmonics,
            kx_band_limit_of_harmonics,
        )

        coordenates.append([index_of_harmonic_height, index_of_harmonic_width])
        zero_fft_region(
            copy_of_fourier_transform, top_limit, bottom_limit, left_limit, right_limit
        )
        # plt.imshow(np.log(1 + np.abs(copy_of_fourier_transform)), cmap = "gray")
        # plt.show()

    return coordenates


def quadrant_loc_sign(y: int, h: int, x: int, w: int, axes: str) -> int:
    sign = 0

    if axes == "y":
        if x > w and y < h:
            sign = 1
        elif x < w and y < h:
            sign = -1
        elif x < w and y > h:
            sign = 1
        elif x > w and y > h:
            sign = -1
        else:
            sign = 0

    elif axes == "x":
        if x > w and y > h:
            sign = 1
        elif x > w and y < h:
            sign = -1
        elif x < w and y < h:
            sign = 1
        elif x < w and y > h:
            sign = -1
        else:
            sign = 0

    else:
        sign = 0

    return sign


def calculating_angles_of_peaks_average(coords: list) -> np.float32:
    main_harmonic_height, main_harmonic_width = coords[0]
    angles = list()
    sign = list()

    for i in range(1, len(coords)):
        height = abs(coords[i][0] - main_harmonic_height)
        width = abs(coords[i][1] - main_harmonic_width)
        if height > width:
            sign.append(
                quadrant_loc_sign(
                    coords[i][0],
                    main_harmonic_height,
                    coords[i][1],
                    main_harmonic_width,
                    axes="y",
                )
            )
            angles.append(np.rad2deg(np.arctan2(width, height)))

        elif height < width:
            sign.append(
                quadrant_loc_sign(
                    coords[i][0],
                    main_harmonic_height,
                    coords[i][1],
                    main_harmonic_width,
                    axes="x",
                )
            )
            angles.append(np.rad2deg(np.arctan2(height, width)))

    return np.mean(np.array(angles) * np.array(sign)).astype(np.float32)

