# import numpy as np#
# import xarray as xr
# import pytest
# import sys
# import os
# from tifffile import imread

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shimexpy")))

# from spatial_harmonics import (
#     shi_fft_linear_and_log,
#     FFTResult,
#     _zero_fft_region,
#     _extracting_harmonic,
#     _identifying_harmonics_x1y1_higher_orders,
#     _identifying_harmonic,
#     spatial_harmonics_of_fourier_spectrum,
#     _unwrapping_phase_gradient_operator,
#     _compute_phase_map,
#     _compute_scattering,
#     _differential_phase_contrast,
#     contrast_retrieval_individual_members
# )


# def test_shi_fft_linear_and_log_basic():
#     """Test básico sin projected_grid ni logspect."""
#     test_image = np.random.rand(10, 10)
#     res = shi_fft_linear_and_log(test_image)

#     assert isinstance(res, FFTResult)

#     assert res.kx is None
#     assert res.ky is None

#     assert isinstance(res.fft, np.ndarray)
#     assert res.fft.shape == test_image.shape


# def test_shi_fft_linear_and_log_with_grid():
#     """Test con projected_grid activado."""
#     test_image = np.random.rand(10, 10)
#     projected_grid = 5.0
#     res = shi_fft_linear_and_log(test_image, projected_grid=projected_grid)

#     assert isinstance(res, FFTResult)
#     assert isinstance(res.kx, np.ndarray)
#     assert isinstance(res.ky, np.ndarray)
#     assert isinstance(res.fft, np.ndarray)

#     assert res.fft.shape == test_image.shape
#     assert res.kx.shape[0] == test_image.shape[1]
#     assert res.ky.shape[0] == test_image.shape[0]


# def test_shi_fft_linear_and_log_with_logspect():
#     """Test solo con logspect activado."""
#     test_image = np.random.rand(10, 10)
#     res = shi_fft_linear_and_log(test_image, logspect=True)

#     assert isinstance(res, FFTResult)
#     assert res.kx is None and res.ky is None

#     assert isinstance(res.fft, np.ndarray)
#     assert res.fft.shape == test_image.shape
#     assert np.all(res.fft >= 0)


# def test_shi_fft_linear_and_log_invalid_dimensions():
#     """Test con imagen 1D: debe lanzar ValueError."""
#     test_image = np.random.rand(10)
#     with pytest.raises(ValueError, match="2D"):
#         shi_fft_linear_and_log(test_image)


# def test_shi_fft_linear_and_log_with_grid_and_logspect():
#     """Test con projected_grid y logspect activados."""
#     test_image = np.random.rand(10, 10)
#     projected_grid = 5.0
#     res = shi_fft_linear_and_log(test_image, projected_grid=projected_grid, logspect=True)

#     assert isinstance(res, FFTResult)
#     assert isinstance(res.kx, np.ndarray)
#     assert isinstance(res.ky, np.ndarray)
#     assert isinstance(res.fft, np.ndarray)

#     assert res.fft.shape == test_image.shape
#     assert res.kx.shape[0] == test_image.shape[1]
#     assert res.ky.shape[0] == test_image.shape[0]

#     assert np.all(res.fft >= 0)


# def test_zero_fft_region():
#     """Test zeroing out a region in a complex array"""
#     # Create a test array filled with ones
#     test_array = np.ones((10, 10), dtype=np.complex128)
    
#     # Zero out the central region
#     result = _zero_fft_region(test_array, 3, 7, 2, 8)
    
#     # Check that dimensions haven't changed
#     assert result.shape == test_array.shape
    
#     # Check that specified region is zero
#     assert np.all(result[3:7, 2:8] == 0)
    
#     # Check that outside region is still ones
#     assert np.all(result[0:3, :] == 1)
#     assert np.all(result[7:, :] == 1) 
#     assert np.all(result[:, 0:2] == 1)
#     assert np.all(result[:, 8:] == 1)
    
#     # Check that output is complex
#     assert np.iscomplexobj(result)


# def test_zero_fft_region_edge_cases():
#     """Test edge cases for region zeroing"""
#     test_array = np.ones((5, 5), dtype=np.complex128)
    
#     # Test zeroing entire array
#     result = _zero_fft_region(test_array, 0, 5, 0, 5)
#     assert np.all(result == 0)
    
#     # Test zeroing single element
#     test_array = np.ones((5, 5), dtype=np.complex128) 
#     result = _zero_fft_region(test_array, 2, 3, 2, 3)
#     assert result[2, 2] == 0
#     assert np.sum(result == 0) == 1


# def test_zero_fft_region_modifies_input():
#     """Test that _zero_fft_region modifies the input array in-place"""
#     test_array = np.ones((5, 5), dtype=np.complex128)
#     original = test_array.copy()

#     # Apply zeroing to a specific region
#     _zero_fft_region(test_array, 1, 3, 1, 3)

#     # The region [1:3, 1:3] should be zeroed out (complex zeros)
#     # The array should be modified
#     assert not np.all(test_array == original)
#     # The zeroed region should be all zeros
#     assert np.all(test_array[1:3, 1:3] == 0)
#     # Unmodified rows should remain unchanged
#     assert np.all(test_array[:1, :] == 1) and np.all(test_array[3:, :] == 1)
#     # Unmodified columns should remain unchanged
#     assert np.all(test_array[:, :1] == 1) and np.all(test_array[:, 3:] == 1)


# @pytest.fixture(scope="module")
# def real_fft_data():
#     # Ruta relativa a la imagen .tif en la misma carpeta que los tests
#     path = os.path.join(os.path.dirname(__file__), "test_reference.tif")
#     image = imread(path).astype(np.float32)

#     img_height, img_width = image.shape
#     projected_grid = 5.0

#     fourier_transform = np.fft.fftshift(np.fft.fft2(image))
#     kx = np.fft.fftfreq(img_width, d=1 / projected_grid)
#     ky = np.fft.fftfreq(img_height, d=1 / projected_grid)

#     return kx, ky, fourier_transform


# def test_extracting_harmonic_basic(real_fft_data):
#     """Test basic functionality of harmonic extraction"""
#     kx, ky, fourier_transform = real_fft_data
#     copy_of_fourier_transform = fourier_transform.copy()
#     limit_band = 0.5

#     # Identify the main maximum harmonic (assumed to be near the center)
#     abs_fft = np.abs(fourier_transform)
#     max_index = np.argmax(abs_fft)
#     main_max_h, main_max_w = np.unravel_index(max_index, abs_fft.shape)

#     # Determine band limits based on the wavevector arrays.
#     ky_band_limit = np.argmin(np.abs(ky - limit_band))
#     kx_band_limit = np.argmin(np.abs(kx - limit_band))

#     assert isinstance(kx_band_limit, np.integer)
#     assert isinstance(ky_band_limit, np.integer)
#     assert isinstance(main_max_h, np.integer)
#     assert isinstance(main_max_w, np.integer)

#     # Extract the 0-order harmonic.
#     top = main_max_h - ky_band_limit
#     bottom = main_max_h + ky_band_limit
#     left = main_max_w - kx_band_limit
#     right = main_max_w + kx_band_limit

#     assert isinstance(top, np.integer)
#     assert isinstance(bottom, np.integer)
#     assert isinstance(left, np.integer)
#     assert isinstance(right, np.integer)

#     # Zero out the extracted region in the copy to avoid re-detection.
#     _zero_fft_region(copy_of_fourier_transform, top, bottom, left, right)

#     # Extract higher-order harmonics (by default, 4 additional harmonics).
#     for i in range(8):
#         top, bottom, left, right, harmonic_h, harmonic_w = _extracting_harmonic(
#         copy_of_fourier_transform, ky_band_limit, kx_band_limit
#         )

#         assert isinstance(top, np.integer)
#         assert isinstance(bottom, np.integer)
#         assert isinstance(left, np.integer)
#         assert isinstance(right, np.integer)

#         label = _identifying_harmonic(main_max_h, main_max_w, harmonic_h, harmonic_w)
#         _zero_fft_region(copy_of_fourier_transform, top, bottom, left, right)

#     assert 0 <= main_max_h < fourier_transform.shape[0]
#     assert 0 <= main_max_w < fourier_transform.shape[1]
#     assert 0 <= top < bottom <= fourier_transform.shape[0]
#     assert 0 <= left < right <= fourier_transform.shape[1]


# @pytest.fixture(scope="module")
# def reference_harmonics(real_fft_data):
#     """
#     Ejecuta spatial_harmonics en modo referencia y devuelve:
#       da_ref, labels_ref, grid_ref, fft, ky, kx
#     """
#     kx, ky, fourier_transform = real_fft_data
#     da_ref, labels_ref, block_grid_ref = spatial_harmonics_of_fourier_spectrum(
#         fourier_transform=fourier_transform,
#         ky=ky,
#         kx=kx,
#         reference=True,
#         limit_band=0.5
#     )
#     return da_ref, labels_ref, block_grid_ref, kx, ky, fourier_transform


# def test_spatial_harmonics_reference_mode(reference_harmonics):
#     """Test spatial harmonics extraction in reference mode"""
#     # Create wavevector arrays
#     da, labels, block_grid, _, _, _ = reference_harmonics

#     # Test return types
#     assert isinstance(da, xr.DataArray)
#     assert isinstance(labels, list)
#     assert isinstance(block_grid, dict)

#     # Test DataArray properties
#     assert "harmonic" in da.dims
#     assert "ky" in da.dims
#     assert "kx" in da.dims

#     # Expected labels and block grid keys
#     expected = {
#         "harmonic_00",
#         "harmonic_horizontal_positive",
#         "harmonic_horizontal_negative",
#         "harmonic_vertical_positive",
#         "harmonic_vertical_negative",
#         "harmonic_diagonal_p1_p1",
#         "harmonic_diagonal_p1_n1",
#         "harmonic_diagonal_n1_p1",
#         "harmonic_diagonal_n1_n1"
#     }

#     assert set(labels) == expected
#     assert set(block_grid) == expected

#     for limits in block_grid.values():
#         assert len(limits) == 4  # [top, bottom, left, right]
#         assert all(isinstance(x, np.integer) for x in limits)


# def test_spatial_harmonics_non_reference_mode(reference_harmonics):
#     """Test en modo reference=False, reutilizando labels y grid del fixture anterior"""
#     da_ref, labels_ref, block_grid_ref, kx, ky, fourier_transform = reference_harmonics

#     da_nonref, labels_nonref, block_grid_nonref = spatial_harmonics_of_fourier_spectrum(
#         fourier_transform=fourier_transform,
#         ky=ky,
#         kx=kx,
#         reference=False,
#         reference_block_grid=block_grid_ref
#     )

#     # They should have the same labels and block_grid
#     assert labels_nonref == labels_ref
#     assert block_grid_nonref == block_grid_ref

#     # Their DataArrays must be the same
#     assert da_nonref.identical(da_ref)


# def test_spatial_harmonics_missing_reference_grid():
#     """Test error raising when reference_block_grid is missing in non-reference mode"""
#     test_ft = np.zeros((50, 50), dtype=np.complex128)
#     kx = np.linspace(-2, 2, 50)
#     ky = np.linspace(-2, 2, 50)
    
#     with pytest.raises(ValueError, match="Reference block grid.*must be provided"):
#         spatial_harmonics_of_fourier_spectrum(
#             fourier_transform=test_ft,
#             ky=ky,
#             kx=kx,
#             reference=False,
#             reference_block_grid=None
#         )


