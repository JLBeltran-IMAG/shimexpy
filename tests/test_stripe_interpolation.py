"""Regression tests for detector stripe interpolation without GUI imports."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "shimexpy_tools/shimexpy_tools/pre_shi/interpolation/interp.py"
)
spec = importlib.util.spec_from_file_location("stripe_interp", MODULE_PATH)
interp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interp)


class StripeInterpolationTests(unittest.TestCase):
    def test_local_models_restore_masked_plane(self):
        rows, cols = np.indices((11, 11))
        original = (2 * rows + 3 * cols + 7).astype(np.float32)
        mask = np.zeros(original.shape, dtype=bool)
        mask[0, 0] = mask[5, 5] = mask[-1, -1] = True
        damaged = original.copy()
        damaged[mask] = np.nan
        for model in (interp.linear_2d, interp.polynomial_2d, interp.spline_2d):
            with self.subTest(model=model.__name__):
                result = model(damaged, mask, 2)
                np.testing.assert_allclose(result, original, atol=1e-5)
                self.assertTrue(np.isnan(damaged[mask]).all())
                self.assertFalse(np.shares_memory(result, damaged))
        result = interp.periodic_2d(damaged, mask, np.array([[0., 0.]]), 2)
        np.testing.assert_allclose(result, original, atol=1e-5)

    def test_methods_restore_profiles_in_both_axes(self):
        rows, cols = np.indices((9, 7))
        original = (3 * rows + 5 * cols).astype(np.float32)
        for method in ("linear", "median", "polynomial", "spline"):
            for axis in (0, 1):
                with self.subTest(method=method, axis=axis):
                    damaged = original.copy()
                    selection = [slice(None), slice(None)]
                    selection[axis] = 3
                    damaged[tuple(selection)] = -100
                    kwargs = {"stripe_rows" if axis == 0 else "stripe_cols": [3]}
                    result = interp.interpolate_stripes(damaged, method, **kwargs)
                    np.testing.assert_allclose(result, original, atol=1e-5)
                    np.testing.assert_equal(damaged[tuple(selection)], -100)

    def test_copy_and_inplace_dtype_contract(self):
        for dtype in (np.float32, np.float64, np.int32):
            for inplace in (False, True):
                with self.subTest(dtype=dtype, inplace=inplace):
                    image = np.zeros((5, 5), dtype=dtype)
                    result = interp.interpolate_stripes(image, "linear", inplace=inplace)
                    self.assertEqual(result.dtype, np.float32)
                    self.assertEqual(result is image, inplace and dtype == np.float32)

    def test_numpy_indices_and_unchanged_edges(self):
        image = np.arange(25, dtype=np.float32).reshape(5, 5)
        result = interp.interpolate_stripes(
            image, "linear", stripe_rows=np.array([0, 4, -1, 9]),
            stripe_cols=np.array([], dtype=int),
        )
        np.testing.assert_array_equal(result, image)

    def test_invalid_parameters(self):
        image = np.zeros((5, 5))
        for kwargs in ({"method": "unknown"}, {"projected_period": 1},
                       {"poly_degree": -1}, {"projected_period": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                interp.interpolate_stripes(image, **dict({"method": "linear"}, **kwargs))
        with self.assertRaises(ValueError):
            interp.interpolate_stripes(image, "polynomial", poly_degree=4, stripe_rows=[1])
        with self.assertRaises(NotImplementedError):
            interp.interpolate_stripes(image, "savgol")
        with self.assertRaises(ValueError):
            interp.interpolate_stripes(np.zeros(5), "linear")

    def test_delete_accepts_numpy_indices_and_preserves_input(self):
        image = np.arange(30).reshape(5, 6)
        result = interp.delete_stripes(image, np.array([1, 3]), np.array([2]))
        expected = image[[0, 2, 4]][:, [0, 1, 3, 4, 5]]
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, image.dtype)
        self.assertFalse(np.shares_memory(result, image))
        self.assertFalse(np.shares_memory(interp.delete_stripes(image, None, None), image))


if __name__ == "__main__":
    unittest.main()
