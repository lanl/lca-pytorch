import unittest

import torch
from torch.testing import assert_close, make_tensor

from lcapt.util import (
    check_equal_shapes,
    to_3d_from_5d,
    to_4d_from_5d,
    to_5d_from_3d,
    to_5d_from_4d,
)


class TestUtil(unittest.TestCase):
    def test_to_5d_from_3d_returns_correct_shape(self):
        inputs_3d = make_tensor((10, 3, 1000), device=None, dtype=torch.float32)
        inputs_5d = to_5d_from_3d(inputs_3d)
        self.assertEqual(len(inputs_5d.shape), 5)
        self.assertTrue(inputs_5d.shape[-2] == 1 and inputs_5d.shape[-1] == 1)

    def test_to_5d_from_3d_keeps_data_intact(self):
        inputs_3d = make_tensor((10, 3, 1000), device=None, dtype=torch.float32)
        inputs_5d = to_5d_from_3d(inputs_3d)
        assert_close(inputs_3d, inputs_5d.squeeze(), rtol=0.0, atol=0.0)

    def test_to_5d_from_3d_raises_AssertionError(self):
        inputs_2d = make_tensor((10, 1000), device=None, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            to_5d_from_3d(inputs_2d)

    def test_to_5d_from_4d_returns_correct_shape(self):
        inputs_4d = make_tensor((10, 3, 100, 100), device=None, dtype=torch.float32)
        inputs_5d = to_5d_from_4d(inputs_4d)
        self.assertEqual(len(inputs_5d.shape), 5)
        self.assertEqual(inputs_5d.shape[-3], 1)

    def test_to_5d_from_4d_keeps_data_intact(self):
        inputs_4d = make_tensor((10, 3, 100, 100), device=None, dtype=torch.float32)
        inputs_5d = to_5d_from_4d(inputs_4d)
        assert_close(inputs_4d, inputs_5d[..., 0, :, :], rtol=0.0, atol=0.0)

    def test_to_5d_from_4d_raises_AssertionError(self):
        inputs_3d = make_tensor((10, 100, 100), device=None, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            to_5d_from_4d(inputs_3d)

    def test_to_4d_from_5d_returns_correct_shape(self):
        inputs_5d = make_tensor((10, 3, 1, 100, 100), device=None, dtype=torch.float32)
        inputs_4d = to_4d_from_5d(inputs_5d)
        self.assertEqual(len(inputs_4d.shape), 4)

    def test_to_4d_from_5d_keeps_data_intact(self):
        inputs_5d = make_tensor((10, 3, 1, 100, 100), device=None, dtype=torch.float32)
        inputs_4d = to_4d_from_5d(inputs_5d)
        assert_close(inputs_4d, inputs_5d[:, :, 0], rtol=0.0, atol=0.0)

    def test_to_4d_from_5d_raises_AssertionError(self):
        inputs_4d = make_tensor((10, 3, 100, 100), device=None, dtype=torch.float32)
        inputs_5d = make_tensor((10, 3, 5, 100, 100), device=None, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            to_4d_from_5d(inputs_4d)
            to_4d_from_5d(inputs_5d)

    def test_to_3d_from_5d_returns_correct_shape(self):
        inputs_5d = make_tensor((10, 3, 100, 1, 1), device=None, dtype=torch.float32)
        inputs_3d = to_3d_from_5d(inputs_5d)
        self.assertEqual(len(inputs_3d.shape), 3)

    def test_to_3d_from_5d_keeps_data_intact(self):
        inputs_5d = make_tensor((10, 3, 100, 1, 1), device=None, dtype=torch.float32)
        inputs_3d = to_3d_from_5d(inputs_5d)
        assert_close(inputs_3d, inputs_5d.squeeze(), rtol=0.0, atol=0.0)

    def test_to_3d_from_5d_raises_AssertionError(self):
        inputs_4d = make_tensor((10, 3, 5, 1), device=None, dtype=torch.float32)
        inputs_5d = make_tensor((10, 3, 100, 100, 1), device=None, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            to_3d_from_5d(inputs_4d)
            to_3d_from_5d(inputs_5d)

    def test_check_equal_shapes_raises_RuntimeError(self):
        tensor1 = torch.zeros(10, 3, 9, 9)
        tensor2 = torch.zeros(10, 4, 9, 9)
        with self.assertRaises(RuntimeError):
            check_equal_shapes(tensor1, tensor2, "test")

    def test_check_equal_shapes_passes(self):
        tensor1 = torch.zeros(10, 3, 9, 9)
        tensor2 = tensor1.clone()
        self.assertIsNone(check_equal_shapes(tensor1, tensor2, "test"))


if __name__ == "__main__":
    unittest.main()
