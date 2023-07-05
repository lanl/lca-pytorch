import unittest

import torch

from lcapt.analysis import make_feature_grid


class TestAnalysis(unittest.TestCase):
    def test_make_feature_grid_returns_correct_shape_3D_input_one_channel(self):
        inputs = torch.randn(5, 1, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (5, 10))

    def test_make_feature_grid_returns_correct_shape_3D_input_with_multi_channel(self):
        inputs = torch.randn(5, 3, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (17, 26))

    def test_make_feature_grid_returns_correct_shape_4D_input(self):
        inputs = torch.randn(5, 3, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (38, 26, 3))

    def test_make_feature_grid_returns_correct_shape_5D_input_time_equals_one(self):
        inputs = torch.randn(5, 3, 1, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (38, 26, 3))

    def test_make_feature_grid_returns_correct_shape_5D_input_time_gt_one(self):
        inputs = torch.randn(5, 3, 3, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (3, 38, 26, 3))

    def test_make_feature_grid_returns_correct_shape_2D_input(self):
        inputs = torch.randn(5, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (5, 10))

    def test_make_feature_grid_raises_RuntimeError_1D_input(self):
        inputs_1d = torch.randn(5)
        inputs_2d = torch.randn(5, 10)
        make_feature_grid(inputs_2d)
        with self.assertRaises(RuntimeError):
            make_feature_grid(inputs_1d)

    def test_make_feature_grid_raises_RuntimeError_6D_input(self):
        inputs_5d = torch.randn(5, 10, 10, 10, 10)
        inputs_6d = torch.randn(5, 1, 10, 10, 10, 10)
        make_feature_grid(inputs_5d)
        with self.assertRaises(RuntimeError):
            make_feature_grid(inputs_6d)


if __name__ == "__main__":
    unittest.main()
