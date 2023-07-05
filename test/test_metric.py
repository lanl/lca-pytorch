import unittest

import numpy as np
import torch
from torch.testing import assert_close

from lcapt.metric import (
    compute_frac_active,
    compute_l1_sparsity,
    compute_l2_error,
    compute_times_active_by_feature,
)


class TestMetrics(unittest.TestCase):
    def test_compute_frac_active_all_active(self):
        inputs = torch.ones(10, 100, 8, 8)
        frac_active = compute_frac_active(inputs)
        self.assertEqual(frac_active, 1.0)

    def test_compute_frac_active_none_active(self):
        inputs = torch.zeros(10, 100, 8, 8)
        frac_active = compute_frac_active(inputs)
        self.assertEqual(frac_active, 0.0)

    def test_compute_frac_active_active_positive_equals_negative(self):
        inputs_pos = torch.randn(10, 100, 8, 8)
        inputs_pos[inputs_pos.abs() < 0.5] = 0
        inputs_neg = inputs_pos * -1
        assert_close(compute_frac_active(inputs_pos), compute_frac_active(inputs_neg))

    def test_compute_frac_active(self):
        inputs = torch.zeros(10, 1, 100)
        for ind in range(10):
            inputs[ind, 0, : ind + 1] = ind + 1
            frac_active = compute_frac_active(inputs[ind])
            self.assertAlmostEqual(frac_active, (ind + 1) / 100)

    def test_compute_frac_active_correct_return_type(self):
        inputs = torch.randn(10, 1, 100)
        self.assertEqual(float, type(compute_frac_active(inputs)))

    def test_compute_l1_sparsity_all_active_and_positive(self):
        inputs = torch.rand(1, 100, 8, 8) + 1
        l1_sparsity = compute_l1_sparsity(inputs, 1.0)
        assert_close(inputs.norm(1), l1_sparsity)

    def test_compute_l1_sparsity_none_active(self):
        inputs = torch.zeros(3, 100, 8, 8)
        l1_sparsity = compute_l1_sparsity(inputs, 1.0)
        assert_close(torch.tensor(0.0), l1_sparsity)

    def test_compute_l1_sparsity_positive_equals_negative(self):
        inputs_pos = torch.randn(1, 100, 8, 8)
        inputs_pos[inputs_pos.abs() < 0.5] = 0
        inputs_neg = inputs_pos * -1
        assert_close(
            compute_l1_sparsity(inputs_pos, 1.0), compute_l1_sparsity(inputs_neg, 1.0)
        )

    def test_compute_l1_sparsity_different_lambdas(self):
        inputs = torch.randn(1, 100, 10, 10)
        inputs[inputs.abs() < 0.5] = 0
        for lambda_ in np.arange(0.1, 3.1, 0.1):
            l1_sparsity = compute_l1_sparsity(inputs, lambda_)
            assert_close(l1_sparsity, inputs.norm(1) * lambda_)

    def test_compute_l1_sparsity_returns_same_dtype_as_inputs(self):
        for dtype in [torch.float16, torch.float32, torch.float64]:
            inputs = torch.randn(10, 100, 8, 8, dtype=dtype)
            l1_sparsity = compute_l1_sparsity(inputs, 1.0)
            self.assertEqual(l1_sparsity.dtype, dtype)

    def test_compute_l1_sparsity_returns_torch_tensor(self):
        inputs = torch.randn(10, 100, 8, 8)
        l1_sparsity = compute_l1_sparsity(inputs, 1.0)
        self.assertEqual(type(l1_sparsity), torch.Tensor)

    def test_compute_l1_sparsity_returns_zero_dim_tensor(self):
        inputs = torch.randn(10, 100, 8, 8)
        l1_sparsity = compute_l1_sparsity(inputs, 1.0)
        self.assertEqual(len(l1_sparsity.shape), 0)
        self.assertEqual(l1_sparsity.numel(), 1)

    def test_compute_l2_error_inputs_not_equal_to_recons(self):
        inputs = torch.zeros(3, 100, 10, 10)
        recons = torch.ones(3, 100, 10, 10)
        assert_close(compute_l2_error(inputs, recons), 0.5 * recons[0].norm(2) ** 2)

    def test_compute_l2_error_inputs_equal_recons(self):
        inputs = torch.randn(3, 10, 100, 100)
        recons = inputs * 1.0
        assert_close(compute_l2_error(inputs, recons), torch.tensor(0.0))

    def test_compute_l2_error_inputs_and_recons_differ_by_one_value(self):
        inputs = torch.zeros(3, 10, 100, 100)
        recons = inputs * 1.0
        recons[:, 0, 0, 50] = 10
        assert_close(compute_l2_error(inputs, recons), torch.tensor(50.0))

    def test_compute_l2_error_returns_same_data_type_as_inputs(self):
        for dtype in [torch.float16, torch.float32, torch.float64]:
            inputs = torch.randn(10, 100, 8, 8, dtype=dtype)
            error = compute_l2_error(inputs, inputs * 10)
            self.assertEqual(error.dtype, dtype)

    def test_compute_l2_error_returns_torch_tensor(self):
        inputs = torch.randn(10, 100, 8, 8)
        error = compute_l2_error(inputs, inputs * 10)
        self.assertEqual(type(error), torch.Tensor)

    def test_compute_l2_error_returns_zero_dim_tensor(self):
        inputs = torch.randn(10, 100, 8, 8)
        error = compute_l2_error(inputs, inputs * 10)
        self.assertEqual(len(error.shape), 0)
        self.assertEqual(error.numel(), 1)

    def test_compute_times_active_by_feature_returns_correct_shape(self):
        inputs = torch.zeros(1, 100, 8, 8)
        times_active = compute_times_active_by_feature(inputs)
        self.assertEqual(times_active.shape[0], 100)
        self.assertEqual(times_active.numel(), 100)
        self.assertEqual(len(times_active.shape), len(inputs.shape))

    def test_compute_times_active_by_feature_returns_torch_tensor(self):
        inputs = torch.zeros(1, 100, 8, 8)
        times_active = compute_times_active_by_feature(inputs)
        self.assertEqual(type(times_active), torch.Tensor)

    def test_compute_times_active_by_feature_none_active(self):
        inputs = torch.zeros(3, 100, 10, 10)
        times_active = compute_times_active_by_feature(inputs)
        assert_close(times_active.sum().item(), 0.0, rtol=0.0, atol=0.0)

    def test_compute_times_active_by_feature_all_active(self):
        inputs = torch.ones(3, 100, 10, 10)
        times_active = compute_times_active_by_feature(inputs)
        self.assertEqual(times_active.sum().item(), 100 * 3 * 10**2)

    def test_compute_times_active_by_feature_variable_active(self):
        inputs = torch.zeros(1, 10, 100)
        for ind in range(10):
            inputs[:, ind, : ind + 1] = torch.randn(1)
        times_active = compute_times_active_by_feature(inputs)
        assert_close(times_active[:, 0, 0], torch.arange(1.0, 11.0))


if __name__ == "__main__":
    unittest.main()
