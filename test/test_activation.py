import unittest

import torch
from torch.testing import assert_close

from lcapt.activation import hard_threshold, soft_threshold


Tensor = torch.Tensor


def create_test_input() -> Tensor:
    inputs = torch.arange(-20, -0.2, 0.1)
    return torch.cat((inputs, inputs * -1))


class TestActivations(unittest.TestCase):
    def test_hard_threshold_above_threshold(self):
        inputs = create_test_input()
        outputs = hard_threshold(inputs, 0.2, nonneg=False)
        assert_close(inputs, outputs, rtol=0.0, atol=0.0)

    def test_nonneg_hard_threshold_above_threshold(self):
        inputs = create_test_input()
        outputs = hard_threshold(inputs, 0.2)
        n_inputs = inputs.numel()
        assert_close(
            inputs[n_inputs // 2 :], outputs[n_inputs // 2 :], rtol=0.0, atol=0.0
        )
        assert_close(
            torch.zeros(n_inputs // 2), outputs[: n_inputs // 2], rtol=0.0, atol=0.0
        )

    def test_hard_threshold_below_threshold(self):
        inputs = torch.arange(-0.2, 0.21, 0.01)
        outputs = hard_threshold(inputs, 0.2, nonneg=False)
        assert_close(outputs, torch.zeros(inputs.numel()), rtol=0.0, atol=0.0)

    def test_nonneg_hard_threshold_below_threshold(self):
        inputs = torch.arange(-10.0, 0.2, 0.01)
        outputs = hard_threshold(inputs, 0.2)
        assert_close(outputs, torch.zeros(inputs.numel()), rtol=0.0, atol=0.0)

    def test_hard_threshold_returns_correct_shape(self):
        inputs = torch.randn(1, 10, 100, 100)
        outputs = hard_threshold(inputs, 0.5, False)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_nonneg_hard_threshold_returns_correct_shape(self):
        inputs = torch.randn(1, 10, 100, 100)
        outputs = hard_threshold(inputs, 0.5)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_hard_threshold_returns_torch_tensor(self):
        inputs = torch.randn(1, 10, 100, 100)
        outputs = hard_threshold(inputs, 0.5)
        self.assertEqual(type(outputs), torch.Tensor)

    def test_soft_threshold_above_threshold(self):
        inputs = create_test_input()
        outputs = soft_threshold(inputs, 0.2, nonneg=False)
        assert_close(inputs - 0.2 * inputs.sign(), outputs, rtol=0.0, atol=0.0)

    def test_nonneg_soft_threshold_above_threshold(self):
        inputs = create_test_input()
        outputs = soft_threshold(inputs, 0.2)
        n_inputs = inputs.numel()
        expected = inputs[n_inputs // 2 :]
        assert_close(
            expected - 0.2 * expected.sign(),
            outputs[n_inputs // 2 :],
            rtol=0.0,
            atol=0.0,
        )
        assert_close(
            torch.zeros(n_inputs // 2), outputs[: n_inputs // 2], rtol=0.0, atol=0.0
        )

    def test_soft_threshold_below_threshold(self):
        inputs = torch.arange(-0.2, 0.21, 0.01)
        outputs = soft_threshold(inputs, 0.2, nonneg=False)
        assert_close(outputs, torch.zeros(inputs.numel()), rtol=0.0, atol=0.0)

    def test_nonneg_soft_threshold_below_threshold(self):
        inputs = torch.arange(-10.0, 0.2, 0.01)
        outputs = soft_threshold(inputs, 0.2)
        assert_close(outputs, torch.zeros(inputs.numel()), rtol=0.0, atol=0.0)

    def test_soft_threshold_returns_correct_shape(self):
        inputs = torch.randn(1, 10, 100, 100)
        outputs = soft_threshold(inputs, 0.5, False)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_nonneg_soft_threshold_returns_correct_shape(self):
        inputs = torch.randn(1, 10, 100, 100)
        outputs = soft_threshold(inputs, 0.5)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_soft_threshold_returns_torch_tensor(self):
        inputs = torch.randn(1, 10, 100, 100)
        outputs = soft_threshold(inputs, 0.5)
        self.assertEqual(type(outputs), torch.Tensor)


if __name__ == "__main__":
    unittest.main()
