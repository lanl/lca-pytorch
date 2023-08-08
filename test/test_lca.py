from tempfile import TemporaryDirectory
import unittest

from sklearn.decomposition import SparseCoder
import torch
from torch.testing import assert_close

from lcapt.lca import LCAConv1D, LCAConv2D, LCAConv3D


class TestLCA(unittest.TestCase):
    def test_LCAConv1D_to_correct_shape_raises_RuntimeError(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir)
            inputs_4d = torch.zeros(1, 3, 10, 10)

            with self.assertRaises(RuntimeError):
                lca._to_correct_shape(inputs_4d)

    def test_LCAConv2D_to_correct_shape_raises_RuntimeError(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir)
            inputs_3d = torch.zeros(1, 3, 10)

            with self.assertRaises(RuntimeError):
                lca._to_correct_shape(inputs_3d)

    def test_LCAConv3D_to_correct_shape_raises_RuntimeError(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir)
            inputs_4d = torch.zeros(1, 3, 10, 10)

            with self.assertRaises(RuntimeError):
                lca._to_correct_shape(inputs_4d)

    def test_LCA_raises_ValueError_given_invalid_return_var(self):
        inputs = torch.zeros(1, 1, 32, 32)
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 1, tmp_dir, lca_iters=1, return_vars=["acts", "conns"])
            lca(inputs)
            lca = LCAConv2D(10, 1, tmp_dir, lca_iters=1, return_vars=("acts", "conns"))
            lca(inputs)
            lca = LCAConv2D(10, 1, tmp_dir, lca_iters=1, return_vars=["atcs", "conns"])
            with self.assertRaises(ValueError):
                lca(inputs)

    def test_LCAConv1D_get_weights_returns_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5)
            weights = lca.get_weights()
            self.assertEqual(len(weights.shape), 3)
            self.assertTupleEqual(weights.numpy().shape, (10, 3, 5))

    def test_LCAConv2D_get_weights_returns_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7))
            weights = lca.get_weights()
            self.assertEqual(len(weights.shape), 4)
            self.assertTupleEqual(weights.numpy().shape, (10, 3, 5, 7))

    def test_LCAConv3D_get_weights_returns_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7))
            weights = lca.get_weights()
            self.assertEqual(len(weights.shape), 5)
            self.assertTupleEqual(weights.numpy().shape, (10, 3, 9, 5, 7))

    def test_LCAConv1D_assign_weight_values_normalize_False(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5)
            new_weights = torch.randn(10, 3, 5)
            lca.assign_weight_values(new_weights, False)
            assert_close(new_weights, lca.get_weights(), rtol=0, atol=0)

    def test_LCAConv2D_assign_weight_values_normalize_False(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7))
            new_weights = torch.randn(10, 3, 5, 7)
            lca.assign_weight_values(new_weights, False)
            assert_close(new_weights, lca.get_weights(), rtol=0, atol=0)

    def test_LCAConv3D_assign_weight_values_normalize_False(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7))
            new_weights = torch.randn(10, 3, 9, 5, 7)
            lca.assign_weight_values(new_weights, False)
            assert_close(new_weights, lca.get_weights(), rtol=0, atol=0)

    def test_LCAConv1D_normalize_weights(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 501)
            new_weights = torch.rand(10, 3, 501) * 10 + 10
            lca.assign_weight_values(new_weights)
            lca.normalize_weights()
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0)

    def test_LCAConv2D_normalize_weights(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7))
            new_weights = torch.rand(10, 3, 5, 7) * 10 + 10
            lca.assign_weight_values(new_weights)
            lca.normalize_weights()
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0)

    def test_LCAConv3D_normalize_weights(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7))
            new_weights = torch.rand(10, 3, 9, 5, 7) * 10 + 10
            lca.assign_weight_values(new_weights)
            lca.normalize_weights()
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0)

    def test_LCAConv1D_initial_weights_are_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 1001)
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0, atol=3e-7, rtol=3e-7)

    def test_LCAConv2D_initial_weights_are_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (17, 15))
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0)

    def test_LCAConv3D_initial_weights_are_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7))
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0)

    def test_compute_input_pad_raises_ValueError(self):
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                LCAConv2D(10, 3, tmp_dir, (5, 7), pad="weird_padding")

    def test_LCAConv1D_input_padding_shape_same_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5)
            self.assertTupleEqual(lca.input_pad, (2, 0, 0))

    def test_LCAConv2D_input_padding_shape_same_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7))
            self.assertTupleEqual(lca.input_pad, (0, 2, 3))

    def test_LCAConv3D_input_padding_shape_same_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7))
            self.assertTupleEqual(lca.input_pad, (4, 2, 3))

    def test_LCAConv1D_input_padding_shape_valid_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, pad="valid")
            self.assertTupleEqual(lca.input_pad, (0, 0, 0))

    def test_LCAConv2D_input_padding_shape_valid_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7), pad="valid")
            self.assertTupleEqual(lca.input_pad, (0, 0, 0))

    def test_LCAConv3D_input_padding_shape_valid_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7), pad="valid")
            self.assertTupleEqual(lca.input_pad, (0, 0, 0))

    def test_LCAConv1D_code_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, lca_iters=3)
            inputs = torch.randn(1, 3, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 100))

    def test_LCAConv2D_code_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7), lca_iters=3)
            inputs = torch.randn(1, 3, 100, 99)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 100, 99))

    def test_LCAConv3D_code_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7), lca_iters=3)
            inputs = torch.randn(1, 3, 8, 100, 101)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 8, 100, 101))

    def test_LCAConv1D_code_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, 2, lca_iters=3)
            inputs = torch.randn(1, 3, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 50))

    def test_LCAConv2D_code_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7), 2, lca_iters=3)
            inputs = torch.randn(1, 3, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 50, 50))

    def test_LCAConv3D_code_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7), 2, lca_iters=3)
            inputs = torch.randn(1, 3, 8, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 4, 50, 50))

    def test_LCAConv1D_code_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, 4, lca_iters=3)
            inputs = torch.randn(1, 3, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 25))

    def test_LCAConv2D_code_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7), 4, lca_iters=3)
            inputs = torch.randn(1, 3, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 25, 25))

    def test_LCAConv3D_code_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7), 4, lca_iters=3)
            inputs = torch.randn(1, 3, 8, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 2, 25, 25))

    def test_LCAConv1D_recon_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, lca_iters=3, return_vars=["recons"])
            inputs = torch.randn(1, 3, 100)
            recon = lca(inputs)
            assert_close(recon.shape, inputs.shape, rtol=0, atol=0)

    def test_LCAConv2D_recon_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7), lca_iters=3, return_vars=["recons"])
            inputs = torch.randn(1, 3, 100, 100)
            recon = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv3D_recon_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, (9, 5, 7), lca_iters=3, return_vars=["recons"]
            )
            inputs = torch.randn(1, 3, 8, 100, 100)
            recon = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv1D_recon_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, 2, lca_iters=3, return_vars=["recons"])
            inputs = torch.randn(1, 3, 100)
            recon = lca(inputs)
            assert_close(recon.shape, inputs.shape, rtol=0, atol=0)

    def test_LCAConv2D_recon_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10, 3, tmp_dir, (5, 7), 2, lca_iters=3, return_vars=["recons"]
            )
            inputs = torch.randn(1, 3, 100, 100)
            recon = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv3D_recon_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, (9, 5, 7), 2, lca_iters=3, return_vars=["recons"]
            )
            inputs = torch.randn(1, 3, 8, 100, 100)
            recon = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv1D_recon_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, 4, lca_iters=3, return_vars=["recons"])
            inputs = torch.randn(1, 3, 100)
            recon = lca(inputs)
            assert_close(recon.shape, inputs.shape, rtol=0, atol=0)

    def test_LCAConv2D_recon_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10, 3, tmp_dir, (5, 7), 4, lca_iters=3, return_vars=["recons"]
            )
            inputs = torch.randn(1, 3, 100, 100)
            recon = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv3D_recon_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, (9, 5, 7), 4, lca_iters=3, return_vars=["recons"]
            )
            inputs = torch.randn(1, 3, 8, 100, 100)
            recon = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv1D_recon_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10, 3, tmp_dir, 100, pad="valid", lca_iters=3, return_vars=["recons"]
            )
            inputs = torch.randn(1, 3, 100)
            recon = lca(inputs)
            self.assertTupleEqual(recon.numpy().shape, (1, 3, 100))

    def test_LCAConv2D_recon_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                3,
                tmp_dir,
                100,
                lca_iters=3,
                return_vars=["recons"],
                pad="valid",
            )
            inputs = torch.randn(1, 3, 100, 100)
            recon = lca(inputs)
            self.assertTupleEqual(recon.numpy().shape, (1, 3, 100, 100))

    def test_LCAConv3D_recon_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                3,
                tmp_dir,
                (4, 20, 20),
                pad="valid",
                lca_iters=3,
                return_vars=["recons"],
            )
            inputs = torch.randn(1, 3, 4, 20, 20)
            recon = lca(inputs)
            self.assertTupleEqual(recon.numpy().shape, (1, 3, 4, 20, 20))

    def test_LCAConv1D_code_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 100, pad="valid", lca_iters=3)
            inputs = torch.randn(1, 3, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 1))

    def test_LCAConv2D_code_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 100, lca_iters=3, pad="valid")
            inputs = torch.randn(1, 3, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 1, 1))

    def test_LCAConv3D_code_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (4, 20, 20), pad="valid", lca_iters=3)
            inputs = torch.randn(1, 3, 4, 20, 20)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 1, 1, 1))

    def test_LCAConv3D_code_shape_no_time_pad(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (5, 7, 7), lca_iters=3, no_time_pad=True)
            inputs = torch.randn(1, 3, 5, 20, 20)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 1, 20, 20))

    def test_LCAConv3D_recon_shape_no_time_pad(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                3,
                tmp_dir,
                (5, 7, 7),
                lca_iters=3,
                no_time_pad=True,
                return_vars=["recons"],
            )
            inputs = torch.randn(1, 3, 5, 20, 20)
            recon = lca(inputs)
            self.assertTupleEqual(recon.numpy().shape, inputs.numpy().shape)

    def test_LCAConv1D_gradient(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, lca_iters=3, req_grad=True)
            inputs = torch.randn(1, 3, 100)
            with torch.no_grad():
                code = lca(inputs)

            loss = code.sum()
            with self.assertRaises(RuntimeError):
                loss.backward()

            code = lca(inputs)
            loss = code.sum()
            loss.backward()

    def test_LCAConv2D_gradient(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (5, 7), lca_iters=3, req_grad=True)
            inputs = torch.randn(1, 3, 20, 20)
            with torch.no_grad():
                code = lca(inputs)

            loss = code.sum()
            with self.assertRaises(RuntimeError):
                loss.backward()

            code = lca(inputs)
            loss = code.sum()
            loss.backward()

    def test_LCAConv3D_gradient(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (9, 5, 7), lca_iters=3, req_grad=True)
            inputs = torch.randn(1, 3, 5, 20, 20)
            with torch.no_grad():
                code = lca(inputs)

            loss = code.sum()
            with self.assertRaises(RuntimeError):
                loss.backward()

            code = lca(inputs)
            loss = code.sum()
            loss.backward()

    def test_LCAConv1D_code_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            for lambda_ in torch.arange(0.1, 1.0, 0.1):
                lca = LCAConv1D(
                    10,
                    3,
                    tmp_dir,
                    100,
                    pad="valid",
                    input_zero_mean=False,
                    input_unit_var=False,
                    lambda_=lambda_,
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                code = lca(inputs)
                code = code.squeeze()
                code = torch.sort(code, descending=True, stable=True)[0]
                self.assertEqual(torch.count_nonzero(code), 1)
                assert_close(code[0], code.max())

    def test_LCAConv2D_code_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            for lambda_ in torch.arange(0.1, 1.0, 0.1):
                lca = LCAConv2D(
                    10,
                    3,
                    tmp_dir,
                    10,
                    pad="valid",
                    input_zero_mean=False,
                    input_unit_var=False,
                    lambda_=lambda_,
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                code = lca(inputs)
                code = code.squeeze()
                code = torch.sort(code, descending=True, stable=True)[0]
                self.assertEqual(torch.count_nonzero(code), 1)
                assert_close(code[0], code.max())

    def test_LCAConv3D_code_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            for lambda_ in torch.arange(0.1, 1.0, 0.1):
                lca = LCAConv3D(
                    10,
                    3,
                    tmp_dir,
                    10,
                    pad="valid",
                    input_zero_mean=False,
                    input_unit_var=False,
                    lambda_=lambda_,
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                code = lca(inputs)
                code = code.squeeze()
                code = torch.sort(code, descending=True, stable=True)[0]
                self.assertTrue(torch.count_nonzero(code), 1)
                assert_close(code[0], code.max())

    def test_LCAConv1D_compute_lateral_connectivity_stride_1_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 50, 2):
                lca = LCAConv1D(15, 3, tmp_dir, ksize)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, 0].numpy().shape, (15, 15, ksize * 2 - 1)
                )

    def test_LCAConv1D_compute_lateral_connectivity_stride_1_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, 0].numpy().shape, (15, 15, ksize * 2 - 1)
                )

    def test_LCAConv1D_compute_lateral_connectivity_stride_2_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 50, 2):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, 2)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns[..., 0, 0].numpy().shape, (15, 15, ksize))

    def test_LCAConv1D_compute_lateral_connectivity_stride_2_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, 2, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns[..., 0, 0].numpy().shape, (15, 15, ksize - 1))

    def test_LCAConv1D_compute_lateral_connectivity_odd_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 7
            for stride, exp_size in zip(range(1, 8), [13, 7, 5, 3, 3, 3, 1]):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, stride)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns[..., 0, 0].numpy().shape, (15, 15, exp_size))

    def test_LCAConv1D_compute_lateral_connectivity_even_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [15, 7, 5, 3, 3, 3, 3, 1]):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, stride, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns[..., 0, 0].numpy().shape, (15, 15, exp_size))

    def test_LCAConv2D_compute_lateral_connectivity_stride_1_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 50, 2):
                ksize2 = ksize + 2
                lca = LCAConv2D(15, 3, tmp_dir, (ksize, ksize2))
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape,
                    (15, 15, ksize * 2 - 1, ksize2 * 2 - 1),
                )

    def test_LCAConv2D_compute_lateral_connectivity_stride_1_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                ksize2 = ksize + 2
                lca = LCAConv2D(15, 3, tmp_dir, (ksize, ksize2), pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape,
                    (15, 15, ksize * 2 - 1, ksize2 * 2 - 1),
                )

    def test_LCAConv2D_compute_lateral_connectivity_stride_2_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 50, 2):
                ksize2 = ksize + 2
                lca = LCAConv2D(15, 3, tmp_dir, (ksize, ksize2), 2)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape, (15, 15, ksize, ksize2)
                )

    def test_LCAConv2D_compute_lateral_connectivity_stride_2_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                ksize2 = ksize + 2
                lca = LCAConv2D(15, 3, tmp_dir, (ksize, ksize2), 2, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape, (15, 15, ksize - 1, ksize2 - 1)
                )

    def test_LCAConv2D_compute_lateral_connectivity_odd_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 7
            for stride, exp_size in zip(range(1, 8), [13, 7, 5, 3, 3, 3, 1]):
                lca = LCAConv2D(15, 3, tmp_dir, ksize, stride)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape, (15, 15, exp_size, exp_size)
                )

    def test_LCAConv2D_compute_lateral_connectivity_even_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [15, 7, 5, 3, 3, 3, 3, 1]):
                lca = LCAConv2D(15, 3, tmp_dir, ksize, stride, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape, (15, 15, exp_size, exp_size)
                )

    def test_LCAConv3D_compute_lateral_connectivity_stride_1_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 11, 2):
                ksize2 = ksize + 2
                ksize3 = ksize + 4
                lca = LCAConv3D(15, 3, tmp_dir, (ksize3, ksize, ksize2))
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape,
                    (15, 15, ksize3 * 2 - 1, ksize * 2 - 1, ksize2 * 2 - 1),
                )

    def test_LCAConv3D_compute_lateral_connectivity_stride_1_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 12, 2):
                ksize2 = ksize + 2
                ksize3 = ksize + 4
                lca = LCAConv3D(15, 3, tmp_dir, (ksize3, ksize, ksize2), pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape,
                    (15, 15, ksize3 * 2 - 1, ksize * 2 - 1, ksize2 * 2 - 1),
                )

    def test_LCAConv3D_compute_lateral_connectivity_stride_2_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 11, 2):
                ksize2 = ksize + 2
                ksize3 = ksize + 4
                lca = LCAConv3D(15, 3, tmp_dir, (ksize3, ksize, ksize2), 2)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns.numpy().shape, (15, 15, ksize3, ksize, ksize2))

    def test_LCAConv3D_compute_lateral_connectivity_stride_2_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 12, 2):
                ksize2 = ksize + 2
                ksize3 = ksize + 4
                lca = LCAConv3D(15, 3, tmp_dir, (ksize3, ksize, ksize2), 2, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape, (15, 15, ksize3 - 1, ksize - 1, ksize2 - 1)
                )

    def test_LCAConv3D_compute_lateral_connectivity_odd_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 7
            for stride, exp_size in zip(range(1, 8), [13, 7, 5, 3, 3, 3, 1]):
                lca = LCAConv3D(15, 3, tmp_dir, ksize, stride)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape, (15, 15, exp_size, exp_size, exp_size)
                )

    def test_LCAConv3D_compute_lateral_connectivity_even_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [15, 7, 5, 3, 3, 3, 3, 1]):
                lca = LCAConv3D(
                    15,
                    3,
                    tmp_dir,
                    ksize,
                    stride,
                    pad="valid",
                )
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape, (15, 15, exp_size, exp_size, exp_size)
                )

    def test_LCAConv3D_no_time_pad(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 7)
            self.assertEqual(lca.input_pad[0], 3)
            lca = LCAConv3D(10, 3, tmp_dir, 7, no_time_pad=True)
            self.assertEqual(lca.input_pad[0], 0)

    def test_l1_norm_of_code_decreases_with_increasing_lambda(self):
        with TemporaryDirectory() as tmp_dir:
            l1_norms = []
            for lambda_ in torch.arange(0.1, 1.0, 0.1):
                lca = LCAConv2D(
                    10,
                    3,
                    tmp_dir,
                    10,
                    lambda_=lambda_,
                    pad="valid",
                    input_zero_mean=False,
                    input_unit_var=False,
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                code = lca(inputs)
                l1_norms.append(code.norm(1).item())
            self.assertEqual(l1_norms, sorted(l1_norms, reverse=True))

    def test_recon_error_increases_with_increasing_lambda(self):
        with TemporaryDirectory() as tmp_dir:
            errors = []
            for lambda_ in torch.arange(0.1, 1.1, 0.1):
                lca = LCAConv2D(
                    10,
                    3,
                    tmp_dir,
                    10,
                    lambda_=lambda_,
                    pad="valid",
                    input_zero_mean=False,
                    input_unit_var=False,
                    return_vars=["recon_errors"],
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                recon_error = lca(inputs)
                errors.append(0.5 * recon_error.norm(2) ** 2)
            self.assertEqual(errors, sorted(errors))

    def test_LCAConv1D_inputs_equal_recon_error_plus_recon(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10,
                5,
                tmp_dir,
                5,
                lca_iters=3,
                input_zero_mean=False,
                input_unit_var=False,
                return_vars=["recons", "recon_errors"],
            )
            inputs = torch.randn(3, 5, 100)
            recon, recon_error = lca(inputs)
            assert_close(inputs, recon_error + recon)

    def test_LCAConv2D_inputs_equal_recon_error_plus_recon(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                5,
                tmp_dir,
                5,
                lca_iters=3,
                input_zero_mean=False,
                input_unit_var=False,
                return_vars=["recons", "recon_errors"],
            )
            inputs = torch.randn(3, 5, 100, 100)
            recon, recon_error = lca(inputs)
            assert_close(inputs, recon_error + recon)

    def test_LCAConv3D_inputs_equal_recon_error_plus_recon(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                5,
                tmp_dir,
                (3, 5, 5),
                (1, 2, 2),
                lca_iters=3,
                input_zero_mean=False,
                input_unit_var=False,
                return_vars=["recons", "recon_errors"],
            )
            inputs = torch.randn(3, 5, 10, 100, 100)
            recon, recon_error = lca(inputs)
            assert_close(inputs, recon_error + recon)

    def test_LCAConv2D_check_conv_params_raises_AssertionError_odd_even_ksizes(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize1 in range(2, 12, 2):
                for ksize2 in range(3, 12, 2):
                    with self.assertRaises(AssertionError):
                        LCAConv2D(10, 1, tmp_dir, (ksize1, ksize2))

    def test_LCAConv3D_check_conv_params_raises_AssertionError_odd_even_ksizes(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize1 in range(2, 12, 2):
                for ksize2 in range(3, 12, 2):
                    for ksize3 in range(2, 12, 2):
                        with self.assertRaises(AssertionError):
                            LCAConv3D(10, 1, tmp_dir, (ksize3, ksize1, ksize2))

    def test_LCAConv1D_compute_inhib_pad_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [7, 6, 6, 4, 5, 6, 7, 0]):
                lca = LCAConv1D(10, 3, tmp_dir, ksize, stride, pad="valid")
                self.assertEqual(lca.lat_conn_pad[0], exp_size)

    def test_LCAConv1D_compute_inhib_pad_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 9
            for stride, exp_size in zip(range(1, 10), [8, 8, 6, 8, 5, 6, 7, 8, 0]):
                lca = LCAConv1D(10, 3, tmp_dir, ksize, stride, pad="valid")
                self.assertEqual(lca.lat_conn_pad[0], exp_size)

    def test_LCAConv2D_compute_inhib_pad_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [7, 6, 6, 4, 5, 6, 7, 0]):
                lca = LCAConv2D(10, 3, tmp_dir, ksize, stride, pad="valid")
                self.assertEqual(lca.lat_conn_pad[1:], (exp_size, exp_size))

    def test_LCAConv2D_compute_inhib_pad_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 9
            for stride, exp_size in zip(range(1, 10), [8, 8, 6, 8, 5, 6, 7, 8, 0]):
                lca = LCAConv2D(10, 3, tmp_dir, ksize, stride, pad="valid")
                self.assertEqual(lca.lat_conn_pad[1:], (exp_size, exp_size))

    def test_LCAConv3D_compute_inhib_pad_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [7, 6, 6, 4, 5, 6, 7, 0]):
                lca = LCAConv3D(
                    10,
                    3,
                    tmp_dir,
                    ksize,
                    stride,
                    pad="valid",
                )
                self.assertEqual(lca.lat_conn_pad, (exp_size,) * 3)

    def test_LCAConv3D_compute_inhib_pad_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 9
            for stride, exp_size in zip(range(1, 10), [8, 8, 6, 8, 5, 6, 7, 8, 0]):
                lca = LCAConv3D(
                    10,
                    3,
                    tmp_dir,
                    ksize,
                    stride,
                    pad="valid",
                )
                self.assertEqual(lca.lat_conn_pad, (exp_size,) * 3)

    def test_LCAConv1D_compute_inhib_pad_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 1)
            self.assertEqual(lca.lat_conn_pad, (0, 0, 0))

    def test_LCAConv2D_compute_inhib_pad_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 1)
            self.assertEqual(lca.lat_conn_pad, (0, 0, 0))

    def test_LCAConv3D_compute_inhib_pad_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 1)
            self.assertEqual(lca.lat_conn_pad, (0, 0, 0))

    def test_LCAConv1D_compute_lateral_connectivity_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 1)
            conns = lca.compute_lateral_connectivity(lca.weights)
            self.assertEqual(conns.numpy().shape, (10, 10, 1, 1, 1))

    def test_LCAConv2D_compute_lateral_connectivity_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 1)
            conns = lca.compute_lateral_connectivity(lca.weights)
            self.assertEqual(conns.numpy().shape, (10, 10, 1, 1, 1))

    def test_LCAConv3D_compute_lateral_connectivity_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 1)
            conns = lca.compute_lateral_connectivity(lca.weights)
            self.assertEqual(conns.numpy().shape, (10, 10, 1, 1, 1))

    def test_LCAConv1D_input_zero_mean_and_unit_var(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(8, 3, tmp_dir, 7, return_vars=["inputs"], lca_iters=1)
            inputs = torch.rand(10, 3, 32)
            inputs_model = lca(inputs)
            for inp in range(10):
                self.assertLess(inputs_model[inp].mean().item(), 1e-5)
                assert_close(inputs_model[inp].std().item(), 1.0)

    def test_LCAConv2D_input_zero_mean_and_unit_var(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(8, 3, tmp_dir, 7, return_vars=["inputs"], lca_iters=1)
            inputs = torch.rand(10, 3, 32, 32)
            inputs_model = lca(inputs)
            for inp in range(10):
                self.assertLess(inputs_model[inp].mean().item(), 1e-5)
                assert_close(inputs_model[inp].std().item(), 1.0)

    def test_LCAConv3D_input_zero_mean_and_unit_var(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(8, 3, tmp_dir, 7, return_vars=["inputs"], lca_iters=1)
            inputs = torch.rand(10, 3, 5, 32, 32)
            inputs_model = lca(inputs)
            for inp in range(10):
                self.assertLess(inputs_model[inp].mean().item(), 1e-5)
                assert_close(inputs_model[inp].std().item(), 1.0)

    def test_LCAConv1D_return_all_ts_return_one_var(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(8, 3, tmp_dir, 7, return_all_ts=True, lca_iters=6)
            inputs = torch.randn(9, 3, 11)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11, 6))

    def test_LCAConv1D_return_all_ts_return_multiple_vars(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                8,
                3,
                tmp_dir,
                7,
                return_all_ts=True,
                return_vars=["acts", "input_drives"],
                lca_iters=6,
            )
            inputs = torch.randn(9, 3, 11)
            code, input_drive = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11, 6))
            self.assertTupleEqual(input_drive.numpy().shape, (9, 8, 11, 6))

    def test_LCAConv1D_return_one_ts_return_multiple_vars(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                8, 3, tmp_dir, 7, return_vars=["acts", "input_drives"], lca_iters=6
            )
            inputs = torch.randn(9, 3, 11)
            code, input_drive = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11))
            self.assertTupleEqual(input_drive.numpy().shape, (9, 8, 11))

    def test_LCAConv2D_return_all_ts_return_one_var(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(8, 3, tmp_dir, 7, return_all_ts=True, lca_iters=6)
            inputs = torch.randn(9, 3, 11, 11)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11, 11, 6))

    def test_LCAConv2D_return_all_ts_return_multiple_vars(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                8,
                3,
                tmp_dir,
                7,
                return_all_ts=True,
                return_vars=["acts", "input_drives"],
                lca_iters=6,
            )
            inputs = torch.randn(9, 3, 11, 11)
            code, input_drive = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11, 11, 6))
            self.assertTupleEqual(input_drive.numpy().shape, (9, 8, 11, 11, 6))

    def test_LCAConv2D_return_one_ts_return_multiple_vars(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                8, 3, tmp_dir, 7, return_vars=["acts", "input_drives"], lca_iters=6
            )
            inputs = torch.randn(9, 3, 11, 11)
            code, input_drive = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11, 11))
            self.assertTupleEqual(input_drive.numpy().shape, (9, 8, 11, 11))

    def test_LCAConv3D_return_all_ts_return_one_var(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(8, 3, tmp_dir, 7, return_all_ts=True, lca_iters=6)
            inputs = torch.randn(9, 3, 11, 11, 11)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11, 11, 11, 6))

    def test_LCAConv3D_return_all_ts_return_multiple_vars(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                8,
                3,
                tmp_dir,
                7,
                return_all_ts=True,
                return_vars=["acts", "input_drives"],
                lca_iters=6,
            )
            inputs = torch.randn(9, 3, 11, 11, 11)
            code, input_drive = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11, 11, 11, 6))
            self.assertTupleEqual(input_drive.numpy().shape, (9, 8, 11, 11, 11, 6))

    def test_LCAConv3D_return_one_ts_return_multiple_vars(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                8, 3, tmp_dir, 7, return_vars=["acts", "input_drives"], lca_iters=6
            )
            inputs = torch.randn(9, 3, 11, 11, 11)
            code, input_drive = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (9, 8, 11, 11, 11))
            self.assertTupleEqual(input_drive.numpy().shape, (9, 8, 11, 11, 11))

    def test_LCAConv1D_transform_conv_params_int(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 11)
            out = lca._transform_conv_params(11)
            self.assertTupleEqual(out, (11, 1, 1))

    def test_LCAConv1D_transform_conv_params_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 11)
            out = lca._transform_conv_params((11,))
            self.assertTupleEqual(out, (11, 1, 1))

    def test_LCAConv2D_transform_conv_params_int(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 11)
            out = lca._transform_conv_params(11)
            self.assertTupleEqual(out, (1, 11, 11))

    def test_LCAConv2D_transform_conv_params_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 11)
            out = lca._transform_conv_params((9, 11))
            self.assertTupleEqual(out, (1, 9, 11))

    def test_LCAConv3D_transform_conv_params_int(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 11)
            out = lca._transform_conv_params(11)
            self.assertTupleEqual(out, (11, 11, 11))

    def test_LCAConv3D_transform_conv_params_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 11)
            out = lca._transform_conv_params((9, 11, 13))
            self.assertTupleEqual(out, (9, 11, 13))

    def test_LCAConv1D_drive_scaling(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10,
                3,
                tmp_dir,
                99,
                lca_iters=3,
                return_vars=["input_drives"],
                input_unit_var=False,
                input_zero_mean=False,
            )
            inputs = torch.ones(1, 3, 1000) * 7
            scaling = torch.zeros(1, 10, 1000)
            scaling[0, 8, 500] = 1.0
            input_drives = lca(inputs)
            input_drives_scaled = lca(inputs, scaling)
            precomputed_drives = lca.compute_input_drive(inputs, lca.weights)
            assert_close(precomputed_drives, input_drives, atol=0, rtol=0)
            assert_close(input_drives_scaled - scaling, input_drives)

    def test_LCAConv2D_drive_scaling(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                3,
                tmp_dir,
                11,
                lca_iters=3,
                return_vars=["input_drives"],
                input_unit_var=False,
                input_zero_mean=False,
            )
            inputs = torch.ones(1, 3, 32, 32) * 7
            scaling = torch.zeros(1, 10, 32, 32)
            scaling[0, 8, 16, 16] = 1.0
            input_drives = lca(inputs)
            input_drives_scaled = lca(inputs, scaling)
            precomputed_drives = lca.compute_input_drive(inputs, lca.weights)
            assert_close(precomputed_drives, input_drives, atol=0, rtol=0)
            assert_close(input_drives_scaled - scaling, input_drives)

    def test_LCAConv3D_drive_scaling(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                3,
                tmp_dir,
                5,
                lca_iters=3,
                return_vars=["input_drives"],
                input_unit_var=False,
                input_zero_mean=False,
            )
            inputs = torch.ones(1, 3, 5, 32, 32) * 7
            scaling = torch.zeros(1, 10, 5, 32, 32)
            scaling[0, 8, 2, 16, 16] = 1.0
            input_drives = lca(inputs)
            input_drives_scaled = lca(inputs, scaling)
            precomputed_drives = lca.compute_input_drive(inputs, lca.weights)
            assert_close(precomputed_drives, input_drives, atol=0, rtol=0)
            assert_close(input_drives_scaled - scaling, input_drives)

    def test_LCAConv1D_gradient_drive_scaling(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, req_grad=True, lca_iters=3)
            inputs = torch.randn(1, 3, 100)
            scaling = torch.randn(1, 10, 100)
            with torch.no_grad():
                acts = lca(inputs, scaling)

            with self.assertRaises(RuntimeError):
                acts.sum().backward()

            acts = lca(inputs, scaling)
            acts.sum().backward()

    def test_LCAConv2D_gradient_drive_scaling(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, req_grad=True, lca_iters=3)
            inputs = torch.randn(1, 3, 11, 11)
            scaling = torch.randn(1, 10, 11, 11)
            with torch.no_grad():
                acts = lca(inputs, scaling)

            with self.assertRaises(RuntimeError):
                acts.sum().backward()

            acts = lca(inputs, scaling)
            acts.sum().backward()

    def test_LCAConv3D_gradient_drive_scaling(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, req_grad=True, lca_iters=3)
            inputs = torch.randn(1, 3, 7, 7, 7)
            scaling = torch.randn(1, 10, 7, 7, 7)
            with torch.no_grad():
                acts = lca(inputs, scaling)

            with self.assertRaises(RuntimeError):
                acts.sum().backward()

            acts = lca(inputs, scaling)
            acts.sum().backward()

    def test_LCA_check_initial_states_raises_RuntimeError(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir)

            with self.assertRaises(RuntimeError):
                lca(torch.zeros(5, 10, 8, 8), initial_states=torch.zeros(5, 10, 7, 7))

    def test_LCA_states_zero_no_initial_states(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                3,
                tmp_dir,
                lca_iters=1,
                input_zero_mean=False,
                input_unit_var=False,
                return_vars=["states"],
            )
            states = lca(torch.zeros(5, 3, 8, 8))
            assert_close(states, torch.zeros_like(states), atol=0, rtol=0)

    def test_LCA_states_equal_initial_states(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                3,
                tmp_dir,
                lca_iters=1,
                input_zero_mean=False,
                input_unit_var=False,
                return_vars=["states"],
            )
            initial_states = torch.ones(5, 10, 8, 8) * -100
            states = lca(torch.zeros(5, 3, 8, 8), initial_states=initial_states)
            assert_close(
                states,
                initial_states + (1 / lca.tau) * -initial_states,
                atol=0,
                rtol=0,
            )

    def test_LCA_initial_states_req_grad_True(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10, 3, tmp_dir, lca_iters=1, return_vars=["states"], req_grad=True
            )
            initial_states = torch.ones(5, 10, 8, 8) * -1
            states = lca(torch.ones(5, 3, 8, 8), initial_states=initial_states)
            self.assertTrue(states.requires_grad)

    def test_LCA_initial_states_req_grad_False(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, lca_iters=1, return_vars=["states"])
            initial_states = torch.ones(5, 10, 8, 8) * -1
            states = lca(torch.ones(5, 3, 8, 8), initial_states=initial_states)
            self.assertEqual(states.requires_grad, False)

    def test_LCA_scale_input_drive(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir)
            input_drive = torch.ones(5, 3, 8, 8)
            scale = torch.zeros(5, 3, 8, 8)
            scale[0, 0, 0, 0] = 1.0
            scaled_drive = lca._scale_input_drive(input_drive, scale)
            unscaled_drive = lca._scale_input_drive(input_drive)
            assert_close(input_drive, scaled_drive, atol=1.0, rtol=0)
            assert_close(input_drive, unscaled_drive, atol=0, rtol=0)

    def test_LCA_gradient_initial_states(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10, 3, tmp_dir, lca_iters=1, return_vars=["states"], req_grad=True
            )
            states = lca(torch.zeros(1, 3, 8, 8))
            states.sum().backward()
            states2 = lca(torch.zeros(1, 3, 8, 8), initial_states=states)
            states2.sum().backward()

    def test_LCAConv1D_transform_conv_params_raises_RuntimeError_2_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv1D(10, 3, tmp_dir, (3,))
            with self.assertRaises(RuntimeError):
                LCAConv1D(10, 3, tmp_dir, (3, 3))

    def test_LCAConv1D_transform_conv_params_raises_RuntimeError_3_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv1D(10, 3, tmp_dir, (3,))
            with self.assertRaises(RuntimeError):
                LCAConv1D(10, 3, tmp_dir, (3, 3, 3))

    def test_LCAConv2D_transform_conv_params_raises_RuntimeError_1_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv2D(10, 3, tmp_dir, (3, 3))
            with self.assertRaises(RuntimeError):
                LCAConv2D(10, 3, tmp_dir, (3,))

    def test_LCAConv2D_transform_conv_params_raises_RuntimeError_3_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv2D(10, 3, tmp_dir, (3, 3))
            with self.assertRaises(RuntimeError):
                LCAConv2D(10, 3, tmp_dir, (3, 3, 3))

    def test_LCAConv3D_transform_conv_params_raises_RuntimeError_1_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv3D(10, 3, tmp_dir, (3, 3, 3))
            with self.assertRaises(RuntimeError):
                LCAConv3D(10, 3, tmp_dir, (3,))

    def test_LCAConv3D_transform_conv_params_raises_RuntimeError_2_tuple(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv3D(10, 3, tmp_dir, (3, 3, 3))
            with self.assertRaises(RuntimeError):
                LCAConv3D(10, 3, tmp_dir, (3, 3))

    def test_LCAConv1D_weights_are_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 101)
            self.assertEqual(lca.get_weights().cpu().numpy().shape, (10, 3, 101))

    def test_LCAConv2D_weights_are_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, (7, 9))
            self.assertEqual(lca.get_weights().cpu().numpy().shape, (10, 3, 7, 9))

    def test_LCAConv3D_weights_are_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, (5, 7, 9))
            self.assertEqual(lca.get_weights().cpu().numpy().shape, (10, 3, 5, 7, 9))

    def test_LCAConv1D_compute_weight_update_produces_correct_shape_odd_kernel(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10, 1, tmp_dir, 101, lca_iters=2, return_vars=["acts", "recon_errors"]
            )
            inputs = torch.ones(1, 1, 1000)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            self.assertEqual(
                update.detach().cpu().numpy().shape,
                lca.get_weights().cpu().numpy().shape,
            )

    def test_LCAConv2D_compute_weight_update_produces_correct_shape_odd_kernel(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10, 1, tmp_dir, 9, lca_iters=2, return_vars=["acts", "recon_errors"]
            )
            inputs = torch.ones(1, 1, 32, 32)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            self.assertEqual(
                update.detach().cpu().numpy().shape,
                lca.get_weights().cpu().numpy().shape,
            )

    def test_LCAConv3D_compute_weight_update_produces_correct_shape_odd_kernel(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 1, tmp_dir, 9, lca_iters=2, return_vars=["acts", "recon_errors"]
            )
            inputs = torch.ones(1, 1, 32, 32, 32)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            self.assertEqual(
                update.detach().cpu().numpy().shape,
                lca.get_weights().cpu().numpy().shape,
            )

    def test_LCAConv1D_compute_weight_update_produces_correct_shape_even_kernel(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10,
                1,
                tmp_dir,
                100,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                pad="valid",
            )
            inputs = torch.ones(1, 1, 1000)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            self.assertEqual(
                update.detach().cpu().numpy().shape,
                lca.get_weights().cpu().numpy().shape,
            )

    def test_LCAConv2D_compute_weight_update_produces_correct_shape_even_kernel(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                1,
                tmp_dir,
                8,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                pad="valid",
            )
            inputs = torch.ones(1, 1, 32, 32)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            self.assertEqual(
                update.detach().cpu().numpy().shape,
                lca.get_weights().cpu().numpy().shape,
            )

    def test_LCAConv3D_compute_weight_update_produces_correct_shape_even_kernel(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                1,
                tmp_dir,
                8,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                pad="valid",
            )
            inputs = torch.ones(1, 1, 32, 32, 32)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            self.assertEqual(
                update.detach().cpu().numpy().shape,
                lca.get_weights().cpu().numpy().shape,
            )

    def test_LCAConv1D_update_weights_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10,
                1,
                tmp_dir,
                101,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
            )
            inputs = torch.ones(1, 1, 1000)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            updated_weights = lca.get_weights() + update
            scale = updated_weights.norm(2, (1, 2), keepdim=True)
            updated_weights_normed = updated_weights / (scale + 1e-8)
            lca.update_weights(acts, recon_error)
            assert_close(updated_weights_normed, lca.get_weights(), atol=0, rtol=0)

    def test_LCAConv2D_update_weights_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                1,
                tmp_dir,
                9,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
            )
            inputs = torch.ones(1, 1, 32, 32)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            updated_weights = lca.get_weights() + update
            scale = updated_weights.norm(2, (1, 2, 3), keepdim=True)
            updated_weights_normed = updated_weights / (scale + 1e-8)
            lca.update_weights(acts, recon_error)
            assert_close(updated_weights_normed, lca.get_weights(), atol=0, rtol=0)

    def test_LCAConv3D_update_weights_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                1,
                tmp_dir,
                9,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
            )
            inputs = torch.ones(1, 1, 32, 32, 32)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            updated_weights = lca.get_weights() + update
            scale = updated_weights.norm(2, (1, 2, 3, 4), keepdim=True)
            updated_weights_normed = updated_weights / (scale + 1e-8)
            lca.update_weights(acts, recon_error)
            assert_close(updated_weights_normed, lca.get_weights(), atol=0, rtol=0)

    def test_LCAConv1D_update_weights_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10,
                1,
                tmp_dir,
                100,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                pad="valid",
            )
            inputs = torch.ones(1, 1, 1000)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            updated_weights = lca.get_weights() + update
            scale = updated_weights.norm(2, (1, 2), keepdim=True)
            updated_weights_normed = updated_weights / (scale + 1e-8)
            lca.update_weights(acts, recon_error)
            assert_close(updated_weights_normed, lca.get_weights(), atol=0, rtol=0)

    def test_LCAConv2D_update_weights_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                1,
                tmp_dir,
                8,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                pad="valid",
            )
            inputs = torch.ones(1, 1, 32, 32)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            updated_weights = lca.get_weights() + update
            scale = updated_weights.norm(2, (1, 2, 3), keepdim=True)
            updated_weights_normed = updated_weights / (scale + 1e-8)
            lca.update_weights(acts, recon_error)
            assert_close(updated_weights_normed, lca.get_weights(), atol=0, rtol=0)

    def test_LCAConv3D_update_weights_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                1,
                tmp_dir,
                8,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                pad="valid",
            )
            inputs = torch.ones(1, 1, 32, 32, 32)
            acts, recon_error = lca(inputs)
            update = lca.compute_weight_update(acts, recon_error)
            updated_weights = lca.get_weights() + update
            scale = updated_weights.norm(2, (1, 2, 3, 4), keepdim=True)
            updated_weights_normed = updated_weights / (scale + 1e-8)
            lca.update_weights(acts, recon_error)
            assert_close(updated_weights_normed, lca.get_weights(), atol=0, rtol=0)

    def test_LCAConv1D_updated_weights_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10,
                1,
                tmp_dir,
                999,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                eta=1e-4,
            )
            inputs = torch.randn(1, 1, 1000)
            acts, recon_error = lca(inputs)
            lca.update_weights(acts, recon_error)
            self.assertEqual(lca.get_weights().norm(2, (1, 2)).sum(), 10.0)

    def test_LCAConv2D_updated_weights_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                1,
                tmp_dir,
                17,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                eta=1e-4,
            )
            inputs = torch.randn(1, 1, 32, 32)
            acts, recon_error = lca(inputs)
            lca.update_weights(acts, recon_error)
            self.assertEqual(lca.get_weights().norm(2, (1, 2, 3)).sum(), 10.0)

    def test_LCAConv3D_updated_weights_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                1,
                tmp_dir,
                9,
                lca_iters=2,
                return_vars=["acts", "recon_errors"],
                eta=1e-4,
            )
            inputs = torch.randn(1, 1, 32, 32, 32)
            acts, recon_error = lca(inputs)
            lca.update_weights(acts, recon_error)
            self.assertEqual(lca.get_weights().norm(2, (1, 2, 3, 4)).sum(), 10.0)

    def test_LCAConv2D_recon_shape_different_strides_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10, 3, tmp_dir, stride=(2, 4), lca_iters=2, return_vars=["recons"]
            )
            recons = lca(torch.randn(1, 3, 32, 32))
            self.assertEqual(recons.cpu().numpy().shape, (1, 3, 32, 32))

    def test_LCAConv3D_recon_shape_different_strides_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, stride=(2, 4, 8), lca_iters=2, return_vars=["recons"]
            )
            recons = lca(torch.randn(1, 3, 32, 32, 32))
            self.assertEqual(recons.cpu().numpy().shape, (1, 3, 32, 32, 32))

    def test_LCAConv2D_code_shape_different_strides_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, stride=(2, 4), lca_iters=2)
            code = lca(torch.randn(1, 3, 32, 32))
            self.assertEqual(code.cpu().numpy().shape, (1, 10, 16, 8))

    def test_LCAConv3D_code_shape_different_strides_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, stride=(2, 4, 8), lca_iters=2)
            code = lca(torch.randn(1, 3, 32, 32, 32))
            self.assertEqual(code.cpu().numpy().shape, (1, 10, 16, 8, 4))

    def test_LCAConv2D_recon_shape_different_strides_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                3,
                tmp_dir,
                8,
                stride=(2, 4),
                lca_iters=2,
                return_vars=["recons"],
                pad="valid",
            )
            recons = lca(torch.randn(1, 3, 32, 32))
            self.assertEqual(recons.cpu().numpy().shape, (1, 3, 32, 32))

    def test_LCAConv3D_recon_shape_different_strides_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                3,
                tmp_dir,
                8,
                stride=(2, 4, 8),
                lca_iters=2,
                return_vars=["recons"],
                pad="valid",
            )
            recons = lca(torch.randn(1, 3, 32, 32, 32))
            self.assertEqual(recons.cpu().numpy().shape, (1, 3, 32, 32, 32))

    def test_LCAConv2D_code_shape_different_strides_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 8, stride=(2, 4), lca_iters=2, pad="valid")
            code = lca(torch.randn(1, 3, 32, 32))
            self.assertEqual(code.cpu().numpy().shape, (1, 10, 13, 7))

    def test_LCAConv3D_code_shape_different_strides_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, 8, stride=(2, 4, 8), lca_iters=2, pad="valid"
            )
            code = lca(torch.randn(1, 3, 32, 32, 32))
            self.assertEqual(code.cpu().numpy().shape, (1, 10, 13, 7, 4))

    def test_LCAConv1D_transform_conv_params_raises_RuntimeError_given_list(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv1D(10, 3, tmp_dir, (3,))
            with self.assertRaises(RuntimeError):
                LCAConv1D(10, 3, tmp_dir, [3])

    def test_LCAConv2D_transform_conv_params_raises_RuntimeError_given_list(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv2D(10, 3, tmp_dir, (3, 3))
            with self.assertRaises(RuntimeError):
                LCAConv2D(10, 3, tmp_dir, [3, 3])

    def test_LCAConv3D_transform_conv_params_raises_RuntimeError_given_list(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv3D(10, 3, tmp_dir, (3, 3, 3))
            with self.assertRaises(RuntimeError):
                LCAConv3D(10, 3, tmp_dir, [3, 3, 3])

    def test_LCA_transfer_raises_RuntimeError_given_list(self):
        with TemporaryDirectory() as tmp_dir:
            inputs = torch.zeros(1, 3, 32, 32)
            lca = LCAConv2D(10, 3, tmp_dir)
            lca.transfer(inputs)
            with self.assertRaises(RuntimeError):
                lca = LCAConv2D(10, 3, tmp_dir, transfer_func=["soft_threshold"])
                lca.transfer(inputs)

    def test_LCA_transfer_raises_RuntimeError_given_tensor(self):
        with TemporaryDirectory() as tmp_dir:
            inputs = torch.zeros(1, 3, 32, 32)
            lca = LCAConv2D(10, 3, tmp_dir)
            lca.transfer(inputs)
            with self.assertRaises(RuntimeError):
                lca = LCAConv2D(10, 3, tmp_dir, transfer_func=torch.zeros(1, 3, 32, 32))
                lca.transfer(inputs)

    def test_LCA_transfer_raises_ValueError(self):
        with TemporaryDirectory() as tmp_dir:
            inputs = torch.zeros(1, 3, 32, 32)
            lca = LCAConv2D(10, 3, tmp_dir)
            lca.transfer(inputs)
            with self.assertRaises(ValueError):
                lca = LCAConv2D(10, 3, tmp_dir, transfer_func="soft_thershold")
                lca.transfer(inputs)

    def test_LCA_transfer_pytorch_activation_func(self):
        with TemporaryDirectory() as tmp_dir:
            inputs = torch.ones(1, 3, 32, 32) * 10.0
            lca = LCAConv2D(10, 3, tmp_dir, transfer_func=torch.tanh)
            output = lca.transfer(inputs)
            assert_close(output.sum().item(), 32.0 * 32.0 * 3.0)

    def test_LCAConv1D_compute_recon_error(self):
        with TemporaryDirectory() as tmp_dir:
            inputs = torch.randn(1, 3, 100)
            recons = torch.randn(1, 3, 100)
            lca = LCAConv1D(10, 3, tmp_dir)
            assert_close(inputs - recons, lca.compute_recon_error(inputs, recons))

    def test_LCAConv2D_compute_recon_error(self):
        with TemporaryDirectory() as tmp_dir:
            inputs = torch.randn(1, 3, 32, 32)
            recons = torch.randn(1, 3, 32, 32)
            lca = LCAConv2D(10, 3, tmp_dir)
            assert_close(inputs - recons, lca.compute_recon_error(inputs, recons))

    def test_LCAConv3D_compute_recon_error(self):
        with TemporaryDirectory() as tmp_dir:
            inputs = torch.randn(1, 3, 32, 32, 32)
            recons = torch.randn(1, 3, 32, 32, 32)
            lca = LCAConv3D(10, 3, tmp_dir)
            assert_close(inputs - recons, lca.compute_recon_error(inputs, recons))

    def test_LCAConv1D_assign_weight_values_normalize_True(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 1001)
            new_weights = torch.randn(*lca.weights.shape) * 100
            lca.assign_weight_values(new_weights)
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0, atol=1e-6, rtol=0)

    def test_LCAConv2D_assign_weight_values_normalize_True(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 11)
            new_weights = torch.randn(*lca.weights.shape) * 100
            lca.assign_weight_values(new_weights)
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0, atol=1e-6, rtol=0)

    def test_LCAConv3D_assign_weight_values_normalize_True(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 11)
            new_weights = torch.randn(*lca.weights.shape) * 100
            lca.assign_weight_values(new_weights)
            for feat in lca.get_weights():
                assert_close(feat.norm(2).item(), 1.0, atol=1e-6, rtol=0)

    def test_LCA_code_vs_sklearn_code_nonneg_acts(self):
        inputs = torch.randn(1, 1, 100)
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                500,
                1,
                tmp_dir,
                inputs.shape[-1],
                inputs.shape[-1],
                0.5,
                100,
                lca_iters=30000,
                pad="valid",
                return_vars=["inputs", "acts"],
                nonneg=True,
            )
            sklearn_solver = SparseCoder(
                lca.get_weights().squeeze().cpu().numpy(),
                transform_algorithm="lasso_lars",
                transform_alpha=lca.lambda_,
                n_jobs=4,
                positive_code=lca.nonneg,
                transform_max_iter=lca.lca_iters,
            )
            inputs, lca_code = lca(inputs)
            sklearn_code = sklearn_solver.fit_transform(inputs[0].cpu().numpy())
            assert_close(
                lca_code.double().squeeze().cpu().numpy(),
                sklearn_code.squeeze(),
                atol=1e-3,
                rtol=0,
            )

    def test_LCA_code_vs_sklearn_code_neg_acts(self):
        inputs = torch.randn(1, 1, 100)
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                500,
                1,
                tmp_dir,
                inputs.shape[-1],
                inputs.shape[-1],
                0.5,
                100,
                lca_iters=30000,
                pad="valid",
                return_vars=["inputs", "acts"],
                nonneg=False,
            )
            sklearn_solver = SparseCoder(
                lca.get_weights().squeeze().cpu().numpy(),
                transform_algorithm="lasso_lars",
                transform_alpha=lca.lambda_,
                n_jobs=4,
                positive_code=lca.nonneg,
                transform_max_iter=lca.lca_iters,
            )
            inputs, lca_code = lca(inputs)
            sklearn_code = sklearn_solver.fit_transform(inputs[0].cpu().numpy())
            assert_close(
                lca_code.double().squeeze().cpu().numpy(),
                sklearn_code.squeeze(),
                atol=1e-3,
                rtol=0,
            )

    def test_LCA_code_vs_sklearn_code_conv(self):
        inputs = torch.randn(1, 1, 500)
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                500,
                1,
                tmp_dir,
                100,
                100,
                0.5,
                100,
                lca_iters=30000,
                pad="valid",
                return_vars=["inputs", "acts"],
                nonneg=True,
                input_unit_var=False,
                input_zero_mean=False,
            )
            sklearn_solver = SparseCoder(
                lca.get_weights().squeeze().cpu().numpy(),
                transform_algorithm="lasso_lars",
                transform_alpha=lca.lambda_,
                n_jobs=4,
                positive_code=lca.nonneg,
                transform_max_iter=lca.lca_iters,
            )
            inputs, lca_code = lca(inputs)
            sklearn_inputs = torch.vstack(
                [
                    inputs.squeeze()[idx : idx + lca.kernel_size]
                    for idx in range(0, inputs.shape[-1], lca.kernel_size)
                ]
            )
            sklearn_code = sklearn_solver.fit_transform(sklearn_inputs.cpu().numpy())
            assert_close(
                lca_code.double().squeeze().cpu().numpy().T,
                sklearn_code.squeeze(),
                atol=1e-3,
                rtol=0,
            )

    def test_LCA_get_weights_keeps_gradient_intact(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 1, tmp_dir, lca_iters=1, req_grad=True)
            lca(torch.zeros(1, 1, 32, 32, requires_grad=True))
            lca.get_weights().sum().backward()

    def test_LCAConv1D_get_weights_clone(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 1, tmp_dir)
            weights = lca.get_weights()
            weights *= 0
            self.assertNotAlmostEqual((lca.get_weights() - weights).sum().item(), 0.0)

    def test_LCAConv2D_get_weights_clone(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 1, tmp_dir)
            weights = lca.get_weights()
            weights *= 0
            self.assertNotAlmostEqual((lca.get_weights() - weights).sum().item(), 0.0)

    def test_LCAConv3D_get_weights_clone(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 1, tmp_dir)
            weights = lca.get_weights()
            weights *= 0
            self.assertNotAlmostEqual((lca.get_weights() - weights).sum().item(), 0.0)

    def test_LCAConv2D_update_weights_normalize_False(self):
        inputs = torch.randn(5, 1, 32, 32)
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                1,
                tmp_dir,
                11,
                4,
                tau=100,
                lca_iters=50,
                return_vars=["acts", "recon_errors"],
            )
            acts, recon_errors = lca(inputs)
            lca.update_weights(acts, recon_errors, False)
            for feat in lca.get_weights():
                self.assertNotAlmostEqual(feat.norm(2).item(), 1.0)

    def test_LCAConv1D_update_weights_normalize_False(self):
        inputs = torch.randn(5, 1, 1000)
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10,
                1,
                tmp_dir,
                101,
                50,
                tau=100,
                lca_iters=50,
                return_vars=["acts", "recon_errors"],
            )
            acts, recon_errors = lca(inputs)
            lca.update_weights(acts, recon_errors, False)
            for feat in lca.get_weights():
                self.assertNotAlmostEqual(feat.norm(2).item(), 1.0)

    def test_LCAConv3D_update_weights_normalize_False(self):
        inputs = torch.randn(5, 1, 32, 32, 32)
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                1,
                tmp_dir,
                11,
                4,
                tau=100,
                lca_iters=50,
                return_vars=["acts", "recon_errors"],
            )
            acts, recon_errors = lca(inputs)
            lca.update_weights(acts, recon_errors, False)
            for feat in lca.get_weights():
                self.assertNotAlmostEqual(feat.norm(2).item(), 1.0)

    def test_init_states_no_initial_states_given(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 1, tmp_dir)
            input_drive = lca.compute_input_drive(
                torch.zeros(5, 1, 32, 32), lca.get_weights()
            )
            states = lca._init_states(input_drive)
            assert_close(states, torch.zeros_like(input_drive), atol=0, rtol=0)

    def test_init_states_initial_states_given(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 1, tmp_dir)
            input_drive = lca.compute_input_drive(
                torch.zeros(5, 1, 32, 32), lca.get_weights()
            )
            initial_states = torch.ones_like(input_drive)
            states = lca._init_states(input_drive, initial_states)
            assert_close(states, initial_states, atol=0, rtol=0)

    def test_LCAConv1D_self_conns_are_zero_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 51, 2):
                for stride in range(1, 51):
                    lca = LCAConv1D(5, 3, tmp_dir, ksize, stride)
                    conns = lca.compute_lateral_connectivity(lca.get_weights() + 1.0)
                    self.assertEqual((conns == 0).sum().item(), lca.out_neurons)
                    for feat_idx in range(lca.out_neurons):
                        assert_close(
                            conns[feat_idx, feat_idx, conns.shape[2] // 2, 0, 0].item(),
                            0.0,
                            atol=0,
                            rtol=0,
                        )

    def test_LCAConv2D_self_conns_are_zero_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 51, 2):
                for stride in range(1, 51):
                    lca = LCAConv2D(5, 3, tmp_dir, (ksize, ksize + 2), stride)
                    conns = lca.compute_lateral_connectivity(lca.get_weights() + 1.0)
                    self.assertEqual((conns == 0).sum().item(), lca.out_neurons)
                    for feat_idx in range(lca.out_neurons):
                        assert_close(
                            conns[
                                feat_idx,
                                feat_idx,
                                0,
                                conns.shape[3] // 2,
                                conns.shape[4] // 2,
                            ].item(),
                            0.0,
                            atol=0,
                            rtol=0,
                        )

    def test_LCAConv3D_self_conns_are_zero_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 23, 2):
                for stride in range(1, 22):
                    lca = LCAConv3D(
                        5, 3, tmp_dir, (ksize, ksize + 2, ksize + 4), stride
                    )
                    conns = lca.compute_lateral_connectivity(lca.get_weights() + 1.0)
                    self.assertEqual((conns == 0).sum().item(), lca.out_neurons)
                    for feat_idx in range(lca.out_neurons):
                        assert_close(
                            conns[
                                feat_idx,
                                feat_idx,
                                conns.shape[2] // 2,
                                conns.shape[3] // 2,
                                conns.shape[4] // 2,
                            ].item(),
                            0.0,
                            atol=0,
                            rtol=0,
                        )

    def test_LCAConv1D_self_conns_are_zero_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                for stride in range(1, 51):
                    lca = LCAConv1D(5, 3, tmp_dir, ksize, stride, pad="valid")
                    conns = lca.compute_lateral_connectivity(lca.get_weights() + 1.0)
                    self.assertEqual((conns == 0).sum().item(), lca.out_neurons)
                    for feat_idx in range(lca.out_neurons):
                        assert_close(
                            conns[feat_idx, feat_idx, conns.shape[2] // 2, 0, 0].item(),
                            0.0,
                            atol=0,
                            rtol=0,
                        )

    def test_LCAConv2D_self_conns_are_zero_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                for stride in range(1, 51):
                    lca = LCAConv2D(
                        5, 3, tmp_dir, (ksize, ksize + 2), stride, pad="valid"
                    )
                    conns = lca.compute_lateral_connectivity(lca.get_weights() + 1.0)
                    self.assertEqual((conns == 0).sum().item(), lca.out_neurons)
                    for feat_idx in range(lca.out_neurons):
                        assert_close(
                            conns[
                                feat_idx,
                                feat_idx,
                                0,
                                conns.shape[3] // 2,
                                conns.shape[4] // 2,
                            ].item(),
                            0.0,
                            atol=0,
                            rtol=0,
                        )

    def test_LCAConv3D_self_conns_are_zero_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 22, 2):
                for stride in range(1, 21):
                    lca = LCAConv3D(
                        5,
                        3,
                        tmp_dir,
                        (ksize, ksize + 2, ksize + 4),
                        stride,
                        pad="valid",
                    )
                    conns = lca.compute_lateral_connectivity(lca.get_weights() + 1.0)
                    self.assertEqual((conns == 0).sum().item(), lca.out_neurons)
                    for feat_idx in range(lca.out_neurons):
                        assert_close(
                            conns[
                                feat_idx,
                                feat_idx,
                                conns.shape[2] // 2,
                                conns.shape[3] // 2,
                                conns.shape[4] // 2,
                            ].item(),
                            0.0,
                            atol=0,
                            rtol=0,
                        )

    def test_LCAConv1D_weight_initialization(self):
        with TemporaryDirectory() as tmp_dir:
            # zeros
            lca = LCAConv1D(1000, 3, tmp_dir, weight_init=(torch.nn.init.zeros_, {}))
            expected_weights = torch.zeros_like(lca.get_weights())
            self.assertEqual((lca.get_weights() - expected_weights).sum().item(), 0.0)
            # ones
            lca = LCAConv1D(1000, 3, tmp_dir, weight_init=(torch.nn.init.ones_, {}))
            expected_weights = torch.ones_like(lca.get_weights())
            expected_weights = expected_weights / (
                expected_weights.norm(2, (1, 2), True) + 1e-12
            )
            self.assertEqual((lca.get_weights() - expected_weights).sum().item(), 0.0)

    def test_LCAConv2D_weight_initialization(self):
        with TemporaryDirectory() as tmp_dir:
            # zeros
            lca = LCAConv2D(1000, 3, tmp_dir, weight_init=(torch.nn.init.zeros_, {}))
            expected_weights = torch.zeros_like(lca.get_weights())
            self.assertEqual((lca.get_weights() - expected_weights).sum().item(), 0.0)
            # ones
            lca = LCAConv2D(1000, 3, tmp_dir, weight_init=(torch.nn.init.ones_, {}))
            expected_weights = torch.ones_like(lca.get_weights())
            expected_weights = expected_weights / (
                expected_weights.norm(2, (1, 2, 3), True) + 1e-12
            )
            self.assertEqual((lca.get_weights() - expected_weights).sum().item(), 0.0)

    def test_LCAConv3D_weight_initialization(self):
        with TemporaryDirectory() as tmp_dir:
            # zeros
            lca = LCAConv3D(1000, 3, tmp_dir, weight_init=(torch.nn.init.zeros_, {}))
            expected_weights = torch.zeros_like(lca.get_weights())
            self.assertEqual((lca.get_weights() - expected_weights).sum().item(), 0.0)
            # ones
            lca = LCAConv3D(1000, 3, tmp_dir, weight_init=(torch.nn.init.ones_, {}))
            expected_weights = torch.ones_like(lca.get_weights())
            expected_weights = expected_weights / (
                expected_weights.norm(2, (1, 2, 3, 4), True) + 1e-12
            )
            self.assertEqual((lca.get_weights() - expected_weights).sum().item(), 0.0)

    def test_LCAConv1D_weight_init_kwargs(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv1D(
                10, 3, tmp_dir, weight_init=(torch.nn.init.constant_, {"val": 25})
            )
            with self.assertRaises(TypeError):
                LCAConv1D(10, 3, tmp_dir, weight_init=(torch.nn.init.constant_, {}))

    def test_LCAConv2D_weight_init_kwargs(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv2D(
                10, 3, tmp_dir, weight_init=(torch.nn.init.constant_, {"val": 25})
            )
            with self.assertRaises(TypeError):
                LCAConv2D(10, 3, tmp_dir, weight_init=(torch.nn.init.constant_, {}))

    def test_LCAConv3D_weight_init_kwargs(self):
        with TemporaryDirectory() as tmp_dir:
            LCAConv3D(
                10, 3, tmp_dir, weight_init=(torch.nn.init.constant_, {"val": 25})
            )
            with self.assertRaises(TypeError):
                LCAConv3D(10, 3, tmp_dir, weight_init=(torch.nn.init.constant_, {}))


if __name__ == "__main__":
    unittest.main()
