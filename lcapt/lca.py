from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Callable, Iterable, Literal, Optional, Union
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .activation import hard_threshold, soft_threshold
from .metric import (
    compute_frac_active,
    compute_l1_sparsity,
    compute_l2_error,
    compute_times_active_by_feature,
)
from .preproc import make_unit_var, make_zero_mean
from .util import (
    check_equal_shapes,
    to_3d_from_5d,
    to_4d_from_5d,
    to_5d_from_3d,
    to_5d_from_4d,
)


Parameter = torch.nn.parameter.Parameter
Tensor = torch.Tensor


class _LCAConvBase(torch.nn.Module):
    def __init__(
        self,
        out_neurons: int,
        in_neurons: int,
        result_dir: str = "./lca_results",
        kernel_size: Union[int, tuple[int], tuple[int, int], tuple[int, int, int]] = 7,
        stride: Union[int, tuple[int], tuple[int, int], tuple[int, int, int]] = 1,
        lambda_: float = 0.25,
        tau: Union[float, int] = 100,
        eta: float = 0.01,
        lca_iters: int = 1000,
        pad: Literal["same", "valid"] = "same",
        return_vars: Iterable[
            Literal[
                "inputs",
                "input_drives",
                "states",
                "acts",
                "recons",
                "recon_errors",
                "conns",
            ]
        ] = ["acts"],
        return_all_ts: bool = False,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            Literal["soft_threshold", "hard_threshold"], Callable[[Tensor], Tensor]
        ] = "soft_threshold",
        input_zero_mean: bool = True,
        input_unit_var: bool = True,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        req_grad: bool = False,
        weight_init: tuple[Callable[[Tensor], Tensor], dict[str, Any]] = (
            torch.nn.init.trunc_normal_,
            {},
        ),
        no_time_pad: bool = False,
    ) -> None:
        self.d_update_clip = d_update_clip
        self.eta = eta
        self.in_neurons = in_neurons
        self.input_unit_var = input_unit_var
        self.input_zero_mean = input_zero_mean
        self.kernel_size = kernel_size
        self.lambda_ = lambda_
        self.lca_iters = lca_iters
        if lr_schedule is not None:
            assert callable(lr_schedule)
        self.lr_schedule = lr_schedule
        self.no_time_pad = no_time_pad
        self.nonneg = nonneg
        self.out_neurons = out_neurons
        self.pad = pad
        self.req_grad = req_grad
        self.result_dir = os.path.abspath(result_dir)
        self.return_all_ts = return_all_ts
        self.return_vars = return_vars
        self.stride = stride
        self.tau = tau
        self.track_metrics = track_metrics
        self.transfer_func = transfer_func
        self.weight_init = weight_init[0]
        self.weight_init_kwargs = weight_init[1]

        os.makedirs(self.result_dir, exist_ok=True)
        self.metric_fpath = os.path.join(self.result_dir, "metrics.xz")
        self._write_params(deepcopy(vars(self)))
        self.kt, self.kh, self.kw = self._transform_conv_params(kernel_size)
        self.stridet, self.strideh, self.stridew = self._transform_conv_params(stride)
        self._check_conv_params()
        self._compute_padding()
        super(_LCAConvBase, self).__init__()
        self._init_weight_tensor()
        self.register_buffer("forward_pass", torch.tensor(1))

        if cudnn_benchmark and torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True

    def assign_weight_values(self, tensor: Tensor, normalize: bool = True) -> None:
        """Manually assign weight tensor"""
        with torch.no_grad():
            check_equal_shapes(self.weights, tensor, "weights")
            self.weights.copy_(tensor)
            if normalize:
                self.normalize_weights()

    def _check_conv_params(self) -> None:
        even_k = [ksize % 2 == 0 for ksize in [self.kt, self.kh, self.kw] if ksize != 1]
        assert all(even_k) or not any(even_k)
        self.kernel_odd = not any(even_k)

    def _compute_identity(self, conns: Tensor) -> Tensor:
        identity = torch.ones_like(conns, requires_grad=False)
        identity[
            range(conns.shape[0]),
            range(conns.shape[1]),
            conns.shape[2] // 2,
            conns.shape[3] // 2,
            conns.shape[4] // 2,
        ] = 0.0
        return identity.requires_grad_(self.req_grad)

    def _compute_inhib_pad(self) -> None:
        """Computes padding for compute_lateral_connectivity"""
        pad = []
        for ksize, stride in zip(
            [self.kt, self.kh, self.kw], [self.stridet, self.strideh, self.stridew]
        ):
            if ksize % 2 != 0:
                pad.append((ksize - 1) // stride * stride)
            else:
                if ksize % stride == 0:
                    pad.append(ksize - stride)
                else:
                    pad.append(ksize // stride * stride)

        self.lat_conn_pad = tuple(pad)

    def _compute_input_pad(self) -> None:
        """Computes padding for forward convolution"""
        if self.pad == "same":
            assert self.kernel_odd
            self.input_pad = (
                0 if self.no_time_pad else self.kt // 2,
                self.kh // 2,
                self.kw // 2,
            )
        elif self.pad == "valid":
            self.input_pad = (0, 0, 0)
        else:
            raise ValueError(
                "Acceptable values for pad are 'same' and 'valid', but got ",
                f"{self.pad}.",
            )

    def _compute_padding(self) -> None:
        self._compute_input_pad()
        self._compute_inhib_pad()
        self._compute_recon_pad()

    def _compute_recon_pad(self) -> None:
        """Computes output padding for recon conv transpose"""
        if self.kernel_odd:
            self.recon_output_pad = (
                self.stridet - 1,
                self.strideh - 1,
                self.stridew - 1,
            )
        else:
            self.recon_output_pad = (0, 0, 0)

    def compute_input_drive(
        self, inputs: Tensor, weights: Union[Tensor, Parameter]
    ) -> Tensor:
        inputs, reshape_func = self._to_correct_shape(inputs)
        drive = F.conv3d(
            inputs,
            self._to_correct_shape(weights)[0],
            stride=(self.stridet, self.strideh, self.stridew),
            padding=self.input_pad,
        )
        return reshape_func(drive)

    def compute_lateral_connectivity(self, weights: Union[Tensor, Parameter]) -> Tensor:
        conns = F.conv3d(
            self._to_correct_shape(weights)[0],
            self._to_correct_shape(weights)[0],
            stride=(self.stridet, self.strideh, self.stridew),
            padding=self.lat_conn_pad,
        )
        if not hasattr(self, "surround"):
            self._compute_n_surround(conns)
        return conns * self._compute_identity(conns)

    def _compute_n_surround(self, conns: Tensor) -> tuple:
        """Computes the number of surround neurons for each dim"""
        conn_shp = conns.shape[2:]
        self.surround = tuple([int(np.ceil((dim - 1) / 2)) for dim in conn_shp])

    def compute_recon(self, acts: Tensor, weights: Union[Tensor, Parameter]) -> Tensor:
        """Computes reconstruction given code"""
        acts, reshape_func = self._to_correct_shape(acts)
        recons = F.conv_transpose3d(
            acts,
            self._to_correct_shape(weights)[0],
            stride=(self.stridet, self.strideh, self.stridew),
            padding=self.input_pad,
            output_padding=self.recon_output_pad,
        )
        return reshape_func(recons)

    def compute_recon_error(self, inputs: Tensor, recons: Tensor) -> Tensor:
        return inputs - recons

    def compute_weight_update(self, acts: Tensor, error: Tensor) -> Tensor:
        acts, reshape_func = self._to_correct_shape(acts)
        error, _ = self._to_correct_shape(error)
        error = F.pad(
            error,
            (
                self.input_pad[2],
                self.input_pad[2],
                self.input_pad[1],
                self.input_pad[1],
                self.input_pad[0],
                self.input_pad[0],
            ),
        )
        error = error.unfold(-3, self.kt, self.stridet)
        error = error.unfold(-3, self.kh, self.strideh)
        error = error.unfold(-3, self.kw, self.stridew)
        update = torch.tensordot(acts, error, dims=([0, 2, 3, 4], [0, 2, 3, 4]))
        return reshape_func(update)

    def _create_trackers(self) -> dict[str, np.ndarray]:
        """Create placeholders to store different metrics"""
        float_tracker = np.zeros([self.lca_iters], dtype=np.float32)
        return {
            "L1": float_tracker.copy(),
            "L2": float_tracker.copy(),
            "TotalEnergy": float_tracker.copy(),
            "FractionActive": float_tracker.copy(),
            "Tau": float_tracker.copy(),
        }

    def encode(
        self,
        inputs: Tensor,
        drive_scaling: Optional[Tensor] = None,
        initial_states: Optional[Tensor] = None,
    ) -> tuple[list[Tensor], ...]:
        """Computes sparse code given data x and dictionary D"""
        input_drive = self.compute_input_drive(inputs, self.weights)
        states = self._init_states(input_drive, initial_states)
        connectivity = self.compute_lateral_connectivity(self.weights)
        return_vars = tuple([[] for _ in range(len(self.return_vars))])
        input_drive = self._scale_input_drive(input_drive, drive_scaling)

        for lca_iter in range(1, self.lca_iters + 1):
            acts = self.transfer(states)
            inhib = self.lateral_competition(acts, connectivity)
            states = states + (1 / self.tau) * (input_drive - states - inhib)

            if self.track_metrics or lca_iter == self.lca_iters or self.return_all_ts:
                recon = self.compute_recon(acts, self.weights)
                recon_error = self.compute_recon_error(inputs, recon)

                if self.return_all_ts or lca_iter == self.lca_iters:
                    for var_idx, var_name in enumerate(self.return_vars):
                        if var_name == "inputs":
                            return_vars[var_idx].append(inputs)
                        elif var_name == "input_drives":
                            return_vars[var_idx].append(input_drive)
                        elif var_name == "states":
                            return_vars[var_idx].append(states)
                        elif var_name == "acts":
                            return_vars[var_idx].append(acts)
                        elif var_name == "recons":
                            return_vars[var_idx].append(recon)
                        elif var_name == "recon_errors":
                            return_vars[var_idx].append(recon_error)
                        elif var_name == "conns":
                            return_vars[var_idx].append(connectivity)
                        else:
                            raise ValueError(
                                f"Invalid value '{var_name}' in return_vars."
                            )

                if self.track_metrics:
                    if lca_iter == 1:
                        tracks = self._create_trackers()
                    tracks = self._update_tracks(
                        tracks, lca_iter, acts, inputs, recon, self.tau
                    )

        if self.track_metrics:
            self._write_tracks(tracks, lca_iter, inputs.device.index)

        return return_vars

    def forward(
        self,
        inputs: Tensor,
        drive_scaling: Optional[Tensor] = None,
        initial_states: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, ...]]:
        if self.input_zero_mean:
            inputs = make_zero_mean(inputs)
        if self.input_unit_var:
            inputs = make_unit_var(inputs)

        inputs, reshape_func = self._to_correct_shape(inputs)
        if drive_scaling is not None:
            drive_scaling, _ = self._to_correct_shape(drive_scaling)
        if initial_states is not None:
            initial_states, _ = self._to_correct_shape(initial_states)

        outputs = self.encode(inputs, drive_scaling, initial_states)
        self.forward_pass += 1

        if self.return_all_ts:
            outputs = tuple(
                [
                    torch.stack([reshape_func(tensor) for tensor in out], -1)
                    for out in outputs
                ]
            )

            if len(self.return_vars) == 1:
                return outputs[0]
            return outputs

        else:
            if len(self.return_vars) == 1:
                return reshape_func(outputs[0][-1])
            return tuple([reshape_func(out[-1]) for out in outputs])

    def _init_states(
        self, input_drive: Tensor, initial_states: Optional[Tensor] = None
    ) -> Tensor:
        if initial_states is None:
            return torch.zeros_like(input_drive, requires_grad=self.req_grad)
        else:
            check_equal_shapes(input_drive, initial_states, "initial_states")
            return initial_states.detach().clone().requires_grad_(self.req_grad)

    def _init_weight_tensor(self) -> None:
        pass

    def lateral_competition(self, acts: Tensor, conns: Tensor) -> Tensor:
        return F.conv3d(acts, conns, stride=1, padding=self.surround)

    def normalize_weights(self, eps: float = 1e-12) -> None:
        """Normalizes features so each one has unit norm"""
        with torch.no_grad():
            dims = tuple(range(1, len(self.weights.shape)))
            scale = self.weights.norm(p=2, dim=dims, keepdim=True)
            self.weights.copy_(self.weights / (scale + eps))

    def _scale_input_drive(
        self, input_drive: Tensor, drive_scaling: Optional[Tensor] = None
    ) -> Tensor:
        """Scale input_drive elementwise with drive_scaling"""
        if drive_scaling is None:
            return input_drive
        else:
            check_equal_shapes(input_drive, drive_scaling, "drive_scaling")
            return input_drive + drive_scaling

    def _to_correct_shape(
        self, inputs: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        pass

    def transfer(self, x: Tensor) -> Tensor:
        if type(self.transfer_func) == str:
            if self.transfer_func == "soft_threshold":
                return soft_threshold(x, self.lambda_, self.nonneg)
            elif self.transfer_func == "hard_threshold":
                return hard_threshold(x, self.lambda_, self.nonneg)
            else:
                raise ValueError(
                    f"If transfer_func is a str, it should be 'soft_threshold' or 'hard_threshold', but got '{self.transfer_func}'."
                )
        elif callable(self.transfer_func):
            return self.transfer_func(x)
        else:
            raise RuntimeError(
                f"transfer_func must be a str or function, but got {type(self.transfer_func)}."
            )

    def _transform_conv_params(
        self, val: Union[int, tuple[int], tuple[int, int], tuple[int, int, int]]
    ) -> tuple[int, int, int]:
        pass

    def update_weights(
        self, acts: Tensor, recon_error: Tensor, normalize: bool = True
    ) -> Tensor:
        """Updates the dictionary given the computed gradient"""
        with torch.no_grad():
            update = self.compute_weight_update(acts, recon_error)
            times_active = compute_times_active_by_feature(acts) + 1
            update *= self.eta / times_active
            update = torch.clamp(
                update, min=-self.d_update_clip, max=self.d_update_clip
            )
            self.weights.copy_(self.weights + update)
            if normalize:
                self.normalize_weights()
            if self.lr_schedule is not None:
                self.eta = self.lr_schedule(self.forward_pass)
            return update

    def _update_tracks(
        self,
        tracks: dict[str, np.ndarray],
        lca_iter: int,
        acts: Tensor,
        inputs: Tensor,
        recons: Tensor,
        tau: Union[int, float],
    ) -> dict[str, np.ndarray]:
        """Update dictionary that stores the tracked metrics"""
        l2_rec_err = compute_l2_error(inputs, recons).item()
        l1_sparsity = compute_l1_sparsity(acts, self.lambda_).item()
        tracks["L2"][lca_iter - 1] = l2_rec_err
        tracks["L1"][lca_iter - 1] = l1_sparsity
        tracks["TotalEnergy"][lca_iter - 1] = l2_rec_err + l1_sparsity
        tracks["FractionActive"][lca_iter - 1] = compute_frac_active(acts)
        tracks["Tau"][lca_iter - 1] = tau
        return tracks

    def _write_params(self, arg_dict: dict[str, Any]) -> None:
        """Writes model params to file"""
        del arg_dict["lr_schedule"]
        for key, val in arg_dict.items():
            if type(val) == tuple:
                arg_dict[key] = list(val)
            elif callable(val):
                arg_dict[key] = val.__name__
        with open(os.path.join(self.result_dir, "params.yaml"), "w") as yamlf:
            yaml.dump(arg_dict, yamlf, sort_keys=True)

    def _write_tracks(
        self, tracker: dict[str, np.ndarray], ts_cutoff: int, dev: Union[int, None]
    ) -> None:
        """Write out objective values to file"""
        for k, v in tracker.items():
            tracker[k] = v[:ts_cutoff]

        obj_df = pd.DataFrame(tracker)
        obj_df["LCAIter"] = np.arange(1, len(obj_df) + 1, dtype=np.int32)
        obj_df["ForwardPass"] = self.forward_pass.item()
        obj_df["Device"] = dev
        obj_df.to_csv(
            self.metric_fpath,
            header=True if not os.path.isfile(self.metric_fpath) else False,
            index=False,
            mode="a",
        )


class LCAConv1D(_LCAConvBase):
    """
    Performs LCA with a 1D Convolution on 3D inputs.

    Args:
        out_neurons (int): Number of LCA neurons/features
        in_neurons (int): Number of input neurons/channels
        result_dir (str, optional): Path where model params and results will
            be saved
        kernel_size (int | tuple[int], optional): Length of the LCA receptive
            fields. Default: 7
        stride (int | tuple[int], optional): Stride of the LCA receptive fields.
            Default: 1
        lambda_ (float, optional): LCA firing threshold. Default: 0.25
        tau (float | int, optional): LCA time constant. Default: 1000
        eta (float, optional): Learning rate for built-in weight updates.
            Default: 0.01
        lca_iters (int, optional): LCA iterations per forward pass.
            Default: 3000
        pad ('same' | 'valid', optional): Input padding for the conv.
            Default: 'same'
        return_vars (list | tuple, optional): Iterable of one or more strings
            determining which variables will be returned during the forward
            pass. Possible values are 'inputs', 'input_drives', 'states',
            'acts', 'recons', 'recon_errors', and 'conns'. Default: ['acts'].
        return_all_ts (bool, optional): Whether to return the value of
            return_vars at every LCA iteration (True) or just the last LCA
            iteration (False). Default: False
        nonneg (bool, optional): Enforces nonnegative activations.
            Default: True
        track_metrics (bool, optional): Whether to track the L1 sparsity
            penalty, L2 reconstruction error, total energy (L1 + L2), and
            fraction of neurons active over the LCA loop at each forward pass
            and write it to a file in result_dir. Default: False
        transfer_func ('soft_threshold' | 'hard_threshold' | callable,
            optional): The function that transforms the LCA membrane
            potentials into activations. If using custom functions, nonneg
            will not be used. Default: 'soft_threshold'
        input_zero_mean (bool, optional): Whether to make each input in the
            batch have zero mean before performing LCA. Default: True
        input_unit_var (bool, optional): Whether to make each input in the
            batch have unit variance before performing LCA. Default: True
        cudnn_benchmark (bool, optional): Use CUDNN benchmark. Default: True
        d_update_clip (float, optional): Clips weight updates from above and
            below when using built-in update method. Default: np.inf
        lr_schedule (callable, optional): Schedule for eta when using the
            built-in weight update method. Should take in the number of
            forward passes at the current update and return a value for eta.
            Default: None
        req_grad (bool, optional): Keep track of the gradients over the LCA
            loop. This is useful when not using the built-in dictionary
            learning method, or for things like adversarial attacks.
            Default: False
        weight_init (tuple(callable, dict), optional): Initialization to use
            for the dictionary weights and a dictionary of keyword arguments
            for that initialization. Default: truncated normal, default args

    Shape:
        Input: (N, in_neurons, L)
        Input Drives, States, Acts: (N, out_neurons, L_out)
        Recons, Recon Errors: (N, in_neurons, L)
        Weights: (out_neurons, in_neurons, kernel_size)
    """

    def __init__(
        self,
        out_neurons: int,
        in_neurons: int,
        result_dir: str = "./lca_results",
        kernel_size: Union[int, tuple[int]] = 7,
        stride: Union[int, tuple[int]] = 1,
        lambda_: float = 0.25,
        tau: Union[float, int] = 100,
        eta: float = 0.01,
        lca_iters: int = 1000,
        pad: Literal["same", "valid"] = "same",
        return_vars: Iterable[
            Literal[
                "inputs",
                "input_drives",
                "states",
                "acts",
                "recons",
                "recon_errors",
                "conns",
            ]
        ] = ["acts"],
        return_all_ts: bool = False,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            Literal["soft_threshold", "hard_threshold"], Callable[[Tensor], Tensor]
        ] = "soft_threshold",
        input_zero_mean: bool = True,
        input_unit_var: bool = True,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        req_grad: bool = False,
        weight_init: tuple[Callable[[Tensor], Tensor], dict[str, Any]] = (
            torch.nn.init.trunc_normal_,
            {},
        ),
    ) -> None:
        super(LCAConv1D, self).__init__(
            out_neurons,
            in_neurons,
            result_dir,
            kernel_size,
            stride,
            lambda_,
            tau,
            eta,
            lca_iters,
            pad,
            return_vars,
            return_all_ts,
            nonneg,
            track_metrics,
            transfer_func,
            input_zero_mean,
            input_unit_var,
            cudnn_benchmark,
            d_update_clip,
            lr_schedule,
            req_grad,
            weight_init,
            False,
        )

    def _init_weight_tensor(self) -> None:
        weights = torch.empty(
            self.out_neurons,
            self.in_neurons,
            self.kernel_size if type(self.kernel_size) == int else self.kernel_size[0],
        )
        weights = self.weight_init(weights, **self.weight_init_kwargs)
        self.weights = torch.nn.Parameter(weights, requires_grad=self.req_grad)
        self.normalize_weights()

    def _to_correct_shape(
        self, inputs: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        if len(inputs.shape) == 3:
            return to_5d_from_3d(inputs), to_3d_from_5d
        elif len(inputs.shape) == 5:
            return inputs, lambda inputs: inputs
        else:
            raise RuntimeError(
                f"Expected 3D inputs, but got {len(inputs.shape)}D inputs."
            )

    def _transform_conv_params(
        self, val: Union[int, tuple[int]]
    ) -> tuple[int, int, int]:
        if type(val) == int:
            return (val, 1, 1)
        elif type(val) == tuple:
            if len(val) == 1:
                return val + (1, 1)
            else:
                raise RuntimeError(
                    f"If tuple given, it should be a 1-tuple (e.g. (7,)), but got {val}."
                )
        else:
            raise RuntimeError(f"Must be given type int or tuple, but got {type(val)}.")

    def get_weights(self) -> None:
        return self.weights.clone()


class LCAConv2D(_LCAConvBase):
    """
    Performs LCA with a 2D Convolution on 4D inputs.

    Args:
        out_neurons (int): Number of LCA neurons/features
        in_neurons (int): Number of input neurons/channels
        result_dir (str, optional): Path where model params and results will
            be saved
        kernel_size (int | tuple[int, int], optional): Spatial size of the LCA
            receptive fields. Default: (7, 7)
        stride (int | tuple[int, int], optional): Stride of the LCA receptive fields.
            Default: (1, 1)
        lambda_ (float, optional): LCA firing threshold. Default: 0.25
        tau (float | int, optional): LCA time constant. Default: 1000
        eta (float, optional): Learning rate for built-in weight updates.
            Default: 0.01
        lca_iters (int, optional): LCA iterations per forward pass.
            Default: 3000
        pad ('same' | 'valid', optional): Input padding for the conv.
            Default: 'same'
        return_vars (list | tuple, optional): Iterable of one or more strings
            determining which variables will be returned during the forward
            pass. Possible values are 'inputs', 'input_drives', 'states',
            'acts', 'recons', 'recon_errors', and 'conns'. Default: ['acts'].
        return_all_ts (bool, optional): Whether to return the value of
            return_vars at every LCA iteration (True) or just the last LCA
            iteration (False). Default: False
        nonneg (bool, optional): Enforces nonnegative activations.
            Default: True
        track_metrics (bool, optional): Whether to track the L1 sparsity
            penalty, L2 reconstruction error, total energy (L1 + L2), and
            fraction of neurons active over the LCA loop at each forward pass
            and write it to a file in result_dir. Default: False
        transfer_func ('soft_threshold' | 'hard_threshold' | callable,
            optional): The function that transforms the LCA membrane
            potentials into activations. If using custom functions, nonneg
            will not be used. Default: 'soft_threshold'
        input_zero_mean (bool, optional): Whether to make each input in the
            batch have zero mean before performing LCA. Default: True
        input_unit_var (bool, optional): Whether to make each input in the
            batch have unit variance before performing LCA. Default: True
        cudnn_benchmark (bool, optional): Use CUDNN benchmark. Default: True
        d_update_clip (float, optional): Clips weight updates from above and
            below when using built-in update method. Default: np.inf
        lr_schedule (callable, optional): Schedule for eta when using the
            built-in weight update method. Should take in the number of
            forward passes at the current update and return a value for eta.
            Default: None
        req_grad (bool, optional): Keep track of the gradients over the LCA
            loop. This is useful when not using the built-in dictionary
            learning method, or for things like adversarial attacks.
            Default: False
        weight_init (tuple(callable, dict), optional): Initialization to use
            for the dictionary weights and a dictionary of keyword arguments
            for that initialization. Default: truncated normal, default args

    Shape:
        Input: (N, in_neurons, H, W)
        Input Drives, States, Acts: (N, out_neurons, H_out, W_out)
        Recons, Recon Errors: (N, in_neurons, H, W)
        Weights: (out_neurons, in_neurons, kernel_size[0], kernel_size[1])
    """

    def __init__(
        self,
        out_neurons: int,
        in_neurons: int,
        result_dir: str = "./lca_results",
        kernel_size: Union[int, tuple[int, int]] = 7,
        stride: Union[int, tuple[int, int]] = 1,
        lambda_: float = 0.25,
        tau: Union[float, int] = 100,
        eta: float = 0.01,
        lca_iters: int = 1000,
        pad: Literal["same", "valid"] = "same",
        return_vars: Iterable[
            Literal[
                "inputs",
                "input_drives",
                "states",
                "acts",
                "recons",
                "recon_errors",
                "conns",
            ]
        ] = ["acts"],
        return_all_ts: bool = False,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            Literal["soft_threshold", "hard_threshold"], Callable[[Tensor], Tensor]
        ] = "soft_threshold",
        input_zero_mean: bool = True,
        input_unit_var: bool = True,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        req_grad: bool = False,
        weight_init: tuple[Callable[[Tensor], Tensor], dict[str, Any]] = (
            torch.nn.init.trunc_normal_,
            {},
        ),
    ) -> None:
        super(LCAConv2D, self).__init__(
            out_neurons,
            in_neurons,
            result_dir,
            kernel_size,
            stride,
            lambda_,
            tau,
            eta,
            lca_iters,
            pad,
            return_vars,
            return_all_ts,
            nonneg,
            track_metrics,
            transfer_func,
            input_zero_mean,
            input_unit_var,
            cudnn_benchmark,
            d_update_clip,
            lr_schedule,
            req_grad,
            weight_init,
            True,
        )

    def _init_weight_tensor(self) -> None:
        weights = torch.empty(
            self.out_neurons,
            self.in_neurons,
            self.kernel_size if type(self.kernel_size) == int else self.kernel_size[0],
            self.kernel_size if type(self.kernel_size) == int else self.kernel_size[1],
        )
        weights = self.weight_init(weights, **self.weight_init_kwargs)
        self.weights = torch.nn.Parameter(weights, requires_grad=self.req_grad)
        self.normalize_weights()

    def _to_correct_shape(
        self, inputs: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        if len(inputs.shape) == 4:
            return to_5d_from_4d(inputs), to_4d_from_5d
        elif len(inputs.shape) == 5:
            return inputs, lambda inputs: inputs
        else:
            raise RuntimeError(
                f"Expected 4D inputs, but got {len(inputs.shape)}D inputs."
            )

    def _transform_conv_params(
        self, val: Union[int, tuple[int, int]]
    ) -> tuple[int, int, int]:
        if type(val) == int:
            return (1, val, val)
        elif type(val) == tuple:
            if len(val) == 2:
                return (1,) + val
            else:
                raise RuntimeError(
                    f"If tuple given, it should be a 2-tuple (e.g. (7, 7)), but got {val}."
                )
        else:
            raise RuntimeError(f"Must be given type int or tuple, but got {type(val)}.")

    def get_weights(self) -> Tensor:
        return self.weights.clone()


class LCAConv3D(_LCAConvBase):
    """
    Performs LCA with a 3D Convolution on 5D inputs.

    Args:
        out_neurons (int): Number of LCA neurons/features
        in_neurons (int): Number of input neurons/channels
        result_dir (str, optional): Path where model params and results will
            be saved
        kernel_size (int | tuple[int, int, int], optional): Spatio-temporal size of the LCA
            receptive fields. Default: (7, 7, 7)
        stride (int | tuple[int, int, int], optional): Stride of the LCA receptive fields.
            Default: (1, 1, 1)
        lambda_ (float, optional): LCA firing threshold. Default: 0.25
        tau (float | int, optional): LCA time constant. Default: 1000
        eta (float, optional): Learning rate for built-in weight updates.
            Default: 0.01
        lca_iters (int, optional): LCA iterations per forward pass.
            Default: 3000
        pad ('same' | 'valid', optional): Input padding for the conv.
            Default: 'same'
        return_vars (list | tuple, optional): Iterable of one or more strings
            determining which variables will be returned during the forward
            pass. Possible values are 'inputs', 'input_drives', 'states',
            'acts', 'recons', 'recon_errors', and 'conns'. Default: ['acts'].
        return_all_ts (bool, optional): Whether to return the value of
            return_vars at every LCA iteration (True) or just the last LCA
            iteration (False). Default: False
        nonneg (bool, optional): Enforces nonnegative activations.
            Default: True
        track_metrics (bool, optional): Whether to track the L1 sparsity
            penalty, L2 reconstruction error, total energy (L1 + L2), and
            fraction of neurons active over the LCA loop at each forward pass
            and write it to a file in result_dir. Default: False
        transfer_func ('soft_threshold' | 'hard_threshold' | callable,
            optional): The function that transforms the LCA membrane
            potentials into activations. If using custom functions, nonneg
            will not be used. Default: 'soft_threshold'
        input_zero_mean (bool, optional): Whether to make each input in the
            batch have zero mean before performing LCA. Default: True
        input_unit_var (bool, optional): Whether to make each input in the
            batch have unit variance before performing LCA. Default: True
        cudnn_benchmark (bool, optional): Use CUDNN benchmark. Default: True
        d_update_clip (float, optional): Clips weight updates from above and
            below when using built-in update method. Default: np.inf
        lr_schedule (callable, optional): Schedule for eta when using the
            built-in weight update method. Should take in the number of
            forward passes at the current update and return a value for eta.
            Default: None
        req_grad (bool, optional): Keep track of the gradients over the LCA
            loop. This is useful when not using the built-in dictionary
            learning method, or for things like adversarial attacks.
            Default: False
        weight_init (tuple(callable, dict), optional): Initialization to use
            for the dictionary weights and a dictionary of keyword arguments
            for that initialization. Default: truncated normal, default args
        no_time_pad (bool, optional): If True, no padding will be performed
            in dimension D regardless of the value of the pad argument.
            Allows for control over padding in the depth dimension that is
            independent of that in the spatial dimensions.

    Shape:
        Input: (N, in_neurons, D, H, W)
        Input Drives, States, Acts: (N, out_neurons, D_out, H_out, W_out)
        Recons, Recon Errors: (N, in_neurons, D, H, W)
        Weights: (out_neurons, in_neurons, kernel_size[0], kernel_size[1],
            kernel_size[2])
    """

    def __init__(
        self,
        out_neurons: int,
        in_neurons: int,
        result_dir: str = "./lca_results",
        kernel_size: Union[int, tuple[int, int, int]] = 7,
        stride: Union[int, tuple[int, int, int]] = 1,
        lambda_: float = 0.25,
        tau: Union[float, int] = 100,
        eta: float = 0.01,
        lca_iters: int = 1000,
        pad: Literal["same", "valid"] = "same",
        return_vars: Iterable[
            Literal[
                "inputs",
                "input_drives",
                "states",
                "acts",
                "recons",
                "recon_errors",
                "conns",
            ]
        ] = ["acts"],
        return_all_ts: bool = False,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            Literal["soft_threshold", "hard_threshold"], Callable[[Tensor], Tensor]
        ] = "soft_threshold",
        input_zero_mean: bool = True,
        input_unit_var: bool = True,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        req_grad: bool = False,
        weight_init: tuple[Callable[[Tensor], Tensor], dict[str, Any]] = (
            torch.nn.init.trunc_normal_,
            {},
        ),
        no_time_pad: bool = False,
    ) -> None:
        super(LCAConv3D, self).__init__(
            out_neurons,
            in_neurons,
            result_dir,
            kernel_size,
            stride,
            lambda_,
            tau,
            eta,
            lca_iters,
            pad,
            return_vars,
            return_all_ts,
            nonneg,
            track_metrics,
            transfer_func,
            input_zero_mean,
            input_unit_var,
            cudnn_benchmark,
            d_update_clip,
            lr_schedule,
            req_grad,
            weight_init,
            no_time_pad,
        )

    def _init_weight_tensor(self) -> None:
        weights = torch.empty(
            self.out_neurons,
            self.in_neurons,
            self.kernel_size if type(self.kernel_size) == int else self.kernel_size[0],
            self.kernel_size if type(self.kernel_size) == int else self.kernel_size[1],
            self.kernel_size if type(self.kernel_size) == int else self.kernel_size[2],
        )
        weights = self.weight_init(weights, **self.weight_init_kwargs)
        self.weights = torch.nn.Parameter(weights, requires_grad=self.req_grad)
        self.normalize_weights()

    def _to_correct_shape(
        self, inputs: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        if len(inputs.shape) == 5:
            return inputs, lambda inputs: inputs
        else:
            raise RuntimeError(
                f"Expected 5D inputs, but got {len(inputs.shape)}D inputs."
            )

    def _transform_conv_params(
        self, val: Union[int, tuple[int, int, int]]
    ) -> tuple[int, int, int]:
        if type(val) == int:
            return (val,) * 3
        elif type(val) == tuple:
            if len(val) == 3:
                return val
            else:
                raise RuntimeError(
                    f"If tuple given, it should be a 3-tuple (e.g. (7, 7, 7)), but got {val}."
                )
        else:
            raise RuntimeError(f"Must be given type int or tuple, but got {type(val)}.")

    def get_weights(self) -> Tensor:
        return self.weights.clone()
