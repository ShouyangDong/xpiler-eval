"""Batch correctness tester for 'deformable_attention' kernels with two-phase
parallelism."""

import argparse
import ctypes
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
    verify_numpy_tensor,
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@torch.no_grad()
def deformable_attention_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """PyTorch reference implementation of multi-scale deformable attention."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split(
        [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # Reshape value: [N_, H_*W_, M_, D_] -> [N_*M_, D_, H_, W_]
        value_l_ = (
            value_list[lid_]
            .flatten(2)
            .transpose(1, 2)
            .reshape(N_ * M_, D_, H_, W_)
        )
        # Reshape sampling grid: [N_, Lq_, M_, P_, 2] -> [N_*M_, Lq_, P_, 2]
        sampling_grid_l_ = (
            sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        )
        # Sample using bilinear interpolation
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [N_*M_, D_, Lq_, P_]
        sampling_value_list.append(sampling_value_l_)

    # Stack and apply attention weights
    # [N_, M_, Lq_, L_, P_] -> [N_*M_, 1, Lq_, L_*P_]
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    # Stack sampled values: [N_*M_, D_, Lq_, L_, P_] -> sum over last two dims
    output = (
        (
            torch.stack(sampling_value_list, dim=-2).flatten(-2)
            * attention_weights
        )
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled deformable_attention kernel."""
    config["file"]
    N, M, D, Lq, L, P = config["args"]
    op_name = config["op_name"]
    # Hardcoded spatial shapes (same as your example)
    # You can make this configurable via JSON if needed
    shapes = torch.tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long
    )
    S = sum((H * W).item() for H, W in shapes)

    # Generate inputs
    value = torch.rand(N, S, M, D, dtype=torch.float32) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2, dtype=torch.float32)
    attention_weights = torch.rand(N, Lq, M, L, P, dtype=torch.float32) + 1e-5
    attention_weights /= attention_weights.sum(dim=-1, keepdim=True).sum(
        dim=-2, keepdim=True
    )

    # Reference output
    torch_da = deformable_attention_pytorch(
        value, shapes, sampling_locations, attention_weights
    )

    # Prepare output buffer
    output_array = np.zeros((N, Lq, M * D), dtype=np.float32)

    # Get pointers
    value_ptr = value.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    shapes_ptr = (
        shapes.numpy()
        .astype(np.int32)
        .ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    sampling_locs_ptr = sampling_locations.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    attn_weights_ptr = attention_weights.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[{op_name}] Function 'deformable' not found in {so_path}",
        )

    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # value
        ctypes.POINTER(ctypes.c_int),  # shapes (int array of [L, 2])
        ctypes.POINTER(ctypes.c_float),  # sampling_locations
        ctypes.POINTER(ctypes.c_float),  # attention_weights
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    func.restype = None

    # Call kernel
    func(
        value_ptr,
        shapes_ptr,
        sampling_locs_ptr,
        attn_weights_ptr,
        output_ptr,
    )

    return verify_numpy_tensor(output_array, torch_da.numpy(), op_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (CPU)")
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the operator to test (used to filter configs).",
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
    )
    parser.add_argument(
        "--source_dir", default="./", help="Directory containing .cpp files"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning(
            "No valid 'deformable_attention' kernels found in config."
        )
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
