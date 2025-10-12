import argparse
import ctypes
import logging
import os
from typing import Tuple

import torch
import torch.nn.functional as F

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
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


def batchnorm_inference(
    input, weight, bias, running_mean, running_var, eps=1e-5
):
    """
    PyTorch golden reference for BatchNorm inference
    input: (N, C, H, W)
    weight (gamma), bias (beta), running_mean, running_var: (C,)
    """
    return F.batch_norm(
        input,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
        eps=eps,
    )


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    N, C, H, W = config["args"]
    op_name = config["op_name"]
    # Generate random input
    input_tensor = torch.rand(N, C, H, W, dtype=torch.float32)

    # Fixed parameters (from training)
    running_mean = torch.rand(C, dtype=torch.float32)
    running_var = torch.rand(C, dtype=torch.float32) + 0.5
    weight = torch.rand(C, dtype=torch.float32)  # gamma
    bias = torch.rand(C, dtype=torch.float32)  # beta
    eps = 1e-5

    # Golden reference
    expected = batchnorm_inference(
        input_tensor, weight, bias, running_mean, running_var, eps
    )

    # Flatten input for C++ (row-major)
    input_flat = input_tensor.flatten()  # (N*C*H*W,)
    input_ptr = input_flat.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Prepare parameter pointers
    mean_ptr = running_mean.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    var_ptr = running_var.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    weight_ptr = weight.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = bias.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Output tensor
    result_ctypes = torch.zeros_like(input_flat)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Load library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    kernel_func = getattr(lib, op_name + "_kernel")  # batchnorm

    # Function signature
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.POINTER(ctypes.c_float),  # mean     (C,)
        ctypes.POINTER(ctypes.c_float),  # var      (C,)
        ctypes.POINTER(ctypes.c_float),  # weight   (C,)
        ctypes.POINTER(ctypes.c_float),  # bias     (C,)
        ctypes.c_int,  # N
        ctypes.c_int,  # C
        ctypes.c_int,  # H
        ctypes.c_int,  # W
        ctypes.c_float,  # epsilon
    ]
    kernel_func.restype = None

    # Call kernel

    kernel_func(
        input_ptr,
        output_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        N,
        C,
        H,
        W,
        eps,
    )

    # Reshape result for comparison
    result_reshaped = result_ctypes.reshape(N, C, H, W)

    # Verify
    is_correct = torch.allclose(
        result_reshaped, expected, rtol=1e-3, atol=1e-3, equal_nan=True
    )

    if is_correct:
        return True, f"[ADD] PASSED✅: {config['file']}"
    else:
        return False, f"[ADD] FAILED❌: {config['file']} (mismatch)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (HIP)")
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the operator to test (used to filter configs).",
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
    )
    parser.add_argument(
        "--source_dir", default="./", help="Directory with .cpp files"
    )
    parser.add_argument(
        "--target",
        default="cpu",
        choices=["cuda", "cpu", "mlu", "hip"],
        help="Target platform",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name, file_type="hip")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
