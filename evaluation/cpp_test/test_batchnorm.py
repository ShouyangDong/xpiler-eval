"""Batch correctness tester for 'batchnorm' kernels with two-phase
parallelism."""

import argparse
import ctypes
import logging
from typing import Tuple

import torch
import torch.nn.functional as F

from evaluation.utils import parse_op_json, run_tests

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


def batchnorm_ref(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference implementation using PyTorch."""
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
    """Run correctness test on compiled batchnorm kernel."""
    try:
        file_name = config["file"]
        N, C, H, W = config["args"]
        op_name = config["op_name"]
        # Generate input and parameters
        input_tensor = torch.rand(N, C, H, W, dtype=torch.float32)
        running_mean = torch.rand(C, dtype=torch.float32)
        running_var = torch.rand(C, dtype=torch.float32) + 0.5
        weight = torch.rand(C, dtype=torch.float32)  # gamma
        bias = torch.rand(C, dtype=torch.float32)  # beta

        # Golden reference
        expected = batchnorm_ref(
            input_tensor, weight, bias, running_mean, running_var
        )

        result_flat = torch.zeros_like(input_tensor)

        # Get pointers
        input_ptr = input_tensor.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = result_flat.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        mean_ptr = running_mean.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        var_ptr = running_var.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        weight_ptr = weight.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        bias_ptr = bias.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[BatchNorm] Function 'batchnorm' not found in {so_path}",
            )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_float),  # mean
            ctypes.POINTER(ctypes.c_float),  # var
            ctypes.POINTER(ctypes.c_float),  # weight
            ctypes.POINTER(ctypes.c_float),  # bias
            ctypes.c_int,  # N
            ctypes.c_int,  # C
            ctypes.c_int,  # H
            ctypes.c_int,  # W
            ctypes.c_float,  # eps
        ]
        func.restype = None
        eps = 1e-5
        # Call kernel
        func(
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

        # Reshape and compare
        if torch.allclose(
            result_flat, expected, rtol=1e-3, atol=1e-3, equal_nan=True
        ):
            return True, f"[BatchNorm] PASSED✅: {file_name}"
        else:
            max_error = (result_flat - expected).abs().max().item()
            return (
                False,
                f"[BatchNorm] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[BatchNorm] Exception in test {file_name}: {str(e)}"


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
        "--source_dir", default="./", help="Directory with .cpp files"
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
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid 'batchnorm' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log results
    log_test_results_and_exit(result, op_name=args.name)
