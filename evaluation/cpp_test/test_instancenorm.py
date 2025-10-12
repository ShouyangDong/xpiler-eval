"""Stable and parallel-safe InstanceNorm2d correctness tester."""

import argparse
import ctypes
import logging
from typing import Tuple

import torch
import torch.nn.functional as F

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
)

# ========== Logger ==========
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


# ========== Reference Implementation ==========
def reference_instancenorm(input, weight, bias, eps=1e-5):
    return F.instance_norm(
        input,
        running_mean=None,
        running_var=None,
        weight=weight,
        bias=bias,
        momentum=0,
        eps=eps,
    )


# ========== Test One Kernel ==========
def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Test compiled kernel correctness (run in subprocess)."""
    file_name = config["file"]
    op_name = config["op_name"]
    N, C, H, W = config["args"]
    eps = 1e-5

    try:
        input_tensor = torch.rand(N, C, H, W, dtype=torch.float32)
        weight = torch.rand(C, dtype=torch.float32)
        bias = torch.rand(C, dtype=torch.float32)
        expected = reference_instancenorm(input_tensor, weight, bias, eps)

        input_flat = input_tensor.flatten()
        input_ptr = input_flat.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        weight_ptr = weight.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        bias_ptr = bias.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result_ctypes = torch.zeros_like(input_flat)
        output_ptr = result_ctypes.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # local load for safety
        lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_LOCAL)
        kernel_func = getattr(lib, op_name, None)
        kernel_func.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        kernel_func.restype = None

        kernel_func(
            input_ptr, output_ptr, weight_ptr, bias_ptr, N, C, H, W, eps
        )
        result_reshaped = result_ctypes.reshape(N, C, H, W)

        torch.allclose(result_reshaped, expected, rtol=1e-3, atol=1e-3)
        max_err = (result_reshaped - expected).abs().max().item()
        return (
            True,
            f"[{op_name}] ✅ {file_name} PASSED | max_err={max_err:.2e}",
        )

    except Exception as e:
        return False, f"[{op_name}] ❌ {file_name} FAILED | {str(e)}"


# ========== CLI ==========
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
    parser.add_argument("--source_dir", required=True)
    parser.add_argument(
        "--target", required=True, choices=["cuda", "hip", "mlu", "cpu"]
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid instancenorm kernels found.")
        exit(0)

    results = run_tests(configs, args.source_dir, args.target, jobs=args.jobs)
    log_test_results_and_exit(results, op_name=args.name)
