"""Batch correctness tester for RMSNorm kernels with parallel compilation and
testing."""

import argparse
import ctypes
import logging
from typing import Tuple

import torch

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
    verify_torch_tensor,
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


def reference_rmsnorm(x: torch.Tensor) -> torch.Tensor:
    """Reference RMSNorm implementation."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled RMSNorm kernel."""
    config["file"]
    shape = config["args"]
    dtype_str = config["dtype"]
    op_name = config["op_name"]

    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[{op_name}] Function '{op_name}' not found in {so_path}",
        )

    # Determine C type and torch dtype
    ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
    torch_dtype = torch.float32 if dtype_str == "float32" else torch.float16

    # Set function signature: void rmsnorm(float* input, float* output)
    func.argtypes = [
        ctypes.POINTER(ctype),
        ctypes.POINTER(ctype),
    ]
    func.restype = None

    # Generate input
    input_tensor = torch.randn(shape, dtype=torch_dtype)
    expected = reference_rmsnorm(input_tensor)

    # Flatten for C
    input_flat = input_tensor.flatten().numpy()
    output_flat = torch.zeros_like(input_tensor).flatten().numpy()
    output_array = (ctype * output_flat.size)()

    input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctype))
    output_ptr = output_array

    # Call kernel
    func(input_ptr, output_ptr)

    # Convert result back
    computed_flat = torch.tensor(
        [output_array[i] for i in range(output_flat.size)],
        dtype=torch_dtype,
    ).view(shape)
    return verify_torch_tensor(computed_flat, expected, op_name)


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
        logger.warning("⚠️ No valid 'rmsnorm' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
