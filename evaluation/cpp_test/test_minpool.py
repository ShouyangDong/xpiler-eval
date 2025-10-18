"""Batch correctness tester for minpool kernels with parallel compilation and
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
    minpool_np,
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


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled minpool kernel."""
    shape = config["args"][:4]
    kernel = config["args"][4:6]
    stride = config["args"][6:8]

    config["file"]
    dtype_str = config.get("dtype", "float32")
    op_name = config["op_name"]
    # Load shared library
    lib = ctypes.CDLL(so_path)

    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[{op_name}] Function '{func_name}' not found in {so_path}",
        )

    # Set function signature
    ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
    torch_dtype = torch.float32 if dtype_str == "float32" else torch.float16

    func.argtypes = [
        ctypes.POINTER(ctype),  # input
        ctypes.POINTER(ctype),  # output
    ]
    func.restype = None

    # Generate input
    torch.manual_seed(1234)
    input_tensor = torch.randn(*shape, dtype=torch_dtype) * 100
    expected = minpool_np(input_tensor, kernel + stride)

    # Flatten and get pointers
    input_flat = input_tensor.flatten().numpy()
    output_flat = (
        torch.zeros(expected.shape, dtype=torch_dtype).flatten().numpy()
    )

    input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctype))
    output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctype))

    # Call kernel
    func(input_ptr, output_ptr)

    # Reshape and compare
    result_reshaped = torch.from_numpy(output_flat).reshape(expected.shape)
    return verify_torch_tensor(result_reshaped, expected, op_name)


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
        logger.warning("No valid 'minpool' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
