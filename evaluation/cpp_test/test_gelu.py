"""Batch correctness tester for GELU kernels with parallel compilation and
testing."""

import argparse
import ctypes
import logging
from typing import Tuple

import numpy as np

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


def reference_gelu(x: np.ndarray) -> np.ndarray:
    """Reference GELU implementation using NumPy."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled GELU kernel."""
    N = np.prod(config["args"])

    op_name = config["op_name"]
    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return False, f"[{op_name}] Function 'gelu' not found in {so_path}"

    # Set function signature
    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    func.restype = None

    # Generate input
    np.random.seed(1234)
    input_data = np.random.uniform(-5, 5, size=N).astype(np.float32)
    expected = reference_gelu(input_data)
    output_data = np.zeros_like(input_data)

    # Get pointers
    input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call kernel
    func(input_ptr, output_ptr)
    return verify_numpy_tensor(output_data, expected, op_name)


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
        logger.warning("No valid 'gelu' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
