"""Batch correctness tester for GEMV kernels with parallel compilation and
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


def reference_gemv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reference GEMV using PyTorch."""
    return torch.matmul(A, x)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled GEMV kernel."""
    try:
        M, K = config["args"]
        file_name = config["file"]
        op_name = config["op_name"]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[{op_name}] Function 'gemv' not found in {so_path}"

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # y
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        A = torch.randn(M, K, dtype=torch.float32)
        x = torch.randn(K, dtype=torch.float32)
        y = torch.zeros(M, dtype=torch.float32)

        # Reference
        y_ref = reference_gemv(A, x)

        # Get pointers
        A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call kernel
        func(A_ptr, x_ptr, y_ptr)

        # Compare
        try:
            torch.allclose(
                y,
                y_ref,
                rtol=1e-3,
                atol=1e-3,
                equal_nan=True,
            )
            return (
                True,
                f"[{op_name}] PASSED✅: {file_name}",
            )
        except Exception as e:
            return False, f"[{op_name}] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[{op_name}] Exception in test {file_name}: {str(e)}"


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
        logger.warning("No valid 'gemv' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
