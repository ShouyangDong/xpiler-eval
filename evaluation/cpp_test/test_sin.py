"""Batch correctness tester for Sin kernels with parallel compilation and
testing."""

import argparse
import ctypes
import logging
import os
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


def ref_program(x: torch.Tensor) -> torch.Tensor:
    """Golden reference: sin(x) using PyTorch."""
    return torch.sin(x)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled sin kernel (fp32 only)."""
    try:
        file_name = config["file"]
        shape = config["args"]
        op_name = config["op_name"]

        # Load shared library
        lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
        func = getattr(lib, op_name + "_kernel", None)
        if not func:
            return False, f"[{op_name}] Function '{op_name}' not found in {so_path}"

        # Set function signature: void sin(float*, float*)
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        func.restype = None

        # Generate input: fp32, contiguous, in range [0, 4π]
        A_torch = (
            torch.rand(*shape, device="cpu", dtype=torch.float32)
            * 4
            * torch.pi
        )
        expected_output = torch.sin(A_torch)  # Golden reference

        # Output tensor
        result_ctypes = torch.zeros(shape, dtype=torch.float32)

        # Get ctypes pointers (ensure contiguous)
        input_ptr = A_torch.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = result_ctypes.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # Call kernel
        func(input_ptr, output_ptr)

        # Compare with tolerance
        if torch.allclose(
            result_ctypes,
            expected_output,
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True,
        ):
            return True, f"[{op_name}] PASSED✅: {file_name}"
        else:
            max_error = (result_ctypes - expected_output).abs().max().item()
            return (
                False,
                f"[{op_name}] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

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

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
