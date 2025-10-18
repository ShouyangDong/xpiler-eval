"""Batch correctness tester for Sign kernels with parallel compilation and
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


def ref_program(x: np.ndarray) -> np.ndarray:
    """Golden reference: sign(x) = +1 if x>0, -1 if x<0, 0 if x==0"""
    return np.sign(x)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled sign kernel."""
    try:
        file_name = config["file"]
        shape = config["args"]
        dtype_str = config["dtype"]
        op_name = config["op_name"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[{op_name}] Function '{op_name}' not found in {so_path}"

        # Determine C type and numpy dtype
        ctype_float = (
            ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        )
        np_dtype = np.float32 if dtype_str == "float32" else np.float16

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctype_float),  # input
            ctypes.POINTER(ctype_float),  # output
        ]
        func.restype = None

        # Generate input
        input_array = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
        expected_output = ref_program(input_array)
        output_array = np.zeros_like(input_array)

        # Prepare pointers
        input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctype_float))
        output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctype_float))

        # Call kernel
        func(input_ptr, output_ptr)

        # Compare
        rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (5e-2, 5e-2)
        if np.allclose(
            output_array, expected_output, rtol=rtol, atol=atol, equal_nan=True
        ):
            max_error = np.max(np.abs(output_array - expected_output))
            return (
                True,
                f"[{op_name}] ✅ {file_name}| Max error: {max_error:.2e}",
            )
        else:
            max_error = np.max(np.abs(output_array - expected_output))
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

    if not configs:
        logger.warning("⚠️ No valid 'sign' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
