"""Batch correctness tester for 'conv1d' kernels with two-phase parallelism."""

import argparse
import ctypes
import logging
from typing import Tuple

import numpy as np

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


def conv1d_ref(input_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reference implementation using NumPy."""
    return np.convolve(input_array, kernel, mode="valid")


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled conv1d kernel."""
    try:
        file_name = config["file"]
        output_size, input_length = config["args"]
        op_name = config["op_name"]
        # Generate random input
        input_array = np.random.uniform(size=input_length).astype(
            np.float32
        )  # shape: (input_length,)

        # Fixed kernel [0.5, 1.0, 0.5]
        kernel = np.array([0.5, 1.0, 0.5], dtype=np.float32)

        # Reference output
        expected = conv1d_ref(input_array, kernel)

        # Prepare output buffer
        output_ctypes = np.zeros(output_size, dtype=np.float32)

        # Get pointers
        input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_ctypes.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[Conv1D] Function 'conv1d' not found in {so_path}"

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # kernel
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        func.restype = None

        # Call kernel
        func(input_ptr, kernel_ptr, output_ptr)

        # Compare
        if np.allclose(
            output_ctypes, expected, rtol=1e-3, atol=1e-3, equal_nan=True
        ):
            return True, f"[Conv1D] PASSED✅: {file_name}"
        else:
            max_error = np.max(np.abs(output_ctypes - expected))
            return (
                False,
                f"[Conv1D] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[Conv1D] Exception in test {file_name}: {str(e)}"


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

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid 'conv1d' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log results
    log_test_results_and_exit(result, op_name=args.name)
