import argparse
import ctypes
import logging
import os
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


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    shape = config["args"]
    # Define the input array and kernel
    input_array = np.random.uniform(size=[shape[1]]).astype("float32")
    kernel = np.array([0.5, 1.0, 0.5]).astype(np.float32)
    # Calculate the output size
    output_size = shape[0]
    # Create an empty output array
    output_ctypes = np.zeros(output_size, dtype=np.float32)
    op_name = config["op_name"]
    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Calculate the result using numpy for comparison
    output_np = np.convolve(input_array, kernel, mode="valid")

    # Load the shared library with the batch matrix multiplication function

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(input_ptr, kernel_ptr, output_ptr, shape[1], shape[0])
    return verify_numpy_tensor(output_ctypes, output_np, op_name)


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
