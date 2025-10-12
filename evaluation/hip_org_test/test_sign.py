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


def ref_program(x):
    """Reference implementation of the sign function using PyTorch.

    Returns: +1 if x > 0, -1 if x < 0, 0 if x == 0.
    """
    return torch.sign(x)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    shape = config["args"]
    # Load the compiled shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")

    # Define function signature
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input array
        ctypes.POINTER(ctypes.c_float),  # output array
        ctypes.c_int,  # total number of elements
    ]
    function.restype = None

    # Create input tensor
    dtype = torch.float32
    # Use a mix of positive, negative, and zero values to test edge cases
    input_tensor = torch.randn(shape, dtype=dtype)
    total_elements = input_tensor.numel()
    # Manually set some values to zero to test edge case
    input_tensor[input_tensor.abs() < 0.1] = 0.0  # Force some zeros

    # Compute reference output using PyTorch
    expected_output = ref_program(input_tensor)

    # Output tensor
    output_tensor = torch.zeros_like(input_tensor)

    # Ensure contiguous memory layout for ctypes access
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # Get raw pointers
    input_ptr = ctypes.cast(
        input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = ctypes.cast(
        output_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

    # Call the Sign kernel
    function(input_ptr, output_ptr, total_elements)

    # Verify results
    if torch.allclose(
        output_tensor, expected_output, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        return True, f"[{op_name}] PASSED✅: {config['file']}"
    else:
        return False, f"[{op_name}] FAILED❌: {config['file']} (mismatch)"


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
