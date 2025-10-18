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


def ref_program(A, B):
    """
    Reference implementation of element-wise subtraction: A - B
    """
    return torch.sub(A, B)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    shape = config["args"]
    total_elements = torch.prod(torch.tensor(shape))
    op_name = config["op_name"]
    # Generate random input tensors
    dtype = torch.float32
    A = torch.rand(shape, dtype=dtype)
    B = torch.rand(shape, dtype=dtype)

    # Compute reference result using PyTorch
    expected_output = ref_program(A, B)

    # Output tensor
    output_tensor = torch.zeros_like(A)

    # Ensure contiguous memory layout for ctypes access
    A = A.contiguous()
    B = B.contiguous()
    output_tensor = output_tensor.contiguous()

    # Get raw pointers
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_float))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_float))
    output_ptr = ctypes.cast(
        output_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

    # Load the compiled shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")

    # Define function signature
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # Output
        ctypes.c_int,  # Total number of elements
    ]
    function.restype = None

    # Call the subtraction kernel
    function(A_ptr, B_ptr, output_ptr, total_elements)

    # Verify results
    return verify_torch_tensor(output_tensor, expected_output, op_name=op_name)


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
    configs = parse_op_json(args.config, args.name, file_type="cu")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
