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


def add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Element-wise addition using PyTorch."""
    return torch.add(A, B)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    shape = config["args"]
    op_name = config["op_name"]

    # Generate random input tensors on CPU using PyTorch
    A = torch.rand(shape, dtype=torch.float32)
    B = torch.rand(shape, dtype=torch.float32)

    # Compute expected result using PyTorch
    result_torch = add(A, B)

    # Ensure tensors are contiguous in memory for ctypes pointer access
    A_cont = A.contiguous()
    B_cont = B.contiguous()
    result_torch.contiguous()

    # Get raw pointers to tensor data
    A_ptr = ctypes.cast(A_cont.data_ptr(), ctypes.POINTER(ctypes.c_float))
    B_ptr = ctypes.cast(B_cont.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Load the compiled shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")

    # Define the function signature
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Input A
        ctypes.POINTER(ctypes.c_float),  # Input B
        ctypes.POINTER(ctypes.c_float),  # Output
        ctypes.c_int,  # Total number of elements
    ]
    function.restype = None

    # Prepare output tensor (contiguous, CPU)
    result_ctypes_torch = torch.zeros(shape, dtype=torch.float32).contiguous()
    output_ptr = ctypes.cast(
        result_ctypes_torch.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

    # Call the HIP kernel function
    # Use .numel() for total elements
    function(A_ptr, B_ptr, output_ptr, A.numel())

    # Compare kernel output with PyTorch result
    return verify_torch_tensor(
        result_ctypes_torch, result_torch, op_name=op_name
    )


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
        logger.warning("No 'add' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
