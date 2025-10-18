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


def batch_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Perform batch matrix multiplication using PyTorch.

    A: (batch_size, i, j)
    B: (batch_size, j, k)
    Output: (batch_size, i, k)
    """
    return torch.matmul(A, B).to(torch.float32)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    shape = config["args"]
    op_name = config["op_name"]
    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
    # Create input tensors on AMD GPU
    A = torch.ones(
        (batch_size, matrix_dim_i, matrix_dim_j),
        dtype=torch.float16,
        device=torch.device("cuda"),
    )
    B = torch.ones(
        (batch_size, matrix_dim_j, matrix_dim_k),
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    # Perform batch matmul on GPU
    result_torch = batch_matmul(A, B)

    # Move reference result to CPU for comparison
    result_torch_cpu = result_torch.cpu().contiguous()

    # Host tensors for kernel (CPU memory)
    A_host = A.cpu().contiguous()
    B_host = B.cpu().contiguous()

    # Output tensor (CPU)
    result_ctypes = torch.zeros(
        (batch_size, matrix_dim_i, matrix_dim_k), dtype=torch.float32
    ).contiguous()

    # Get raw pointers (CPU memory)
    A_ptr = ctypes.cast(A_host.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    B_ptr = ctypes.cast(B_host.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    output_ptr = ctypes.cast(
        result_ctypes.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    # Load and call kernel
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    kernel_func = getattr(lib, op_name + "_kernel")

    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel_func.restype = None

    kernel_func(
        A_ptr,
        B_ptr,
        output_ptr,
        batch_size,
        matrix_dim_i,
        matrix_dim_j,
        matrix_dim_k,
    )

    # Verify results
    return verify_torch_tensor(
        result_ctypes, result_torch_cpu, op_name=op_name
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
        "--source_dir", default="./", help="Directory with .cu files"
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
    configs = parse_op_json(args.config, args.name, file_type="cuda")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
