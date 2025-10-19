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


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    M, N = config["args"]

    # Set device to AMD GPU if available

    # Create tensors on AMD GPU
    A = torch.randn(M, N, dtype=torch.float32, device=torch.device("cuda"))
    x = torch.randn(N, dtype=torch.float32, device=torch.device("cuda"))

    # Perform matmul on AMD GPU
    y_torch = torch.matmul(A, x)  # Shape: (M,)

    # Move reference result back to CPU for comparison
    y_torch_cpu = y_torch.cpu().contiguous()

    # Host tensors for kernel input (float32)
    A_host = A.cpu().contiguous()
    x_host = x.cpu().contiguous()

    # Output tensor on CPU
    y_ctypes = torch.zeros(M, dtype=torch.float32).contiguous()

    # Get raw pointers (CPU memory)
    A_ptr = ctypes.cast(A_host.data_ptr(), ctypes.POINTER(ctypes.c_float))
    x_ptr = ctypes.cast(x_host.data_ptr(), ctypes.POINTER(ctypes.c_float))
    y_ptr = ctypes.cast(y_ctypes.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Load and call kernel
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    kernel_func = getattr(lib, op_name + "_kernel")

    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel_func.restype = None

    kernel_func(A_ptr, x_ptr, y_ptr, M, N)

    # Verify results
    return verify_torch_tensor(y_ctypes, y_torch_cpu, op_name=op_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (CUDA)")
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
