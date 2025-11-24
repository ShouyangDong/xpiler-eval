"""Parallel tester for MatMul kernels on CPU/GPU/MLU.

Supports two-phase pipeline:
1. Parallel compilation
2. Parallel correctness testing
"""

import argparse
import ctypes
import logging
from typing import Tuple

import torch

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    verify_torch_tensor,
)

# ------------------ Logging setup ------------------
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

# ------------------ Testing ------------------


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled MatMul kernel."""
    M, K, N = config["args"]

    op_name = config["op_name"]

    # Generate input
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    ref = torch.matmul(A, B)

    # Prepare C buffer
    C = torch.zeros((M, N), dtype=torch.float32)

    # Convert to ctypes
    A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Load library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # C
    ]
    func.restype = None

    # Call kernel
    func(A_ptr, B_ptr, C_ptr)
    return verify_torch_tensor(C, ref, op_name)


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
        "--jobs", type=int, default=4, help="Number of parallel jobs"
    )
    args = parser.parse_args()

    # Load config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid 'gemm' kernels found.")
        exit(0)

    # Run test pipeline
    log_test_results_and_exit(results, op_name=args.name)
