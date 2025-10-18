"""Batch correctness tester for 'dense_int16_bias_int32' kernels with two-phase
parallelism."""

import argparse
import ctypes
import logging
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


def reference_dense_int16(
    A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Reference implementation: int16 GEMM + int32 bias.
    A: [M, K], B: [K, N], bias: [N]
    Returns: [M, N] int32
    """
    return torch.matmul(A, B).to(torch.int32) + bias.unsqueeze(0)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled dense_int16_bias_int32 kernel."""
    M, N, K = config["args"]
    config["file"]
    op_name = config["op_name"]
    # Generate inputs
    A = torch.randint(-10, 10, (M, K), dtype=torch.int16)
    B = torch.randint(-10, 10, (K, N), dtype=torch.int16)
    bias = torch.randint(-100, 100, (N,), dtype=torch.int32)

    A = A.contiguous()
    B = B.contiguous()
    bias = bias.contiguous()

    # Reference output
    ref = reference_dense_int16(A, B, bias)

    # Prepare output buffer
    output = torch.zeros((M, N), dtype=torch.int32).contiguous()

    # Get pointers
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_int16))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_int16))
    bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_int32))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_int32))

    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[{op_name}] Function 'dense' not found in {so_path}",
        )

    func.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    func.restype = None

    # Call kernel
    func(A_ptr, B_ptr, bias_ptr, output_ptr)
    return verify_torch_tensor(output, ref, op_name)


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
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid 'dense' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
