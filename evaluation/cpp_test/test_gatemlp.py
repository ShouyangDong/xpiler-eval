"""Batch correctness tester for gatemlp kernels (int16 → int32) with two-phase
parallelism."""

import argparse
import ctypes
import logging
from typing import Tuple

import numpy as np
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


def reference_gatemlp(
    X: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Reference implementation using PyTorch.

    X, A, B: int16 arrays
    Output: O = SiLU(X @ A) * (X @ B), converted to int32
    """
    X_t = torch.from_numpy(X).to(torch.int16)
    A_t = torch.from_numpy(A).to(torch.int16)
    B_t = torch.from_numpy(B).to(torch.int16)

    # Forward: O = silu(X @ A) * (X @ B)
    C = torch.matmul(X_t, A_t).to(torch.float32)  # Promote to float for SiLU
    O2 = torch.matmul(X_t, B_t).to(torch.int32)  # Keep as int32
    O1 = torch.nn.functional.silu(C)
    O_fp32 = O1 * O2.float()
    return O_fp32.cpu().numpy()


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled GateMLP kernel."""
    try:
        B, K, N = config["args"]
        file_name = config["file"]
        op_name = config["op_name"]
        # Generate inputs in int16 range [-10, 10]
        torch.manual_seed(1234)
        X_np = torch.randint(
            low=-10, high=11, size=(B, K), dtype=torch.int16
        ).numpy()
        A_np = torch.randint(
            low=-10, high=11, size=(K, N), dtype=torch.int16
        ).numpy()
        B_np = torch.randint(
            low=-10, high=11, size=(K, N), dtype=torch.int16
        ).numpy()

        # Ensure contiguous
        X_np = np.ascontiguousarray(X_np, dtype=np.int16)
        A_np = np.ascontiguousarray(A_np, dtype=np.int16)
        B_np = np.ascontiguousarray(B_np, dtype=np.int16)

        # Reference output
        ref = reference_gatemlp(X_np, A_np, B_np)

        # Prepare output buffer: float32 to hold int32 values
        output = np.zeros((B, N), dtype=np.float32)
        output = np.ascontiguousarray(output)

        # Get pointers
        X_ptr = X_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        O_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[{op_name}] Function 'gatemlp' not found in {so_path}",
            )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_float),  # int32 stored as float
        ]
        func.restype = None

        # Call kernel
        func(X_ptr, A_ptr, B_ptr, O_ptr)

        # Compare: allow ±2 error due to integer rounding
        diff = np.abs(output - ref)
        max_diff = diff.max()
        mean_diff = diff.mean()

        if max_diff <= 2.0:
            return (
                True,
                f"[{op_name}] ✅ {file_name}| Max diff: {max_diff:.2f}",
            )
        else:
            return (
                False,
                f"[{op_name}] FAILED❌: {file_name} | Max diff: {
                    max_diff:.2f}, Mean: {
                    mean_diff:.2f}",
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
        logger.warning("No valid 'gatemlp' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
