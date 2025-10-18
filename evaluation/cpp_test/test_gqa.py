"""Batch correctness tester for GQA kernels with parallel compilation and
testing."""

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


def reference_gqa(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """Reference GQA using PyTorch."""
    # Q: [B, H, Sq, D]
    # K: [B, H, D, Skv]
    # V: [B, H, Skv, D]
    # S = softmax(Q @ K)
    # O = S @ V
    with torch.no_grad():
        S = torch.matmul(Q, K)  # [B, H, Sq, Skv]
        S = torch.softmax(S, dim=-1)
        O = torch.matmul(S, V)  # [B, H, Sq, D]
    return O


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled GQA kernel."""
    B = config["batch"]
    H = config["num_heads"]
    Sq = config["seq_q"]
    Skv = config["seq_kv"]
    D = config["head_dim"]
    config["file"]
    op_name = config["op_name"]
    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)

    # Set function signature
    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # K
        ctypes.POINTER(ctypes.c_float),  # V
        ctypes.POINTER(ctypes.c_float),  # O
        ctypes.c_int,  # batch
        ctypes.c_int,  # num_heads
        ctypes.c_int,  # seq_q
        ctypes.c_int,  # seq_kv
        ctypes.c_int,  # head_dim
    ]
    func.restype = None

    # Generate input
    torch.manual_seed(1234)
    Q = torch.randn(B, H, Sq, D, dtype=torch.float32)
    # Note: K is [B, H, D, Skv]
    K = torch.randn(B, H, D, Skv, dtype=torch.float32)
    V = torch.randn(B, H, Skv, D, dtype=torch.float32)
    O = torch.zeros(B, H, Sq, D, dtype=torch.float32)

    # Reference
    O_ref = reference_gqa(Q, K, V)

    # Get pointers
    Q_ptr = Q.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    K_ptr = K.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    V_ptr = V.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    O_ptr = O.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call kernel
    func(Q_ptr, K_ptr, V_ptr, O_ptr, B, H, Sq, Skv, D)
    return verify_torch_tensor(O, O_ref, op_name)


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
        logger.warning("No valid 'gqa' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
