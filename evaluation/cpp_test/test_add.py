"""Batch correctness tester for 'add' kernels with two-phase parallelism."""

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


def add_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.add(A, B)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    shape = config["args"]
    op_name = config["op_name"]
    A = torch.rand(*shape, device="cpu")
    B = torch.rand(*shape, device="cpu")
    ref = add_ref(A, B)

    A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_tensor = torch.zeros_like(ref)
    out_ptr = output_tensor.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Load and call kernel
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
    func.restype = None
    func(A_ptr, B_ptr, out_ptr)
    return verify_torch_tensor(output_tensor, ref, op_name)


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
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No 'add' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name,
        configs,
        args.source_dir,
        __file__,
        args.target,
        num_workers=args.jobs,
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
