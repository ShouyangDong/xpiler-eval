"""Batch correctness tester for 'concat' kernels with two-phase parallelism."""

import argparse
import ctypes
import logging
import sys
from typing import List, Tuple

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


def concat_ref(tensors: List[torch.Tensor], axis: int) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.cat(tensors, dim=axis)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled concat kernel."""

    shape = config["args"]
    axis = config["axis"]
    op_name = config["op_name"]
    # Create input tensors
    input1 = torch.randn(*shape, dtype=torch.float32)
    input2 = torch.randn(*shape, dtype=torch.float32)

    # Reference output
    expected = concat_ref([input1, input2], axis=axis)

    # Flatten for C++ (row-major)
    flat1 = input1.flatten().numpy()
    flat2 = input2.flatten().numpy()
    output_flat = torch.zeros_like(expected.flatten())
    out_ptr = output_flat.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Get pointers
    ptr1 = flat1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr2 = flat2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Load library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[{op_name}] Function 'concat' not found in {so_path}",
        )

    func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
    func.restype = None

    # Call kernel
    func(ptr1, ptr2, out_ptr)

    # Reshape and compare
    result_reshaped = output_flat.reshape(expected.shape)

    return verify_torch_tensor(result_reshaped, expected, op_name)


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
        logger.warning("No valid 'concat' kernels found in config.")
        sys.exit(0)

    # Run two-phase test
    results = run_tests(
        args.name,
        configs,
        args.source_dir,
        __file__,
        args.target,
        num_workers=args.jobs,
    )

    # Log all results
    log_test_results_and_exit(results, op_name=args.name)
