"""Batch correctness tester for GATHER kernels with two-phase parallelism."""

import argparse
import ctypes
import logging
from typing import Tuple

import torch

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
    verify_numpy_tensor,
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


def reference_gather(
    params: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Reference implementation using PyTorch.

    Out-of-bound indices are clamped and set to zero.
    """
    clamped_indices = torch.clamp(indices, 0, params.size(0) - 1)
    result = params[clamped_indices]
    # Zero out out-of-bound values
    out_of_bound = (indices < 0) | (indices >= params.size(0))
    if out_of_bound.any():
        result[out_of_bound] = 0.0
    return result


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled Gather kernel."""
    B, L, I = config["args"]

    op_name = config["op_name"]
    # Generate inputs
    torch.manual_seed(1234)
    params = torch.randn(B, L, dtype=torch.float32)
    indices = torch.randint(low=-1, high=B + 1, size=(I,), dtype=torch.int32)

    # Ensure contiguous
    params = params.contiguous()
    indices = indices.contiguous()

    # Reference output
    ref = reference_gather(params, indices).numpy()

    # Output buffer
    output = torch.zeros(I, L, dtype=torch.float32).contiguous().numpy()

    # Get pointers
    params_ptr = params.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    indices_ptr = indices.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32)
    )
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[{op_name}] Function 'gather' not found in {so_path}",
        )

    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # params
        ctypes.POINTER(ctypes.c_int32),  # indices
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,  # params_batch
        ctypes.c_int,  # params_len
        ctypes.c_int,  # indices_len
    ]
    func.restype = None

    # Call kernel
    func(params_ptr, indices_ptr, output_ptr, B, L, I)
    return verify_numpy_tensor(output, ref, op_name)


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
        logger.warning("No valid 'gather' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name,
        configs,
        args.source_dir,
        __file__,
        args.target,
        num_workers=args.jobs,
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
