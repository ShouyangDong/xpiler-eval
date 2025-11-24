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


def scatter_reference(self_tensor, indices_tensor, src_tensor, dim):
    """Mimic torch.Tensor.scatter_ self.scatter_(dim, indices, src)"""
    result = self_tensor.clone()
    result.scatter_(dim, indices_tensor, src_tensor)
    return result


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    dim = config["axis"]
    shape = config["args"]
    # Create tensors
    self_tensor = torch.rand(*shape, dtype=torch.float32)
    src_tensor = torch.rand(*shape, dtype=torch.float32)
    # indices must be within valid range for the target dimension
    size_dim = shape[dim]
    indices_tensor = torch.randint(0, size_dim, shape, dtype=torch.int64)

    # Golden reference
    expected = scatter_reference(self_tensor, indices_tensor, src_tensor, dim)

    # Prepare pointers
    self_ptr = (
        self_tensor.flatten()
        .numpy()
        .ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    indices_ptr = (
        indices_tensor.flatten()
        .numpy()
        .ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    src_ptr = (
        src_tensor.flatten()
        .numpy()
        .ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    output_flat = torch.zeros_like(self_tensor).flatten()
    output_ptr = output_flat.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Load shared library
    lib = ctypes.CDLL(so_path)
    kernel_func = getattr(lib, op_name + "_kernel")
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # self
        ctypes.POINTER(ctypes.c_int),  # indices
        ctypes.POINTER(ctypes.c_float),  # src
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    kernel_func.restype = None

    kernel_func(self_ptr, indices_ptr, src_ptr, output_ptr)

    result_reshaped = output_flat.reshape(expected.shape)
    return verify_torch_tensor(result_reshaped, expected, op_name)


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
        "--source_dir", default="./", help="Directory with .hip files"
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
    configs = parse_op_json(args.config, args.name, file_type="hip")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
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
