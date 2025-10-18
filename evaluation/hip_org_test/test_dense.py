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
    shape = config["args"]
    op_name = config["op_name"]
    batch_size, in_features, out_features = shape
    # Generate input and parameters
    x = torch.randn(
        batch_size,
        in_features,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    weight = torch.randn(
        out_features,
        in_features,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    bias = torch.randn(
        out_features, dtype=torch.float32, device=torch.device("cuda")
    )

    # Reference: PyTorch Linear forward
    y_torch = torch.nn.functional.linear(
        x, weight, bias
    )  # shape: (batch_size, out_features)

    # Move reference result to CPU for comparison
    y_torch_cpu = y_torch.cpu().contiguous()

    # Host tensors for kernel input
    x_host = x.cpu().contiguous()
    weight_host = weight.t().cpu().contiguous()
    bias_host = bias.cpu().contiguous()

    # Output buffer (CPU)
    y_kernel = torch.zeros(
        batch_size, out_features, dtype=torch.float32
    ).contiguous()

    # Get raw pointers
    x_ptr = ctypes.cast(x_host.data_ptr(), ctypes.POINTER(ctypes.c_float))
    weight_ptr = ctypes.cast(
        weight_host.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    bias_ptr = ctypes.cast(
        bias_host.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    y_ptr = ctypes.cast(y_kernel.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Load shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")

    # Define function signature
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input (x)
        ctypes.POINTER(ctypes.c_float),  # weight (W)
        ctypes.POINTER(ctypes.c_float),  # bias (b)
        ctypes.POINTER(ctypes.c_float),  # output (y)
        ctypes.c_int,  # batch_size
        ctypes.c_int,  # in_features
        ctypes.c_int,  # out_features
    ]
    function.restype = None

    # Call the Dense kernel
    function(
        x_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        batch_size,
        in_features,
        out_features,
    )
    return verify_torch_tensor(y_kernel, y_torch_cpu, op_name)


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
    configs = parse_op_json(args.config, args.name, file_type="hip")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
