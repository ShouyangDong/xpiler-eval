"""Batch correctness tester for depthwise_conv2d kernels with two-phase
parallelism."""

import argparse
import ctypes
import logging
from typing import Tuple

import torch
import torch.nn.functional as F

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


def depthwise_conv2d(input, w):
    """Torch depthwise conv2d with SAME padding."""
    x = input.permute(2, 0, 1).unsqueeze(0).float()
    weight = w.permute(2, 0, 1).unsqueeze(1).float()
    y = F.conv2d(x, weight, stride=1, padding=0, groups=x.shape[1])
    return y.squeeze(0).permute(1, 2, 0)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled depthwise_conv2d kernel."""
    input_height, kernel_size, input_channels = config["args"][:3]

    op_name = config["op_name"]
    # Define the input tensor, kernel, and parameters
    input_tensor = torch.rand(
        input_height, input_height, input_channels, dtype=torch.float32
    )
    kernel = torch.rand(
        kernel_size, kernel_size, input_channels, dtype=torch.float32
    )

    # Calculate the output tensor shape
    output_height = input_height - kernel_size + 1
    output_width = input_height - kernel_size + 1

    # Create an empty output tensor
    output_ctypes = torch.zeros(
        (output_height, output_width, input_channels), dtype=torch.float32
    )

    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_tensor.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    kernel_ptr = kernel.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    # Calculate the result using numpy for comparison
    output_np = depthwise_conv2d(input_tensor, kernel)

    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[{op_name}] Function 'depthwiseconv' not found in {so_path}",
        )

    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    func.restype = None

    # Call kernel
    func(input_ptr, kernel_ptr, output_ptr)
    return verify_torch_tensor(output_ctypes, output_np, op_name)


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
        logger.warning("No valid 'depthwiseconv' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
