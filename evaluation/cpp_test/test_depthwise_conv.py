"""Batch correctness tester for depthwise_conv2d kernels with two-phase
parallelism."""

import argparse
import ctypes
import logging
from typing import Tuple

import numpy as np

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


def depthwise_conv2d(input, w):
    """Two-dimensional depthwise convolution.

    Uses SAME padding with 0s, a stride of 1 and no dilation. A single output
    channel is used per input channel (channel_multiplier=1).

    // before: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth)

    Returns a result with shape (height, width, in_depth).
    """
    height, width, in_depth = input.shape
    output_height = height - w.shape[0] + 1
    output_width = width - w.shape[1] + 1
    output = np.zeros((output_height, output_width, in_depth))
    for c in range(in_depth):
        # For each input channel separately, apply its corresponsing filter
        # to the input.
        for i in range(output_height):
            for j in range(output_width):
                for fi in range(w.shape[0]):
                    for fj in range(w.shape[1]):
                        w_element = w[fi, fj, c]
                        output[i, j, c] += input[i + fi, j + fj, c] * w_element
    return output


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled depthwise_conv2d kernel."""
    try:
        input_height, kernel_size, input_channels = config["args"][:3]
        file_name = config["file"]
        op_name = config["op_name"]
        # Define the input tensor, kernel, and parameters
        input_tensor = np.random.rand(
            input_height, input_height, input_channels
        ).astype(np.float32)
        kernel = np.random.rand(
            kernel_size, kernel_size, input_channels
        ).astype(np.float32)

        # Calculate the output tensor shape
        output_height = input_height - kernel_size + 1
        output_width = input_height - kernel_size + 1

        # Create an empty output tensor
        output_ctypes = np.zeros(
            (output_height, output_width, input_channels), dtype=np.float32
        )

        # Convert the arrays to contiguous memory for ctypes
        input_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_ctypes.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        # Calculate the result using numpy for comparison
        output_np = depthwise_conv2d(input_tensor, kernel).astype("float32")

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[DWConv] Function 'depthwiseconv' not found in {so_path}",
            )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        func.restype = None

        # Call kernel
        func(input_ptr, kernel_ptr, output_ptr)

        if np.allclose(
            output_ctypes, output_np, rtol=1e-03, atol=1e-03, equal_nan=True
        ):
            return (True, f"[DWConv] ✅ {file_name}")
        else:
            diff = np.abs(output_ctypes - output_np)
            max_diff = diff.max()
            return (
                False,
                f"[DWConv] FAILED❌: {file_name} | Max diff: {max_diff:.2e}",
            )

    except Exception as e:
        return False, f"[DWConv]❌ Exception in test {file_name}: {str(e)}"


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
