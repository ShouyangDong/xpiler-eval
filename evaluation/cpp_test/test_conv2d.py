"""Batch correctness tester for 'conv2d' kernels with two-phase parallelism."""

import argparse
import ctypes
import logging
from typing import Tuple

import torch

from evaluation.utils import (
    conv2d_nhwc,
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


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled conv2d kernel."""
    try:
        file_name = config["file"]
        data_shape = config["args"][:4]
        kernel_shape = config["args"][4:8]
        stride, pad = config["args"][8], config["args"][9]
        op_name = config["op_name"]
        # generate data
        data_np = torch.rand(data_shape)
        kernel_np = torch.rand(kernel_shape)
        # cpu compute
        result_cpu = conv2d_nhwc(
            data_np,
            kernel_np,
            stride,
            pad,
        )

        # Prepare output buffer
        result_ctypes = torch.zeros(result_cpu.shape)

        # Get pointers
        input_ptr = data_np.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        kernel_ptr = kernel_np.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = result_ctypes.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[Conv2D] Function 'conv2d' not found in {so_path}"

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # kernel
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        func.restype = None

        # Call kernel
        func(input_ptr, kernel_ptr, output_ptr)

        # Compare
        if torch.allclose(
            result_ctypes, result_cpu, rtol=1e-3, atol=1e-3, equal_nan=True
        ):
            return True, f"[Conv2D] PASSED✅: {file_name}"
        else:
            max_error = torch.max(torch.abs(result_ctypes - result_cpu)).item()
            return (
                False,
                f"[Conv2D] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[Conv2D] Exception in test {file_name}: {str(e)}"


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
        logger.warning("No valid 'conv2d' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log results
    log_test_results_and_exit(results, op_name=args.name)
