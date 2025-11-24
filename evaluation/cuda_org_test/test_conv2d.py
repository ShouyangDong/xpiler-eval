import argparse
import ctypes
import logging
import os
from typing import Tuple

import torch

from evaluation.utils import (
    conv2d_nhwc,
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
    op_name = config["op_name"]
    data_shape = config["args"][:4]
    kernel_shape = config["args"][4:8]
    stride_h = stride_w = config["args"][8]
    pad = config["args"][9]

    # generate data
    data_np = torch.rand(data_shape)
    kernel_np = torch.rand(kernel_shape)
    # cpu compute
    result_cpu = conv2d_nhwc(
        data_np,
        kernel_np,
        stride_h,
        pad,
    )

    # Load the shared library with the conv2d function

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Convert the matrices to contiguous memory for ctypes
    input_ptr = data_np.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_ptr = kernel_np.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    result_ctypes = torch.zeros(result_cpu.shape)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    # Call the function with the matrices and dimensions
    function(
        input_ptr,
        kernel_ptr,
        output_ptr,
        data_shape[0],
        data_shape[1],
        data_shape[3],
        kernel_shape[0],
        kernel_shape[1],
        stride_h,
    )
    # Check if the results match
    return verify_torch_tensor(result_ctypes, result_cpu, op_name=op_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (CUDA)")
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the operator to test (used to filter configs).",
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
    )
    parser.add_argument(
        "--source_dir", default="./", help="Directory with .cu files"
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
    configs = parse_op_json(args.config, args.name, file_type="cu")

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
