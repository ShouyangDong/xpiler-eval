import argparse
import ctypes
import logging
import os
from typing import Tuple

import torch

from evaluation.utils import (
    avgpool_np,
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
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    shape = config["args"][:4]
    kernel_stride = shape[4:]
    config["file"]
    input_array = torch.randn(*shape, device="cpu")
    # Calculate the result using numpy for comparison
    output_np = avgpool_np(input_array, kernel_stride)
    output_array = torch.zeros(output_np.shape, dtype=torch.float32)
    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(
        input_ptr,
        output_ptr,
        shape[0],
        shape[3],
        shape[1],
        kernel_stride[0],
        kernel_stride[2],
    )
    # Check if the results match
    if torch.allclose(
        output_array,
        output_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    ):
        return True, f"[ADD] PASSED✅: {config['file']}"
    else:
        return False, f"[ADD] FAILED❌: {config['file']} (mismatch)"


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
