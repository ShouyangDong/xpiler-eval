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
    op_name = config["op_name"]
    shape = config["args"]
    axis = config["axis"]
    # ✅ Generate input tensor
    A = torch.rand(shape, device="cpu", dtype=torch.float32)

    expected_tensor = torch.sum(A, dim=axis)  # shape: reduced
    expected_numpy = expected_tensor.numpy()
    expected_flat = expected_numpy.flatten()

    A_flat = A.numpy()
    A_ptr = A_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    output_size = expected_flat.size  # int
    result_array = (ctypes.c_float * output_size)()
    # load shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    kernel_func = getattr(lib, op_name + "_kernel")

    # ✅ Function  signature
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    kernel_func.restype = None

    # ✅ invoke kernel
    kernel_func(A_ptr, result_array)

    # ✅ Get output
    computed_array = [result_array[i] for i in range(output_size)]
    computed_tensor = torch.tensor(computed_array).view_as(
        torch.from_numpy(expected_numpy)
    )

    # ✅ verification
    return verify_torch_tensor(
        computed_tensor, expected_tensor, op_name=op_name
    )


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
