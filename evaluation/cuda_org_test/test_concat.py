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


def concat_reference(tensors, axis):
    return torch.cat(tensors, dim=axis)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    axis = config["axis"]

    N, C, H, W = config["args"]

    input1 = torch.rand(N, C, H, W, dtype=torch.float32)
    input2 = torch.rand(N, C, H, W, dtype=torch.float32)

    expected = concat_reference([input1, input2], axis=axis)

    input1_ptr = (
        input1.flatten().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    input2_ptr = (
        input2.flatten().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    output_flat = torch.zeros_like(expected.flatten())
    output_ptr = output_flat.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    lib = ctypes.CDLL(so_path)
    kernel_func = getattr(lib, op_name + "_kernel")
    kernel_func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
    kernel_func.restype = None

    kernel_func(input1_ptr, input2_ptr, output_ptr)

    result_reshaped = output_flat.reshape(expected.shape)

    is_correct = torch.allclose(
        result_reshaped, expected, rtol=1e-4, atol=1e-4
    )

    return verify_torch_tensor(result_reshaped, expected, op_name=op_name)


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
    configs = parse_op_json(args.config, args.name, file_type="cu")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
