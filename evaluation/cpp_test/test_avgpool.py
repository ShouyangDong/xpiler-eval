"""Batch correctness tester for 'avgpool' kernels with two-phase
parallelism."""

import argparse
import ctypes
import logging
from typing import List, Tuple

import torch

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


def avgpool_ref(
    input_tensor: torch.Tensor, kernel_stride: List[int]
) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    kh, kw, sh, sw = kernel_stride
    input_nhwc = input_tensor.permute(0, 3, 1, 2)  # NCHW
    pool = torch.nn.AvgPool2d(kernel_size=(kh, kw), stride=(sh, sw))
    output_tensor = pool(input_nhwc)
    return output_tensor.permute(0, 2, 3, 1)  # back to NHWC


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled avgpool kernel."""
    file_name = config["file"]
    shape = config["args"][:4]
    kernel_stride = config["args"][4:8]
    kh, kw, sh, sw = kernel_stride
    op_name = config["op_name"]
    # Generate input
    input_tensor = torch.randn(*shape, dtype=torch.float32, device="cpu")
    ref_output = avgpool_ref(input_tensor, kernel_stride)

    # Prepare output buffer
    out_shape = [
        shape[0],
        (shape[1] - kh) // sh + 1,
        (shape[2] - kw) // sw + 1,
        shape[3],
    ]
    output_tensor = torch.zeros(out_shape, dtype=torch.float32)
    input_ptr = input_tensor.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_tensor.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Load and call kernel
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[AvgPool] Function 'avgpool' not found in {so_path}",
        )

    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    func.restype = None
    func(input_ptr, output_ptr)

    try:
        # Verify
        torch.allclose(
            output_tensor, ref, rtol=1e-3, atol=1e-3, equal_nan=True
        )
        return True, f"[{op_name}] PASSED✅: {config['file']}"

    except Exception as e:
        return False, f"[{op_name}] FAILED❌: {config['file']} (mismatch)"


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
        logger.warning("No valid 'avgpool' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log results
    log_test_results_and_exit(results, op_name=args.name)
