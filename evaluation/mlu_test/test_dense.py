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

    device = torch.device("cuda" if torch.mluda.is_available() else "cpu")

    x = torch.ones(batch_size, in_features, dtype=torch.float16, device=device)
    weight = torch.ones(
        in_features, out_features, dtype=torch.float16, device=device
    )
    bias = torch.zeros(out_features, dtype=torch.float32, device=device)

    with torch.amp.autocast(
        enabled=True, device_type="cuda", dtype=torch.float16
    ):
        matmul_out = x @ weight
        y_torch = matmul_out + bias
    y_torch = y_torch.float()  # ensure output is fp32 for comparison
    y_torch_cpu = y_torch.cpu().contiguous()

    x_host = x.cpu().contiguous()  # fp16

    weight_host = weight.cpu().contiguous()
    bias_host = bias.cpu().contiguous()  # fp32
    y_kernel = torch.zeros(
        batch_size, out_features, dtype=torch.float32
    ).contiguous()

    x_ptr = ctypes.cast(x_host.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    weight_ptr = ctypes.cast(
        weight_host.data_ptr(), ctypes.POINTER(ctypes.c_uint16)
    )
    bias_ptr = ctypes.cast(
        bias_host.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    y_ptr = ctypes.cast(y_kernel.data_ptr(), ctypes.POINTER(ctypes.c_float))

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    kernel_func = getattr(lib, op_name + "_kernel")

    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel_func.restype = None
    kernel_func(
        x_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        batch_size,
        in_features,
        out_features,
    )
    return verify_torch_tensor(y_kernel, y_torch_cpu, op_name=op_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (MLU)")
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the operator to test (used to filter configs).",
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
    )
    parser.add_argument(
        "--source_dir", default="./", help="Directory with .mlu files"
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
    configs = parse_op_json(args.config, args.name, file_type="mlu")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
