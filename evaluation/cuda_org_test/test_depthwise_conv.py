import argparse
import ctypes
import logging
import os
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
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    shape = config["args"]
    input_height, kernel_size, input_channels = shape[0], shape[1], shape[2]
    device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )

    input_tensor = torch.rand(
        (input_height, input_height, input_channels),
        dtype=torch.float32,
        device=device,
    )
    kernel = torch.rand(
        (kernel_size, kernel_size, input_channels),
        dtype=torch.float32,
        device=device,
    )

    output_height = input_height - kernel_size + 1
    output_width = input_height - kernel_size + 1
    output_ctypes = torch.zeros(
        (output_height, output_width, input_channels), dtype=torch.float32
    )

    input_np = input_tensor.cpu().contiguous().numpy()
    kernel_np = kernel.cpu().contiguous().numpy()
    input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_ptr = kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_ctypes.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_torch = depthwise_conv2d(input_tensor, kernel)

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None

    function(
        input_ptr,
        kernel_ptr,
        output_ptr,
        input_height,
        kernel_size,
        input_channels,
    )
    return verify_torch_tensor(
        output_ctypes, output_torch.cpu(), op_name=op_name
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
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
