import argparse
import ctypes
import logging
import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
    verify_numpy_tensor,
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


def ref_program(q, k, v, causal=False):
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        mask = torch.triu(
            torch.ones(score.shape[-2], score.shape[-1]), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, torch.finfo(q.dtype).min)
        mask = mask.to(q.device, q.dtype)
        score = score + mask
    attn = F.softmax(score, dim=-1)
    output = torch.matmul(attn, v)
    return output


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    shape = config["args"]
    dtype = torch.float32

    query = torch.randn(shape).to(dtype).contiguous()
    key = torch.randn(shape).to(dtype).contiguous()
    value = torch.randn(shape).to(dtype).contiguous()
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    # Obtain function handle
    function = getattr(lib, op_name + "_kernel")
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Create the input array.
    expected_output = ref_program(query, key, value)
    # Create the output array.
    output_array = np.zeros_like(query.numpy())
    # Convert the input and output arrays to C pointer types.
    input_ptr_q = query.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_k = key.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_v = value.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Invoke the HIP kernel.
    function(
        input_ptr_q,
        input_ptr_k,
        input_ptr_v,
        output_ptr,
        np.prod(shape),
        np.prod(shape),
        np.prod(shape),
        np.prod(shape),
    )
    # Verification results
    return verify_numpy_tensor(
        output_array, expected_output.numpy(), op_name=op_name
    )


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
