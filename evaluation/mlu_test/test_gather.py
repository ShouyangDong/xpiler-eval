import argparse
import ctypes
import logging
import random
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


def element_wise_gather(params, indices, axis=0):
    # invoke gather
    result = torch.gather(params, dim=axis, index=indices)
    return result


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    PARAMS_SHAPE = config["args"]
    AXIS = config["axis"]

    # === 2. Generate input tensors ===
    params = torch.randn(*PARAMS_SHAPE, dtype=torch.float32, device="cpu")
    axis_dim_size = params.size(AXIS)

    # Randomly generate indices_len: between 1 and axis_dim_size * 2
    min_len = 1
    max_len = axis_dim_size
    indices_len = random.randint(min_len, max_len)

    # Generate indices: include valid and out-of-bound (-1 or >= axis_dim_size)
    indices = torch.randint(
        low=0,
        high=axis_dim_size,
        size=(indices_len,),
        dtype=torch.int64,
        device="cpu",
    )

    output_shape = list(params.shape)
    output_shape[AXIS] = indices.size(0)  # M

    indices_expanded = indices.view(
        *[1 if i != AXIS else -1 for i in range(params.ndim)]
    )
    indices = indices_expanded.expand(*output_shape)
    # === 3. Golden reference using PyTorch ===
    expected = element_wise_gather(params, indices, axis=AXIS)

    # === 4. Prepare ctypes pointers ===

    def to_ptr(tensor, dtype):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(dtype))

    params_ptr = to_ptr(params, ctypes.c_float)
    indices_ptr = to_ptr(indices, ctypes.c_int64)
    result_ctypes = torch.zeros_like(expected, dtype=torch.float32)
    output_ptr = to_ptr(result_ctypes, ctypes.c_float)

    # === 8. Load shared library ===
    lib = ctypes.CDLL(so_path)

    # Look for 'gather' function
    kernel_func = getattr(lib, op_name + "_kernel")

    # === 9. Set function signature ===
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_int64),  # indices
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,  # N (number of indices)
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel_func.restype = None

    # === 10. Call C++ kernel ===
    kernel_func(params_ptr, indices_ptr, output_ptr, indices_len)

    # === 11. Verify result ===
    return verify_torch_tensor(result_ctypes, expected, op_name=op_name)


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
        args.name,
        configs,
        args.source_dir,
        __file__,
        args.target,
        num_workers=args.jobs,
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
