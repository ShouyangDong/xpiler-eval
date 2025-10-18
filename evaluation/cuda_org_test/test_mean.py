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

    input_shape, reduce_dim = config["args"], config["axis"]

    input_tensor = torch.rand(
        input_shape, dtype=torch.float32, requires_grad=False
    )
    input_ptr = input_tensor.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # PyTorch mean
    expected = torch.mean(input_tensor, dim=reduce_dim).contiguous()
    expected_flat = expected.numpy()
    output_shape = expected.shape
    output_numel = expected_flat.size

    # output buffer
    result_array = (ctypes.c_float * output_numel)()
    # load shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    kernel_func = getattr(lib, op_name + "_kernel")

    # 动态Construct argtypes：支持任意 rank 的 shape 和 reduce_dim
    rank = len(input_shape)

    if rank not in [2, 3, 4]:
        raise NotImplementedError(
            f"Rank {rank} not supported. Only 2D/3D/4D supported."
        )

    # Function  signature：void mean(float* input, float* output, int d0, ...,
    # int reduce_dim)
    argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    # Add每个dim大小
    argtypes += [ctypes.c_int] * rank
    # Add reduce_dim 参数
    argtypes.append(ctypes.c_int)

    kernel_func.argtypes = argtypes
    kernel_func.restype = None

    # Construct input arguments
    args_list = [input_ptr, result_array] + input_shape + [reduce_dim]

    # invoke kernel

    kernel_func(*args_list)

    # Get output
    computed_flat = torch.tensor(
        [result_array[i] for i in range(output_numel)]
    )
    computed_tensor = computed_flat.view(output_shape)

    # verification
    return verify_torch_tensor(computed_tensor, expected, op_name=op_name)


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
