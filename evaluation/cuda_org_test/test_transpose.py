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
    # ✅ Get the actual shape and perm from config
    input_shape, perm = config["args"], config["perm"]
    op_name = config["op_name"]
    output_shape = [input_shape[i] for i in perm]  # permuted shape

    # ✅ Generate input tensor
    input_tensor = torch.rand(
        input_shape, dtype=torch.float32, requires_grad=False
    )
    input_flat = input_tensor.numpy()  # C-order
    input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # ✅ Reference：PyTorch permute
    expected = input_tensor.permute(*perm).contiguous()
    expected_flat = expected.numpy()

    # ✅ output buffer
    output_numel = expected_flat.size
    result_array = (ctypes.c_float * output_numel)()  # 分配空间

    # load shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    kernel_func = getattr(lib, op_name + "_kernel")

    rank = len(input_shape)

    if rank not in [2, 3, 4]:
        raise NotImplementedError(
            f"Rank {rank} not supported. Only 2D, 3D, and 4D are supported."
        )

    # Construct argtypes: [float*, float*, d0, d1, ..., p0, p1, ...]
    argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    # Add shape dim (d0, d1, ...)
    argtypes += [ctypes.c_int] * rank
    # Add perm dim (p0, p1, ...)
    argtypes += [ctypes.c_int] * rank

    kernel_func.argtypes = argtypes

    # ✅ Construct input arguments
    args_list = [input_ptr, result_array] + input_shape + perm

    kernel_func(*args_list)

    # ✅ Get output并 reshape
    computed_flat = torch.tensor(
        [result_array[i] for i in range(output_numel)]
    )
    computed_tensor = computed_flat.view(output_shape)

    # ✅ verification
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
