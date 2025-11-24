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


def ref_program(X_fp16, A_fp16, B_fp16):
    """Golden reference using autocast to mimic real inference behavior.

    Inputs: fp16 tensors on CUDA
    Output: fp32 tensor on CUDA
    """
    with torch.amp.autocast(
        enabled=True, device_type="cuda", dtype=torch.float16
    ):
        O1 = torch.nn.functional.silu(torch.matmul(X_fp16, A_fp16))
        O2 = torch.matmul(X_fp16, B_fp16)
        O = O1 * O2
    return O.float()


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    batch, dim_k, dim_n = config["args"]
    lib = ctypes.CDLL(so_path)
    kernel_func = getattr(lib, op_name + "_kernel")
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),  # X (fp16)
        ctypes.POINTER(ctypes.c_uint16),  # A (fp16)
        ctypes.POINTER(ctypes.c_uint16),  # B (fp16)
        ctypes.POINTER(ctypes.c_float),  # O (fp32)
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel_func.restype = None
    torch.manual_seed(1234)
    device = torch.device("cuda")

    # Generate fp16 inputs
    X_fp16 = torch.ones(batch, dim_k, dtype=torch.float16, device=device) / 16
    A_fp16 = torch.ones(dim_k, dim_n, dtype=torch.float16, device=device) / 16
    B_fp16 = torch.ones(dim_k, dim_n, dtype=torch.float16, device=device) / 16

    # Reference output (fp32 on CUDA)
    O_ref = ref_program(X_fp16, A_fp16, B_fp16)

    # Output buffer: allocate fp32 tensor on CUDA
    O = torch.zeros(batch, dim_n, dtype=torch.float32, device=device)

    X_fp16 = X_fp16.contiguous()
    A_fp16 = A_fp16.contiguous()
    B_fp16 = B_fp16.contiguous()
    O = O.contiguous()

    X_ptr = ctypes.cast(X_fp16.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    A_ptr = ctypes.cast(A_fp16.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    B_ptr = ctypes.cast(B_fp16.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    O_ptr = ctypes.cast(O.data_ptr(), ctypes.POINTER(ctypes.c_float))
    kernel_func(X_ptr, A_ptr, B_ptr, O_ptr, batch, dim_k, dim_n)
    return verify_torch_tensor(O, O_ref, op_name=op_name)


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
