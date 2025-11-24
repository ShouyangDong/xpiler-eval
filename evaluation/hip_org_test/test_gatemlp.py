import argparse
import ctypes
import logging
import os
from typing import Tuple

import numpy as np
import torch

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


def ref_program(X, A, B):
    """Golden reference using PyTorch.

    Inputs: X, A, B as PyTorch tensors (on CUDA)
    Output: O = SiLU(X @ A) * (X @ B) as NumPy array
    """
    O1 = torch.nn.functional.silu(torch.matmul(X, A))
    O2 = torch.matmul(X, B)
    O = O1 * O2
    return O.cpu().numpy()


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    batch, dim_k, dim_n = config["args"]
    # -------------------------------
    # 2. Load kernel
    # -------------------------------
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    function = getattr(lib, op_name + "_kernel")
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # O
        ctypes.c_int,  # batch
        ctypes.c_int,  # dim
        ctypes.c_int,  # k
    ]
    function.restype = None

    # -------------------------------
    # ✅ 3. Generate random inputs
    # -------------------------------
    torch.manual_seed(1234)

    #
    X = torch.randn(batch, dim_k, dtype=torch.float32, device="cuda")
    A = torch.randn(dim_k, dim_n, dtype=torch.float32, device="cuda")
    B = torch.randn(dim_k, dim_n, dtype=torch.float32, device="cuda")

    # Calculate output
    O_ref = ref_program(X, A, B)

    X_np = X.cpu().numpy()
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()
    O = np.zeros_like(O_ref)  # output buffer

    # Get pointer
    X_ptr = X_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # -------------------------------
    # 4. invoke C++/CUDA kernel
    # -------------------------------
    function(X_ptr, A_ptr, B_ptr, O_ptr, batch, dim_k, dim_n)
    return verify_numpy_tensor(O, O_ref, op_name)


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
        "--source_dir", default="./", help="Directory with .hip files"
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
    configs = parse_op_json(args.config, args.name, file_type="hip")

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
