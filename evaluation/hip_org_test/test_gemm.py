import argparse
import ctypes
import logging
from typing import Tuple

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


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    M, K, N = config["args"]
    torch.manual_seed(1234)
    A = torch.randn(M, K, dtype=torch.float32, device=torch.device("cuda"))
    x = torch.randn(K, N, dtype=torch.float32, device=torch.device("cuda"))

    # Reference output using PyTorch
    y_torch = torch.matmul(A, x).cpu()  # Move to CPU for comparison

    # Output buffer (on CPU, to be filled by the kernel if needed)
    y_kernel = torch.zeros(
        M, N, dtype=torch.float32, device="cpu"
    ).contiguous()

    # --------------------------------------------------
    # 5. Load the compiled shared library
    # --------------------------------------------------

    lib = ctypes.CDLL(so_path)
    # Assume kernel function name is: {prefix}_kernel (e.g., gemm_kernel)
    kernel_func = getattr(lib, op_name + "_kernel")

    # Define argument types
    # Inputs A, x, y are raw pointers (void*), followed by M, K, N
    kernel_func.argtypes = [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # y (output pointer, could be CPU or GPU)
        ctypes.c_int,  # M
        ctypes.c_int,  # K
        ctypes.c_int,  # N
    ]
    kernel_func.restype = None

    # --------------------------------------------------
    # 6. Call the HIP kernel
    # --------------------------------------------------
    # Use .data_ptr() to get raw GPU/CPU memory address (as int,
    # auto-converted by ctypes)
    kernel_func(A.data_ptr(), x.data_ptr(), y_kernel.data_ptr(), M, K, N)
    # --------------------------------------------------
    # 7. Verify correctness
    # --------------------------------------------------
    if torch.allclose(y_kernel, y_torch, rtol=1e-3, atol=1e-3, equal_nan=True):
        return True, f"[ADD] PASSED✅: {config['file']}"
    else:
        return False, f"[ADD] FAILED❌: {config['file']} (mismatch)"


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
    configs = parse_op_json(args.config, args.name, file_type="hip")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
