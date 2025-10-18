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


# --- Golden Reference: reduce min along axis ---
def reduce_min(input_tensor, axis):
    return torch.min(input_tensor, dim=axis)[0]  # 返回 values，忽略 indices


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    op_name = config["op_name"]
    shape = config["args"]
    axis = config["axis"]
    dtype_str = config.get("dtype", "float32")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # --- Generate input tensors on CPU (for ctypes) ---
    A = torch.randn(*shape, dtype=dtype, device="cpu") * 10
    expected = reduce_min(A, axis)

    # --- Flatten and get ctypes pointers ---
    A_flat = A.flatten().numpy()
    result_flat = torch.zeros_like(expected).flatten().numpy()

    ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
    A_ptr = A_flat.ctypes.data_as(ctypes.POINTER(ctype))
    out_ptr = result_flat.ctypes.data_as(ctypes.POINTER(ctype))

    # ========================================================
    # ✅ Step 2: Load the .so library
    # ========================================================

    lib = ctypes.CDLL(so_path)
    kernel_func = lib[op_name + "_kernel"]  # Get function by name

    # ========================================================
    # ✅ Step 3: Set function signature
    # ========================================================
    kernel_func.argtypes = [
        ctypes.POINTER(ctype),  # A
        ctypes.POINTER(ctype),  # out
    ]
    kernel_func.restype = None

    # ========================================================
    # ✅ Step 4: Call the kernel
    # ========================================================
    kernel_func(A_ptr, out_ptr)

    # ========================================================
    # ✅ Step 5: Reshape and verify
    # ========================================================
    result_tensor = (
        torch.from_numpy(result_flat).reshape(expected.shape).to("cpu")
    )

    # Verification
    rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (1e-2, 5e-2)
    if torch.allclose(result_tensor, expected, rtol=rtol, atol=atol):
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
