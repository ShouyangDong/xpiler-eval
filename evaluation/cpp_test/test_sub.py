"""Batch correctness tester for Element-wise Subtraction (A - B) kernels."""

import argparse
import ctypes
import logging
from typing import Tuple

import numpy as np
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


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Golden reference: A - B using PyTorch."""
    return torch.sub(A, B)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled sub kernel."""
    try:
        file_name = config["file"]
        shape = config["args"]
        dtype_str = config["dtype"]
        op_name = config["op_name"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[{op_name}] Function '{op_name}' not found in {so_path}"

        # Determine C type and numpy dtype
        ctype_float = (
            ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        )
        np_dtype = np.float32 if dtype_str == "float32" else np.float16
        torch_dtype = (
            torch.float32 if dtype_str == "float32" else torch.float16
        )

        # Set function signature: void sub(float* A, float* B, float* out)
        func.argtypes = [
            ctypes.POINTER(ctype_float),  # input A
            ctypes.POINTER(ctype_float),  # input B
            ctypes.POINTER(ctype_float),  # output
        ]
        func.restype = None

        # Generate input
        A_np = np.random.rand(*shape).astype(np_dtype)
        B_np = np.random.rand(*shape).astype(np_dtype)
        A_torch = torch.from_numpy(A_np).to(dtype=torch_dtype)
        B_torch = torch.from_numpy(B_np).to(dtype=torch_dtype)
        expected_output = ref_program(A_torch, B_torch)

        # Prepare output array
        output_np = np.zeros_like(A_np)
        A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctype_float))
        B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctype_float))
        output_ptr = output_np.ctypes.data_as(ctypes.POINTER(ctype_float))

        # Call kernel
        func(A_ptr, B_ptr, output_ptr)

        # Convert result back to torch tensor
        output_torch = torch.from_numpy(output_np).to(dtype=torch_dtype)

        # Compare
        rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (5e-2, 5e-2)
        if torch.allclose(
            output_torch, expected_output, rtol=rtol, atol=atol, equal_nan=True
        ):
            max_error = (output_torch - expected_output).abs().max().item()
            return (
                True,
                f"[{op_name}] ✅ {file_name}| Max error: {max_error:.2e}",
            )
        else:
            max_error = (output_torch - expected_output).abs().max().item()
            return (
                False,
                f"[{op_name}] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[{op_name}] Exception in test {file_name}: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (CPU)")
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the operator to test (used to filter configs).",
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
    )
    parser.add_argument(
        "--source_dir", default="./", help="Directory containing .cpp files"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("⚠️ No valid 'sub' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
