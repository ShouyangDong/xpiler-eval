"""Batch correctness tester for Reduction Sum kernels."""

import argparse
import ctypes
import logging
from typing import List, Tuple

import numpy as np
import torch

from evaluation.utils import parse_op_json, run_tests

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


def ref_program(A: torch.Tensor, axes: List[int]) -> torch.Tensor:
    """Golden reference: torch.sum(A, dim=axes)"""
    return torch.sum(A, dim=axes)


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled sum kernel."""
    try:
        file_name = config["file"]
        shape = config["args"]
        axes = config["axis"]
        dtype_str = config["dtype"]
        op_name = config["op_name"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[SUM] Function '{op_name}' not found in {so_path}"

        # Determine C type and numpy dtype
        ctype_float = (
            ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        )
        np_dtype = np.float32 if dtype_str == "float32" else np.float16
        torch_dtype = (
            torch.float32 if dtype_str == "float32" else torch.float16
        )

        # Set function signature: void sum(float* input, float* output)
        func.argtypes = [
            ctypes.POINTER(ctype_float),  # input
            ctypes.POINTER(ctype_float),  # output
        ]
        func.restype = None

        # Generate input
        input_np = np.random.rand(*shape).astype(np_dtype)
        input_torch = torch.from_numpy(input_np).to(dtype=torch_dtype)
        expected_output = ref_program(input_torch, axes)

        # Compute output size
        output_shape = expected_output.shape
        output_size = expected_output.numel()
        output_np = np.zeros(output_size, dtype=np_dtype)

        # Get pointers
        input_ptr = input_np.ctypes.data_as(ctypes.POINTER(ctype_float))
        output_ptr = output_np.ctypes.data_as(ctypes.POINTER(ctype_float))

        # Call kernel
        func(input_ptr, output_ptr)

        # Convert back to tensor
        output_torch = (
            torch.from_numpy(output_np)
            .view(output_shape)
            .to(dtype=torch_dtype)
        )

        # Compare
        rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (5e-2, 5e-2)
        if torch.allclose(
            output_torch, expected_output, rtol=rtol, atol=atol, equal_nan=True
        ):
            max_error = (output_torch - expected_output).abs().max().item()
            return (
                True,
                f"[SUM] ✅ {file_name}| Shape: {shape} → {list(output_shape)} | Max error: {max_error:.2e}",
            )
        else:
            max_error = (output_torch - expected_output).abs().max().item()
            return (
                False,
                f"[SUM] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[SUM] Exception in test {file_name}: {str(e)}"


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
        logger.warning("⚠️ No valid 'sum' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(result, op_name=args.name)