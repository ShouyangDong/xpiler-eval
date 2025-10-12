"""Batch correctness tester for min reduction kernels with parallel compilation
and testing."""

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


def reference_min(input: torch.Tensor, axis: int) -> torch.Tensor:
    """Reference min reduction using PyTorch."""
    return torch.min(input, dim=axis)[0]  # [0] = values, [1] = indices


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled min kernel."""
    try:
        shape = config["args"]
        axis = config["axis"]
        file_name = config["file"]
        dtype_str = config.get("dtype", "float32")
        op_name = config["op_name"]
        output_shape = [
            1 if i == axis else size for i, size in enumerate(shape)
        ]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[{op_name}] Function 'min' not found in {so_path}"

        # Set function signature
        ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        torch_dtype = (
            torch.float32 if dtype_str == "float32" else torch.float16
        )

        func.argtypes = [
            ctypes.POINTER(ctype),  # input
            ctypes.POINTER(ctype),  # output
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        input_tensor = torch.randn(*shape, dtype=torch_dtype) * 100
        expected = reference_min(input_tensor, axis)

        # Flatten and get pointers
        input_flat = input_tensor.flatten().numpy()
        output_flat = (
            torch.zeros(output_shape, dtype=torch_dtype).flatten().numpy()
        )

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctype))
        output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctype))

        # Call kernel
        func(input_ptr, output_ptr)

        # Reshape and compare
        result_reshaped = torch.from_numpy(output_flat).reshape(expected.shape)

        try:
            rtol, atol = (
                (1e-3, 1e-3) if dtype_str == "float32" else (1e-2, 5e-2)
            )
            torch.testing.assert_close(
                result_reshaped,
                expected,
                rtol=rtol,
                atol=atol,
                check_dtype=True,
                equal_nan=False,
                msg=lambda msg: f"[{op_name}] {file_name} failed: {msg}",
            )
            max_abs_err = (result_reshaped - expected).abs().max().item()
            return (
                True,
                f"[{op_name}] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[{op_name}] FAILED❌: {file_name} | {str(e)}"

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
        logger.warning("No valid 'min' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
