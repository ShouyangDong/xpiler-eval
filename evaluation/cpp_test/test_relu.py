"""Batch correctness tester for ReLU kernels with parallel compilation and
testing."""

import argparse
import ctypes
import logging
from typing import Tuple

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


def reference_relu(input_array: np.ndarray) -> np.ndarray:
    """Reference ReLU using PyTorch."""
    x = torch.from_numpy(input_array)
    return torch.nn.functional.relu(x).numpy()


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled ReLU kernel."""
    try:
        shape = config["args"]
        file_name = config["file"]
        dtype_str = config.get("dtype", "float32")
        op_name = config["op_name"]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[RELU] Function '{func_name}' not found in {so_path}",
            )

        # Set function signature
        ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        np_dtype = np.float32 if dtype_str == "float32" else np.float16

        func.argtypes = [
            ctypes.POINTER(ctype),  # input
            ctypes.POINTER(ctype),  # output
        ]
        func.restype = None

        # Generate input data
        np.random.seed(1234)
        input_array = np.random.uniform(-5.0, 5.0, size=shape).astype(np_dtype)
        expected = reference_relu(input_array)

        # Flatten and get pointers
        input_flat = input_array.flatten()
        output_flat = np.zeros_like(input_flat)

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctype))
        output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctype))

        # Call kernel
        func(input_ptr, output_ptr)

        # Reshape result
        result_reshaped = output_flat.reshape(shape)

        # Compare
        try:
            rtol, atol = (
                (1e-3, 1e-3) if dtype_str == "float32" else (1e-2, 1e-2)
            )
            np.testing.assert_allclose(
                result_reshaped,
                expected,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
                err_msg=f"[RELU] {file_name} failed",
            )
            max_abs_err = np.max(np.abs(result_reshaped - expected))
            return (
                True,
                f"[RELU] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except AssertionError as e:
            return (
                False,
                f"[RELU] FAILED❌: {file_name} | {str(e).splitlines()[0]}",
            )

    except Exception as e:
        return False, f"[RELU] Exception in test {file_name}: {str(e)}"


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
        logger.warning("No valid 'relu' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    passed = sum(1 for r in results if r[0])
    total = len(results)

    for success, msg in results:
        if success:
            logger.info(msg)
        else:
            logger.error(msg)

    # Final summary
    if passed == total:
        logger.info(f"🎉 All {total} ReLU tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} ReLU tests failed.")
        exit(1)
