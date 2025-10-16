"""Batch correctness tester for Mean reduction kernels with parallel
compilation and testing."""

import argparse
import ctypes
import logging
from typing import Tuple

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


def reference_mean(
    input_tensor: torch.Tensor, reduce_dim: int
) -> torch.Tensor:
    """Reference mean using PyTorch."""
    return torch.mean(input_tensor, dim=reduce_dim).contiguous()


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled mean kernel."""
    try:
        file_name = config["file"]
        shape = config["args"]
        reduce_dim = config["axis"]
        op_name = config["op_name"]
        dtype_str = config["dtype"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[MEAN] Function '{op_name}' not found in {so_path}"

        # Determine C type
        ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        torch_dtype = (
            torch.float32 if dtype_str == "float32" else torch.float16
        )

        # Set function signature
        rank = len(shape)
        if rank not in [2, 3, 4]:
            return False, f"[MEAN] Rank {rank} not supported (only 2D/3D/4D)"

        argtypes = [
            ctypes.POINTER(ctype),  # input
            ctypes.POINTER(ctype),  # output
        ]

        func.argtypes = argtypes
        func.restype = None

        # Generate input
        input_tensor = torch.rand(shape, dtype=torch_dtype)
        expected = reference_mean(input_tensor, reduce_dim)
        output_shape = expected.shape
        output_numel = expected.numel()

        # Flatten for C
        input_flat = input_tensor.flatten().numpy()
        result_array = (ctype * output_numel)()  # allocate C array

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctype))

        # Build args list
        args_list = [input_ptr, result_array] + list(shape) + [reduce_dim]

        # Call kernel
        func(*args_list)

        # Convert result back
        computed_flat = torch.tensor(
            [result_array[i] for i in range(output_numel)], dtype=torch_dtype
        )
        computed = computed_flat.view(output_shape)

        # Compare
        rtol, atol = (1e-5, 1e-5) if dtype_str == "float32" else (1e-3, 1e-3)
        if torch.allclose(
            computed, expected, rtol=rtol, atol=atol, equal_nan=True
        ):
            max_error = (computed - expected).abs().max().item()
            return (
                True,
                f"[MEAN] ✅ {file_name}| Max error: {max_error:.2e}",
            )
        else:
            max_error = (computed - expected).abs().max().item()
            return (
                False,
                f"[MEAN] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[MEAN] Exception in test {file_name}: {str(e)}"


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
        logger.warning("⚠️ No valid 'mean' kernels found in config.")
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
        logger.info(f"🎉 All {total} Mean tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} Mean tests failed.")
        exit(1)
