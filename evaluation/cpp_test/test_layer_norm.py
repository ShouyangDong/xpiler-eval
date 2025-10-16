"""Batch correctness tester for LayerNorm kernels with parallel compilation and
testing."""

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


def ref_program(x, gamma, beta, eps=1e-5):
    # Using PyTorch to compute layer normalization
    x_tensor = torch.tensor(x)
    layer_norm = torch.nn.LayerNorm(
        x_tensor.size()[1:]
    )  # Initialize LayerNorm, maintaining dimensions.
    x_normalized = layer_norm(x_tensor)

    # Calculate output
    out = gamma * x_normalized + beta
    # Return the output in numpy format to maintain interface consistency.
    return out


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled LayerNorm kernel."""
    try:
        shape = config["args"]
        file_name = config["file"]
        op_name = config["op_name"]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[LAYERNORM] Function 'layernorm' not found in {so_path}",
            )

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # weight (gamma)
            ctypes.POINTER(ctypes.c_float),  # bias (beta)
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        # Create the input array.
        input_array = torch.randn(shape)
        gamma_array = torch.randn(shape[-1:])
        beta_array = torch.randn(shape[-1:])

        # Use the modified ref_program for layer normalization calculation.
        expected_output = ref_program(input_array, gamma_array, beta_array)

        # Create the output array.
        output_array = torch.zeros(shape)

        # Convert the input and output arrays to C pointer types.
        input_ptr = input_array.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        gamma_ptr = gamma_array.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        beta_ptr = beta_array.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = output_array.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # If the call to the C func can be preserved:
        func(input_ptr, gamma_ptr, beta_ptr, output_ptr)

        try:
            torch.allclose(
                output_array,
                expected_output,
                rtol=1e-3,
                atol=1e-3,
                equal_nan=False,
            )
            max_abs_err = (output_array - expected_output).abs().max().item()
            return (
                True,
                f"[LAYERNORM] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[LAYERNORM] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[LAYERNORM] Exception in test {file_name}: {str(e)}"


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
        logger.warning("No valid 'layernorm' kernels found in config.")
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
        logger.info(f"🎉 All {total} LayerNorm tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} LayerNorm tests failed.")
        exit(1)
