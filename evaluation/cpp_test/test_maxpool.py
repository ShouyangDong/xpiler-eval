#!/usr/bin/env python3
"""Correctness tester for maxpool kernels with config support and error handling."""

import argparse
import ctypes
import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, Tuple, Optional

import torch

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import maxpool_np
from evaluation.utils import run_cpp_compilation as run_compilation


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def reference_maxpool(input_tensor: torch.Tensor, kernel: int, stride: int) -> torch.Tensor:
    """Reference maxpool using custom numpy function (or PyTorch)."""
    return torch.from_numpy(maxpool_np(input_tensor.numpy(), [kernel, stride]))


def parse_config(args_config: str) -> Dict:
    """Parse config from file or JSON string."""
    if os.path.exists(args_config) and args_config.endswith(".json"):
        with open(args_config, "r") as f:
            return json.load(f)
    else:
        return json.loads(args_config)


def parse_filename(file_name: str) -> Dict[str, int]:
    """
    Parse filename: maxpool_H_W_K_S.cpp → H, W, K, S
    Returns: dict with shape=[H, W], kernel, stride
    """
    base = os.path.splitext(file_name)[0]
    parts = base.split("_")
    if len(parts) != 6 or parts[0] != "maxpool":
        raise ValueError(f"Invalid maxpool filename: {file_name}")
    H, W, K, S = map(int, parts[1:5])
    return {
        "shape": [H, W],
        "kernel": K,
        "stride": S,
        "output_shape": [(H - K) // S + 1, (W - K) // S + 1]
    }


def compile_cpp_to_so(cpp_path: str, so_path: str) -> Tuple[bool, str]:
    """Compile .cpp to .so with macro injection."""
    try:
        with open(cpp_path, "r") as f:
            code = f.read()
        code = macro + code

        # Use secure temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as tmp:
            tmp.write(code)
            temp_cpp = tmp.name

        success, msg = run_compilation(so_path, temp_cpp)

        # Clean up
        os.remove(temp_cpp)
        if not success:
            return False, f"Compilation failed: {msg}"
        return True, so_path

    except Exception as e:
        return False, f"Compile error: {e}"


def test_maxpool_kernel(cpp_file: str, config: Dict, target: str) -> bool:
    """Test one maxpool kernel."""
    try:
        # Prefer config, fallback to filename
        shape = config.get("args")
        kernel = config.get("kernel_size")
        stride = config.get("stride")

        if not shape or not kernel or not stride:
            parsed = parse_filename(os.path.basename(cpp_file))
            shape = parsed["shape"]
            kernel = parsed["kernel"]
            stride = parsed["stride"]

        H, W = shape
        KH, KW = kernel if isinstance(kernel, (list, tuple)) else (kernel, kernel)
        SH, SW = stride if isinstance(stride, (list, tuple)) else (stride, stride)

        logger.info(f"Testing maxpool on {cpp_file} | shape={shape}, kernel=({KH},{KW}), stride=({SH},{SW})")

        # Generate input
        input_tensor = torch.randn(H, W, dtype=torch.float32, device="cpu") * 10
        expected = reference_maxpool(input_tensor, KH, SH)  # assuming square kernel

        # Prepare output buffer
        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1
        output_tensor = torch.zeros(OH, OW, dtype=torch.float32)

        # Get pointers
        input_ptr = input_tensor.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_tensor.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Compile
        so_file = cpp_file.replace(".cpp", ".so")
        success, msg = compile_cpp_to_so(cpp_file, so_file)
        if not success:
            logger.error(msg)
            return False

        # Load and call
        lib = ctypes.CDLL(so_file)
        func = getattr(lib, "maxpool", None)
        if not func:
            logger.error(f"Function 'maxpool' not found in {so_file}")
            return False

        func.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        func.restype = None

        logger.debug(f"Calling maxpool kernel: {cpp_file}")
        func(input_ptr, output_ptr)

        # Verify
        try:
            torch.testing.assert_close(
                output_tensor, expected,
                rtol=1e-3, atol=1e-3,
                check_dtype=True,
                msg=lambda m: f"Verification failed: {m}"
            )
            max_err = (output_tensor - expected).abs().max().item()
            logger.info(f"✅ PASSED: {cpp_file} | Max error: {max_err:.2e}")
            return True
        except Exception as e:
            logger.error(f"❌ FAILED: {cpp_file} | {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Exception during test {cpp_file}: {e}")
        return False
    finally:
        # Optional: clean up .so
        so_file = cpp_file.replace(".cpp", ".so")
        if os.path.exists(so_file):
            try:
                os.remove(so_file)
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MaxPool2D kernel")
    parser.add_argument("--file", required=True, help="Path to .cpp kernel file")
    parser.add_argument("--config", required=True, help="Path to JSON config or JSON string")
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform"
    )

    args = parser.parse_args()

    # Parse config
    try:
        config = parse_config(args.config)
    except Exception as e:
        logger.error(f"Failed to parse config: {e}")
        exit(1)

    # Run test
    success = test_maxpool_kernel(args.file, config, args.target)

    if success:
        logger.info("🎉 Verification successful!")
        exit(0)
    else:
        logger.error("❌ Verification failed!")
        exit(1)