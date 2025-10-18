"""Batch correctness tester for Sum Pooling kernels."""

import argparse
import ctypes
import logging
import os
from typing import Tuple

import numpy as np
import torch

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import parse_op_json
from evaluation.utils import run_cpp_compilation as run_compilation
from evaluation.utils import run_tests, sumpool_np, verify_torch_tensor

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


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one sumpool kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[{op_name}] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[{op_name}] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[{op_name}] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled sumpool kernel."""
    config["file"]
    input_shape = config["args"][:4]
    kernel_size = config["args"][4:6]
    stride = config["args"][6:8]
    dtype_str = config["dtype"]
    op_name = config["op_name"]

    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return (
            False,
            f"[{op_name}] Function '{op_name}' not found in {so_path}",
        )

    # Determine C type and numpy dtype
    ctype_float = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
    np_dtype = np.float32 if dtype_str == "float32" else np.float16
    torch_dtype = torch.float32 if dtype_str == "float32" else torch.float16

    # Set function signature
    func.argtypes = [
        ctypes.POINTER(ctype_float),  # input
        ctypes.POINTER(ctype_float),  # output
    ]
    func.restype = None

    # Generate input
    input_np = np.random.rand(*input_shape).astype(np_dtype)
    input_torch = torch.from_numpy(input_np).to(dtype=torch_dtype)

    # Compute reference output
    expected_output = sumpool_np(input_torch, kernel_size + stride)
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
        torch.from_numpy(output_np).view(output_shape).to(dtype=torch_dtype)
    )
    return verify_torch_tensor(output_torch, expected_output, op_name)


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
        logger.warning("⚠️ No valid 'sumpool' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        "sumpool",
        args.name,
        configs,
        args.source_dir,
        args.target,
        num_workers=args.jobs,
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
