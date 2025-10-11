#!/usr/bin/env python3
"""Batch correctness tester for 'matmul' kernels with two-phase parallelism."""

import argparse
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

import torch

from evaluation.macros import CPP_MACROS as macro
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


def matmul_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.matmul(A, B)


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: matmul_BATCH_I_J_K.cpp
    Returns shape and metadata.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 5 or parts[0] != "matmul":
            raise ValueError(f"Invalid matmul filename: {file_name}")
        batch, i, j, k = map(int, parts[1:5])
        return {
            "shape": [batch, i, j, k],
            "file": file_name,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """
    Compile one matmul kernel.
    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[MatMul] File not found: {file_path}"

    # Patch source with macros
    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[MatMul] Patch failed {file_name}: {e}"

    # Compile
    success, msg = run_compilation(so_path, temp_file)
    if success:
        os.remove(temp_file)
        return config, True, so_path
    else:
        os.remove(temp_file)
        return config, False, f"[MatMul] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """
    Run correctness test on compiled matmul kernel.
    """
    try:
        file_name = config["file"]
        batch, i, j, k = config["shape"]

        # Generate random input tensors
        A = torch.randn(batch, i, j, dtype=torch.float32)
        B = torch.randn(batch, j, k, dtype=torch.float32)

        # Golden reference
        expected = matmul_ref(A, B)

        # Flatten for C++ (row-major order)
        A_flat = A.flatten()
        B_flat = B.flatten()
        result_flat = torch.zeros(batch * i * k, dtype=torch.float32)

        # Get pointers
        A_ptr = A_flat.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B_flat.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = result_flat.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "matmul", None)
        if not func:
            return False, f"[MatMul] Function 'matmul' not found in {so_path}"

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C (output)
        ]
        func.restype = None

        # Call kernel
        func(A_ptr, B_ptr, output_ptr)

        # Reshape and compare
        result_reshaped = result_flat.reshape(batch, i, k)
        if torch.allclose(result_reshaped, expected, rtol=1e-3, atol=1e-3, equal_nan=True):
            return True, f"[MatMul] PASSED: {file_name}"
        else:
            max_error = (result_reshaped - expected).abs().max().item()
            return False, f"[MatMul] FAILED: {file_name} | Max error: {max_error:.2e}"

    except Exception as e:
        return False, f"[MatMul] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Phase 1: Compile all matmul kernels in parallel.
    Phase 2: Test only successful ones.
    """
    logger.info(f"[MatMul] Starting two-phase test for {len(configs)} kernels...")

    compiled_map = {}  # file -> so_path
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[MatMul] Phase 1/2: Compiling {len(configs)} kernels...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(compile_kernel, config, source_dir)
            for config in configs
        ]

        for future in as_completed(futures):
            config, success, msg = future.result()
            if success:
                compiled_map[config["file"]] = msg
            else:
                results.append((False, msg))

    logger.info(f"[MatMul] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed.")

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(f"[MatMul] Phase 2/2: Testing {len(compiled_map)} compiled kernels...")
        test_configs = [
            (config, compiled_map[config["file"]])
            for config in configs
            if config["file"] in compiled_map
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(test_kernel, config, so_path)
                for config, so_path in test_configs
            ]

            for future in as_completed(futures):
                results.append(future.result())

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON string or path to config file")
    parser.add_argument("--source_dir", default="./", help="Directory with .cpp files")
    parser.add_argument("--target", required=True, choices=["cuda", "hip", "mlu", "cpu"], help="Target platform")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    # Parse config
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
    else:
        try:
            configs = json.loads(args.config)
        except Exception as e:
            logger.error(f"Invalid config: {e}")
            exit(1)

    if isinstance(configs, dict):
        configs = [configs]

    # Filter only 'matmul' kernels
    matmul_configs = []
    for c in configs:
        file_name = c.get("file", "")
        if not file_name.endswith(".cpp"):
            continue
        if file_name.startswith("matmul"):
            try:
                parsed = parse_filename(file_name)
                c.update(parsed)
                matmul_configs.append(c)
            except Exception as e:
                logger.warning(f"Skipping invalid matmul config {file_name}: {e}")

    if not matmul_configs:
        logger.warning("No valid 'matmul' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(matmul_configs, args.source_dir, args.target, num_workers=args.jobs)

    # Log results
    passed = sum(1 for r in results if r[0])
    total = len(results)

    for success, msg in results:
        if success:
            logger.info(msg)
        else:
            logger.error(msg)

    # Final summary
    if passed == total:
        logger.info(f"🎉 All {total} matmul tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} matmul tests failed.")
        exit(1)