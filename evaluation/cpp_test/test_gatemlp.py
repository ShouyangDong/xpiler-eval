"""Batch correctness tester for gatemlp kernels (int16 → int32) with two-phase
parallelism."""

import argparse
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import torch

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation
from evaluation.utils import parse_op_json

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


def reference_gatemlp(
    X: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Reference implementation using PyTorch.

    X, A, B: int16 arrays
    Output: O = SiLU(X @ A) * (X @ B), converted to int32
    """
    X_t = torch.from_numpy(X).to(torch.int16)
    A_t = torch.from_numpy(A).to(torch.int16)
    B_t = torch.from_numpy(B).to(torch.int16)

    # Forward: O = silu(X @ A) * (X @ B)
    C = torch.matmul(X_t, A_t).to(torch.float32)  # Promote to float for SiLU
    O2 = torch.matmul(X_t, B_t).to(torch.int32)  # Keep as int32
    O1 = torch.nn.functional.silu(C)
    O_fp32 = O1 * O2.float()
    return O_fp32.cpu().numpy()


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one GateMLP kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[GateMLP] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[GateMLP] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    if success:
        try:
            os.remove(temp_file)
        except BaseException:
            pass
        return config, True, so_path
    else:
        try:
            os.remove(temp_file)
        except BaseException:
            pass
        return config, False, f"[GateMLP] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled GateMLP kernel."""
    try:
        B, K, N = config["args"]
        file_name = config["file"]
        op_name = config["op_name"]
        # Generate inputs in int16 range [-10, 10]
        torch.manual_seed(1234)
        X_np = torch.randint(
            low=-10, high=11, size=(B, K), dtype=torch.int16
        ).numpy()
        A_np = torch.randint(
            low=-10, high=11, size=(K, N), dtype=torch.int16
        ).numpy()
        B_np = torch.randint(
            low=-10, high=11, size=(K, N), dtype=torch.int16
        ).numpy()

        # Ensure contiguous
        X_np = np.ascontiguousarray(X_np, dtype=np.int16)
        A_np = np.ascontiguousarray(A_np, dtype=np.int16)
        B_np = np.ascontiguousarray(B_np, dtype=np.int16)

        # Reference output
        ref = reference_gatemlp(X_np, A_np, B_np)

        # Prepare output buffer: float32 to hold int32 values
        output = np.zeros((B, N), dtype=np.float32)
        output = np.ascontiguousarray(output)

        # Get pointers
        X_ptr = X_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        O_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[GateMLP] Function 'gatemlp' not found in {so_path}",
            )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_float),  # int32 stored as float
        ]
        func.restype = None

        # Call kernel
        func(X_ptr, A_ptr, B_ptr, O_ptr)

        # Compare: allow ±2 error due to integer rounding
        diff = np.abs(output - ref)
        max_diff = diff.max()
        mean_diff = diff.mean()

        if max_diff <= 2.0:
            return (
                True,
                f"[GateMLP] ✅ {file_name}| Max diff: {max_diff:.2f}",
            )
        else:
            return (
                False,
                f"[GateMLP] FAILED❌: {file_name} | Max diff: {max_diff:.2f}, Mean: {mean_diff:.2f}",
            )

    except Exception as e:
        return False, f"[GateMLP] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[GateMLP] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[GateMLP] Phase 1/2: Compiling {len(configs)} kernels...")
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

    logger.info(
        f"[GateMLP] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[GateMLP] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
        )
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

        logger.debug("[GateMLP] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[GateMLP] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (CPU)")
    parser.add_argument(
        "--name", required=True, 
        help="Name of the operator to test (used to filter configs)."
    )
    parser.add_argument(
        "--config", required=True, 
        help="JSON string or path to config file"
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
        logger.warning("No valid 'gatemlp' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} GateMLP tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} GateMLP tests failed.")
        exit(1)
