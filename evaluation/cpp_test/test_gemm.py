"""Parallel tester for MatMul kernels on CPU/GPU/MLU.

Supports two-phase pipeline:
1. Parallel compilation
2. Parallel correctness testing
"""

import argparse
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import torch

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation
from evaluation.utils import parse_op_json

# ------------------ Logging setup ------------------
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


# ------------------ Compilation ------------------
def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one gemm kernel and return (config, success, so_path or error
    msg)."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))
    tmp_path = os.path.join(
        source_dir, file_name.replace(".cpp", "_patched.cpp")
    )

    if not os.path.isfile(file_path):
        return config, False, f"[GEMM] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(tmp_path, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[GEMM] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, tmp_path)

    try:
        os.remove(tmp_path)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[GEMM] Compile failed {file_name}: {msg}"


# ------------------ Testing ------------------
def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled MatMul kernel."""
    try:
        M, K, N = config["args"]
        file_name = config["file"]
        op_name = config["op_name"]

        # Generate input
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        ref = torch.matmul(A, B)

        # Prepare C buffer
        C = torch.zeros((M, N), dtype=torch.float32)

        # Convert to ctypes
        A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C
        ]
        func.restype = None

        # Call kernel
        func(A_ptr, B_ptr, C_ptr)

        # Compare
        try:
            torch.allclose(
                C,
                ref,
                rtol=1e-3,
                atol=1e-3,
                equal_nan=True,
            )
            return True, f"[GEMM] PASSED✅: {file_name}"
        except Exception as e:
            return False, f"[GEMM] FAILED❌: {file_name} | {e}"

    except Exception as e:
        return False, f"[GEMM] Exception in test {config['file']}: {e}"


# ------------------ Pipeline ------------------
def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[GEMM] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[GEMM] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[GEMM] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[GEMM] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[GEMM] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[GEMM] Failed to delete {so_path}: {e}")
    return results


# ------------------ Main ------------------
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
        "--jobs", type=int, default=4, help="Number of parallel jobs"
    )
    args = parser.parse_args()

    # Load config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid 'gemm' kernels found.")
        exit(0)

    # Run test pipeline
    results = run_tests(configs, args.source_dir, args.target, args.jobs)

    # Summarize results
    passed = sum(1 for r in results if r[0])
    total = len(results)
    for success, msg in results:
        if success:
            logger.info(msg)
        else:
            logger.error(msg)

    if passed == total:
        logger.info(f"🎉 All {total} MatMul tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} MatMul tests failed.")
        exit(1)
