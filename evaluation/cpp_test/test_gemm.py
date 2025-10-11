"""Batch correctness tester for GEMM kernels with parallel compilation and
testing."""

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


def reference_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Reference GEMM using PyTorch."""
    return torch.matmul(A, B)


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: gemm_64_32_16.cpp
    Format: gemm_M_K_N.cpp
    Returns: dict with shape and metadata.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) != 4 or parts[0] != "gemm":
            raise ValueError(f"Invalid GEMM filename: {file_name}")
        M, K, N = map(int, parts[1:4])
        return {
            "file": file_name,
            "shape": [M, K, N],
            "M": M,
            "K": K,
            "N": N,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one GEMM kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[GEMM] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[GEMM] Patch failed {file_name}: {e}"

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
        return config, False, f"[GEMM] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled GEMM kernel."""
    try:
        M, K, N = config["args"]
        file_name = config["file"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "gemm", None)
        if not func:
            return False, f"[GEMM] Function 'gemm' not found in {so_path}"

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        C = torch.zeros(M, N, dtype=torch.float32)

        # Reference
        C_ref = reference_gemm(A, B)

        # Get pointers
        A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call kernel
        func(A_ptr, B_ptr, C_ptr)

        # Compare
        try:
            torch.testing.assert_close(
                C,
                C_ref,
                rtol=1e-3,
                atol=1e-3,
                check_dtype=True,
                equal_nan=False,
                msg=lambda msg: f"[GEMM] {file_name} failed: {msg}",
            )
            max_abs_err = (C - C_ref).abs().max().item()
            return (
                True,
                f"[GEMM] PASSED: {file_name} | Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[GEMM] FAILED: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[GEMM] Exception in test {file_name}: {str(e)}"


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

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GEMM kernels (CPU)")
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
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
    else:
        try:
            configs = json.loads(args.config)
        except Exception as e:
            logger.error(f"Invalid config JSON: {e}")
            exit(1)

    if isinstance(configs, dict):
        configs = [configs]

    # Filter and parse GEMM kernels
    configs = [c for c in configs if c.get("op_name") == "gemm"]
    gemm_configs = [
        {**config, "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp"}
        for config in configs
    ]


    if not gemm_configs:
        logger.warning("No valid 'gemm' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        gemm_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} GEMM tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} GEMM tests failed.")
        exit(1)
