"""Batch correctness tester for 'dense_int16_bias_int32' kernels with two-phase
parallelism."""

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


def reference_dense_int16(
    A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Reference implementation: int16 GEMM + int32 bias.
    A: [M, K], B: [K, N], bias: [N]
    Returns: [M, N] int32
    """
    return torch.matmul(A, B).to(torch.int32) + bias.unsqueeze(0)


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one dense_int16 kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[Dense] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[Dense] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    if success:
        os.remove(temp_file)
        return config, True, so_path
    else:
        os.remove(temp_file)
        return config, False, f"[Dense] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled dense_int16_bias_int32 kernel."""
    try:
        M, N, K = config["args"]
        file_name = config["file"]
        op_name = config["op_name"]
        # Generate inputs
        A = torch.randint(-10, 10, (M, K), dtype=torch.int16)
        B = torch.randint(-10, 10, (K, N), dtype=torch.int16)
        bias = torch.randint(-100, 100, (N,), dtype=torch.int32)

        A = A.contiguous()
        B = B.contiguous()
        bias = bias.contiguous()

        # Reference output
        ref = reference_dense_int16(A, B, bias)

        # Prepare output buffer
        output = torch.zeros((M, N), dtype=torch.int32).contiguous()

        # Get pointers
        A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_int16))
        B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_int16))
        bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_int32))
        output_ptr = ctypes.cast(
            output.data_ptr(), ctypes.POINTER(ctypes.c_int32)
        )

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[Dense] Function 'dense' not found in {so_path}"

        func.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        func.restype = None

        # Call kernel
        func(A_ptr, B_ptr, bias_ptr, output_ptr)

        # Compare
        abs_diff = torch.abs(output - ref)
        max_diff = abs_diff.max().item()

        if max_diff <= 1:
            return True, f"[Dense] ✅ {file_name}| Max diff: {max_diff}"
        else:
            return (
                False,
                f"[Dense] FAILED❌: {file_name} | Max diff: {max_diff}",
            )

    except Exception as e:
        return False, f"[Dense] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[DenseInt16] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[Dense] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[Dense] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[Dense] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[Dense] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[Dense] Failed to delete {so_path}: {e}")
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
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid 'dense' kernels found in config.")
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
        logger.info(f"🎉 All {total} Dense Int16 tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} Dense Int16 tests failed.")
        exit(1)
