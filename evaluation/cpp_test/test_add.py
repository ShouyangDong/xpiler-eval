"""Batch correctness tester for 'add' kernels with two-phase parallelism."""

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


def add_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.add(A, B)


def patch_source(src_path: str, dst_path: str) -> bool:
    """Insert macros and save patched source."""
    try:
        with open(src_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(dst_path, "w") as f:
            f.write(code)
        return True
    except Exception as e:
        logger.error(f"Failed to patch {src_path}: {e}")
        return False


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one kernel.

    Returns: (config, success, message)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[ADD] Source not found: {file_path}"

    # Patch source
    if not patch_source(file_path, temp_file):
        return config, False, f"[ADD] Patch failed: {file_path}"

    # Compile
    success, msg = run_compilation(so_path, temp_file)
    if success:
        # Cleanup temp file only on success
        os.remove(temp_file)
        return config, True, so_path  # Return path to .so for next phase
    else:
        os.remove(temp_file)
        return config, False, f"[ADD] Compile failed: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    try:
        shape = config["args"]
        if isinstance(shape, int):
            shape = [shape]
        else:
            shape = [int(s) for s in shape]

        A = torch.rand(*shape, device="cpu")
        B = torch.rand(*shape, device="cpu")
        ref = add_ref(A, B)

        A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_tensor = torch.zeros_like(ref)
        out_ptr = output_tensor.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # Load and call kernel
        lib = ctypes.CDLL(so_path)
        func = lib.add
        func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
        func.restype = None
        func(A_ptr, B_ptr, out_ptr)

        # Verify
        if torch.allclose(
            output_tensor, ref, rtol=1e-3, atol=1e-3, equal_nan=True
        ):
            return True, f"[ADD] PASSED✅: {config['file']}"
        else:
            return False, f"[ADD] FAILED❌: {config['file']} (mismatch)"

    except Exception as e:
        return False, f"[ADD] Test error {config['file']}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Phase 1: Compile all in parallel.
    Phase 2: Test only successful ones in parallel.
    """
    logger.info(f"[ADD] Starting two-phase test for {len(configs)} kernels...")

    compiled_so_map: Dict[str, str] = {}  # file -> so_path
    failed_results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(
        f"[ADD] Phase 1/2: Compiling {len(configs)} kernels with {num_workers} workers..."
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(compile_kernel, config, source_dir)
            for config in configs
        ]

        for future in as_completed(futures):
            config, success, msg = future.result()
            if success:
                compiled_so_map[config["file"]] = msg  # msg == so_path
            else:
                failed_results.append((False, msg))

    logger.info(
        f"[ADD] Compilation: {len(compiled_so_map)} succeeded, {len(failed_results)} failed."
    )

    # === PHASE 2: Parallel Testing (only for compiled kernels) ===
    if compiled_so_map:
        logger.info(
            f"[ADD] Phase 2/2: Testing {len(compiled_so_map)} compiled kernels..."
        )
        test_configs = [
            (config, compiled_so_map[config["file"]])
            for config in configs
            if config["file"] in compiled_so_map
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(test_kernel, config, so_path)
                for config, so_path in test_configs
            ]

            for future in as_completed(futures):
                result = future.result()
                failed_results.append(result)

    return failed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
    )
    parser.add_argument(
        "--source_dir", default="./", help="Directory with .cpp files"
    )
    parser.add_argument(
        "--target",
        default="cpu",
        choices=["cuda", "cpu", "mlu", "hip"],
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
            logger.error(f"Invalid config: {e}")
            exit(1)

    if isinstance(configs, dict):
        configs = [configs]

    # Filter only 'add' kernels
    add_configs = [c for c in configs if c.get("op_name") == "add"]

    add_configs = [
        {**config, "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp"}
        for config in add_configs
    ]

    if not add_configs:
        logger.warning("No 'add' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        add_configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    passed = sum(1 for r in results if r[0])
    total = len(results)
    rate = passed / total if total else 1.0

    for success, msg in results:
        if success:
            logger.info(msg)
        else:
            logger.error(msg)

    if rate == 1.0:
        logger.info(f"🎉 All {total} add tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} add tests failed.")
        exit(1)
