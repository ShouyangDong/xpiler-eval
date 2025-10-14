"""Batch correctness tester for GELU kernels with parallel compilation and
testing."""

import argparse
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

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


def reference_gelu(x: np.ndarray) -> np.ndarray:
    """Reference GELU implementation using NumPy."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: gelu_1024.cpp, gelu_4096.cpp, etc.
    Format: gelu_N.cpp
    Returns: dict with shape and metadata.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) != 2 or parts[0] != "gelu":
            raise ValueError(f"Invalid GELU filename: {file_name}")
        N = int(parts[1])
        return {
            "file": file_name,
            "shape": [N],
            "size": N,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one GELU kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[GELU] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[GELU] Patch failed {file_name}: {e}"

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
        return config, False, f"[GELU] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled GELU kernel."""
    try:
        N = np.prod(config["args"])
        file_name = config["file"]
        op_name = config["op_name"]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[GELU] Function 'gelu' not found in {so_path}"

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        func.restype = None

        # Generate input
        np.random.seed(1234)
        input_data = np.random.uniform(-5, 5, size=N).astype(np.float32)
        expected = reference_gelu(input_data)
        output_data = np.zeros_like(input_data)

        # Get pointers
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call kernel
        func(input_ptr, output_ptr)

        # Compare
        max_abs_err = np.max(np.abs(output_data - expected))
        max_rel_err = np.max(
            np.abs(output_data - expected) / (np.abs(expected) + 1e-8)
        )

        if max_abs_err < 1e-3:
            return (
                True,
                f"[GELU] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        else:
            return False, (
                f"[GELU] FAILED❌: {file_name} | Max abs error: {max_abs_err:.2e}, "
                f"Max rel error: {max_rel_err:.2e}"
            )

    except Exception as e:
        return False, f"[GELU] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[GELU] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[GELU] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[GELU] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[GELU] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[Gelu] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[Gelu] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GELU kernels (CPU)")
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

    # Filter and parse GELU kernels
    configs = [c for c in configs if c.get("op_name") == "gelu"]
    gelu_configs = [
        {
            **config,
            "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp",
        }
        for config in configs
    ]

    if not gelu_configs:
        logger.warning("No valid 'gelu' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        gelu_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} GELU tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} GELU tests failed.")
        exit(1)
