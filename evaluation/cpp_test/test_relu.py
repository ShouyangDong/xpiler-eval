"""Batch correctness tester for ReLU kernels with parallel compilation and
testing."""

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


def reference_relu(input_array: np.ndarray) -> np.ndarray:
    """Reference ReLU using PyTorch."""
    x = torch.from_numpy(input_array)
    return torch.nn.functional.relu(x).numpy()


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: relu_dim1_dim2_...dimN.cpp
    Example: relu_32_64.cpp → shape=[32, 64]
    Returns config dict.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 1 or parts[0] != "relu":
            raise ValueError(f"Invalid ReLU filename: {file_name}")

        shape = [int(p) for p in parts[1:]]
        if not shape:
            raise ValueError("Empty shape in ReLU filename")

        return {
            "file": file_name,
            "shape": shape,
            "dtype": "float32",  # default
            "ndim": len(shape),
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one ReLU kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[RELU] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[RELU] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[RELU] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled ReLU kernel."""
    try:
        shape = config["args"]
        file_name = config["file"]
        dtype_str = config.get("dtype", "float32")
        op_name = config["op_name"]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[RELU] Function '{func_name}' not found in {so_path}",
            )

        # Set function signature
        ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        np_dtype = np.float32 if dtype_str == "float32" else np.float16

        func.argtypes = [
            ctypes.POINTER(ctype),  # input
            ctypes.POINTER(ctype),  # output
        ]
        func.restype = None

        # Generate input data
        np.random.seed(1234)
        input_array = np.random.uniform(-5.0, 5.0, size=shape).astype(np_dtype)
        expected = reference_relu(input_array)

        # Flatten and get pointers
        input_flat = input_array.flatten()
        output_flat = np.zeros_like(input_flat)

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctype))
        output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctype))

        # Call kernel
        func(input_ptr, output_ptr)

        # Reshape result
        result_reshaped = output_flat.reshape(shape)

        # Compare
        try:
            rtol, atol = (
                (1e-3, 1e-3) if dtype_str == "float32" else (1e-2, 1e-2)
            )
            np.testing.assert_allclose(
                result_reshaped,
                expected,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
                err_msg=f"[RELU] {file_name} failed",
            )
            max_abs_err = np.max(np.abs(result_reshaped - expected))
            return (
                True,
                f"[RELU] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except AssertionError as e:
            return (
                False,
                f"[RELU] FAILED❌: {file_name} | {str(e).splitlines()[0]}",
            )

    except Exception as e:
        return False, f"[RELU] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(
        f"[RELU] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[RELU] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[RELU] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[RELU] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[RELU] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[RELU] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ReLU kernels")
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

    # Filter and parse relu kernels
    relu_configs = []
    for c in configs:
        shapes = c.get("args")
        op_name = c.get("op_name")
        # Construct filename
        file_name = f"{op_name}_{'_'.join(map(str, shapes))}.cpp"

        if file_name.startswith("relu"):
            try:
                parsed = parse_filename(file_name)
                c.update(parsed)
                if "dtype" in c:
                    parsed["dtype"] = c["dtype"]
                relu_configs.append(parsed)
            except Exception as e:
                logger.warning(
                    f"Skipping invalid RELU config {file_name}: {e}"
                )

    if not relu_configs:
        logger.warning("No valid 'relu' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        relu_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} ReLU tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} ReLU tests failed.")
        exit(1)
