"""Batch correctness tester for Sigmoid kernels with parallel compilation and
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


def ref_program(x: np.ndarray) -> np.ndarray:
    """Golden reference: sigmoid = 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + np.exp(-x))


def parse_config(config_input: str) -> List[Dict]:
    """Parse config: either JSON file or JSON string."""
    if os.path.isfile(config_input):
        with open(config_input, "r") as f:
            config_data = json.load(f)
    else:
        try:
            config_data = json.loads(config_input)
        except Exception as e:
            raise ValueError(f"Invalid JSON config: {e}")

    if isinstance(config_data, dict):
        config_data = [config_data]

    parsed_configs = []
    for idx, c in enumerate(config_data):
        try:
            shape = c.get("args")
            if (
                not isinstance(shape, list)
                or len(shape) < 1
                or not all(isinstance(d, int) for d in shape)
            ):
                raise ValueError(
                    f"Invalid 'args' (must be list of int): {shape}"
                )

            op_name =  c.get("op_name")
            # Construct filename
            file_name = f"{op_name}_{'_'.join(map(str, shape))}.cpp"
            if not file_name or not file_name.endswith(".cpp"):
                raise ValueError(f"Invalid or missing 'file': {file_name}")

            dtype = c.get("dtype", "float32")
            if dtype not in ["float32", "float16"]:
                raise ValueError(f"Unsupported dtype: {dtype}")


            if op_name not in ["sigmoid"]:
                logger.warning(
                    f"[SIGMOID] Expected op='sigmoid', got {op_name}"
                )

            parsed_configs.append(
                {
                    "file": file_name,
                    "shape": shape,
                    "dtype": dtype,
                    "op": op_name,
                }
            )
        except Exception as e:
            logger.warning(f"[SIGMOID] Skip invalid config #{idx}: {e}")

    return parsed_configs


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one sigmoid kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[SIGMOID] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[SIGMOID] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[SIGMOID] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled sigmoid kernel."""
    try:
        file_name = config["file"]
        shape = config["args"]
        dtype_str = config["dtype"]
        op_name = config["op_name"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[SIGMOID] Function '{op_name}' not found in {so_path}",
            )

        # Determine C type and numpy dtype
        ctype_float = (
            ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        )
        np_dtype = np.float32 if dtype_str == "float32" else np.float16

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctype_float),  # input
            ctypes.POINTER(ctype_float),  # output
        ]
        func.restype = None

        # Generate input
        input_array = np.random.uniform(-5, 5, size=shape).astype(np_dtype)
        expected_output = ref_program(input_array)
        output_array = np.zeros_like(input_array)

        # Prepare pointers
        input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctype_float))
        output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctype_float))

        # Call kernel
        func(input_ptr, output_ptr)

        # Compare
        rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (5e-2, 5e-2)
        if np.allclose(
            output_array, expected_output, rtol=rtol, atol=atol, equal_nan=True
        ):
            max_error = np.max(np.abs(output_array - expected_output))
            return (
                True,
                f"[SIGMOID] ✅ {file_name}| Max error: {max_error:.2e}",
            )
        else:
            max_error = np.max(np.abs(output_array - expected_output))
            return (
                False,
                f"[SIGMOID] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[SIGMOID] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(
        f"[SIGMOID] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[SIGMOID] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[SIGMOID] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[SIGMOID] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

                
        logger.debug("[SIGMOID] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[SIGMOID] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Sigmoid kernels")
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
    try:
        configs = parse_config(args.config)
    except Exception as e:
        logger.error(f"❌ Config parsing failed: {e}")
        exit(1)

    if not configs:
        logger.warning("⚠️ No valid 'sigmoid' kernels found in config.")
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
        logger.info(f"🎉 All {total} Sigmoid tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} Sigmoid tests failed.")
        exit(1)
