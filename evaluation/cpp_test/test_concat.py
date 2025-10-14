"""Batch correctness tester for 'concat' kernels with two-phase parallelism."""

import argparse
import ctypes
import json
import logging
import os
import sys
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


def concat_ref(tensors: List[torch.Tensor], axis: int) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.cat(tensors, dim=axis)


def parse_config(config: dict, file_name: str) -> Dict:
    """Validate and enrich concat config."""
    if "args" not in config or "axis" not in config:
        raise ValueError(f"Missing 'args' or 'axis' in config for {file_name}")
    shape = config["args"]
    axis = config["axis"]
    if not isinstance(shape, list) or not all(
        isinstance(d, int) for d in shape
    ):
        raise ValueError(f"Invalid shape in config: {shape}")
    if not isinstance(axis, int):
        raise ValueError(f"Invalid axis: {axis}")
    return {
        "file": file_name,
        "shape": shape,
        "axis": axis,
    }


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one concat kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[Concat] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[Concat] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    if success:
        os.remove(temp_file)
        return config, True, so_path
    else:
        os.remove(temp_file)
        return config, False, f"[Concat] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled concat kernel."""
    try:
        file_name = config["file"]
        shape = config["args"]
        axis = config["axis"]
        op_name = config["op_name"]
        # Create input tensors
        input1 = torch.randn(*shape, dtype=torch.float32)
        input2 = torch.randn(*shape, dtype=torch.float32)

        # Reference output
        expected = concat_ref([input1, input2], axis=axis)

        # Flatten for C++ (row-major)
        flat1 = input1.flatten().numpy()
        flat2 = input2.flatten().numpy()
        output_flat = torch.zeros_like(expected.flatten())
        out_ptr = output_flat.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # Get pointers
        ptr1 = flat1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr2 = flat2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[Concat] Function 'concat' not found in {so_path}"

        func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
        func.restype = None

        # Call kernel
        func(ptr1, ptr2, out_ptr)

        # Reshape and compare
        result_reshaped = output_flat.reshape(expected.shape)
        if torch.allclose(
            result_reshaped, expected, rtol=1e-4, atol=1e-4, equal_nan=True
        ):
            return True, f"[Concat] PASSED✅: {file_name}"
        else:
            max_error = (result_reshaped - expected).abs().max().item()
            return (
                False,
                f"[Concat] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[Concat] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile in parallel, then test in parallel."""
    logger.info(
        f"[Concat] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}  # file -> so_path
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[Concat] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[Concat] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[Concat] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[CONCAT] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[CONCAT] Failed to delete {so_path}: {e}")
    return results


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
            raw_configs = json.load(f)
    else:
        try:
            raw_configs = json.loads(args.config)
        except Exception as e:
            logger.error(f"Invalid config JSON: {e}")
            sys.exit(1)

    if isinstance(raw_configs, dict):
        raw_configs = [raw_configs]

    # Filter and parse concat kernels
    configs = [c for c in raw_configs if c.get("op_name") == "concat"]
    concat_configs = [
        {
            **config,
            "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp",
        }
        for config in configs
    ]

    if not concat_configs:
        logger.warning("No valid 'concat' kernels found in config.")
        sys.exit(0)

    # Run two-phase test
    results = run_tests(
        concat_configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log all results
    passed = sum(1 for r in results if r[0])
    total = len(results)

    for success, msg in results:
        if success:
            logger.info(msg)
        else:
            logger.error(msg)

    # Final summary
    if passed == total:
        logger.info(f"🎉 All {total} concat tests passed!")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} concat tests failed.")
        sys.exit(1)
