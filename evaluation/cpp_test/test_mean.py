"""Batch correctness tester for Mean reduction kernels with parallel
compilation and testing."""

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


def reference_mean(
    input_tensor: torch.Tensor, reduce_dim: int
) -> torch.Tensor:
    """Reference mean using PyTorch."""
    return torch.mean(input_tensor, dim=reduce_dim).contiguous()



def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one mean kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[MEAN] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[MEAN] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[MEAN] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled mean kernel."""
    try:
        file_name = config["file"]
        shape = config["args"]
        reduce_dim = config["axis"]
        op_name = config["op_name"]
        dtype_str = config["dtype"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[MEAN] Function '{op_name}' not found in {so_path}"

        # Determine C type
        ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        torch_dtype = (
            torch.float32 if dtype_str == "float32" else torch.float16
        )

        # Set function signature
        rank = len(shape)
        if rank not in [2, 3, 4]:
            return False, f"[MEAN] Rank {rank} not supported (only 2D/3D/4D)"

        argtypes = [
            ctypes.POINTER(ctype),  # input
            ctypes.POINTER(ctype),  # output
        ]

        func.argtypes = argtypes
        func.restype = None

        # Generate input
        input_tensor = torch.rand(shape, dtype=torch_dtype)
        expected = reference_mean(input_tensor, reduce_dim)
        output_shape = expected.shape
        output_numel = expected.numel()

        # Flatten for C
        input_flat = input_tensor.flatten().numpy()
        result_array = (ctype * output_numel)()  # allocate C array

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctype))

        # Build args list
        args_list = [input_ptr, result_array] + list(shape) + [reduce_dim]

        # Call kernel
        func(*args_list)

        # Convert result back
        computed_flat = torch.tensor(
            [result_array[i] for i in range(output_numel)], dtype=torch_dtype
        )
        computed = computed_flat.view(output_shape)

        # Compare
        rtol, atol = (1e-5, 1e-5) if dtype_str == "float32" else (1e-3, 1e-3)
        if torch.allclose(
            computed, expected, rtol=rtol, atol=atol, equal_nan=True
        ):
            max_error = (computed - expected).abs().max().item()
            return (
                True,
                f"[MEAN] ✅ {file_name}| Max error: {max_error:.2e}",
            )
        else:
            max_error = (computed - expected).abs().max().item()
            return (
                False,
                f"[MEAN] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[MEAN] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(
        f"[MEAN] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[MEAN] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[MEAN] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[MEAN] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[MEAN] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[MEAN] Failed to delete {so_path}: {e}")
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
        logger.warning("⚠️ No valid 'mean' kernels found in config.")
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
        logger.info(f"🎉 All {total} Mean tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} Mean tests failed.")
        exit(1)
