"""Batch correctness tester for Element-wise Subtraction (A - B) kernels."""

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


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Golden reference: A - B using PyTorch."""
    return torch.sub(A, B)


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

            if op_name not in ["sub"]:
                logger.warning(f"[SUB] Expected op='sub', got {op_name}")

            parsed_configs.append(
                {
                    "file": file_name,
                    "shape": shape,
                    "dtype": dtype,
                    "op": op_name,
                }
            )
        except Exception as e:
            logger.warning(f"[SUB] Skip invalid config #{idx}: {e}")

    return parsed_configs


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one sub kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[SUB] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[SUB] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[SUB] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled sub kernel."""
    try:
        file_name = config["file"]
        shape = config["args"]
        dtype_str = config["dtype"]
        op_name = config["op_name"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[SUB] Function '{op_name}' not found in {so_path}"

        # Determine C type and numpy dtype
        ctype_float = (
            ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        )
        np_dtype = np.float32 if dtype_str == "float32" else np.float16
        torch_dtype = (
            torch.float32 if dtype_str == "float32" else torch.float16
        )

        # Set function signature: void sub(float* A, float* B, float* out)
        func.argtypes = [
            ctypes.POINTER(ctype_float),  # input A
            ctypes.POINTER(ctype_float),  # input B
            ctypes.POINTER(ctype_float),  # output
        ]
        func.restype = None

        # Generate input
        A_np = np.random.rand(*shape).astype(np_dtype)
        B_np = np.random.rand(*shape).astype(np_dtype)
        A_torch = torch.from_numpy(A_np).to(dtype=torch_dtype)
        B_torch = torch.from_numpy(B_np).to(dtype=torch_dtype)
        expected_output = ref_program(A_torch, B_torch)

        # Prepare output array
        output_np = np.zeros_like(A_np)
        A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctype_float))
        B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctype_float))
        output_ptr = output_np.ctypes.data_as(ctypes.POINTER(ctype_float))

        # Call kernel
        func(A_ptr, B_ptr, output_ptr)

        # Convert result back to torch tensor
        output_torch = torch.from_numpy(output_np).to(dtype=torch_dtype)

        # Compare
        rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (5e-2, 5e-2)
        if torch.allclose(
            output_torch, expected_output, rtol=rtol, atol=atol, equal_nan=True
        ):
            max_error = (output_torch - expected_output).abs().max().item()
            return (
                True,
                f"[SUB] ✅ {file_name}| Max error: {max_error:.2e}",
            )
        else:
            max_error = (output_torch - expected_output).abs().max().item()
            return (
                False,
                f"[SUB] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[SUB] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(f"[SUB] Starting two-phase test for {len(configs)} kernels...")

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[SUB] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[SUB] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[SUB] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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
    parser = argparse.ArgumentParser(
        description="Test Element-wise Subtraction (A - B) kernels"
    )
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
        logger.warning("⚠️ No valid 'sub' kernels found in config.")
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
        logger.info(f"🎉 All {total} Subtraction tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} Subtraction tests failed.")
        exit(1)
