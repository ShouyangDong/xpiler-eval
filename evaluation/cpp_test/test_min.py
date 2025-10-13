"""Batch correctness tester for min reduction kernels with parallel compilation
and testing."""

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


def reference_min(input: torch.Tensor, axis: int) -> torch.Tensor:
    """Reference min reduction using PyTorch."""
    return torch.min(input, dim=axis)[0]  # [0] = values, [1] = indices


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: min_64_64.cpp → shape=[64,64], axis=1
    Format: min_M_N.cpp → reduce along axis=1 (last dim)
    or:     min_B_L_D.cpp → reduce along axis=2
    Returns: dict with shape, axis, and metadata.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 3 or parts[0] != "min":
            raise ValueError(f"Invalid min filename: {file_name}")

        dims = [int(p) for p in parts[1:]]
        if len(dims) == 2:
            M, N = dims
            shape = [M, N]
            axis = 1  # reduce along N
        elif len(dims) == 3:
            B, L, D = dims
            shape = [B, L, D]
            axis = 2  # reduce along D
        else:
            raise ValueError(f"Unsupported min shape: {dims}")

        output_shape = shape[:axis] + shape[axis + 1 :]
        total_input = 1
        for d in shape:
            total_input *= d
        total_output = 1
        for d in output_shape:
            total_output *= d

        return {
            "file": file_name,
            "shape": shape,
            "axis": axis,
            "output_shape": output_shape,
            "total_input": total_input,
            "total_output": total_output,
            "ndim": len(shape),
            "dtype": "float32",  # default
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one min kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    # Use unique temp file to avoid race
    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[MIN] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[MIN] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[MIN] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled min kernel."""
    try:
        shape = config["args"]
        axis = config["axis"]
        file_name = config["file"]
        dtype_str = config.get("dtype", "float32")
        output_shape = [
            1 if i == axis else size for i, size in enumerate(shape)
        ]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "min", None)  # expect function named 'min'
        if not func:
            return False, f"[MIN] Function 'min' not found in {so_path}"

        # Set function signature
        ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        torch_dtype = (
            torch.float32 if dtype_str == "float32" else torch.float16
        )

        func.argtypes = [
            ctypes.POINTER(ctype),  # input
            ctypes.POINTER(ctype),  # output
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        input_tensor = torch.randn(*shape, dtype=torch_dtype) * 100
        expected = reference_min(input_tensor, axis)

        # Flatten and get pointers
        input_flat = input_tensor.flatten().numpy()
        output_flat = (
            torch.zeros(output_shape, dtype=torch_dtype)
            .flatten()
            .numpy()
        )

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctype))
        output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctype))

        # Call kernel
        func(input_ptr, output_ptr)

        # Reshape and compare
        result_reshaped = torch.from_numpy(output_flat).reshape(expected.shape)

        try:
            rtol, atol = (
                (1e-3, 1e-3) if dtype_str == "float32" else (1e-2, 5e-2)
            )
            torch.testing.assert_close(
                result_reshaped,
                expected,
                rtol=rtol,
                atol=atol,
                check_dtype=True,
                equal_nan=False,
                msg=lambda msg: f"[MIN] {file_name} failed: {msg}",
            )
            max_abs_err = (result_reshaped - expected).abs().max().item()
            return (
                True,
                f"[MIN] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[MIN] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[MIN] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(f"[MIN] Starting two-phase test for {len(configs)} kernels...")

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[MIN] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[MIN] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[MIN] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

                
        logger.debug("[MIN] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[MIN] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Min reduction kernels")
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

    # Filter and parse min kernels
    configs = [c for c in configs if c.get("op_name") == "min"]
    min_configs = [
        {**config, "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp"}
        for config in configs
    ]

    if not min_configs:
        logger.warning("No valid 'min' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        min_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} MIN reduction tests passed!")
        exit(0)
    else:
        logger.error(
            f"❌ {total - passed}/{total} MIN reduction tests failed."
        )
        exit(1)
