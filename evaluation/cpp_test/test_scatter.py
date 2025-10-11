"""Batch correctness tester for scatter kernels with parallel compilation and
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


def reference_scatter(
    self_tensor: torch.Tensor,
    indices_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Mimic torch.Tensor.scatter_ semantics."""
    result = self_tensor.clone()
    result.scatter_(dim=dim, index=indices_tensor, src=src_tensor)
    return result


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: scatter_D1_D2_..._Dn.cpp
    Example: scatter_4_5_6.cpp, scatter_2_3_8_8.cpp
    Returns dict with shape and axis.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 2 or parts[0] != "scatter":
            raise ValueError(f"Invalid scatter filename: {file_name}")

        shape = [int(p) for p in parts[1:]]
        if len(shape) < 1:
            raise ValueError(f"Invalid shape in {file_name}")

        return {
            "file": file_name,
            "shape": shape,
            "output_shape": shape,  # scatter in-place
            "dtype": "float32",     # default
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one scatter kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[SCATTER] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[SCATTER] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[SCATTER] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled scatter kernel."""
    try:
        shape = config["args"]
        dim = config.get("axis")  # must be provided in config
        file_name = config["file"]
        dtype_str = config.get("dtype", "float32")

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func_name = "scatter"  # assume function name is 'scatter'
        func = getattr(lib, func_name, None)
        if not func:
            return False, f"[SCATTER] Function '{func_name}' not found in {so_path}"

        # Set ctypes and torch dtype
        ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        torch_dtype = torch.float32 if dtype_str == "float32" else torch.float16

        func.argtypes = [
            ctypes.POINTER(ctype),  # self
            ctypes.POINTER(ctypes.c_int),  # indices
            ctypes.POINTER(ctype),  # src
            ctypes.POINTER(ctype),  # output
        ]
        func.restype = None

        # Generate input tensors
        torch.manual_seed(1234)
        # Create tensors
        self_tensor = torch.rand(*shape, dtype=torch.float32)
        src_tensor = torch.rand(*shape, dtype=torch.float32)
        # indices must be within valid range for the target dimension
        size_dim = shape[dim]
        indices_tensor = torch.randint(0, size_dim, shape, dtype=torch.int64)

        # Reference output
        expected = reference_scatter(self_tensor, indices_tensor, src_tensor, dim)

        # Flatten and get pointers
        self_flat = self_tensor.flatten().numpy()
        indices_flat = indices_tensor.flatten().numpy()
        src_flat = src_tensor.flatten().numpy()
        output_flat = torch.zeros_like(self_tensor).flatten().numpy()

        self_ptr = self_flat.ctypes.data_as(ctypes.POINTER(ctype))
        indices_ptr = indices_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        src_ptr = src_flat.ctypes.data_as(ctypes.POINTER(ctype))
        output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctype))

        # Call kernel
        func(self_ptr, indices_ptr, src_ptr, output_ptr)

        # Reshape and compare
        result_reshaped = torch.from_numpy(output_flat).reshape(expected.shape)

        try:
            rtol, atol = (1e-4, 1e-4) if dtype_str == "float32" else (1e-2, 5e-2)
            torch.testing.assert_close(
                result_reshaped,
                expected,
                rtol=rtol,
                atol=atol,
                check_dtype=True,
                equal_nan=False,
                msg=lambda msg: f"[SCATTER] {file_name} failed: {msg}",
            )
            max_abs_err = (result_reshaped - expected).abs().max().item()
            return (
                True,
                f"[SCATTER] PASSED: {file_name} | Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[SCATTER] FAILED: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[SCATTER] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(
        f"[SCATTER] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[SCATTER] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[SCATTER] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[SCATTER] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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
    parser = argparse.ArgumentParser(description="Test Scatter kernels")
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

    # Filter and parse scatter kernels
    scatter_configs = []
    for c in configs:
        shapes = c.get("args")
        op_name = c.get("op_name")
        axis = c.get("axis")

        if op_name != "scatter":
            continue
        if not shapes or axis is None:
            logger.warning(f"Skipping invalid scatter config: {c}")
            continue

        # Construct filename
        file_name = f"{op_name}_{'_'.join(map(str, shapes))}.cpp"

        try:
            parsed = parse_filename(file_name)
            c.update(parsed)
            if "dtype" in c:
                parsed["dtype"] = c["dtype"]
            c["axis"] = axis  # explicitly add axis
            scatter_configs.append(c)
        except Exception as e:
            logger.warning(f"Skipping invalid SCATTER config {file_name}: {e}")

    if not scatter_configs:
        logger.warning("No valid 'scatter' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        scatter_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} SCATTER tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} SCATTER tests failed.")
        exit(1)