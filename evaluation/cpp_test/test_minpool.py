"""Batch correctness tester for minpool kernels with parallel compilation and
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
from evaluation.utils import minpool_np
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


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: minpool_N_C_H_W_kx_ky_sx_sy.cpp
    Example: minpool_1_3_224_224_2_2_2_2.cpp
    Returns dict with shape, kernel, stride.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 6 or parts[0] != "minpool":
            raise ValueError(f"Invalid minpool filename: {file_name}")

        # Format: minpool_N_C_H_W_[kx_ky...]_[sx_sy...]
        dims = [int(p) for p in parts[1:]]

        if len(dims) < 4:
            raise ValueError("Too few dimensions in filename")

        # Assume at least 4: NCHW
        spatial_dims = (len(dims) - 4) // 2
        if 4 + 2 * spatial_dims != len(dims):
            raise ValueError(f"Invalid minpool dims count: {len(dims)}")

        shape = dims[:4]  # N, C, H, W
        kernel = dims[4 : 4 + spatial_dims]  # kx, ky, ...
        stride = dims[4 + spatial_dims :]  # sx, sy, ...

        output_np = minpool_np(torch.randn(*shape), kernel + stride)
        output_shape = list(output_np.shape)

        return {
            "file": file_name,
            "shape": shape,
            "kernel": kernel,
            "stride": stride,
            "output_shape": output_shape,
            "dtype": "float32",  # default
            "ndim": 2 + spatial_dims,  # 2D pool, 3D pool, etc.
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one minpool kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[MINPOOL] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[MINPOOL] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[MINPOOL] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled minpool kernel."""
    try:
        shape = config["args"][:4]
        kernel = config["args"][4:6]
        stride = config["args"][6:8]

        file_name = config["file"]
        dtype_str = config.get("dtype", "float32")

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func_name = "minpool"  # assume function name is 'minpool'
        func = getattr(lib, func_name, None)
        if not func:
            return (
                False,
                f"[MINPOOL] Function '{func_name}' not found in {so_path}",
            )

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
        expected = minpool_np(input_tensor, kernel + stride)

        # Flatten and get pointers
        input_flat = input_tensor.flatten().numpy()
        output_flat = (
            torch.zeros(expected.shape, dtype=torch_dtype).flatten().numpy()
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
                msg=lambda msg: f"[MINPOOL] {file_name} failed: {msg}",
            )
            max_abs_err = (result_reshaped - expected).abs().max().item()
            return (
                True,
                f"[MINPOOL] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[MINPOOL] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[MINPOOL] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(
        f"[MINPOOL] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[MINPOOL] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[MINPOOL] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[MINPOOL] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[MINPOOL] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[MINPOOL] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MinPool kernels")
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

    # Filter and parse minpool kernels
    minpool_configs = []
    for c in configs:
        shapes = c.get("args")
        op_name = c.get("op_name")
        # Construct filename
        file_name = f"{op_name}_{'_'.join(map(str, shapes))}.cpp"

        if file_name.startswith("minpool"):
            try:
                parsed = parse_filename(file_name)
                c.update(parsed)
                if "dtype" in c:
                    parsed["dtype"] = c["dtype"]
                minpool_configs.append(parsed)
            except Exception as e:
                logger.warning(
                    f"Skipping invalid MINPOOL config {file_name}: {e}"
                )

    if not minpool_configs:
        logger.warning("No valid 'minpool' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        minpool_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} MINPOOL tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} MINPOOL tests failed.")
        exit(1)
