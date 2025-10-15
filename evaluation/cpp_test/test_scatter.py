"""Batch correctness tester for scatter kernels with parallel compilation and
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
from evaluation.utils import parse_op_json

# ----------------- Logger -----------------
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


# ----------------- Reference Implementation -----------------
def reference_scatter(
    self_tensor: torch.Tensor,
    indices_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    dim: int,
):
    """Mimic torch.Tensor.scatter_ semantics."""
    result = self_tensor.clone()
    result.scatter_(dim=dim, index=indices_tensor, src=src_tensor)
    return result



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


# ----------------- Core: Kernel Test -----------------
def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled scatter kernel."""
    try:
        dim = config["axis"]
        shape = config["args"]
        config["op_name"]
        file_name = config["file"]
        op_name = config["op_name"]
        # ----------- Prepare tensors -----------
        self_tensor = torch.rand(*shape, dtype=torch.float32)
        src_tensor = torch.rand(*shape, dtype=torch.float32)
        size_dim = shape[dim]
        indices_tensor = torch.randint(0, size_dim, shape, dtype=torch.int64)

        expected = reference_scatter(
            self_tensor, indices_tensor, src_tensor, dim
        )

        # Convert to contiguous numpy arrays
        self_np = (
            self_tensor.contiguous().numpy().astype(np.float32, copy=False)
        )
        src_np = src_tensor.contiguous().numpy().astype(np.float32, copy=False)
        indices_np = (
            indices_tensor.contiguous().numpy().astype(np.int32, copy=False)
        )
        out_np = np.zeros_like(self_np, dtype=np.float32)

        # ----------- Load shared lib -----------
        lib = ctypes.CDLL(so_path)
        kernel_func = getattr(lib, op_name, None)
        kernel_func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # self
            ctypes.POINTER(ctypes.c_int32),  # indices
            ctypes.POINTER(ctypes.c_float),  # src
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        kernel_func.restype = None

        # ----------- Call kernel -----------
        kernel_func(
            self_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            src_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        # Compare
        result_reshaped = torch.from_numpy(out_np).reshape(expected.shape)
        try:
            torch.allclose(result_reshaped, expected, rtol=1e-4, atol=1e-4)
            return True, f"[SCATTER] PASSED✅: {file_name}"
        except Exception as e:
            return False, f"[SCATTER] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[SCATTER] Exception in test {file_name}: {str(e)}"


# ----------------- Run Batch -----------------
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

        logger.debug("[SCATTER] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[SCATTER] Failed to delete {so_path}: {e}")
    return results


# ----------------- CLI -----------------
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
        "--target", required=True, choices=["cuda", "hip", "mlu", "cpu"]
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel workers")

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid scatter kernels found.")
        exit(0)

    results = run_tests(
        configs, args.source_dir, args.target, num_workers=args.jobs
    )
    passed = sum(1 for r in results if r[0])
    total = len(results)

    for ok, msg in results:
        (logger.info if ok else logger.error)(msg)

    if passed == total:
        logger.info(f"🎉 All {total} SCATTER tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} SCATTER tests failed.")
        exit(1)
