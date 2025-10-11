"""
Parallel tester for BatchMatMul (BMM) kernels on CPU/GPU/MLU.
Supports two-phase pipeline:
1. Parallel compilation
2. Parallel correctness testing
"""

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


# ------------------ Logging setup ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ------------------ Utility ------------------
def parse_bmm_filename(file_name: str) -> Dict:
    """
    Parse filename like: bmm_8_64_128_256.cpp
    Return {"file": ..., "B":8, "M":64, "K":128, "N":256}
    """
    base = os.path.splitext(file_name)[0]
    parts = base.split("_")
    if len(parts) < 5 or parts[0] != "bmm":
        raise ValueError(f"Invalid bmm filename: {file_name}")
    B, M, K, N = map(int, parts[1:5])
    return {
        "file": file_name,
        "op_name": parts[0],
        "shape": [B, M, K, N],
        "B": B,
        "M": M,
        "K": K,
        "N": N,
    }


# ------------------ Compilation ------------------
def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one BMM kernel and return (config, success, so_path or error msg)."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))
    tmp_path = os.path.join(source_dir, file_name.replace(".cpp", "_patched.cpp"))

    if not os.path.isfile(file_path):
        return config, False, f"[BMM] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(tmp_path, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[BMM] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, tmp_path)

    try:
        os.remove(tmp_path)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[BMM] Compile failed {file_name}: {msg}"


# ------------------ Testing ------------------
def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled BMM kernel."""
    try:
        B, M, K, N = config["args"]
        file_name = config["file"]
        name = config["op_name"]

        # Generate input tensors
        A = torch.randn(B, M, K, dtype=torch.float32)
        B_ = torch.randn(B, K, N, dtype=torch.float32)
        ref = torch.matmul(A, B_)  # reference result

        # Prepare output buffer
        C = torch.zeros((B, M, N), dtype=torch.float32)

        # Convert to ctypes
        A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B_.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, name)
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C
        ]
        func.restype = None

        # Run kernel
        func(A_ptr, B_ptr, C_ptr)

        # Compare with reference
        try:
            torch.allclose(
                C,
                ref,
                rtol=1e-3,
                atol=1e-3,
                equal_nan=True,
            )
            return True, f"[BMM] PASSED✅: {file_name}"
        except Exception as e:
            return False, f"[BMM] FAILED❌: {file_name} | {e}"

    except Exception as e:
        return False, f"[BMM] Exception in test {config['file']}: {e}"


# ------------------ Pipeline ------------------
def run_tests(configs: List[dict], source_dir: str, target: str, num_workers: int = 4):
    logger.info(f"[BMM] Starting two-phase test for {len(configs)} kernels...")

    compiled_map = {}
    results = []

    # === Phase 1: Compilation ===
    logger.info(f"[BMM] Phase 1/2: Compiling {len(configs)} kernels...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compile_kernel, cfg, source_dir) for cfg in configs]
        for future in as_completed(futures):
            cfg, success, msg = future.result()
            if success:
                compiled_map[cfg["file"]] = msg
            else:
                results.append((False, msg))

    logger.info(
        f"[BMM] Compilation done: {len(compiled_map)} succeeded, "
        f"{len([r for r in results if not r[0]])} failed."
    )

    # === Phase 2: Testing ===
    if compiled_map:
        logger.info(f"[BMM] Phase 2/2: Testing {len(compiled_map)} kernels...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(test_kernel, cfg, compiled_map[cfg["file"]])
                for cfg in configs
                if cfg["file"] in compiled_map
            ]
            for future in as_completed(futures):
                results.append(future.result())

    return results


# ------------------ Main ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel BatchMatMul kernel tester")
    parser.add_argument("--config", required=True, help="JSON string or path to config")
    parser.add_argument(
        "--source_dir", default="./", help="Directory containing .cpp files"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel jobs")
    args = parser.parse_args()

    # Load config
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
    else:
        configs = json.loads(args.config)

    if isinstance(configs, dict):
        configs = [configs]

    # Filter bmm configs
    configs = [c for c in configs if c.get("op_name") == "bmm"]
    bmm_configs = []
    for c in configs:
        file_name = f"{c['op_name']}_{'_'.join(map(str, c['args']))}.cpp"
        bmm_configs.append({**c, "file": file_name})

    if not bmm_configs:
        logger.warning("No valid 'bmm' kernels found.")
        exit(0)

    # Run test pipeline
    results = run_tests(bmm_configs, args.source_dir, args.target, args.jobs)

    # Summarize results
    passed = sum(1 for r in results if r[0])
    total = len(results)
    for success, msg in results:
        if success:
            logger.info(msg)
        else:
            logger.error(msg)

    if passed == total:
        logger.info(f"🎉 All {total} BMM tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} BMM tests failed.")
        exit(1)
