"""Batch correctness tester for GQA kernels with parallel compilation and
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


def reference_gqa(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """Reference GQA using PyTorch."""
    # Q: [B, H, Sq, D]
    # K: [B, H, D, Skv]
    # V: [B, H, Skv, D]
    # S = softmax(Q @ K)
    # O = S @ V
    with torch.no_grad():
        S = torch.matmul(Q, K)  # [B, H, Sq, Skv]
        S = torch.softmax(S, dim=-1)
        O = torch.matmul(S, V)  # [B, H, Sq, D]
    return O


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: gqa_1_128_128_64.cpp
    Format: gqa_B_Sq_Skv_Hd.cpp
    Returns: dict with shape and metadata.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) != 5 or parts[0] != "gqa":
            raise ValueError(f"Invalid GQA filename: {file_name}")
        B, Sq, Skv, Hd = map(int, parts[1:5])
        H = 2  # Assume fixed num_heads=2 (or infer from macro/config)
        return {
            "file": file_name,
            "shape": [B, Sq, Skv, Hd],
            "batch": B,
            "seq_q": Sq,
            "seq_kv": Skv,
            "head_dim": Hd,
            "num_heads": H,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one GQA kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[GQA] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[GQA] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    if success:
        try:
            os.remove(temp_file)
        except BaseException:
            pass
        return config, True, so_path
    else:
        try:
            os.remove(temp_file)
        except BaseException:
            pass
        return config, False, f"[GQA] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled GQA kernel."""
    try:
        B = config["batch"]
        H = config["num_heads"]
        Sq = config["seq_q"]
        Skv = config["seq_kv"]
        D = config["head_dim"]
        file_name = config["file"]
        op_name = config["op_name"]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[GQA] Function 'gqa' not found in {so_path}"

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Q
            ctypes.POINTER(ctypes.c_float),  # K
            ctypes.POINTER(ctypes.c_float),  # V
            ctypes.POINTER(ctypes.c_float),  # O
            ctypes.c_int,  # batch
            ctypes.c_int,  # num_heads
            ctypes.c_int,  # seq_q
            ctypes.c_int,  # seq_kv
            ctypes.c_int,  # head_dim
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        Q = torch.randn(B, H, Sq, D, dtype=torch.float32)
        # Note: K is [B, H, D, Skv]
        K = torch.randn(B, H, D, Skv, dtype=torch.float32)
        V = torch.randn(B, H, Skv, D, dtype=torch.float32)
        O = torch.zeros(B, H, Sq, D, dtype=torch.float32)

        # Reference
        O_ref = reference_gqa(Q, K, V)

        # Get pointers
        Q_ptr = Q.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        K_ptr = K.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        V_ptr = V.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        O_ptr = O.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call kernel
        func(Q_ptr, K_ptr, V_ptr, O_ptr, B, H, Sq, Skv, D)

        # Compare
        try:
            torch.testing.assert_close(
                O,
                O_ref,
                rtol=5e-3,
                atol=5e-3,
                check_dtype=True,
                equal_nan=False,
                msg=lambda msg: f"[GQA] {file_name} failed: {msg}",
            )
            max_abs_err = (O - O_ref).abs().max().item()
            return (
                True,
                f"[GQA] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[GQA] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[GQA] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(f"[GQA] Starting two-phase test for {len(configs)} kernels...")

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[GQA] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[GQA] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[GQA] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[GQA] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[GQA] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GQA kernels")
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

    # Filter and parse GQA kernels
    configs = [c for c in configs if c.get("op_name") == "gqa"]
    gqa_configs = [
        {
            **config,
            "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp",
        }
        for config in configs
    ]

    if not gqa_configs:
        logger.warning("No valid 'gqa' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        gqa_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} GQA tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} GQA tests failed.")
        exit(1)
