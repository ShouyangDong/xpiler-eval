"""Batch correctness tester for Multi-Head Attention (MHA) kernels with
parallel compilation and testing."""

import argparse
import ctypes
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

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


def reference_mha(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Reference MHA forward pass using PyTorch."""
    # q: [B, H, L, D], k/v: [B, H, S, D]
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
        q.size(-1)
    )  # [B,H,L,S]
    if causal:
        mask = torch.triu(torch.ones(L, S, device=q.device), diagonal=1).bool()
        score = score.masked_fill(mask, torch.finfo(score.dtype).min)
    attn = F.softmax(score, dim=-1)
    output = torch.matmul(attn, v)  # [B,H,L,D]
    return output


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: mha_8_16_64_64.cpp → B=8, H=16, L=S=64, D=64
    Format: mha_B_H_L_D.cpp (assume L == S)
    or:     mha_B_H_L_S_D.cpp (explicit L and S)

    Returns: dict with shape info.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 5 or parts[0] != "mha":
            raise ValueError(f"Invalid MHA filename: {file_name}")

        dims = [int(p) for p in parts[1:]]
        if len(dims) == 4:
            B, H, L, D = dims
            S = L
        elif len(dims) == 5:
            B, H, L, S, D = dims
        else:
            raise ValueError(f"Unsupported MHA shape: {dims}")

        q_shape = [B, H, L, D]
        k_shape = [B, H, S, D]
        v_shape = [B, H, S, D]
        out_shape = [B, H, L, D]

        total_q = B * H * L * D
        total_kv = B * H * S * D
        total_out = B * H * L * D

        return {
            "file": file_name,
            "q_shape": q_shape,
            "k_shape": k_shape,
            "v_shape": v_shape,
            "out_shape": out_shape,
            "total_q": total_q,
            "total_kv": total_kv,
            "total_out": total_out,
            "causal": False,  # default; override by config
            "dtype": "float32",  # default
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one MHA kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))

    # Use unique temp file per process to avoid race
    temp_file = os.path.join(
        source_dir,
        f"{file_name.replace('.cpp', '')}_patched_{os.getpid()}.cpp",
    )

    if not os.path.isfile(file_path):
        return config, False, f"[MHA] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[MHA] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[MHA] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled MHA kernel."""
    try:
        shape = config["args"]
        causal = config.get("causal", False)
        dtype_str = config.get("dtype", "float32")
        file_name = config["file"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "mha", None)
        if not func:
            return False, f"[MHA] Function 'mha' not found in {so_path}"

        # Set function signature
        ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
        torch_dtype = (
            torch.float32 if dtype_str == "float32" else torch.float16
        )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        q = torch.randn(shape, dtype=torch_dtype)
        k = torch.randn(shape, dtype=torch_dtype)
        v = torch.randn(shape, dtype=torch_dtype)

        expected = reference_mha(q, k, v, causal=causal).cpu()

        # Flatten and get pointers
        q_flat = q.flatten().numpy()
        k_flat = k.flatten().numpy()
        v_flat = v.flatten().numpy()
        out_flat = (
            torch.zeros(expected.shape, dtype=torch_dtype).flatten().numpy()
        )

        ptr_q = q_flat.ctypes.data_as(ctypes.POINTER(ctype))
        ptr_k = k_flat.ctypes.data_as(ctypes.POINTER(ctype))
        ptr_v = v_flat.ctypes.data_as(ctypes.POINTER(ctype))
        ptr_out = out_flat.ctypes.data_as(ctypes.POINTER(ctype))

        # Call kernel
        func(ptr_q, ptr_k, ptr_v, ptr_out, causal)

        # Reshape and compare
        result = torch.from_numpy(out_flat).reshape(expected.shape).cpu()

        try:
            torch.allclose(
                result,
                expected,
                rtol=1e-3,
                atol=1e-3,
                equal_nan=False,
            )
            max_abs_err = (result - expected).abs().max().item()
            return (
                True,
                f"[MHA] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[MHA] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[MHA] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(f"[MHA] Starting two-phase test for {len(configs)} kernels...")

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[MHA] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[MHA] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[MHA] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[MHA] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[MHA] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MHA kernels")
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

    # Filter and parse MHA kernels
    configs = [c for c in configs if c.get("op_name") == "max"]
    max_configs = [
        {
            **config,
            "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp",
        }
        for config in configs
    ]

    if not mha_configs:
        logger.warning("No valid 'mha' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        mha_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} MHA tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} MHA tests failed.")
        exit(1)
