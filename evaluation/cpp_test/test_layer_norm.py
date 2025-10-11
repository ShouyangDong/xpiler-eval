#!/usr/bin/env python3
"""Batch correctness tester for LayerNorm kernels with parallel compilation and testing."""

import argparse
import ctypes
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

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
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def reference_layernorm(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference LayerNorm using PyTorch."""
    normalized_shape = input.size()[-weight.size(0):]
    return F.layer_norm(input, normalized_shape, weight, bias, eps)


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: layernorm_2_128_768.cpp
    Format: layernorm_B_L_H.cpp  (Batch, Length, Hidden)
    or:     layernorm_N_D.cpp     (N, D)
    Returns: dict with shape and metadata.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 3 or parts[0] != "layernorm":
            raise ValueError(f"Invalid LayerNorm filename: {file_name}")

        dims = [int(p) for p in parts[1:]]
        if len(dims) == 3:
            B, L, H = dims
            shape = [B, L, H]
            normalized_dim = H
        elif len(dims) == 2:
            N, D = dims
            shape = [N, D]
            normalized_dim = D
        else:
            raise ValueError(f"Unsupported LayerNorm shape: {dims}")

        total = 1
        for d in shape:
            total *= d

        return {
            "file": file_name,
            "shape": shape,
            "B": dims[0], "H": dims[-1],
            "normalized_dim": dims[-1],
            "total_elements": total,
            "ndim": len(shape)
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one LayerNorm kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))
    temp_file = os.path.join(source_dir, file_name.replace(".cpp", "_patched.cpp"))

    if not os.path.isfile(file_path):
        return config, False, f"[LAYERNORM] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[LAYERNORM] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    if success:
        try:
            os.remove(temp_file)
        except:
            pass
        return config, True, so_path
    else:
        try:
            os.remove(temp_file)
        except:
            pass
        return config, False, f"[LAYERNORM] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled LayerNorm kernel."""
    try:
        shape = config["shape"]
        file_name = config["file"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "layernorm", None)
        if not func:
            return False, f"[LAYERNORM] Function 'layernorm' not found in {so_path}"

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # weight (gamma)
            ctypes.POINTER(ctypes.c_float),  # bias (beta)
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        input_tensor = torch.randn(*shape, dtype=torch.float32)
        weight = torch.randn(config["normalized_dim"], dtype=torch.float32)
        bias = torch.randn(config["normalized_dim"], dtype=torch.float32)
        eps = 1e-5

        # Reference
        expected = reference_layernorm(input_tensor, weight, bias, eps)

        # Flatten and get pointers
        input_flat = input_tensor.flatten().numpy()
        weight_flat = weight.numpy()
        bias_flat = bias.numpy()
        output_flat = torch.zeros_like(input_tensor).flatten().numpy()

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        gamma_ptr = weight_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        beta_ptr = bias_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call kernel
        func(input_ptr, gamma_ptr, beta_ptr, output_ptr)

        # Reshape and compare
        result_reshaped = torch.from_numpy(output_flat).reshape(shape)

        try:
            torch.testing.assert_close(
                result_reshaped, expected,
                rtol=1e-3, atol=1e-3,
                check_dtype=True,
                equal_nan=False,
                msg=lambda msg: f"[LAYERNORM] {file_name} failed: {msg}"
            )
            max_abs_err = (result_reshaped - expected).abs().max().item()
            return True, f"[LAYERNORM] PASSED: {file_name} | Max error: {max_abs_err:.2e}"
        except Exception as e:
            return False, f"[LAYERNORM] FAILED: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[LAYERNORM] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(f"[LAYERNORM] Starting two-phase test for {len(configs)} kernels...")

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[LAYERNORM] Phase 1/2: Compiling {len(configs)} kernels...")
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

    logger.info(f"[LAYERNORM] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed.")

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(f"[LAYERNORM] Phase 2/2: Testing {len(compiled_map)} compiled kernels...")
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
    parser = argparse.ArgumentParser(description="Test LayerNorm kernels")
    parser.add_argument("--config", required=True, help="JSON string or path to config file")
    parser.add_argument("--source_dir", default="./", help="Directory containing .cpp files")
    parser.add_argument("--target", required=True, choices=["cuda", "hip", "mlu", "cpu"], help="Target platform")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel workers")

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

    # Filter and parse LayerNorm kernels
    norm_configs = []
    for c in configs:
        file_name = c.get("file", "")
        if not file_name.endswith(".cpp"):
            continue
        if file_name.startswith("layernorm"):
            try:
                parsed = parse_filename(file_name)
                c.update(parsed)
                norm_configs.append(c)
            except Exception as e:
                logger.warning(f"Skipping invalid LayerNorm config {file_name}: {e}")

    if not norm_configs:
        logger.warning("No valid 'layernorm' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(norm_configs, args.source_dir, args.target, num_workers=args.jobs)

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
        logger.info(f"🎉 All {total} LayerNorm tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} LayerNorm tests failed.")
        exit(1)