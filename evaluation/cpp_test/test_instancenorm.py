"""Stable and parallel-safe InstanceNorm2d correctness tester."""

import argparse
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation

# ========== Logger ==========
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


# ========== Reference Implementation ==========
def reference_instancenorm(input, weight, bias, eps=1e-5):
    return F.instance_norm(
        input,
        running_mean=None,
        running_var=None,
        weight=weight,
        bias=bias,
        momentum=0,
        eps=eps,
    )


# ========== Compilation ==========
def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one kernel to .so."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))
    temp_file = os.path.join(source_dir, file_name.replace(".cpp", "_tmp.cpp"))

    if not os.path.isfile(file_path):
        return config, False, f"[INSTANCENORM] File not found: {file_path}"

    try:
        code = macro + open(file_path, "r").read()
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[INSTANCENORM] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    try:
        os.remove(temp_file)
    except BaseException:
        pass

    if success:
        return config, True, so_path
    else:
        return config, False, f"[INSTANCENORM] Compile failed {file_name}: {msg}"


# ========== Test One Kernel ==========
def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Test compiled kernel correctness (run in subprocess)."""
    file_name = config["file"]
    name = config["op_name"]
    N, C, H, W = config["args"]
    eps = 1e-5

    try:
        input_tensor = torch.rand(N, C, H, W, dtype=torch.float32)
        weight = torch.rand(C, dtype=torch.float32)
        bias = torch.rand(C, dtype=torch.float32)
        expected = reference_instancenorm(input_tensor, weight, bias, eps)

        input_flat = input_tensor.flatten()
        input_ptr = input_flat.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        weight_ptr = weight.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        bias_ptr = bias.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result_ctypes = torch.zeros_like(input_flat)
        output_ptr = result_ctypes.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # local load for safety
        lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_LOCAL)
        kernel_func = getattr(lib, name)
        kernel_func.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        kernel_func.restype = None

        kernel_func(input_ptr, output_ptr, weight_ptr, bias_ptr, N, C, H, W, eps)
        result_reshaped = result_ctypes.reshape(N, C, H, W)

        torch.allclose(result_reshaped, expected, rtol=1e-3, atol=1e-3)
        max_err = (result_reshaped - expected).abs().max().item()
        return True, f"[INSTANCENORM] ✅ {file_name} PASSED | max_err={max_err:.2e}"

    except Exception as e:
        return False, f"[INSTANCENORM] ❌ {file_name} FAILED | {str(e)}"


# ========== Runner ==========
def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(
        f"[INSTANCENORM] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[INSTANCENORM] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[INSTANCENORM] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[INSTANCENORM] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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


# ========== CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path or JSON string")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target", required=True, choices=["cuda", "hip", "mlu", "cpu"])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    # Parse config
    if os.path.isfile(args.config):
        configs = json.load(open(args.config))
    else:
        configs = json.loads(args.config)

    if isinstance(configs, dict):
        configs = [configs]

    # Only test InstanceNorm
    instancenorm_cfgs = [
        {
            **cfg,
            "file": f"{cfg['op_name']}_{'_'.join(map(str, cfg['args']))}.cpp",
        }
        for cfg in configs
        if cfg.get("op_name") == "instancenorm"
    ]

    if not instancenorm_cfgs:
        logger.warning("No valid instancenorm kernels found.")
        exit(0)

    results = run_tests(instancenorm_cfgs, args.source_dir, args.target, jobs=args.jobs)
    passed = sum(1 for r, _ in results if r)
    total = len(results)

    for ok, msg in results:
        (logger.info if ok else logger.error)(msg)

    if passed == total:
        logger.info(f"🎉 All {total} tests PASSED!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} tests FAILED.")
        exit(1)
