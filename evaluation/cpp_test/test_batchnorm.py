"""Batch correctness tester for 'batchnorm' kernels with two-phase
parallelism."""

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
from evaluation.utils import parse_op_json

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


def batchnorm_ref(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return F.batch_norm(
        input,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
        eps=eps,
    )


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one batchnorm kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[BatchNorm] File not found: {file_path}"

    # Patch source
    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[BatchNorm] Patch failed {file_name}: {e}"

    # Compile
    success, msg = run_compilation(so_path, temp_file)
    if success:
        os.remove(temp_file)
        return config, True, so_path
    else:
        os.remove(temp_file)
        return config, False, f"[BatchNorm] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled batchnorm kernel."""
    try:
        file_name = config["file"]
        N, C, H, W = config["args"]
        op_name = config["op_name"]
        # Generate input and parameters
        input_tensor = torch.rand(N, C, H, W, dtype=torch.float32)
        running_mean = torch.rand(C, dtype=torch.float32)
        running_var = torch.rand(C, dtype=torch.float32) + 0.5
        weight = torch.rand(C, dtype=torch.float32)  # gamma
        bias = torch.rand(C, dtype=torch.float32)  # beta

        # Golden reference
        expected = batchnorm_ref(
            input_tensor, weight, bias, running_mean, running_var
        )

        result_flat = torch.zeros_like(input_tensor)

        # Get pointers
        input_ptr = input_tensor.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = result_flat.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        mean_ptr = running_mean.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        var_ptr = running_var.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        weight_ptr = weight.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        bias_ptr = bias.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[BatchNorm] Function 'batchnorm' not found in {so_path}",
            )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_float),  # mean
            ctypes.POINTER(ctypes.c_float),  # var
            ctypes.POINTER(ctypes.c_float),  # weight
            ctypes.POINTER(ctypes.c_float),  # bias
            ctypes.c_int,  # N
            ctypes.c_int,  # C
            ctypes.c_int,  # H
            ctypes.c_int,  # W
            ctypes.c_float,  # eps
        ]
        func.restype = None
        eps = 1e-5
        # Call kernel
        func(
            input_ptr,
            output_ptr,
            mean_ptr,
            var_ptr,
            weight_ptr,
            bias_ptr,
            N,
            C,
            H,
            W,
            eps,
        )

        # Reshape and compare
        if torch.allclose(
            result_flat, expected, rtol=1e-3, atol=1e-3, equal_nan=True
        ):
            return True, f"[BatchNorm] PASSED✅: {file_name}"
        else:
            max_error = (result_flat - expected).abs().max().item()
            return (
                False,
                f"[BatchNorm] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[BatchNorm] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Phase 1: Compile all batchnorm kernels in parallel.
    Phase 2: Test only successful ones.
    """
    logger.info(
        f"[BatchNorm] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}  # file -> so_path
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[BatchNorm] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[BatchNorm] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[BatchNorm] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[BATCHNORM] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[BATCHNORM] Failed to delete {so_path}: {e}")
    return results


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
        "--source_dir", default="./", help="Directory with .cpp files"
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
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid 'batchnorm' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log results
    passed = sum(1 for r in results if r[0])
    total = len(results)

    for success, msg in results:
        if success:
            logger.info(msg)
        else:
            logger.error(msg)

    # Final summary
    if passed == total:
        logger.info(f"🎉 All {total} batchnorm tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} batchnorm tests failed.")
        exit(1)
