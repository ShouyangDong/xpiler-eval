"""Batch correctness tester for InstanceNorm2d kernels with parallel
compilation and testing."""

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


def reference_instancenorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference InstanceNorm using PyTorch."""
    return F.instance_norm(
        input,
        running_mean=None,
        running_var=None,
        weight=weight,
        bias=bias,
        momentum=0,
        eps=eps,
    )


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: instancenorm_1_64_56_56.cpp
    Format: instancenorm_N_C_H_W.cpp
    Returns: dict with shape and metadata.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) != 5 or parts[0] != "instancenorm":
            raise ValueError(f"Invalid InstanceNorm filename: {file_name}")
        N, C, H, W = map(int, parts[1:5])
        total = N * C * H * W
        return {
            "file": file_name,
            "shape": [N, C, H, W],
            "N": N,
            "C": C,
            "H": H,
            "W": W,
            "total_elements": total,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one InstanceNorm kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))
    temp_file = os.path.join(
        source_dir, file_name.replace(".cpp", "_patched.cpp")
    )

    if not os.path.isfile(file_path):
        return config, False, f"[INSTANCENORM] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[INSTANCENORM] Patch failed {file_name}: {e}"

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
        return (
            config,
            False,
            f"[INSTANCENORM] Compile failed {file_name}: {msg}",
        )


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled InstanceNorm kernel."""
    try:
        N, C, H, W = config["args"]
        file_name = config["file"]

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "instancenorm", None)
        if not func:
            return (
                False,
                f"[INSTANCENORM] Function 'instancenorm' not found in {so_path}",
            )

        # Set function signature
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input     (N*C*H*W)
            ctypes.POINTER(ctypes.c_float),  # output    (N*C*H*W)
            ctypes.POINTER(ctypes.c_float),  # weight    (C,)
            ctypes.POINTER(ctypes.c_float),  # bias      (C,)
            ctypes.c_int,  # N
            ctypes.c_int,  # C
            ctypes.c_int,  # H
            ctypes.c_int,  # W
            ctypes.c_float,  # eps
        ]
        func.restype = None

        # Generate input
        torch.manual_seed(1234)
        input_tensor = torch.rand(N, C, H, W, dtype=torch.float32)
        weight = torch.rand(C, dtype=torch.float32)
        bias = torch.rand(C, dtype=torch.float32)
        eps = 1e-5

        # Reference
        expected = reference_instancenorm(input_tensor, weight, bias, eps)

        # Flatten input/output
        input_flat = input_tensor.flatten().numpy()
        output_flat = torch.zeros_like(input_tensor).flatten().numpy()

        input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        weight_ptr = weight.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        bias_ptr = bias.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Call kernel
        func(input_ptr, output_ptr, weight_ptr, bias_ptr, N, C, H, W, eps)

        # Reshape and compare
        result_reshaped = torch.from_numpy(output_flat).reshape(N, C, H, W)

        try:
            torch.testing.assert_close(
                result_reshaped,
                expected,
                rtol=1e-3,
                atol=1e-3,
                check_dtype=True,
                equal_nan=False,
                msg=lambda msg: f"[INSTANCENORM] {file_name} failed: {msg}",
            )
            max_abs_err = (result_reshaped - expected).abs().max().item()
            return (
                True,
                f"[INSTANCENORM] PASSED: {file_name} | Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[INSTANCENORM] FAILED: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[INSTANCENORM] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[INSTANCENORM] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(
        f"[INSTANCENORM] Phase 1/2: Compiling {len(configs)} kernels..."
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test InstanceNorm kernels")
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

    # Filter and parse InstanceNorm kernels
    configs = [c for c in configs if c.get("op_name") == "instancenorm"]
    norm_configs = [
        {**config, "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp"}
        for config in configs
    ]

    if not norm_configs:
        logger.warning("No valid 'instancenorm' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        norm_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} InstanceNorm tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} InstanceNorm tests failed.")
        exit(1)
