"""Batch correctness tester for 'conv2d_nchw' kernels with two-phase
parallelism."""

import argparse
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import torch

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import conv2d_nchw
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
    Parse filename: conv2d_nchw_N_C_H_W_KH_KW_CI_CO_SH_PW.cpp
    Example: conv2d_nchw_1_64_28_28_3_3_64_128_1_1.cpp
    Returns: N, C, H, W, KH, KW, CI, CO, stride, pad
    Note: CI should equal C.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")

        # Skip "conv2d_nchw"
        N, C, H, W = map(int, parts[1:5])
        KH, KW, CI, CO = map(int, parts[5:9])
        stride = int(parts[9])
        pad = int(parts[10])
        return {
            "file": file_name,
            "args": map(int, parts[1:])
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one conv2d_nchw kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[Conv2D-NCHW] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[Conv2D-NCHW] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    if success:
        os.remove(temp_file)
        return config, True, so_path
    else:
        os.remove(temp_file)
        return (
            config,
            False,
            f"[Conv2D-NCHW] Compile failed {file_name}: {msg}",
        )


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled conv2d_nchw kernel."""
    try:
        file_name = config["file"]
        data_shape = config["args"][:4]
        kernel_shape = config["args"][4:8]
        stride, pad = config["args"][8], config["args"][9]
        dtype = "float32"
        wtype = "float32"

        # generate data
        data_np = torch.rand(data_shape)
        kernel_np = torch.rand(kernel_shape)
        # cpu compute
        result_cpu = conv2d_nchw(
            data_np,
            kernel_np,
            stride,
            pad,
        )

        result_ctypes = torch.zeros(result_cpu.shape)

        # Get pointers
        input_ptr = data_np.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        kernel_ptr = kernel_np.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = result_ctypes.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "conv2dnchw")
        if not func:
            return (
                False,
                f"[Conv2D-NCHW] Function 'conv2dnchw' not found in {so_path}",
            )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # kernel
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        func.restype = None

        # Call kernel
        func(input_ptr, kernel_ptr, output_ptr)

        # Compare
        if torch.allclose(
            result_ctypes, result_cpu, rtol=1e-3, atol=1e-3, equal_nan=True
        ):
            return True, f"[Conv2D-NCHW] PASSED✅: {file_name}"
        else:
            max_error = torch.max(torch.abs(result_ctypes - result_cpu)).item()
            return (
                False,
                f"[Conv2D-NCHW] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[Conv2D-NCHW] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Phase 1: Compile all conv2d_nchw kernels in parallel.
    Phase 2: Test only successful ones.
    """
    logger.info(
        f"[Conv2D-NCHW] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}  # file -> so_path
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(
        f"[Conv2D-NCHW] Phase 1/2: Compiling {len(configs)} kernels..."
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
        f"[Conv2D-NCHW] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[Conv2D-NCHW] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
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

    # Parse config
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
    else:
        try:
            configs = json.loads(args.config)
        except Exception as e:
            logger.error(f"Invalid config: {e}")
            exit(1)

    if isinstance(configs, dict):
        configs = [configs]

    # Filter only 'conv2d_nchw' kernels
    configs = [c for c in configs if c.get("op_name") == "conv2dnchw"]
    conv2d_nchw_configs = [
        {**config, "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp"}
        for config in configs
    ]


    if not conv2d_nchw_configs:
        logger.warning("No valid 'conv2d_nchw' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        conv2d_nchw_configs,
        args.source_dir,
        args.target,
        num_workers=args.jobs,
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
        logger.info(f"🎉 All {total} conv2d_nchw tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} conv2d_nchw tests failed.")
        exit(1)
