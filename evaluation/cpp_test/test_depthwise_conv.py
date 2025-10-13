"""Batch correctness tester for depthwise_conv2d kernels with two-phase
parallelism."""

import argparse
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

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


def depthwise_conv2d(input, w):
    """Two-dimensional depthwise convolution.

    Uses SAME padding with 0s, a stride of 1 and no dilation. A single output
    channel is used per input channel (channel_multiplier=1).

    // before: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth)

    Returns a result with shape (height, width, in_depth).
    """
    height, width, in_depth = input.shape
    output_height = height - w.shape[0] + 1
    output_width = width - w.shape[1] + 1
    output = np.zeros((output_height, output_width, in_depth))
    for c in range(in_depth):
        # For each input channel separately, apply its corresponsing filter
        # to the input.
        for i in range(output_height):
            for j in range(output_width):
                for fi in range(w.shape[0]):
                    for fj in range(w.shape[1]):
                        w_element = w[fi, fj, c]
                        output[i, j, c] += input[i + fi, j + fj, c] * w_element
    return output


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: dwconv_32_3_64.cpp
    Format: dwconv_H_K_C.cpp
    Returns: H, K, C, input_shape, kernel_shape
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) != 4 or parts[0] != "dwconv":
            raise ValueError(f"Invalid depthwise conv filename: {file_name}")

        H, K, C = map(int, parts[1:4])
        return {
            "file": file_name,
            "input_height": H,
            "kernel_size": K,
            "input_channels": C,
            "input_shape": [H, H, C],
            "kernel_shape": [K, K, C],
            "output_height": H - K + 1,
            "output_width": H - K + 1,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one depthwise_conv2d kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[DWConv] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[DWConv] Patch failed {file_name}: {e}"

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
        return config, False, f"[DWConv] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled depthwise_conv2d kernel."""
    try:
        input_height, kernel_size, input_channels = config["args"][:3]
        file_name = config["file"]

        # Define the input tensor, kernel, and parameters
        input_tensor = np.random.rand(
            input_height, input_height, input_channels
        ).astype(np.float32)
        kernel = np.random.rand(kernel_size, kernel_size, input_channels).astype(
            np.float32
        )

        # Calculate the output tensor shape
        output_height = input_height - kernel_size + 1
        output_width = input_height - kernel_size + 1

        # Create an empty output tensor
        output_ctypes = np.zeros(
            (output_height, output_width, input_channels), dtype=np.float32
        )

        # Convert the arrays to contiguous memory for ctypes
        input_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # Calculate the result using numpy for comparison
        output_np = depthwise_conv2d(input_tensor, kernel).astype("float32")

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "depthwiseconv", None)
        if not func:
            return False, f"[DWConv] Function 'depthwiseconv' not found in {so_path}"

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        func.restype = None

        # Call kernel
        func(input_ptr, kernel_ptr, output_ptr)

        if np.allclose(
            output_ctypes,
            output_np,
            rtol=1e-03,
            atol=1e-03,
            equal_nan=True
        ):
            return (True, f"[DWConv] ✅ {file_name}")
        else:
            diff = np.abs(output_ctypes - output_np)
            max_diff = diff.max()
            return (
                False,
                f"[DWConv] FAILED❌: {file_name} | Max diff: {max_diff:.2e}",
            )

    except Exception as e:
        return False, f"[DWConv]❌ Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[DepthwiseConv2D] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[DWConv] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[DWConv] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[DWConv] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[Depthconv] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[Depthconv] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test depthwise_conv2d kernels"
    )
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

    # Filter and parse dwconv kernels
    configs = [c for c in configs if c.get("op_name") == "depthwiseconv"]
    dwconv_configs = [
        {**config, "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp"}
        for config in configs
    ]

    if not dwconv_configs:
        logger.warning("No valid 'depthwiseconv' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        dwconv_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} Depthwise Conv2D tests passed!")
        exit(0)
    else:
        logger.error(
            f"❌ {total - passed}/{total} Depthwise Conv2D tests failed."
        )
        exit(1)
