"""Batch correctness tester for 'avgpool' kernels with two-phase
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


def avgpool_ref(
    input_tensor: torch.Tensor, kernel_stride: List[int]
) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    kh, kw, sh, sw = kernel_stride
    input_nhwc = input_tensor.permute(0, 3, 1, 2)  # NCHW
    pool = torch.nn.AvgPool2d(kernel_size=(kh, kw), stride=(sh, sw))
    output_tensor = pool(input_nhwc)
    return output_tensor.permute(0, 2, 3, 1)  # back to NHWC


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename like: avgpool_N_H_W_C_kh_kw_sh_sw.cpp
    Returns shape and kernel_stride.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 6:
            raise ValueError(f"Invalid filename format: {file_name}")

        N, H, W, C = map(int, parts[1:5])
        kh, kw, sh, sw = map(int, parts[5:9])

        return {
            "shape": [N, H, W, C],
            "kernel_stride": [kh, kw, sh, sw],
            "file": file_name,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one avgpool kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[AvgPool] File not found: {file_path}"

    # Patch source with macros
    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[AvgPool] Patch failed {file_name}: {e}"

    # Compile
    success, msg = run_compilation(so_path, temp_file)
    if success:
        os.remove(temp_file)
        return config, True, so_path  # return .so path for testing
    else:
        os.remove(temp_file)
        return config, False, f"[AvgPool] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled avgpool kernel."""
    try:
        file_name = config["file"]
        shape = config["args"][:4]
        kernel_stride = config["args"][4:8]
        kh, kw, sh, sw = kernel_stride

        # Generate input
        input_tensor = torch.randn(*shape, dtype=torch.float32, device="cpu")
        ref_output = avgpool_ref(input_tensor, kernel_stride)

        # Prepare output buffer
        out_shape = [
            shape[0],
            (shape[1] - kh) // sh + 1,
            (shape[2] - kw) // sw + 1,
            shape[3],
        ]
        output_tensor = torch.zeros(out_shape, dtype=torch.float32)
        input_ptr = input_tensor.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = output_tensor.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # Load and call kernel
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "avgpool", None)
        if not func:
            return (
                False,
                f"[AvgPool] Function 'avgpool' not found in {so_path}",
            )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        func.restype = None
        func(input_ptr, output_ptr)

        # Verify
        if torch.allclose(
            output_tensor, ref_output, rtol=1e-3, atol=1e-3, equal_nan=True
        ):
            return True, f"[AvgPool] PASSED✅: {file_name}"
        else:
            return False, f"[AvgPool] FAILED❌: {file_name} (mismatch)"

    except Exception as e:
        return False, f"[AvgPool] Exception in test {config['file']}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Phase 1: Compile all avgpool kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[AvgPool] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}  # file -> so_path
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[AvgPool] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[AvgPool] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[AvgPool] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[AVGPOOL] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[AVGPOOL] Failed to delete {so_path}: {e}")
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

    # Filter only 'avgpool' kernels
    avgpool_configs = [c for c in configs if c.get("op_name") == "avgpool"]

    avgpool_configs = [
        {**config, "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp"}
        for config in avgpool_configs
    ]

    if not avgpool_configs:
        logger.warning("No valid 'avgpool' kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        avgpool_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} avgpool tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} avgpool tests failed.")
        exit(1)
