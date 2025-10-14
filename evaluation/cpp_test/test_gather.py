"""Batch correctness tester for GATHER kernels with two-phase parallelism."""

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


def reference_gather(
    params: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Reference implementation using PyTorch.

    Out-of-bound indices are clamped and set to zero.
    """
    clamped_indices = torch.clamp(indices, 0, params.size(0) - 1)
    result = params[clamped_indices]
    # Zero out out-of-bound values
    out_of_bound = (indices < 0) | (indices >= params.size(0))
    if out_of_bound.any():
        result[out_of_bound] = 0.0
    return result


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: gather_100_32_16.cpp
    Format: gather_params_batch_params_len_indices_len.cpp
    Returns: dict with shapes and metadata.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) != 4 or parts[0] != "gather":
            raise ValueError(f"Invalid Gather filename: {file_name}")

        params_batch, params_len, indices_len = map(int, parts[1:4])
        return {
            "file": file_name,
            "params_batch": params_batch,
            "params_len": params_len,
            "indices_len": indices_len,
            "params_shape": [params_batch, params_len],
            "indices_shape": [indices_len],
            "output_shape": [indices_len, params_len],
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one Gather kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[Gather] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[Gather] Patch failed {file_name}: {e}"

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
        return config, False, f"[Gather] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled Gather kernel."""
    try:
        B, L, I = config["args"]
        file_name = config["file"]
        op_name = config["op_name"]
        # Generate inputs
        torch.manual_seed(1234)
        params = torch.randn(B, L, dtype=torch.float32)
        indices = torch.randint(
            low=-1, high=B + 1, size=(I,), dtype=torch.int32
        )

        # Ensure contiguous
        params = params.contiguous()
        indices = indices.contiguous()

        # Reference output
        ref = reference_gather(params, indices).numpy()

        # Output buffer
        output = torch.zeros(I, L, dtype=torch.float32).contiguous().numpy()

        # Get pointers
        params_ptr = params.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        indices_ptr = indices.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)
        )
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return False, f"[Gather] Function 'gather' not found in {so_path}"

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # params
            ctypes.POINTER(ctypes.c_int32),  # indices
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,  # params_batch
            ctypes.c_int,  # params_len
            ctypes.c_int,  # indices_len
        ]
        func.restype = None

        # Call kernel
        func(params_ptr, indices_ptr, output_ptr, B, L, I)

        # Compare
        diff = torch.from_numpy(output) - torch.from_numpy(ref)
        max_abs_err = diff.abs().max().item()

        if max_abs_err < 1e-5:
            return (
                True,
                f"[Gather] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        else:
            return (
                False,
                f"[Gather] FAILED❌: {file_name} | Max error: {max_abs_err:.2e}",
            )

    except Exception as e:
        return False, f"[Gather] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[Gather] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[Gather] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[Gather] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[Gather] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[Gather] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[Gather] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GATHER kernels (CPU)")
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

    # Filter and parse gather kernels
    configs = [c for c in configs if c.get("op_name") == "gather"]
    gather_configs = [
        {
            **config,
            "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp",
        }
        for config in configs
    ]

    if not gather_configs:
        logger.warning("No valid 'gather' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        gather_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} GATHER tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} GATHER tests failed.")
        exit(1)
