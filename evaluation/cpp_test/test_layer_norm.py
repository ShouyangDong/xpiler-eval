"""Batch correctness tester for LayerNorm kernels with parallel compilation and
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


def ref_program(x, gamma, beta, eps=1e-5):
    # Using PyTorch to compute layer normalization
    x_tensor = torch.tensor(x)
    layer_norm = torch.nn.LayerNorm(
        x_tensor.size()[1:]
    )  # Initialize LayerNorm, maintaining dimensions.
    x_normalized = layer_norm(x_tensor)

    # Calculate output
    out = gamma * x_normalized + beta
    # Return the output in numpy format to maintain interface consistency.
    return out


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
        elif len(dims) == 2:
            N, D = dims
            shape = [N, D]
        else:
            raise ValueError(f"Unsupported LayerNorm shape: {dims}")

        total = 1
        for d in shape:
            total *= d

        return {
            "file": file_name,
            "shape": shape,
            "B": dims[0],
            "H": dims[-1],
            "normalized_dim": dims[-1],
            "total_elements": total,
            "ndim": len(shape),
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one LayerNorm kernel."""
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = os.path.join(source_dir, file_name.replace(".cpp", ".so"))
    temp_file = os.path.join(
        source_dir, file_name.replace(".cpp", "_patched.cpp")
    )

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
        except BaseException:
            pass
        return config, True, so_path
    else:
        try:
            os.remove(temp_file)
        except BaseException:
            pass
        return config, False, f"[LAYERNORM] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled LayerNorm kernel."""
    try:
        shape = config["args"]
        file_name = config["file"]
        op_name = config["op_name"]
        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, op_name, None)
        if not func:
            return (
                False,
                f"[LAYERNORM] Function 'layernorm' not found in {so_path}",
            )

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
        # Create the input array.
        input_array = torch.randn(shape)
        gamma_array = torch.randn(shape[-1:])
        beta_array = torch.randn(shape[-1:])

        # Use the modified ref_program for layer normalization calculation.
        expected_output = ref_program(input_array, gamma_array, beta_array)

        # Create the output array.
        output_array = torch.zeros(shape)

        # Convert the input and output arrays to C pointer types.
        input_ptr = input_array.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        gamma_ptr = gamma_array.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        beta_ptr = beta_array.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = output_array.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # If the call to the C func can be preserved:
        func(input_ptr, gamma_ptr, beta_ptr, output_ptr)

        try:
            torch.allclose(
                output_array,
                expected_output,
                rtol=1e-3,
                atol=1e-3,
                equal_nan=False,
            )
            max_abs_err = (output_array - expected_output).abs().max().item()
            return (
                True,
                f"[LAYERNORM] ✅ {file_name}| Max error: {max_abs_err:.2e}",
            )
        except Exception as e:
            return False, f"[LAYERNORM] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[LAYERNORM] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """Two-phase test: compile all → test all."""
    logger.info(
        f"[LAYERNORM] Starting two-phase test for {len(configs)} kernels..."
    )

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

    logger.info(
        f"[LAYERNORM] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[LAYERNORM] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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

        logger.debug("[LAYERNORM] Cleaning up generated .so files...")
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[LAYERNORM] Failed to delete {so_path}: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LayerNorm kernels")
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

    # Filter and parse LayerNorm kernels
    configs = [c for c in configs if c.get("op_name") == "layernorm"]
    norm_configs = [
        {
            **config,
            "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp",
        }
        for config in configs
    ]

    if not norm_configs:
        logger.warning("No valid 'layernorm' kernels found in config.")
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
        logger.info(f"🎉 All {total} LayerNorm tests passed!")
        exit(0)
    else:
        logger.error(f"❌ {total - passed}/{total} LayerNorm tests failed.")
        exit(1)
