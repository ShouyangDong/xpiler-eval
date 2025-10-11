"""Batch correctness tester for 'deformable_attention' kernels with two-phase
parallelism."""

import argparse
import ctypes
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
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


@torch.no_grad()
def deformable_attention_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """PyTorch reference implementation of multi-scale deformable attention."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split(
        [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # Reshape value: [N_, H_*W_, M_, D_] -> [N_*M_, D_, H_, W_]
        value_l_ = (
            value_list[lid_]
            .flatten(2)
            .transpose(1, 2)
            .reshape(N_ * M_, D_, H_, W_)
        )
        # Reshape sampling grid: [N_, Lq_, M_, P_, 2] -> [N_*M_, Lq_, P_, 2]
        sampling_grid_l_ = (
            sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        )
        # Sample using bilinear interpolation
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [N_*M_, D_, Lq_, P_]
        sampling_value_list.append(sampling_value_l_)

    # Stack and apply attention weights
    # [N_, M_, Lq_, L_, P_] -> [N_*M_, 1, Lq_, L_*P_]
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    # Stack sampled values: [N_*M_, D_, Lq_, L_, P_] -> sum over last two dims
    output = (
        (
            torch.stack(sampling_value_list, dim=-2).flatten(-2)
            * attention_weights
        )
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


def parse_filename(file_name: str) -> Dict:
    """
    Parse filename: deformable_attention_N_M_D_Lq_L_P.cpp
    Example: deformable_attention_1_8_64_300_4_4.cpp
    Returns: N, M, D, Lq, L, P
    Note: spatial shapes are hardcoded here (as in original), or can be loaded from config.
    """
    try:
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        if len(parts) < 7 or parts[0] != "deformable_attention":
            raise ValueError(
                f"Invalid deformable_attention filename: {file_name}"
            )

        N, M, D, Lq, L, P = map(int, parts[1:7])
        return {
            "file": file_name,
            "N": N,
            "M": M,
            "D": D,
            "Lq": Lq,
            "L": L,
            "P": P,
            "input_shape": [N, None, M, D],  # S will be inferred from shapes
        }
    except Exception as e:
        raise ValueError(f"Failed to parse {file_name}: {e}")


def compile_kernel(config: dict, source_dir: str) -> Tuple[dict, bool, str]:
    """Compile one deformable_attention kernel.

    Returns: (config, success, message_or_so_path)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)
    so_path = file_path.replace(".cpp", ".so")
    temp_file = file_path.replace(".cpp", "_patched.cpp")

    if not os.path.isfile(file_path):
        return config, False, f"[DA] File not found: {file_path}"

    try:
        with open(file_path, "r") as f:
            code = f.read()
        code = macro + code
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        return config, False, f"[DA] Patch failed {file_name}: {e}"

    success, msg = run_compilation(so_path, temp_file)
    if success:
        os.remove(temp_file)
        return config, True, so_path
    else:
        os.remove(temp_file)
        return config, False, f"[DA] Compile failed {file_name}: {msg}"


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled deformable_attention kernel."""
    try:
        file_name = config["file"]
        N, M, D, Lq, L, P = config["args"]

        # Hardcoded spatial shapes (same as your example)
        # You can make this configurable via JSON if needed
        shapes = torch.tensor(
            [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long
        )
        S = sum((H * W).item() for H, W in shapes)

        # Generate inputs
        value = torch.rand(N, S, M, D, dtype=torch.float32) * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2, dtype=torch.float32)
        attention_weights = (
            torch.rand(N, Lq, M, L, P, dtype=torch.float32) + 1e-5
        )
        attention_weights /= attention_weights.sum(dim=-1, keepdim=True).sum(
            dim=-2, keepdim=True
        )

        # Reference output
        torch_da = deformable_attention_pytorch(
            value, shapes, sampling_locations, attention_weights
        )

        # Prepare output buffer
        output_array = np.zeros((N, Lq, M * D), dtype=np.float32)

        # Get pointers
        value_ptr = value.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        shapes_ptr = (
            shapes.numpy()
            .astype(np.int32)
            .ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )
        sampling_locs_ptr = sampling_locations.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        attn_weights_ptr = attention_weights.numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        output_ptr = output_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )

        # Load shared library
        lib = ctypes.CDLL(so_path)
        func = getattr(lib, "deformable", None)
        if not func:
            return (
                False,
                f"[DA] Function 'deformable' not found in {so_path}",
            )

        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # value
            ctypes.POINTER(ctypes.c_int),  # shapes (int array of [L, 2])
            ctypes.POINTER(ctypes.c_float),  # sampling_locations
            ctypes.POINTER(ctypes.c_float),  # attention_weights
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        func.restype = None

        # Call kernel
        func(
            value_ptr,
            shapes_ptr,
            sampling_locs_ptr,
            attn_weights_ptr,
            output_ptr,
        )

        # Compare
        if np.allclose(
            output_array,
            torch_da.numpy(),
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True,
        ):
            return True, f"[DA] PASSED✅: {file_name}"
        else:
            max_error = np.max(np.abs(output_array - torch_da.numpy()))
            return (
                False,
                f"[DA] FAILED❌: {file_name} | Max error: {max_error:.2e}",
            )

    except Exception as e:
        return False, f"[DA] Exception in test {file_name}: {str(e)}"


def run_tests(
    configs: List[dict], source_dir: str, target: str, num_workers: int = 4
) -> List[Tuple[bool, str]]:
    """
    Two-phase test:
    Phase 1: Compile all kernels in parallel.
    Phase 2: Test only successfully compiled ones.
    """
    logger.info(
        f"[DeformableAttention] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_map = {}  # file -> so_path
    results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(f"[DA] Phase 1/2: Compiling {len(configs)} kernels...")
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
        f"[DA] Compilation: {len(compiled_map)} succeeded, {len([r for r in results if not r[0]])} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_map:
        logger.info(
            f"[DA] Phase 2/2: Testing {len(compiled_map)} compiled kernels..."
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
    parser = argparse.ArgumentParser(
        description="Test Deformable Attention kernels"
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

    # Filter and parse deformable_attention kernels
    configs = [c for c in configs if c.get("op_name") == "deformable"]
    da_configs = [
        {**config, "file": f"{config['op_name']}_{'_'.join(map(str, config['args']))}.cpp"}
        for config in configs
    ]

    if not da_configs:
        logger.warning(
            "No valid 'deformable_attention' kernels found in config."
        )
        exit(0)

    # Run tests
    results = run_tests(
        da_configs, args.source_dir, args.target, num_workers=args.jobs
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
        logger.info(f"🎉 All {total} Deformable Attention tests passed!")
        exit(0)
    else:
        logger.error(
            f"❌ {total - passed}/{total} Deformable Attention tests failed."
        )
        exit(1)
