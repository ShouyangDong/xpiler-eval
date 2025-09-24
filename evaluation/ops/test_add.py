#!/usr/bin/env python3
"""Unified test script for 'add' operator across 4 platforms: cuda, hip, bang,
cpu Uses --config to get shape/dtype, compiles .so, runs and verifies with
PyTorch."""
import argparse
import ctypes
import json
import os
import sys

import torch

from evaluation.macros import DLBOOST_MACROS as macro
from evaluation.utils import run_dlboost_compilation as run_compilation


# --- Golden Reference Function ---
def add(A, B):
    return torch.add(A, B)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--so-file", required=True, help="Path to compiled .so file"
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to kernel config"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform",
    )

    args = parser.parse_args()

    # --- Parse config ---
    try:
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                config = json.load(f)
        else:
            config = json.loads(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to parse config: {e}")
        sys.exit(1)

    op_name = config["op_name"]
    shape = config["args"]
    dtype_str = config.get("dtype", "float32")

    # Map dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # Device mapping
    device_map = {
        "cpu": "cpu",
        "cuda": "cuda",
        "hip": "cuda",  # ROCm uses cuda device in PyTorch
        "bang": "cpu",  # MLU data generated on CPU
    }
    device_map[args.target]

    # --- Generate input tensors ---
    A = torch.randn(*shape, dtype=dtype, device="cpu")
    B = torch.randn(*shape, dtype=dtype, device="cpu")
    golden = add(A, B).to("cpu")

    # --- Compile .so if not exists ---
    so_file = args.so_file
    if not os.path.exists(so_file):
        src_file = so_file.replace(".so", "")

        if not os.path.exists(src_file):
            print(f"[ERROR] Source file not found: {src_file}")
            sys.exit(1)

        # Read and patch source
        with open(src_file, "r") as f:
            code = f.read()

        patched_code = macro + code
        temp_file = src_file + "_patched"
        with open(temp_file, "w") as f:
            f.write(patched_code)

        success, output = run_compilation(so_file, temp_file)
        os.remove(temp_file)

        if not success:
            print(f"[ERROR] Compilation failed:\n{output}")
            sys.exit(1)

    # --- Load .so ---
    try:
        lib = ctypes.CDLL(so_file)
        kernel_func = getattr(lib, op_name)  # e.g., 'add'
    except Exception as e:
        print(f"[ERROR] Failed to load {so_file}: {e}")
        sys.exit(1)

    # --- Set ctypes types ---
    ctype_map = {
        "float32": ctypes.c_float,
        "float16": ctypes.c_ushort,  # FP16 stored as uint16
        "bfloat16": ctypes.c_ushort,
        "int32": ctypes.c_int,
    }
    ctype = ctype_map.get(dtype_str, ctypes.c_float)

    # Flatten and get pointers
    A_flat = A.flatten().numpy()
    B_flat = B.flatten().numpy()
    result_flat = torch.zeros_like(golden).flatten().numpy()

    A_ptr = A_flat.ctypes.data_as(ctypes.POINTER(ctype))
    B_ptr = B_flat.ctypes.data_as(ctypes.POINTER(ctype))
    out_ptr = result_flat.ctypes.data_as(ctypes.POINTER(ctype))

    # Set function signature
    kernel_func.argtypes = [
        ctypes.POINTER(ctype),
        ctypes.POINTER(ctype),
        ctypes.POINTER(ctype),
    ]
    kernel_func.restype = None

    # --- Call kernel ---
    try:
        kernel_func(A_ptr, B_ptr, out_ptr)
    except Exception as e:
        print(f"[ERROR] Kernel call failed: {e}")
        sys.exit(1)

    # --- Reshape result ---
    result_tensor = (
        torch.from_numpy(result_flat).reshape(golden.shape).to("cpu")
    )

    # --- Verify ---
    if torch.allclose(result_tensor, golden, rtol=1e-3, atol=1e-3):
        print("Verification successful!")
        sys.exit(0)
    else:
        print("Verification failed!")
        print(f"Max diff: {(result_tensor - golden).abs().max().item()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
