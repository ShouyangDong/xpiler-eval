"""
CUDA-specific unit test for element-wise min kernel.
Supports --config and --target cuda, compiles .cu to .so using nvcc.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys

import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


# --- Golden Reference ---
def element_wise_min(A, B):
    return torch.minimum(A, B)


def main():
    parser = argparse.ArgumentParser(description="CUDA min kernel tester")
    parser.add_argument("--so-file", required=True, help="Path to compiled .so file (e.g., min_64_64.so)")
    parser.add_argument("--config", required=True, help="JSON string or path to kernel config")
    parser.add_argument("--target", required=True, choices=["cuda"], help="Must be 'cuda' for this script")

    args = parser.parse_args()

    # --- Parse config ---
    try:
        if os.path.exists(args.config) and args.config.endswith(".json"):
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            config = json.loads(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to parse config: {e}", file=sys.stderr)
        sys.exit(1)

    op_name = config["op_name"]
    shape = config["args"]
    dtype_str = config.get("dtype", "float32")

    print(f"🔍 Testing {op_name.upper()} on CUDA with shape {shape}, dtype={dtype_str}")

    # --- Device and dtype setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU for verification.")
        device = "cpu"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # --- Generate input tensors on CPU (for ctypes) ---
    A = torch.randn(*shape, dtype=dtype, device="cpu") * 10
    B = torch.randn(*shape, dtype=dtype, device="cpu") * 10
    expected = element_wise_min(A, B).to("cpu")

    # --- Get ctypes pointers ---
    A_flat = A.flatten().numpy()
    B_flat = B.flatten().numpy()
    result_flat = torch.zeros_like(expected).flatten().numpy()

    ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
    A_ptr = A_flat.ctypes.data_as(ctypes.POINTER(ctype))
    B_ptr = B_flat.ctypes.data_as(ctypes.POINTER(ctype))
    out_ptr = result_flat.ctypes.data_as(ctypes.POINTER(ctype))

    # --- Compile .so if not exists ---
    so_file = args.so_file
    src_file = so_file.replace(".so", ".cu")  # assume .cu source

    if not os.path.exists(so_file):
        if not os.path.exists(src_file):
            print(f"[ERROR] Source file not found: {src_file}", file=sys.stderr)
            sys.exit(1)

        print(f"⚙️ Compiling {src_file} -> {so_file} using nvcc")
        with open(src_file, "r") as f:
            code = f.read()

        patched_code = macro + code
        temp_file = src_file.replace(".cu", "_patched.cu")
        with open(temp_file, "w") as f:
            f.write(patched_code)

        success, output = run_compilation(so_file, temp_file)
        os.remove(temp_file)

        if not success:
            print(f"[ERROR] Compilation failed:\n{output}", file=sys.stderr)
            sys.exit(1)

    # --- Load .so ---
    try:
        lib = ctypes.CDLL(so_file)
        kernel_func = getattr(lib, op_name)
    except Exception as e:
        print(f"[ERROR] Failed to load {so_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Set function signature ---
    kernel_func.argtypes = [ctypes.POINTER(ctype), ctypes.POINTER(ctype), ctypes.POINTER(ctype)]
    kernel_func.restype = None

    # --- Call kernel ---
    try:
        print(f"🚀 Running {op_name} kernel on CUDA (via .so)...")
        kernel_func(A_ptr, B_ptr, out_ptr)
    except Exception as e:
        print(f"[ERROR] Kernel call failed: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Reshape and verify ---
    result_tensor = torch.from_numpy(result_flat).reshape(expected.shape).to("cpu")

    # --- Verification ---
    rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (1e-2, 5e-2)
    if torch.allclose(result_tensor, expected, rtol=rtol, atol=atol):
        print("✅ Verification successful! CUDA min kernel matches PyTorch.")
        sys.exit(0)
    else:
        print("❌ Verification failed!")
        diff = (result_tensor - expected).abs()
        print(f"min error: {diff.min().item()}")
        print(f"Sample (expected vs got):\n{expected[0, :5]}\n{result_tensor[0, :5]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
