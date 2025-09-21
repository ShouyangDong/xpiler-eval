"""
CUDA-specific unit test for element-wise max kernel.
Supports --config and --target cuda, compiles .cu to .so using nvcc.
"""
import argparse
import ctypes
import json
import os
import sys

import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


# --- Golden Reference ---
def element_wise_max(A, B):
    return torch.maximum(A, B)


def main():
    parser = argparse.ArgumentParser(description="CUDA max kernel tester")
    parser.add_argument("--file", required=True, help="Path to the .cu source file (e.g., max_64_64.cu)")
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

    op_name = config["op_name"]        # e.g., "max_kernel"
    shape = config["args"]             # e.g., [64, 64]
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
    expected = element_wise_max(A, B).to("cpu")

    # --- Flatten and get ctypes pointers ---
    A_flat = A.flatten().numpy()
    B_flat = B.flatten().numpy()
    result_flat = torch.zeros_like(expected).flatten().numpy()

    ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
    A_ptr = A_flat.ctypes.data_as(ctypes.POINTER(ctype))
    B_ptr = B_flat.ctypes.data_as(ctypes.POINTER(ctype))
    out_ptr = result_flat.ctypes.data_as(ctypes.POINTER(ctype))

    # ========================================================
    # ✅ Step 1: Input is .cu → Output is .so
    # ========================================================
    cu_file = args.file

    if not os.path.exists(cu_file):
        print(f"[ERROR] Source file not found: {cu_file}", file=sys.stderr)
        sys.exit(1)

    if not cu_file.endswith(".cu"):
        print(f"[ERROR] Expected a .cu file, got: {cu_file}", file=sys.stderr)
        sys.exit(1)

    so_file = cu_file.replace(".cu", ".so")

    # --- Compile only if .so missing or older than .cu ---
    compile_needed = True
    if os.path.exists(so_file):
        cu_mtime = os.path.getmtime(cu_file)
        so_mtime = os.path.getmtime(so_file)
        if so_mtime > cu_mtime:
            compile_needed = False

    if compile_needed:
        print(f"⚙️ Compiling {cu_file} → {so_file} using nvcc")
        with open(cu_file, "r") as f:
            code = f.read()

        # Patch with macros
        patched_code = macro + code
        temp_file = cu_file.replace(".cu", "_patched.cu")
        with open(temp_file, "w") as f:
            f.write(patched_code)

        success, output = run_compilation(so_file, temp_file)
        os.remove(temp_file)

        if not success:
            print(f"[ERROR] Compilation failed:\n{output}", file=sys.stderr)
            sys.exit(1)

    # ========================================================
    # ✅ Step 2: Load the .so library
    # ========================================================
    try:
        lib = ctypes.CDLL(so_file)
        kernel_func = lib[op_name + "_kernel"]  # Use op_name from config
    except KeyError:
        print(f"[ERROR] Function '{op_name}' not found in {so_file}.", file=sys.stderr)
        print(f"Available symbols may not include '{op_name}'. Check kernel export.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load {so_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # ✅ Step 3: Set function signature
    # ========================================================
    kernel_func.argtypes = [
        ctypes.POINTER(ctype),  # A
        ctypes.POINTER(ctype),  # B
        ctypes.POINTER(ctype)   # out
    ]
    kernel_func.restype = None

    # ========================================================
    # ✅ Step 4: Call the kernel
    # ========================================================
    try:
        print(f"🚀 Running {op_name} kernel on CUDA (via .so)...")
        kernel_func(A_ptr, B_ptr, out_ptr)
    except Exception as e:
        print(f"[ERROR] Kernel call failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # ✅ Step 5: Reshape and verify
    # ========================================================
    result_tensor = torch.from_numpy(result_flat).reshape(expected.shape).to("cpu")

    rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (1e-2, 5e-2)
    if torch.allclose(result_tensor, expected, rtol=rtol, atol=atol):
        print("✅ Verification successful! CUDA max kernel matches PyTorch.")
        sys.exit(0)
    else:
        print("❌ Verification failed!")
        diff = (result_tensor - expected).abs()
        print(f"Max error: {diff.max().item()}")
        print(f"Sample (expected vs got):\n{expected[0, :5]}\n{result_tensor[0, :5]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
