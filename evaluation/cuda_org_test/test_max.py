"""CUDA-specific unit test for element-wise max kernel.

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


# --- Golden Reference: reduce max along axis ---
def reduce_max(input_tensor, axis):
    return torch.max(input_tensor, dim=axis)[0]  # 返回 values，忽略 indices


def main():
    parser = argparse.ArgumentParser(description="CUDA max kernel tester")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the .cu source file (e.g., max_64_64.cu)",
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to kernel config"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda"],
        help="Must be 'cuda' for this script",
    )

    args = parser.parse_args()

    # --- Parse config ---
    try:
        if os.path.exists(args.config) and args.config.endswith(".json"):
            with open(args.config, "r") as f:
                config = json.load(f)
        else:
            config = json.loads(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to parse config: {e}", file=sys.stderr)
        sys.exit(1)

    op_name = config["op_name"]
    shape = config["args"]
    axis = config["axes"]
    dtype_str = config.get("dtype", "float32")

    print(
        f"🔍 Testing {op_name.upper()} on CUDA with shape {shape}, dtype={dtype_str}, axes={axis}"
    )

    # --- Device and dtype setup ---
    "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print(
            "[WARNING] CUDA not available, falling back to CPU for verification."
        )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # --- Generate input tensors on CPU (for ctypes) ---
    A = torch.randn(*shape, dtype=dtype, device="cpu") * 10
    expected = reduce_max(A, axis)

    # --- Flatten and get ctypes pointers ---
    A_flat = A.flatten().numpy()
    result_flat = torch.zeros_like(expected).flatten().numpy()

    ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
    A_ptr = A_flat.ctypes.data_as(ctypes.POINTER(ctype))
    out_ptr = result_flat.ctypes.data_as(ctypes.POINTER(ctype))

    # ========================================================
    # ✅ Step 1: Input is .cu file → Output is .so file
    # ========================================================
    cu_file = args.file
    if not os.path.exists(cu_file):
        print(f"[ERROR] Source file not found: {cu_file}", file=sys.stderr)
        sys.exit(1)

    if not cu_file.endswith(".cu"):
        print(f"[ERROR] Expected a .cu file, got: {cu_file}", file=sys.stderr)
        sys.exit(1)

    # e.g., max_64_64.cu → max_64_64.so
    so_file = cu_file.replace(".cu", ".so")

    # --- Compile .cu to .so if .so doesn't exist or is older ---
    compile_needed = True
    if os.path.exists(so_file):
        cu_mtime = os.path.getmtime(cu_file)
        so_mtime = os.path.getmtime(so_file)
        if so_mtime > cu_mtime:
            compile_needed = False  # .so is up-to-date

    if compile_needed:
        print(f"⚙️ Compiling {cu_file} → {so_file} using nvcc")
        with open(cu_file, "r") as f:
            code = f.read()

        # Patch with macros
        patched_code = macro + code
        temp_file = cu_file.replace(".cu", "_patched.cu")
        with open(temp_file, "w") as f:
            f.write(patched_code)

        # Run compilation: nvcc -> .so
        success, output = run_compilation(so_file, temp_file)
        os.remove(temp_file)  # Clean up

        if not success:
            print(f"[ERROR] Compilation failed:\n{output}", file=sys.stderr)
            sys.exit(1)

    # ========================================================
    # ✅ Step 2: Load the .so library
    # ========================================================
    try:
        lib = ctypes.CDLL(so_file)
        kernel_func = lib[op_name + "_kernel"]  # Get function by name
    except Exception as e:
        print(
            f"[ERROR] Failed to load or find function '{op_name}' in {so_file}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ========================================================
    # ✅ Step 3: Set function signature
    # ========================================================
    kernel_func.argtypes = [
        ctypes.POINTER(ctype),  # A
        ctypes.POINTER(ctype),  # out
    ]
    kernel_func.restype = None

    # ========================================================
    # ✅ Step 4: Call the kernel
    # ========================================================
    try:
        print(f"🚀 Running {op_name} kernel on CUDA (via .so)...")
        kernel_func(A_ptr, out_ptr)
    except Exception as e:
        print(f"[ERROR] Kernel call failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # ✅ Step 5: Reshape and verify
    # ========================================================
    result_tensor = (
        torch.from_numpy(result_flat).reshape(expected.shape).to("cpu")
    )

    # Verification
    rtol, atol = (1e-3, 1e-3) if dtype_str == "float32" else (1e-2, 5e-2)
    if torch.allclose(result_tensor, expected, rtol=rtol, atol=atol):
        print("✅ Verification successful! CUDA max kernel matches PyTorch.")
        subprocess.run(["rm", so_file], check=False)
        sys.exit(0)
    else:
        print("❌ Verification failed!")
        diff = (result_tensor - expected).abs()
        print(f"Max error: {diff.max().item()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
