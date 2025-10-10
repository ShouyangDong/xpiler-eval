import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate HIP GEMM kernel output against PyTorch reference"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the input .hip source file",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to kernel configuration file",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform for compilation",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Parse M, K, N from filename (e.g., gemm_128_256_64.hip)
    # --------------------------------------------------
    base_name = os.path.basename(args.file)
    name_prefix = base_name.split("_")[0]  # e.g., "gemm"
    shape_str = base_name.split(".")[0]  # e.g., "gemm_128_256_64"

    try:
        M, K, N = [int(dim) for dim in shape_str.split("_")[1:]]
    except Exception as e:
        print(
            f"[ERROR] Failed to parse M, K, N from filename '{base_name}': {e}"
        )
        exit(1)

    so_name = args.file.replace(".hip", ".so")
    temp_file = args.file.replace(".hip", "_bak.hip")

    # --------------------------------------------------
    # 2. Check if ROCm (HIP) is available
    # --------------------------------------------------
    if not torch.cuda.is_available():
        print(
            "[ERROR] ROCm is not available. Please install PyTorch with ROCm support."
        )
        exit(1)

    # PyTorch uses "cuda" even for AMD GPUs under ROCm
    device = torch.device("cuda")

    # --------------------------------------------------
    # 3. Generate input tensors and compute reference output
    # --------------------------------------------------
    torch.manual_seed(1234)
    A = torch.randn(M, K, dtype=torch.float32, device=device)
    x = torch.randn(K, N, dtype=torch.float32, device=device)

    # Reference output using PyTorch
    y_torch = torch.matmul(A, x).cpu()  # Move to CPU for comparison

    # Output buffer (on CPU, to be filled by the kernel if needed)
    y_kernel = torch.zeros(
        M, N, dtype=torch.float32, device="cpu"
    ).contiguous()

    # --------------------------------------------------
    # 4. Inject macros and compile the HIP file
    # --------------------------------------------------
    try:
        with open(args.file, "r") as f:
            code = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read file '{args.file}': {e}")
        exit(1)

    code = macro + code

    try:
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        print(f"[ERROR] Failed to write temporary file '{temp_file}': {e}")
        exit(1)

    success, compile_output = run_compilation(so_name, temp_file)
    if not success:
        print("[ERROR] Compilation failed:")
        print(compile_output)
        exit(1)

    # Clean up temporary .hip file
    os.remove(temp_file)

    # --------------------------------------------------
    # 5. Load the compiled shared library
    # --------------------------------------------------
    try:
        lib = ctypes.CDLL(so_name)
    except Exception as e:
        print(f"[ERROR] Failed to load shared library '{so_name}': {e}")
        exit(1)

    # Assume kernel function name is: {prefix}_kernel (e.g., gemm_kernel)
    kernel_func_name = f"{name_prefix}_kernel"
    try:
        kernel_func = getattr(lib, kernel_func_name)
    except AttributeError:
        print(
            f"[ERROR] Function '{kernel_func_name}' not found in the compiled library."
        )
        print(f"Available symbols may not include '{kernel_func_name}'.")
        exit(1)

    # Define argument types
    # Inputs A, x, y are raw pointers (void*), followed by M, K, N
    kernel_func.argtypes = [
        ctypes.c_void_p,  # A (GPU pointer)
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # y (output pointer, could be CPU or GPU)
        ctypes.c_int,  # M
        ctypes.c_int,  # K
        ctypes.c_int,  # N
    ]
    kernel_func.restype = None

    # --------------------------------------------------
    # 6. Call the HIP kernel
    # --------------------------------------------------
    # Use .data_ptr() to get raw GPU/CPU memory address (as int,
    # auto-converted by ctypes)
    kernel_func(A.data_ptr(), x.data_ptr(), y_kernel.data_ptr(), M, K, N)

    # Synchronize to ensure kernel completion
    torch.cuda.synchronize()

    # --------------------------------------------------
    # 7. Verify correctness
    # --------------------------------------------------
    if torch.allclose(y_kernel, y_torch, rtol=1e-3, atol=1e-3, equal_nan=True):
        print(
            "✅ Verification successful! Results match between HIP kernel and PyTorch."
        )
    else:
        max_diff = (y_kernel - y_torch).abs().max().item()
        print("❌ Verification failed! Results do not match.")
        print(f"Expected shape: {y_torch.shape}, Got: {y_kernel.shape}")
        print(f"Max absolute difference: {max_diff:.2e}")
        exit(1)

    # --------------------------------------------------
    # 8. Cleanup: Remove the generated .so file
    # --------------------------------------------------
    try:
        subprocess.run(["rm", so_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Failed to remove '{so_name}': {e}")
