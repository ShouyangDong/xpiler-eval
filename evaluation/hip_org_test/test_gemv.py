import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate GEMV HIP kernel output against PyTorch"
    )
    parser.add_argument(
        "--file", type=str, help="Path to the source .hip file"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to kernel configuration file",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform for compilation",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # Kernel name, e.g., "gemv"
    shapes = base_name.split(".")[0]
    shape = [int(dim) for dim in shapes.split("_")[1:]]  # [M, N]

    M, N = shape

    # Set device to AMD GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(
            "[ERROR] ROCm not available. Please install PyTorch with ROCm support."
        )
        exit(1)

    # Create tensors on AMD GPU
    A = torch.randn(M, N, dtype=torch.float32, device=device)
    x = torch.randn(N, dtype=torch.float32, device=device)

    # Perform matmul on AMD GPU
    y_torch = torch.matmul(A, x)  # Shape: (M,)

    # Move reference result back to CPU for comparison
    y_torch_cpu = y_torch.cpu().contiguous()

    # Host tensors for kernel input (float32)
    A_host = A.cpu().contiguous()
    x_host = x.cpu().contiguous()

    # Output tensor on CPU
    y_ctypes = torch.zeros(M, dtype=torch.float32).contiguous()

    # Get raw pointers (CPU memory)
    A_ptr = ctypes.cast(A_host.data_ptr(), ctypes.POINTER(ctypes.c_float))
    x_ptr = ctypes.cast(x_host.data_ptr(), ctypes.POINTER(ctypes.c_float))
    y_ptr = ctypes.cast(y_ctypes.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Shared library name
    so_name = args.file.replace(".hip", ".so")

    # Read and inject macros
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    # Write to temporary .hip file
    file_name = args.file.replace(
        base_name.replace(".hip", ""), base_name + "_bak.hip"
    )
    with open(file_name, "w") as f:
        f.write(code)

    # Compile kernel
    success, output = run_compilation(so_name, file_name)
    if not success:
        print("[ERROR] Compilation failed:")
        print(output)
        exit(1)

    os.remove(file_name)

    # Load and call kernel
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    kernel_func = getattr(lib, name + "_kernel")

    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel_func.restype = None

    kernel_func(A_ptr, x_ptr, y_ptr, M, N)

    # Verify results
    if torch.allclose(
        y_ctypes, y_torch_cpu, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        exit(1)

    # Clean up
    subprocess.run(["rm", so_name], check=False)
