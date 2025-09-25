import argparse
import ctypes
import os
import subprocess

import torch  

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GEMV HIP kernel output against PyTorch")
    parser.add_argument("--file", type=str, help="Path to the source .hip file")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to kernel configuration file"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform for compilation"
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # Kernel name, e.g., "gemv"
    shapes = base_name.split(".")[0]
    shape = [int(dim) for dim in shapes.split("_")[1:]]  # [M, N] where A is (M x N), x is (N,)
    
    M, N = shape  # M: rows, N: cols

    # Create input tensors using PyTorch
    A = torch.randn(M, N, dtype=torch.float32)
    x = torch.randn(N, dtype=torch.float32)

    # Reference result using PyTorch matmul
    y_torch = torch.matmul(A, x)  # Shape: (M,)

    # Ensure tensors are contiguous for ctypes
    A_cont = A.contiguous()
    x_cont = x.contiguous()
    y_torch_cont = y_torch.contiguous()

    # Allocate output tensor
    y_ctypes_torch = torch.zeros(M, dtype=torch.float32).contiguous()

    # Get raw pointers
    A_ptr = ctypes.cast(A_cont.data_ptr(), ctypes.POINTER(ctypes.c_float))
    x_ptr = ctypes.cast(x_cont.data_ptr(), ctypes.POINTER(ctypes.c_float))
    y_ptr = ctypes.cast(y_ctypes_torch.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Shared library name
    so_name = args.file.replace(".hip", ".so")

    # Read and modify source code
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code  # Inject macros (e.g., config constants)

    # Write to temporary .hip file
    file_name = args.file.replace(base_name.replace(".hip", ""), base_name + "_bak.hip")
    with open(file_name, "w") as f:
        f.write(code)

    # Compile the kernel
    success, output = run_compilation(so_name, file_name)
    if not success:
        print("[ERROR] Compilation failed:")
        print(output)
        exit(1)

    # Clean up temporary source file
    os.remove(file_name)

    # Load the compiled shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")

    # Define function signature
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A (M x N matrix)
        ctypes.POINTER(ctypes.c_float),  # x (N vector)
        ctypes.POINTER(ctypes.c_float),  # y (M vector, output)
        ctypes.c_int,                    # M (rows)
        ctypes.c_int,                    # N (cols)
    ]
    function.restype = None

    # Call the GEMV kernel
    function(A_ptr, x_ptr, y_ptr, M, N)

    # Verify results
    if torch.allclose(y_ctypes_torch, y_torch, rtol=1e-3, atol=1e-3, equal_nan=True):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        print("Expected (PyTorch):", y_torch.tolist())
        print("Got (Kernel):", y_ctypes_torch.tolist())
        exit(1)

    # Clean up compiled library
    subprocess.run(["rm", so_name], check=False)
