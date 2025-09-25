import argparse
import ctypes
import os
import subprocess

import torch  

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GEMM HIP kernel output against PyTorch")
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
    shapes = base_name.split(".")[0]
    shape = [int(dim) for dim in shapes.split("_")[1:]]  # [M, K, N]
    name = base_name.split("_")[0]  # Kernel name prefix (e.g., 'gemm')

    M, K, N = shape  # Matrix dimensions: A (M x K), x (K x N), y (M x N)

    # Create input tensors using PyTorch (float16 for input, common in mixed-precision)
    A = torch.ones((M, K), dtype=torch.float16)
    x = torch.ones((K, N), dtype=torch.float16)

    # Compute reference result using PyTorch matmul
    y_torch = torch.matmul(A, x).to(torch.float32)  # Result is in float32 (promotion)

    # Ensure tensors are contiguous for ctypes pointer access
    A_cont = A.contiguous()
    x_cont = x.contiguous()
    y_torch_cont = y_torch.contiguous()

    # Get raw pointers (float16 stored as uint16 in memory)
    A_ptr = ctypes.cast(A_cont.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    x_ptr = ctypes.cast(x_cont.data_ptr(), ctypes.POINTER(ctypes.c_uint16))

    # Output tensor (float32)
    y_ctypes_torch = torch.zeros((M, N), dtype=torch.float32).contiguous()
    y_ptr = ctypes.cast(y_ctypes_torch.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Shared library name
    so_name = args.file.replace(".hip", ".so")

    # Read and modify source code
    with open(args.file, "r") as f:
        code = f.read()

    # Inject macros (e.g., config constants)
    code = macro + code

    # Write to temporary .hip file
    file_name = args.file.replace(base_name.replace(".hip", ""), base_name + "_bak.hip")
    with open(file_name, "w") as f:
        f.write(code)

    # Compile the HIP kernel
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
        ctypes.POINTER(ctypes.c_uint16),  # A (float16 data)
        ctypes.POINTER(ctypes.c_uint16),  # x (float16 data)
        ctypes.POINTER(ctypes.c_float),   # y (float32 output)
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_int,                     # N
    ]
    function.restype = None

    # Call the GEMM kernel
    function(A_ptr, x_ptr, y_ptr, M, K, N)

    # Verify results
    if torch.allclose(y_ctypes_torch, y_torch, rtol=1e-3, atol=1e-3, equal_nan=True):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        print("Expected (PyTorch):", y_torch.flatten().tolist())
        print("Got (Kernel):", y_ctypes_torch.flatten().tolist())
        exit(1)

    # Clean up compiled library
    subprocess.run(["rm", so_name], check=False)
