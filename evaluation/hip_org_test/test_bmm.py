import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation


def batch_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform batch matrix multiplication using PyTorch.
    A: (batch_size, i, j)
    B: (batch_size, j, k)
    Output: (batch_size, i, k)
    """
    return torch.matmul(A, B).to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate batch GEMM HIP kernel output against PyTorch")
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
    shape = [int(dim) for dim in shapes.split("_")[1:]]
    name = base_name.split("_")[0]  # Kernel name prefix

    # Parse shape: batch_size, i, j, k
    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape

    # Generate input tensors using PyTorch (float16 for input, as in original)
    A = torch.ones((batch_size, matrix_dim_i, matrix_dim_j), dtype=torch.float16)
    B = torch.ones((batch_size, matrix_dim_j, matrix_dim_k), dtype=torch.float16)

    # Compute expected result using PyTorch
    result_torch = batch_matmul(A, B)  # Result is in float32 (matmul promotes)

    # Ensure tensors are contiguous for ctypes pointer access
    A_cont = A.contiguous()
    B_cont = B.contiguous()
    result_torch_cont = result_torch.contiguous()

    # Get raw pointers to tensor data
    # float16 is stored as uint16 in memory
    A_ptr = ctypes.cast(A_cont.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    B_ptr = ctypes.cast(B_cont.data_ptr(), ctypes.POINTER(ctypes.c_uint16))

    # Output tensor uses float32
    result_ctypes_torch = torch.zeros(
        (batch_size, matrix_dim_i, matrix_dim_k), dtype=torch.float32
    ).contiguous()
    output_ptr = ctypes.cast(result_ctypes_torch.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Shared library name
    so_name = args.file.replace(".hip", ".so")

    # Read original HIP source
    with open(args.file, "r") as f:
        code = f.read()

    # Inject macros (e.g., config definitions)
    code = macro + code

    # Create temporary .hip file with macros
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
        ctypes.POINTER(ctypes.c_uint16),  # A (float16 data as uint16)
        ctypes.POINTER(ctypes.c_uint16),  # B (float16 data as uint16)
        ctypes.POINTER(ctypes.c_float),   # Output (float32)
        ctypes.c_int,                     # batch_size
        ctypes.c_int,                     # i (matrix_dim_i)
        ctypes.c_int,                     # j (matrix_dim_j)
        ctypes.c_int,                     # k (matrix_dim_k)
    ]
    function.restype = None

    # Call the HIP kernel
    function(A_ptr, B_ptr, output_ptr, batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k)

    # Verify results
    if torch.allclose(result_ctypes_torch, result_torch, rtol=1e-3, atol=1e-3, equal_nan=True):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        # Optional: print first few elements
        print("Expected (PyTorch):", result_torch.flatten()[:10].tolist())
        print("Got (Kernel):", result_ctypes_torch.flatten()[:10].tolist())
        exit(1)

    # Clean up compiled library
    subprocess.run(["rm", so_name], check=False)
