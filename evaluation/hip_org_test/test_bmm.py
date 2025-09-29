import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation


def batch_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Perform batch matrix multiplication using PyTorch.

    A: (batch_size, i, j)
    B: (batch_size, j, k)
    Output: (batch_size, i, k)
    """
    return torch.matmul(A, B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate batch GEMM HIP kernel output against PyTorch"
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
    shapes = base_name.split(".")[0]
    shape = [int(dim) for dim in shapes.split("_")[1:]]
    name = base_name.split("_")[0]

    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(
            "[ERROR] ROCm not available. Please install PyTorch with ROCm support."
        )
        exit(1)

    # Create input tensors on AMD GPU
    A = torch.ones(
        (batch_size, matrix_dim_i, matrix_dim_j),
        dtype=torch.float32,
        device=device,
    )
    B = torch.ones(
        (batch_size, matrix_dim_j, matrix_dim_k),
        dtype=torch.float32,
        device=device,
    )

    # Perform batch matmul on GPU
    result_torch = batch_matmul(A, B)

    # Move reference result to CPU for comparison
    result_torch_cpu = result_torch.cpu().contiguous()

    # Host tensors for kernel (CPU memory)
    A_host = A.cpu().contiguous()
    B_host = B.cpu().contiguous()

    # Output tensor (CPU)
    result_ctypes = torch.zeros(
        (batch_size, matrix_dim_i, matrix_dim_k), dtype=torch.float32
    ).contiguous()

    # Get raw pointers (CPU memory)
    A_ptr = ctypes.cast(A_host.data_ptr(), ctypes.POINTER(ctypes.c_float))
    B_ptr = ctypes.cast(B_host.data_ptr(), ctypes.POINTER(ctypes.c_float))
    output_ptr = ctypes.cast(
        result_ctypes.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

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
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel_func.restype = None

    kernel_func(
        A_ptr,
        B_ptr,
        output_ptr,
        batch_size,
        matrix_dim_i,
        matrix_dim_j,
        matrix_dim_k,
    )

    # Verify results
    if torch.allclose(
        result_ctypes, result_torch_cpu, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        exit(1)

    # Clean up
    subprocess.run(["rm", so_name], check=False)
