import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Element-wise addition using PyTorch."""
    return torch.add(A, B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate CUDA kernel output against PyTorch"
    )
    parser.add_argument("--file", type=str, help="Path to the source .cu file")
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
    shape = [
        int(dim) for dim in shapes.split("_")[1:]
    ]  # Extract shape from filename
    name = base_name.split("_")[0]  # Extract kernel name

    # Generate random input tensors on CPU using PyTorch
    A = torch.rand(shape, dtype=torch.float32)
    B = torch.rand(shape, dtype=torch.float32)

    # Compute expected result using PyTorch
    result_torch = add(A, B)

    # Ensure tensors are contiguous in memory for ctypes pointer access
    A_cont = A.contiguous()
    B_cont = B.contiguous()
    result_torch_cont = result_torch.contiguous()

    # Get raw pointers to tensor data
    A_ptr = ctypes.cast(A_cont.data_ptr(), ctypes.POINTER(ctypes.c_float))
    B_ptr = ctypes.cast(B_cont.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Output shared library name
    so_name = args.file.replace(".cu", ".so")

    # Read original CUDA source
    with open(args.file, "r") as f:
        code = f.read()

    # Prepend macro definitions (e.g., for configuration)
    code = macro + code

    # Create a backup .cu file with macros injected
    file_name = args.file.replace(
        base_name.replace(".cu", ""), base_name + "_bak.cu"
    )
    with open(file_name, "w") as f:
        f.write(code)

    # Compile the CUDA kernel into a shared library (.so)
    success, output = run_compilation(so_name, file_name)
    if not success:
        print("[ERROR] Compilation failed:")
        print(output)
        exit(1)

    # Clean up the temporary source file
    os.remove(file_name)

    # Load the compiled shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")

    # Define the function signature
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Input A
        ctypes.POINTER(ctypes.c_float),  # Input B
        ctypes.POINTER(ctypes.c_float),  # Output
        ctypes.c_int,  # Total number of elements
    ]
    function.restype = None

    # Prepare output tensor (contiguous, CPU)
    result_ctypes_torch = torch.zeros(shape, dtype=torch.float32).contiguous()
    output_ptr = ctypes.cast(
        result_ctypes_torch.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

    # Call the CUDA kernel function
    # Use .numel() for total elements
    function(A_ptr, B_ptr, output_ptr, A.numel())

    # Compare kernel output with PyTorch result
    if torch.allclose(
        result_ctypes_torch, result_torch, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        # Optional: print first few elements for debugging
        print("Expected (PyTorch):", result_torch.flatten()[:10].tolist())
        print("Got (Kernel):", result_ctypes_torch.flatten()[:10].tolist())
        exit(1)

    # Clean up the compiled shared library
    subprocess.run(["rm", so_name], check=False)
