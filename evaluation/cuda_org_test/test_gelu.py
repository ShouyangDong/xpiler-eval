import argparse
import ctypes
import os
import subprocess
from ctypes import CDLL

import torch
import torch.nn as nn

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def ref_program(x: torch.Tensor) -> torch.Tensor:
    """GELU activation function reference implementation using PyTorch.

    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    gelu = nn.GELU()
    return gelu(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate GELU CUDA kernel output against PyTorch"
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
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform for compilation",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = "gelu"  # Kernel name
    shapes = base_name.split(".")[0]
    shape = [int(dim) for dim in shapes.split("_")[1:]]

    so_name = args.file.replace(".cu", ".so")

    # Read and modify source code
    with open(args.file, "r") as f:
        code = f.read()

    # Inject macros (e.g., configuration constants)
    code = macro + code

    # Write modified code to temporary file
    file_name = args.file.replace(
        base_name.replace(".cu", ""), base_name + "_bak.cu"
    )
    with open(file_name, "w") as f:
        f.write(code)

    # Compile the CUDA kernel
    success, output = run_compilation(so_name, file_name)
    if not success:
        print("[ERROR] Compilation failed:")
        print(output)
        exit(1)

    # Clean up temporary source file
    os.remove(file_name)

    # Load the compiled shared library
    lib = CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")

    # Define function signature
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Input array (float32)
        ctypes.POINTER(ctypes.c_float),  # Output array (float32)
        ctypes.c_int,  # Total number of elements
    ]
    function.restype = None

    # Generate input tensor using PyTorch
    input_tensor = torch.rand(shape, dtype=torch.float32)
    total_elements = input_tensor.numel()
    expected_output = ref_program(input_tensor)

    # Prepare output tensor
    output_tensor = torch.zeros_like(input_tensor).contiguous()

    # Ensure input is contiguous and get raw pointers
    input_ptr = ctypes.cast(
        input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = ctypes.cast(
        output_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

    # Call the compiled GELU kernel
    function(input_ptr, output_ptr, total_elements)

    # Verify results
    if torch.allclose(
        output_tensor, expected_output, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        # Optional: print first few values
        print("Expected (PyTorch):", expected_output.flatten()[:10].tolist())
        print("Got (Kernel):", output_tensor.flatten()[:10].tolist())
        exit(1)

    # Clean up shared library
    subprocess.run(["rm", so_name], check=False)
