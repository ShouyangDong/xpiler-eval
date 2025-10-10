import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation


def ref_program(A, B):
    """
    Reference implementation of element-wise subtraction: A - B
    """
    return torch.sub(A, B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate HIP element-wise subtraction kernel"
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
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform for compilation",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [
        int(dim) for dim in shapes.split("_")[1:]
    ]  # e.g., [1024], [32, 768]
    total_elements = int(
        torch.prod(torch.tensor(shape))
    )  # Total number of elements
    name = base_name.split("_")[0]  # e.g., "sub" from "sub_1024.hip"
    so_name = args.file.replace(".hip", ".so")

    # Generate random input tensors
    dtype = torch.float32
    A = torch.rand(shape, dtype=dtype)
    B = torch.rand(shape, dtype=dtype)

    # Compute reference result using PyTorch
    expected_output = ref_program(A, B)

    # Output tensor
    output_tensor = torch.zeros_like(A)

    # Ensure contiguous memory layout for ctypes access
    A = A.contiguous()
    B = B.contiguous()
    output_tensor = output_tensor.contiguous()

    # Get raw pointers
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_float))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_float))
    output_ptr = ctypes.cast(
        output_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

    # Read and modify source code
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code  # Inject macros

    # Write to temporary .hip file
    file_name = args.file.replace(
        base_name.replace(".hip", ""), base_name + "_bak.hip"
    )
    with open(file_name, "w") as f:
        f.write(code)

    # Compile the HIP kernel
    success, output = run_compilation(so_name, file_name)
    if not success:
        print("[ERROR] Compilation failed:")
        print(output)
        exit(1)

    os.remove(file_name)

    # Load the compiled shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")

    # Define function signature
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # Output
        ctypes.c_int,  # Total number of elements
    ]
    function.restype = None

    # Call the subtraction kernel
    function(A_ptr, B_ptr, output_ptr, total_elements)

    # Verify results
    if torch.allclose(
        output_tensor, expected_output, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        # Debug: Print first 10 elements
        print("A (first 10):", A.flatten()[:10].tolist())
        print("B (first 10):", B.flatten()[:10].tolist())
        print("Expected (A-B):", expected_output.flatten()[:10].tolist())
        print("Got (Kernel):", output_tensor.flatten()[:10].tolist())
        exit(1)

    # Clean up compiled library
    subprocess.run(["rm", so_name], check=False)
