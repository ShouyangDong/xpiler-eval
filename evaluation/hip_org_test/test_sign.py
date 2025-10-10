import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation


def ref_program(x):
    """Reference implementation of the sign function using PyTorch.

    Returns: +1 if x > 0, -1 if x < 0, 0 if x == 0.
    """
    return torch.sign(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Sign HIP kernel output against PyTorch"
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
    name = "sign"
    shapes = base_name.split(".")[0]
    shape = [
        int(dim) for dim in shapes.split("_")[1:]
    ]  # e.g., [1024], [32, 768]
    so_name = args.file.replace(".hip", ".so")

    # Read and modify source code
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code  # Inject macros (e.g., config constants)

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
        ctypes.POINTER(ctypes.c_float),  # input array
        ctypes.POINTER(ctypes.c_float),  # output array
        ctypes.c_int,  # total number of elements
    ]
    function.restype = None

    # Create input tensor
    dtype = torch.float32
    # Use a mix of positive, negative, and zero values to test edge cases
    input_tensor = torch.randn(shape, dtype=dtype)
    total_elements = input_tensor.numel()
    # Manually set some values to zero to test edge case
    input_tensor[input_tensor.abs() < 0.1] = 0.0  # Force some zeros

    # Compute reference output using PyTorch
    expected_output = ref_program(input_tensor)

    # Output tensor
    output_tensor = torch.zeros_like(input_tensor)

    # Ensure contiguous memory layout for ctypes access
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # Get raw pointers
    input_ptr = ctypes.cast(
        input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = ctypes.cast(
        output_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

    # Call the Sign kernel
    function(input_ptr, output_ptr, total_elements)

    # Verify results
    if torch.allclose(
        output_tensor, expected_output, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        # Print first 10 elements for debugging
        print("Input (first 10):", input_tensor.flatten()[:10].tolist())
        print("Expected (Ref):", expected_output.flatten()[:10].tolist())
        print("Got (Kernel):", output_tensor.flatten()[:10].tolist())
        exit(1)

    # Clean up compiled library
    subprocess.run(["rm", so_name], check=False)
