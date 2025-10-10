import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation


def ref_program(x):
    """Reference Softmax function using PyTorch.

    Applies softmax along the last dimension. Numerically stable (uses log-sum-
    exp trick internally).
    """
    return torch.softmax(x, dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Softmax HIP kernel output against PyTorch"
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
    name = "softmax"
    shapes = base_name.split(".")[0]
    shape = [
        int(dim) for dim in shapes.split("_")[1:]
    ]  # e.g., [32, 100], [2, 8, 64]
    # Total number of rows
    batch_size = int(torch.prod(torch.tensor(shape[:-1])))
    # Size of last dimension
    hidden_size = shape[-1]
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
        ctypes.c_int,  # batch_size (number of rows)
        ctypes.c_int,  # hidden_size (softmax dimension)
    ]
    function.restype = None

    # Create input tensor
    dtype = torch.float32
    # Use a wider range to test numerical stability
    # Larger values to stress-test exp overflow
    input_tensor = torch.randn(shape, dtype=dtype) * 10

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

    # Call the Softmax kernel
    function(input_ptr, output_ptr, batch_size, hidden_size)

    # Verify results
    if torch.allclose(
        output_tensor, expected_output, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")

        # Debug: Print first row of input and outputs
        print(
            f"Input (first row): {input_tensor.view(-1, hidden_size)[0].tolist()}"
        )
        print(
            f"Expected (Ref):    {expected_output.view(-1, hidden_size)[0].tolist()}"
        )
        print(
            f"Got (Kernel):      {output_tensor.view(-1, hidden_size)[0].tolist()}"
        )

        # Additional check: sum of softmax should be ~1.0
        kernel_row_sum = output_tensor.view(-1, hidden_size)[0].sum().item()
        ref_row_sum = expected_output.view(-1, hidden_size)[0].sum().item()
        print(
            f"Sum of first row (Kernel): {kernel_row_sum:.6f} (should be ~1.0)"
        )
        print(f"Sum of first row (Ref):    {ref_row_sum:.6f} (should be ~1.0)")

        exit(1)

    # Clean up compiled library
    subprocess.run(["rm", so_name], check=False)
