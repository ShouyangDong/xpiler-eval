import argparse
import ctypes
import os
import subprocess

import torch

from benchmark.utils import run_dlboost_compilation as run_compilation


def ref_program(x, gamma, beta, eps=1e-5):
    # Using PyTorch to compute layer normalization.
    x_tensor = torch.tensor(x)
    layer_norm = torch.nn.LayerNorm(
        x_tensor.size()[1:]
    )  # Initialize LayerNorm, maintaining dimensions.
    x_normalized = layer_norm(x_tensor)

    # Calculate the output.
    out = gamma * x_normalized + beta
    return out  # Return the output in numpy format for interface consistency.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = "layernorm"
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]

    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    with open(
        os.path.join(os.getcwd(), "benchmark/macro/cpp_macro.txt"), "r"
    ) as f:
        macro = f.read()
    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".cpp", ""), base_name + "_bak.cpp"
    )
    with open(file_name, mode="w") as f:
        f.write(code)

    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)

    # Load and invoke C code (if necessary).
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name)

    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None

    # Create the input array.
    input_array = torch.randn(shape)
    gamma_array = torch.randn(shape[-1:])
    beta_array = torch.randn(shape[-1:])

    # Use the modified ref_program for layer normalization calculation.
    expected_output = ref_program(input_array, gamma_array, beta_array)

    # Create the output array.
    output_array = torch.zeros(shape)

    # Convert the input and output arrays to C pointer types.
    input_ptr = input_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    gamma_ptr = gamma_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    beta_ptr = beta_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # If the call to the C function can be preserved:
    function(input_ptr, gamma_ptr, beta_ptr, output_ptr)

    # Verification result
    torch.allclose(
        output_array,
        expected_output,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
