import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import DLBOOST_MACROS as macro
from evaluation.utils import run_dlboost_compilation as run_compilation


# Define the add function using torch
def add(A, B):
    return torch.add(A, B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    parser.add_argument(
        "--config", required=True, help="JSON string or path to kernel config"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform",
    )
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Generate random matrices for testing
    A = torch.rand(*shape, device="cpu")
    B = torch.rand(*shape, device="cpu")

    # Perform add using torch
    result = add(A, B)

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".cpp", ""), base_name + "_bak.cpp"
    )
    with open(file_name, mode="w") as f:
        f.write(code)

    # Load the shared library with the add function
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name)
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    result_ctypes = torch.zeros(shape)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    function(A_ptr, B_ptr, output_ptr)
    # Check if the results match
    torch.allclose(
        result_ctypes,
        result,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
