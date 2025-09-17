import argparse
import ctypes
import os
import subprocess

import torch

from benchmark.utils import run_dlboost_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Generate random matrices for testing
    # Define the input matrix A and vector x
    A = torch.randn(shape[0], shape[1])
    x = torch.randn(shape[1], shape[2])

    # Create an empty vector y
    y_ctypes = torch.zeros((shape[0], shape[2]))

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    x_ptr = x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_ptr = y_ctypes.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Perform gemm using numpydd
    y_np = torch.matmul(A, x)

    # Load the shared library with the batch matrix multiplication function
    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open(
        os.path.join(os.getcwd(), "benchmark/macro/dlboost_macro.txt"), "r"
    ) as f:
        macro = f.read()
        f.close()
    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".cpp", ""), base_name + "_bak.cpp"
    )
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()
    # Load the shared library with the batch matrix multiplication function
    success, output = run_compilation(so_name, file_name)

    os.remove(file_name)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name)
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(A_ptr, x_ptr, y_ptr)
    # Check if the results match
    torch.allclose(
        y_ctypes,
        y_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
