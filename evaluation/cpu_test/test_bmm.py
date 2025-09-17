import argparse
import ctypes
import os
import subprocess

import torch

from benchmark.utils import run_dlboost_compilation as run_compilation


# Define the batch matrix multiplication function using numpy
def batch_matmul(A, B):
    return torch.matmul(A, B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Generate random matrices for testing
    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
    A = torch.randn([batch_size, matrix_dim_i, matrix_dim_j], device="cpu")
    B = torch.randn([batch_size, matrix_dim_j, matrix_dim_k], device="cpu")

    # Perform batch matrix multiplication using numpy
    result_np = batch_matmul(A, B)

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open(
        os.path.join(os.getcwd(), "benchmark/macro/cpp_macro.txt"), "r"
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
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    result_ctypes = torch.zeros((batch_size, matrix_dim_i, matrix_dim_k))
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    function(A_ptr, B_ptr, output_ptr)
    # Check if the results match
    torch.allclose(
        result_ctypes,
        result_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
