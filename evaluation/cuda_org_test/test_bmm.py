import argparse
import ctypes
import os
import subprocess

import numpy as np

from benchmark.utils import run_cuda_compilation as run_compilation

# Define the batch matrix multiplication function using numpy


def batch_matmul(A, B):
    return np.matmul(A, B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Generate random matrices for testing
    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
    A = np.ones((batch_size, matrix_dim_i, matrix_dim_j)).astype("float16")
    B = np.ones((batch_size, matrix_dim_j, matrix_dim_k)).astype("float16")

    # Perform batch matrix multiplication using numpy
    result_np = batch_matmul(A, B)

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    name = base_name.split("_")[0]
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open(
        os.path.join(os.getcwd(), "benchmark/macro/cuda_macro.txt"), "r"
    ) as f:
        macro = f.read()
        f.close()
    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".cu", ""), base_name + "_bak.cu"
    )
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()

    # Load the shared library with the batch matrix multiplication function
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    result_ctypes = np.zeros(
        (batch_size, matrix_dim_i, matrix_dim_k), dtype=np.float32
    )
    output_ptr = result_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    function(A_ptr, B_ptr, output_ptr, *shape)
    # Check if the results match
    np.testing.assert_allclose(
        result_ctypes,
        result_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
