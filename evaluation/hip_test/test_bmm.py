import argparse
import ctypes
import os
import subprocess

import numpy as np

from benchmark.template.hip_host_template import create_hip_func
from benchmark.utils import run_hip_compilation as run_compilation


# Define the batch matrix multiplication function using numpy
def batch_matmul(A, B):
    return np.matmul(A.astype(np.int16), B.astype(np.int16)).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Generate random matrices for testing
    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
    A = np.ones((batch_size, matrix_dim_i, matrix_dim_j), dtype=np.int8)
    B = np.ones((batch_size, matrix_dim_j, matrix_dim_k), dtype=np.int8)

    # Perform batch matrix multiplication using numpy
    result_np = batch_matmul(A, B)

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    name = base_name.split("_")[0]
    so_name = args.file.replace(".hip", ".so")
    file_name = create_hip_func(args.file, op_type="matmul")
    # Load the shared library with the batch matrix multiplication function
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
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
    function(
        A_ptr,
        B_ptr,
        output_ptr,
        np.prod([batch_size, matrix_dim_i, matrix_dim_j]),
        np.prod([batch_size, matrix_dim_j, matrix_dim_k]),
        np.prod((batch_size, matrix_dim_i, matrix_dim_k)),
    )
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
