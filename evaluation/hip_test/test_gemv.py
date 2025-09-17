import argparse
import ctypes
import os
import subprocess

import numpy as np

from benchmark.template.hip_host_template import create_hip_func
from benchmark.utils import run_hip_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Generate random matrices for testing
    # Define the input matrix A and vector x
    A = np.random.rand(*shape).astype(np.float32)
    x = np.random.rand(shape[1]).astype(np.float32)

    # Create an empty vector y
    y_ctypes = np.zeros(shape[0], dtype=np.float32)

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_ptr = y_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Perform gemm using numpy
    y_np = np.matmul(A, x)

    # Load the shared library with the batch matrix multiplication function
    so_name = args.file.replace(".hip", ".so")
    file_name = create_hip_func(args.file, op_type="matmul")
    # Load the shared library with the batch matrix multiplication function
    success, output = run_compilation(so_name, file_name)

    os.remove(file_name)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, "gemv_kernel")
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(A_ptr, x_ptr, y_ptr, np.prod(shape), shape[1], shape[0])
    # Check if the results match
    np.testing.assert_allclose(
        y_ctypes,
        y_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
