import argparse
import ctypes
import os
import subprocess

import numpy as np

from benchmark.template.cuda_host_template import create_cuda_func
from benchmark.utils import run_cuda_compilation as run_compilation


def ref_program(x):
    # Apply softmax operation to the last dimension.
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def verify_softmax(base_name, file, shape):
    A = np.random.rand(*shape).astype("float32")
    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_np = ref_program(A)

    so_name = file.replace(".cu", ".so")

    file_name = create_cuda_func(file)
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    name = base_name.split("_")[0]
    function = getattr(lib, name + "_kernel")
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    result_ctypes = np.zeros(shape, dtype=np.float32)
    output_ptr = result_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    function(A_ptr, output_ptr, np.prod(shape))
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
    subprocess.run(["rm", so_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    verify_softmax(base_name, args.file, shape)
