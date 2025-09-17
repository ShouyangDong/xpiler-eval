import argparse
import ctypes
import os
import subprocess

import numpy as np

from benchmark.utils import run_dlboost_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Define the input array and kernel
    input_array = np.random.uniform(size=[shape[1]]).astype("float32")
    kernel = np.array([0.5, 1.0, 0.5]).astype(np.float32)
    # Calculate the output size
    output_size = shape[0]
    # Create an empty output array
    output_ctypes = np.zeros(output_size, dtype=np.float32)

    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Calculate the result using numpy for comparison
    output_np = np.convolve(input_array, kernel, mode="valid")

    # Load the shared library with the batch matrix multiplication function
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
    function(input_ptr, kernel_ptr, output_ptr)
    # Check if the results match
    np.testing.assert_allclose(
        output_ctypes,
        output_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
