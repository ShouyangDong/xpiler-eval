import argparse
import ctypes
import os
import subprocess
from ctypes import CDLL

import numpy as np
import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def ref_program(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


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
    name = "rmsnorm"
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".cu", ""), base_name + "_bak.cu"
    )
    with open(file_name, mode="w") as f:
        f.write(code)

    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    lib = CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Create the input array.
    dtype = "float32"
    input_array = np.random.uniform(size=shape).astype(dtype)
    expected_output = ref_program(torch.from_numpy(input_array))

    # Create the output array.
    output_array = np.zeros_like(input_array)

    # Convert the input and output arrays into C pointer types.
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Invoke the C function
    function(input_ptr, output_ptr, shape[0], shape[1])
    # Verification results

    np.testing.assert_allclose(
        output_array,
        expected_output,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
