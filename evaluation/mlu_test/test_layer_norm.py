import argparse
import ctypes
import os
import subprocess

import numpy as np
from benchmark.template.mlu_host_template import create_mlu_func

from evaluation.utils import run_mlu_compilation as run_compilation


def ref_program(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / (std + eps)
    out = gamma * x_normalized + beta
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = "layernorm"
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    so_name = args.file.replace(".mlu", ".so")
    file_name = create_mlu_func(args.file, op_type="layer_norm")
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Create the input array.
    dtype = "float32"
    input_array = np.random.uniform(size=shape).astype(dtype)
    gamma_array = np.random.uniform(size=shape[-1:]).astype(dtype)
    beta_array = np.random.uniform(size=shape[-1:]).astype(dtype)
    expected_output = ref_program(input_array, gamma_array, beta_array)

    # Create the output array.
    output_array = np.zeros_like(input_array)

    # Convert the input and output arrays to C pointer types.
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    gamma_ptr = gamma_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    beta_ptr = beta_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Calling a C function
    function(
        input_ptr,
        gamma_ptr,
        beta_ptr,
        output_ptr,
        np.prod(shape),
        np.prod(shape[-1:]),
    )
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
