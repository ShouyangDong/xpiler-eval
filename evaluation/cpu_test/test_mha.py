import argparse
import ctypes
import os
import subprocess
from ctypes import CDLL

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import run_dlboost_compilation as run_compilation


def ref_program(q, k, v, causal=False):
    return F.scaled_dot_product_attention(q, k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = "mha"
    causal = False
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    dtype = torch.float32

    query = torch.randn(shape).to(dtype)
    key = torch.randn(shape).to(dtype)
    value = torch.randn(shape).to(dtype)

    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".cpp", ""), base_name + "_bak.cpp"
    )
    with open(file_name, mode="w") as f:
        f.write(code)

    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    lib = CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name)
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # Create the input array.
    expected_output = ref_program(query, key, value)

    # Create the output array.
    output_array = np.zeros_like(query.numpy())
    # Convert the input and output arrays to C pointer types.
    input_ptr_q = query.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_k = key.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_v = value.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Invoke the C function
    function(input_ptr_q, input_ptr_k, input_ptr_v, output_ptr)
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
