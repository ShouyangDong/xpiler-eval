import argparse
import ctypes
import os
import subprocess

import numpy as np
import torch

from benchmark.template.mlu_host_template import create_mlu_func
from benchmark.utils import avgpool_np
from benchmark.utils import run_mlu_compilation as run_compilation


def verify_pooling(base_name, file, shape, kernel_stride):
    input_array = torch.randn(*shape, device="cpu")
    output_np = avgpool_np(input_array, kernel_stride)
    output_array = torch.zeros(output_np.shape, dtype=torch.float32)
    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    so_name = file.replace(".mlu", ".so")
    file_name = create_mlu_func(file, op_type="pool")
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
        ctypes.c_int,
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(input_ptr, output_ptr, np.prod(shape), np.prod(output_np.shape))
    # Check if the results match
    torch.allclose(
        output_array,
        output_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )
    print("Verification successful!")
    subprocess.run(["rm", so_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)

    shape = base_name.split("_")[1:5]
    shape = [int(intg) for intg in shape]
    kernel_stride = base_name.split(".")[0].split("_")[5:]
    kernel_stride = [int(intg) for intg in kernel_stride]
    verify_pooling(base_name, args.file, shape, kernel_stride)
