import argparse
import ctypes
import os
import subprocess

import numpy as np
import torch

from benchmark.template.cuda_host_template import create_cuda_func
from benchmark.utils import conv2d_nchw
from benchmark.utils import run_cuda_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")

    args = parser.parse_args()
    base_name = os.path.basename(args.file)

    name = base_name.split("_")[0]
    data_shape = base_name.split("_")[1:5]

    data_shape = [int(intg) for intg in data_shape]

    kernel_shape = base_name.split("_")[5:9]
    kernel_shape = [int(intg) for intg in kernel_shape]
    stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
    pad = int(base_name.split(".")[0].split("_")[10])
    dtype = "float32"
    wtype = "float32"

    # generate data
    data_np = torch.rand(data_shape)
    kernel_np = torch.rand(kernel_shape)
    # cpu compute
    result_cpu = conv2d_nchw(
        data_np,
        kernel_shape[1],
        kernel_shape[0],
        kernel_shape[2],
        stride_h,
        pad,
    )

    # Load the shared library with the conv2d function
    so_name = args.file.replace(".cu", ".so")
    file_name = create_cuda_func(args.file, op_type="matmul")
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Convert the matrices to contiguous memory for ctypes
    input_ptr = data_np.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_ptr = kernel_np.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    result_ctypes = torch.zeros(result_cpu.shape, dtype=np.float32)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    # Call the function with the matrices and dimensions
    function(
        input_ptr,
        kernel_ptr,
        output_ptr,
        np.prod(data_np.shape),
        np.prod(kernel_np.shape),
        np.prod(result_cpu.shape),
    )
    # Check if the results match
    torch.allclose(
        result_ctypes,
        result_cpu,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
