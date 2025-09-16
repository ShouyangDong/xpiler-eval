import argparse
import ctypes
import math
import os
import subprocess
from ctypes import CDLL

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.template.mlu_host_template import create_mlu_func
from benchmark.utils import run_mlu_compilation as run_compilation


def ref_program(q, k, v, causal=False):
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        mask = torch.triu(
            torch.ones(score.shape[-2], score.shape[-1]), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, torch.finfo(q.dtype).min)
        mask = mask.to(q.device, q.dtype)
        score = score + mask
    attn = F.softmax(score, dim=-1)
    output = torch.matmul(attn, v)
    return output


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

    query = torch.randn(shape).to(dtype).contiguous()
    key = torch.randn(shape).to(dtype).contiguous()
    value = torch.randn(shape).to(dtype).contiguous()
    file_name = create_mlu_func(args.file)
    so_name = args.file.replace(".mlu", ".so")
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    lib = CDLL(os.path.join(os.getcwd(), so_name))
    # Obtain function handle
    function = getattr(lib, name + "_kernel")
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
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

    # Invoke the CUDA kernel.
    function(input_ptr_q, input_ptr_k, input_ptr_v, output_ptr, np.prod(shape))
    # Verification results

    # Verification results
    np.testing.assert_allclose(
        output_array,
        expected_output.numpy(),
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
