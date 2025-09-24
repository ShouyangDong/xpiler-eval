import argparse
import ctypes
import os
import subprocess

import torch
import torch.nn.functional as F

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def batchnorm_inference(
    input, weight, bias, running_mean, running_var, eps=1e-5
):
    """
    PyTorch golden reference for BatchNorm inference
    input: (N, C, H, W)
    weight (gamma), bias (beta), running_mean, running_var: (C,)
    """
    return F.batch_norm(
        input,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
        eps=eps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the C++ source file (e.g., batchnorm_1_3_224_224.cu)",
    )
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
    name = base_name.split("_")[0]  # "batchnorm"
    shapes_str = base_name.split(".")[0]
    shape_parts = [int(x) for x in shapes_str.split("_")[1:]]

    # 假设形状为 NCHW
    N, C, H, W = shape_parts
    total_elements = N * C * H * W

    print(
        f"🔍 Testing {name.upper()} with shape [N,C,H,W] = [{N},{C},{H},{W}]"
    )

    # Generate random input
    input_tensor = torch.rand(N, C, H, W, dtype=torch.float32)

    # Fixed parameters (from training)
    running_mean = torch.rand(C, dtype=torch.float32)
    running_var = torch.rand(C, dtype=torch.float32) + 0.5
    weight = torch.rand(C, dtype=torch.float32)  # gamma
    bias = torch.rand(C, dtype=torch.float32)  # beta
    eps = 1e-5

    # Golden reference
    expected = batchnorm_inference(
        input_tensor, weight, bias, running_mean, running_var, eps
    )

    # Flatten input for C++ (row-major)
    input_flat = input_tensor.flatten()  # (N*C*H*W,)
    input_ptr = input_flat.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Prepare parameter pointers
    mean_ptr = running_mean.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    var_ptr = running_var.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    weight_ptr = weight.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = bias.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Output tensor
    result_ctypes = torch.zeros_like(input_flat)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Shared library name
    so_name = args.file.replace(".cu", ".so")

    # Read and patch C++ code
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    temp_file_name = args.file.replace(".cu", "_bak.cu")
    with open(temp_file_name, "w") as f:
        f.write(code)

    # Compile
    print(f"⚙️ Compiling {temp_file_name} -> {so_name}")
    success, compile_output = run_compilation(so_name, temp_file_name)
    if not success:
        print("❌ Compilation failed:")
        print(compile_output)
        exit(1)

    os.remove(temp_file_name)

    # Load library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    kernel_func = getattr(lib, name + "_kernel")  # batchnorm

    # Function signature
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.POINTER(ctypes.c_float),  # mean     (C,)
        ctypes.POINTER(ctypes.c_float),  # var      (C,)
        ctypes.POINTER(ctypes.c_float),  # weight   (C,)
        ctypes.POINTER(ctypes.c_float),  # bias     (C,)
        ctypes.c_int,  # N
        ctypes.c_int,  # C
        ctypes.c_int,  # H
        ctypes.c_int,  # W
        ctypes.c_float,  # epsilon
    ]
    kernel_func.restype = None

    # Call kernel
    print(f"🚀 Running {name.upper()} kernel...")
    kernel_func(
        input_ptr,
        output_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        N,
        C,
        H,
        W,
        eps,
    )

    # Reshape result for comparison
    result_reshaped = result_ctypes.reshape(N, C, H, W)

    # Verify
    is_correct = torch.allclose(
        result_reshaped, expected, rtol=1e-3, atol=1e-3, equal_nan=True
    )

    if is_correct:
        print("✅ Verification successful! C++ BatchNorm matches PyTorch.")
    else:
        print("❌ Verification failed!")
        diff = (result_reshaped - expected).abs()
        print("Max error:", diff.max().item())
        print("Sample expected:", expected[0, 0, :3, :3])
        print("Sample got:     ", result_reshaped[0, 0, :3, :3])

    # Clean up
    subprocess.run(["rm", so_name], check=False)
