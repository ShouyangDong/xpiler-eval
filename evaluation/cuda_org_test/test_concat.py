import argparse
import ctypes
import json
import os
import subprocess

import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def concat_reference(tensors, axis):
    return torch.cat(tensors, dim=axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", type=str, required=True, help="Path to C++ file"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target", choices=["cuda", "hip", "bang", "cpu"], required=True
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]
    config = json.loads(args.config)
    axis = config["axis"]
    try:
        shape_parts = base_name.replace(f".cu", "").split("_")
        N, C, H, W = map(int, shape_parts[1:5])
    except Exception as e:
        raise ValueError(f"Invalid filename format: {base_name}") from e

    print(f"Testing {name.upper()} | Shape: [{N},{C},{H},{W}] | Axis: {axis}")

    input1 = torch.rand(N, C, H, W, dtype=torch.float32)
    input2 = torch.rand(N, C, H, W, dtype=torch.float32)

    expected = concat_reference([input1, input2], axis=axis)

    input1_ptr = (
        input1.flatten().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    input2_ptr = (
        input2.flatten().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    output_flat = torch.zeros_like(expected.flatten())
    output_ptr = output_flat.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    so_name = args.file.replace(".cu", ".so")

    with open(args.file, "r") as f:
        code = f.read()
    code = macro + code

    temp_file = args.file.replace(".cu", "_bak.cu")
    with open(temp_file, "w") as f:
        f.write(code)

    print(f"Compiling {temp_file} -> {so_name}")
    success, log = run_compilation(so_name, temp_file)
    if not success:
        print("Compilation failed:")
        print(log)
        exit(1)

    os.remove(temp_file)

    lib = ctypes.CDLL(so_name)
    kernel_func = getattr(lib, name + "_kernel")
    kernel_func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
    kernel_func.restype = None

    print("Running static concat kernel...")
    kernel_func(input1_ptr, input2_ptr, output_ptr)

    result_reshaped = output_flat.reshape(expected.shape)

    is_correct = torch.allclose(
        result_reshaped, expected, rtol=1e-4, atol=1e-4
    )

    if is_correct:
        print("Verification successful!")
    else:
        print("Verification failed!")
        diff = (result_reshaped - expected).abs()
        print("Max error:", diff.max().item())
        print(
            "Sample expected:\n",
            expected[0, :3, :2, :2] if axis == 1 else expected[0, 0, :2, :2],
        )
        print(
            "Sample got:\n",
            (
                result_reshaped[0, :3, :2, :2]
                if axis == 1
                else result_reshaped[0, 0, :2, :2]
            ),
        )

    subprocess.run(["rm", so_name], check=False)
