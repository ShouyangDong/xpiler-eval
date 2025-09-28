import argparse
import ctypes
import os
import subprocess
import re
import json
import torch

from evaluation.macros import MLU_MACROS as macro
from evaluation.utils import run_mlu_compilation as run_compilation


def scatter_reference(self_tensor, indices_tensor, src_tensor, dim):
    """
    Mimic torch.Tensor.scatter_
    self.scatter_(dim, indices, src)
    """
    result = self_tensor.clone()
    result.scatter_(dim, indices_tensor, src_tensor)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to C++ source file")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=["cuda", "hip", "bang", "cpu"], required=True)
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    kernel_name = base_name.split("_")[0]
    config =json.loads(args.config)
    dim = config["axis"]
    try:
        shape_str = base_name.replace(f".mlu", "")
        N, C, H, W = map(int, shape_str.split("_")[1:5])
    except Exception as e:
        raise ValueError(f"Invalid filename format: {base_name}") from e

    print(f"Testing {kernel_name.upper()} | Shape: [{N},{C},{H},{W}] | Dim: {dim}")

    # Create tensors
    self_tensor = torch.rand(N, C, H, W, dtype=torch.float32)
    src_tensor = torch.rand(N, C, H, W, dtype=torch.float32)
    # indices must be within valid range for the target dimension
    size_dim = [N, C, H, W][dim]
    indices_tensor = torch.randint(0, size_dim, (N, C, H, W), dtype=torch.int32)

    # Golden reference
    expected = scatter_reference(self_tensor, indices_tensor, src_tensor, dim)

    # Prepare pointers
    self_ptr = self_tensor.flatten().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    indices_ptr = indices_tensor.flatten().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    src_ptr = src_tensor.flatten().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_flat = torch.zeros_like(self_tensor).flatten()
    output_ptr = output_flat.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    so_name = args.file.replace(".mlu", ".so")

    # Inject macros and save temp file
    with open(args.file, "r") as f:
        code = f.read()
    code = macro + code

    temp_file = args.file.replace(".mlu", "_bak.mlu")
    with open(temp_file, "w") as f:
        f.write(code)

    print(f"Compiling {temp_file} -> {so_name}")
    success, log = run_compilation(so_name, temp_file, target=args.target)
    if not success:
        print("Compilation failed:")
        print(log)
        exit(1)

    os.remove(temp_file)

    # Load shared library
    lib = ctypes.CDLL(so_name)
    kernel_func = getattr(lib, kernel_name)
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # self
        ctypes.POINTER(ctypes.c_int),    # indices
        ctypes.POINTER(ctypes.c_float),  # src
        ctypes.POINTER(ctypes.c_float)   # output
    ]
    kernel_func.restype = None

    print("Running scatter kernel (torch.scatter_ semantics)...")
    kernel_func(self_ptr, indices_ptr, src_ptr, output_ptr)

    result_reshaped = output_flat.reshape(expected.shape)

    # Verify
    is_correct = torch.allclose(result_reshaped, expected, rtol=1e-4, atol=1e-4)

    if is_correct:
        print("Verification successful!")
    else:
        print("Verification failed!")
        diff = (result_reshaped - expected).abs()
        print("Max error:", diff.max().item())
        print("Sample expected (top-left):", expected[0, 0, :2, :2])
        print("Sample got (top-left):     ", result_reshaped[0, 0, :2, :2])

    # Cleanup
    subprocess.run(["rm", so_name], check=False)
