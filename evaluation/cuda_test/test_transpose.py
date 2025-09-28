import argparse
import ctypes
import os

import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation

# Define the transpose function using torch


def transpose(A, axes):
    return A.permute(*axes).contiguous()


def parse_filename(filename):
    """
    Parse: transpose_3_4_to_4_3.cu -> (shape=[3,4], axes=[1,0])
           transpose_2_3_4_to_4_2_3.cu -> (shape=[2,3,4], axes=[2,0,1])
    """
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., transpose_3_4_to_4_3

    # Split by '_to_'
    if "_to_" not in stem:
        raise ValueError(f"Invalid filename format: {filename}")

    in_part, out_part = stem.split("_to_")
    in_part = in_part.replace("transpose_", "")
    try:
        in_shape = list(map(int, in_part.split("_")))
        out_shape = list(map(int, out_part.split("_")))
    except ValueError as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    # Infer axes: out_shape[i] == in_shape[axes[i]]
    axes = []
    for d in out_shape:
        found = False
        for idx, sz in enumerate(in_shape):
            if sz == d and idx not in axes:
                axes.append(idx)
                found = True
                break
        if not found:
            raise ValueError(
                f"Cannot infer axes for {in_shape} -> {out_shape}"
            )
    permuted = [in_shape[i] for i in axes]
    if permuted != out_shape:
        raise ValueError(
            f"Axis inference failed: {in_shape} with {axes} -> {permuted}"
        )

    return in_shape, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test transpose CUDA kernel against PyTorch"
    )
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

    # === 1. Parse shape and axes from filename ===
    try:
        shape, axes = parse_filename(args.file)
        print(f"🔍 Parsed shape: {shape}, axes: {axes}")
    except Exception as e:
        raise RuntimeError(f"❌ Filename parsing failed: {e}")

    # === 2. Generate random input tensor ===
    A = torch.rand(*shape, dtype=torch.float32, device="cpu")
    print(f"🧪 Input tensor shape: {A.shape}")

    # === 3. Perform transpose using PyTorch ===
    result_torch = transpose(A, axes)
    print(f"✅ PyTorch output shape: {result_torch.shape}")

    # === 4. Prepare ctypes pointers ===
    A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result_torch.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # === 5. Prepare output buffer for CUDA kernel ===
    total_elements = int(torch.numel(result_torch))
    result_cuda = torch.zeros_like(result_torch)
    output_ptr = result_cuda.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # === 6. Inject macro and compile ===
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    # Write temporary file
    temp_file = args.file.replace(".cu", "_temp.cu")
    with open(temp_file, "w") as f:
        f.write(code)

    # Compile using cpp
    success, compile_output = run_compilation(so_name, temp_file)
    if not success:
        print(f"❌ Compilation failed:\n{compile_output}")
        exit(1)
    os.remove(temp_file)
    print(f"✅ Compiled to {so_name}")

    # === 7. Load shared library ===
    lib = ctypes.CDLL(so_name)
    kernel_name = f"transpose_kernel_{'_'.join(map(str, shape))}_to_{'_'.join(map(str, [shape[i] for i in axes]))}"
    try:
        kernel_func = getattr(lib, kernel_name)
    except AttributeError:
        available = [attr for attr in dir(lib) if attr.startswith("transpose")]
        print(f"❌ Kernel function '{kernel_name}' not found in .so")
        print(f"Available functions: {available}")
        exit(1)

    # === 8. Set function signature ===
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    kernel_func.restype = None

    # === 9. Call CUDA kernel ===
    print(f"🚀 Running CUDA kernel: {kernel_name}")
    kernel_func(A_ptr, output_ptr)

    # === 10. Compare results ===
    if torch.allclose(
        result_cuda, result_torch, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! CUDA result matches PyTorch.")
    else:
        print("❌ Verification failed!")
        print(f"PyTorch sum: {result_torch.flatten()[:5]}")
        print(f"CUDA result: {result_cuda.flatten()[:5]}")
        exit(1)

    # Cleanup (optional)
    if os.path.exists(so_name):
        os.remove(so_name)
        print(f"🗑️  Removed {so_name}")
