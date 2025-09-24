import argparse
import ctypes
import os

import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_dlboost_compilation as run_compilation

# Define the min function using torch


def min_op(x, dim):
    return torch.min(x, dim=dim, keepdim=True).values  # keep same rank


def parse_filename(filename):
    """Parse shape and dim from filename like min_3_4_dim1.cu."""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., min_3_4_dim1

    if not stem.startswith("min_"):
        raise ValueError(f"Invalid filename: {filename}")

    # Split by '_dim'
    if "_dim" not in stem:
        raise ValueError(f"Missing '_dim' in filename: {stem}")

    shape_part, dim_part = stem.rsplit("_dim", 1)
    try:
        shape = list(map(int, shape_part.split("_")[1:]))  # skip 'min'
        dim = int(dim_part)
    except ValueError as e:
        raise ValueError(f"Cannot parse shape or dim from {stem}: {e}")

    if dim < 0 or dim >= len(shape):
        raise ValueError(f"Invalid dim={dim} for shape {shape}")

    print(f"🔍 Parsed shape: {shape}, dim: {dim}")
    return shape, dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test min CUDA kernel against PyTorch"
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

    # === 1. Parse shape and dim from filename ===
    try:
        shape, dim = parse_filename(args.file)
    except Exception as e:
        raise RuntimeError(f"❌ Filename parsing failed: {e}")

    # === 2. Generate random input tensor ===
    x = torch.randn(*shape, dtype=torch.float32, device="cpu")
    print(f"🧪 Input tensor shape: {x.shape}")

    # === 3. Perform min using PyTorch ===
    result_torch = min_op(x, dim)
    output_shape = result_torch.shape
    print(f"✅ PyTorch output shape: {output_shape}")

    # === 4. Prepare ctypes pointers ===
    def to_ptr(tensor):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    x_ptr = to_ptr(x)
    result_cuda = torch.zeros_like(result_torch)
    output_ptr = to_ptr(result_cuda)

    # === 5. Inject macro and compile ===
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    # Write temporary file
    temp_file = args.file.replace(".cu", "_temp.cu")
    with open(temp_file, "w") as f:
        f.write(code)

    # Compile using dlboost
    success, compile_output = run_compilation(so_name, temp_file)
    if not success:
        print(f"❌ Compilation failed:\n{compile_output}")
        exit(1)
    os.remove(temp_file)
    print(f"✅ Compiled to {so_name}")

    # === 6. Load shared library ===
    lib = ctypes.CDLL(so_name)
    # Kernel name example: min_kernel_3_4_dim1
    kernel_name = f"min_kernel_{'_'.join(map(str, shape))}_dim{dim}"
    try:
        kernel_func = getattr(lib, kernel_name)
    except AttributeError:
        available = [attr for attr in dir(lib) if "min" in attr.lower()]
        print(f"❌ Kernel function '{kernel_name}' not found in .so")
        print(f"Available functions: {available}")
        exit(1)

    # === 7. Set function signature ===
    # void min_kernel_N_M_dimK(
    #   const float* input, float* output,
    #   int* shape, int rank, int dim
    # );
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.POINTER(ctypes.c_int),  # shape (array)
        ctypes.c_int,  # rank (ndim)
        ctypes.c_int,  # dim
    ]
    kernel_func.restype = None

    # === 8. Call CUDA kernel ===
    shape_arr = (ctypes.c_int * len(shape))(*shape)
    print(f"🚀 Running CUDA kernel: {kernel_name}")
    kernel_func(x_ptr, output_ptr, shape_arr, len(shape), dim)

    # === 9. Compare results ===
    if torch.allclose(
        result_cuda, result_torch, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! CUDA result matches PyTorch.")
    else:
        print("❌ Verification failed!")
        print(f"PyTorch output (first 5): {result_torch.flatten()[:5]}")
        print(f"CUDA result (first 5): {result_cuda.flatten()[:5]}")
        diff = (result_torch - result_cuda).abs()
        print(f"min diff: {diff.min().item():.2e}")
        exit(1)

    # Cleanup
    if os.path.exists(so_name):
        os.remove(so_name)
        print(f"🗑️  Removed {so_name}")
