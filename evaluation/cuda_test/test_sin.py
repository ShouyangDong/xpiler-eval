import argparse
import ctypes
import os

import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation

# Define the sin function using torch


def sin_op(x):
    return torch.sin(x)  # element-wise sine


def parse_filename(filename):
    """Parse shape from filename like sin_3_4.cu."""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., sin_3_4

    if not stem.startswith("sin_"):
        raise ValueError(f"Invalid filename: {filename}")

    try:
        shape = list(map(int, stem.split("_")[1:]))  # skip 'sin'
    except ValueError as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    if len(shape) == 0:
        raise ValueError(f"No shape found in filename: {stem}")

    print(f"🔍 Parsed shape: {shape}")
    return shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test sin CUDA kernel against PyTorch"
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

    # === 1. Parse shape from filename ===
    try:
        shape = parse_filename(args.file)
    except Exception as e:
        raise RuntimeError(f"❌ Filename parsing failed: {e}")

    # === 2. Generate random input tensor ===
    x = torch.randn(*shape, dtype=torch.float32, device="cpu")
    # 或者使用更广范围：x = torch.linspace(-2*torch.pi, 2*torch.pi,
    # steps=prod(shape)).reshape(shape)
    print(f"🧪 Input tensor shape: {x.shape}")

    # === 3. Perform sin using PyTorch ===
    result_torch = sin_op(x)
    print(f"✅ PyTorch output shape: {result_torch.shape}")

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

    # Inject macro (e.g., for CUDA headers)
    try:
        with open(
            os.path.join(os.getcwd(), "benchmark/macro/cpp_macro.txt"), "r"
        ) as f:
            macro = f.read()
        code = macro + code
    except FileNotFoundError:
        print("⚠️  Macro file not found, proceeding without injection.")

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

    # === 6. Load shared library ===
    lib = ctypes.CDLL(so_name)
    # Kernel name example: sin_kernel_3_4
    kernel_name = f"sin_kernel_{'_'.join(map(str, shape))}"
    try:
        kernel_func = getattr(lib, kernel_name)
    except AttributeError:
        available = [attr for attr in dir(lib) if "sin" in attr.lower()]
        print(f"❌ Kernel function '{kernel_name}' not found in .so")
        print(f"Available functions: {available}")
        exit(1)

    # === 7. Set function signature ===
    # void sin_kernel_N_M(
    #   const float* input,
    #   float* output,
    #   int* shape, int rank
    # );
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.POINTER(ctypes.c_int),  # shape
        ctypes.c_int,  # rank
    ]
    kernel_func.restype = None

    # === 8. Call CUDA kernel ===
    shape_arr = (ctypes.c_int * len(shape))(*shape)
    print(f"🚀 Running CUDA kernel: {kernel_name}")
    kernel_func(x_ptr, output_ptr, shape_arr, len(shape))

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
        print(f"Max diff: {diff.max().item():.2e}")
        exit(1)

    # Cleanup
    if os.path.exists(so_name):
        os.remove(so_name)
        print(f"🗑️  Removed {so_name}")
