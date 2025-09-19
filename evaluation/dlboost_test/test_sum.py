import argparse
import ctypes
import os
import subprocess

import torch
from evaluation.utils import run_dlboost_compilation as run_compilation


# Define the sum function using torch
def global_sum(A):
    return torch.sum(A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the C++ source file (e.g., sum_64_64.cpp)",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # 应该是 "sum"
    shapes_str = base_name.split(".")[0]  # e.g., "sum_64_64"
    shape = [
        int(x) for x in shapes_str.split("_")[1:]
    ]  # 提取尺寸，如 [64, 64]

    print(f"🔍 Testing {name.upper()} with input shape {shape}")

    # Generate random input matrix
    A = torch.rand(*shape, device="cpu", dtype=torch.float32)

    # Golden reference using PyTorch
    expected_scalar = global_sum(A).item()  # 得到 Python float

    # Convert input to NumPy and get pointer
    A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Output: single float (by reference)
    result_ctypes = ctypes.c_float(0.0)

    # Shared library name
    so_name = args.file.replace(".cpp", ".so")

    # Read original C++ code
    with open(args.file, "r") as f:
        code = f.read()

    # Inject macros
    with open(
        os.path.join(os.getcwd(), "benchmark/macro/cpp_macro.txt"), "r"
    ) as f:
        macro = f.read()

    code = macro + code

    # Create temporary modified file
    temp_file_name = args.file.replace(".cpp", "_bak.cpp")
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

    # Load shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    kernel_func = getattr(lib, name)  # e.g., sum

    # Function signature: void sum(float* input, float* output)
    # output 是一个 float 的地址，用于返回结果
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input array
        ctypes.POINTER(ctypes.c_float),  # output scalar (by reference)
    ]
    kernel_func.restype = None

    # Call kernel
    print(f"🚀 Running {name.upper()} kernel...")
    kernel_func(A_ptr, ctypes.byref(result_ctypes))

    # Get result as float
    computed = result_ctypes.value

    # Verify
    abs_error = abs(computed - expected_scalar)
    if abs_error <= 1e-3:
        print(f"✅ Verification successful! C++ sum matches PyTorch.")
        print(
            f"   Expected: {expected_scalar:.6f}, Got: {computed:.6f}, Error: {abs_error:.2e}"
        )
    else:
        print(f"❌ Verification failed!")
        print(
            f"   Expected: {expected_scalar:.6f}, Got: {computed:.6f}, Error: {abs_error:.2e}"
        )
        exit(1)

    # Clean up
    subprocess.run(["rm", so_name], check=False)