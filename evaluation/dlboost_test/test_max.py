import argparse
import ctypes
import os
import subprocess

import torch
from evaluation.utils import run_dlboost_compilation as run_compilation


# Define the max (element-wise) function using torch
def element_wise_max(A, B):
    return torch.maximum(A, B)  # 黄金标准


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the C++ source file (e.g., max_64_64.cpp)",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # 应该是 "max"
    shapes_str = base_name.split(".")[0]  # e.g., "max_64_64"
    shape = [int(x) for x in shapes_str.split("_")[1:]]  # 提取尺寸

    print(f"🔍 Testing {name.upper()} with shape {shape}")

    # Generate random input matrices
    A = (
        torch.rand(*shape, device="cpu", dtype=torch.float32) * 10 - 5
    )  # [-5, 5]
    B = (
        torch.rand(*shape, device="cpu", dtype=torch.float32) * 10 - 5
    )  # [-5, 5]

    # Golden reference using PyTorch
    expected = element_wise_max(A, B)

    # Convert to NumPy and get ctypes pointers
    A_ptr = A.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Output tensor
    result_ctypes = torch.zeros(shape, dtype=torch.float32)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

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
    kernel_func = getattr(lib, name)  # e.g., max

    # Function signature: void max(float*, float*, float*)
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    kernel_func.restype = None

    # Call kernel
    print(f"🚀 Running {name.upper()} kernel...")
    kernel_func(A_ptr, B_ptr, output_ptr)

    # Verify
    is_correct = torch.allclose(
        result_ctypes,
        expected,
        rtol=1e-3,
        atol=1e-3,
        equal_nan=True,
    )

    if is_correct:
        print("✅ Verification successful! C++ max matches PyTorch.")
    else:
        print("❌ Verification failed!")
        print("Expected (first 10):", expected.flatten()[:10])
        print("Got (first 10):", result_ctypes.flatten()[:10])
        diff = (result_ctypes - expected).abs()
        print("Max error:", diff.max().item())

    # Clean up
    subprocess.run(["rm", so_name], check=False)