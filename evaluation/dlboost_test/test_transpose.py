import argparse
import ctypes
import os
import subprocess

import torch
from evaluation.utils import run_dlboost_compilation as run_compilation


def transpose_2d(input_tensor):
    return input_tensor.t().contiguous()  # 或 torch.transpose(input, 0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the C++ source file (e.g., transpose_64_64.cpp)",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # 应该是 "transpose"
    shapes_str = base_name.split(".")[0]
    shape = [int(x) for x in shapes_str.split("_")[1:]]  # e.g., [64, 64]

    if len(shape) != 2:
        print("❌ Only 2D transpose is supported in this test.")
        exit(1)

    M, N = shape
    print(
        f"🔍 Testing {name.upper()} with input shape [{M}, {N}] -> output shape [{N}, {M}]"
    )

    # 生成输入数据
    input_tensor = torch.rand(M, N, dtype=torch.float32)
    input_ptr = input_tensor.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # 黄金标准：PyTorch transpose
    expected = transpose_2d(input_tensor)  # (N, M)

    # 输出张量
    result_ctypes = torch.zeros(N, M, dtype=torch.float32)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # 共享库名
    so_name = args.file.replace(".cpp", ".so")

    # 读取并注入宏
    with open(args.file, "r") as f:
        code = f.read()

    with open(
        os.path.join(os.getcwd(), "benchmark/macro/cpp_macro.txt"), "r"
    ) as f:
        macro = f.read()

    code = macro + code

    temp_file_name = args.file.replace(".cpp", "_bak.cpp")
    with open(temp_file_name, "w") as f:
        f.write(code)

    # 编译
    print(f"⚙️ Compiling {temp_file_name} -> {so_name}")
    success, compile_output = run_compilation(so_name, temp_file_name)
    if not success:
        print("❌ Compilation failed:")
        print(compile_output)
        exit(1)

    os.remove(temp_file_name)

    # 加载共享库
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    kernel_func = getattr(lib, name)  # transpose

    # 函数签名：void transpose(float* in, float* out, int M, int N)
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input [M, N]
        ctypes.POINTER(ctypes.c_float),  # output [N, M]
        ctypes.c_int,  # M
        ctypes.c_int,  # N
    ]
    kernel_func.restype = None

    # 调用 kernel
    print(f"🚀 Running {name.upper()} kernel...")
    kernel_func(input_ptr, output_ptr, M, N)

    # 验证
    is_correct = torch.allclose(
        result_ctypes, expected, rtol=1e-5, atol=1e-8, equal_nan=True
    )

    if is_correct:
        print("✅ Verification successful! C++ transpose matches PyTorch.")
    else:
        print("❌ Verification failed!")
        print("Expected (top-left 3x3):")
        print(expected[:3, :3])
        print("Got (top-left 3x3):")
        print(result_ctypes[:3, :3])
        diff = (result_ctypes - expected).abs()
        print("Max error:", diff.max().item())

    # 清理
    subprocess.run(["rm", so_name], check=False)