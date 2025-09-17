import argparse
import ctypes
import os
import subprocess

import torch
from benchmark.utils import run_dlboost_compilation as run_compilation


def reshape_tensor(input_tensor, output_shape):
    return input_tensor.reshape(*output_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the C++ source file (e.g., reshape_64x64_to_8x512.cpp)",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # 应该是 "reshape"
    shapes_str = base_name.split(".")[0]  # e.g., reshape_64x64_to_8x512

    # 解析输入和输出形状，支持多种分隔符
    if "_to_" in shapes_str:
        in_shape_str, out_shape_str = shapes_str.split("_to_")
        in_shape = [int(x) for x in in_shape_str.split("x")]
        out_shape = [int(x) for x in out_shape_str.split("x")]
    else:
        # fallback: 单一形状（如 reshape_64_64.cpp 表示从一维拉平再 reshape）
        shape = [int(x) for x in base_name.split(".")[0].split("_")[1:]]
        in_shape = [-1]  # 表示从 flat 拉伸
        out_shape = shape

    print(
        f"🔍 Testing {name.upper()} with input shape {in_shape} -> output shape {out_shape}"
    )

    # 总元素数必须一致
    def prod(shape):
        result = 1
        for dim in shape:
            result *= dim if dim != -1 else 1
        return result

    total_elements = prod(out_shape)  # 假设 -1 能被推断

    # 生成输入数据（1D，便于 C++ 处理）
    input_flat = torch.rand(total_elements, dtype=torch.float32)
    input_ptr = input_flat.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # PyTorch 黄金标准：reshape
    input_view = input_flat.reshape(
        in_shape if in_shape != [-1] else (total_elements,)
    )
    expected = reshape_tensor(input_view, out_shape)  # (out_shape)

    # 输出张量
    result_ctypes = torch.zeros(total_elements, dtype=torch.float32)
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
    kernel_func = getattr(lib, name)  # reshape

    # 函数签名：void reshape(float* in, float* out, int total_size)
    # 注意：reshape 在连续内存上只是复制（memcpy），不改变顺序
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,  # total number of elements
    ]
    kernel_func.restype = None

    # 调用 kernel
    print(f"🚀 Running {name.upper()} kernel...")
    kernel_func(input_ptr, output_ptr, total_elements)

    # 重塑结果用于比较
    result_reshaped = result_ctypes.reshape(out_shape)

    # 验证
    is_correct = torch.allclose(
        result_reshaped, expected, rtol=1e-5, atol=1e-8, equal_nan=True
    )

    if is_correct:
        print("✅ Verification successful! C++ reshape matches PyTorch.")
    else:
        print("❌ Verification failed!")
        print("Expected (first 10):", expected.flatten()[:10])
        print("Got (first 10):", result_reshaped.flatten()[:10])
        diff = (result_reshaped - expected).abs()
        print("Max error:", diff.max().item())

    # 清理
    subprocess.run(["rm", so_name], check=False)
