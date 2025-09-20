import argparse
import ctypes
import json
import os
import subprocess

import torch
from evaluation.utils import run_cuda_compilation as run_compilation
from evaluation.macros import CUDA_MACROS as macro


def parse_config(config_input):
    """
    Parse config: either a JSON string or a file path.
    Expected format:
    {
        "op_name": "transpose",
        "dtype": "float32",
        "args": [36, 16, 48],
        "perm": [0, 2, 1]
    }
    """
    if os.path.isfile(config_input):
        with open(config_input, 'r') as f:
            config = json.load(f)
    else:
        config = json.loads(config_input)
    
    shape = config["args"]
    perm = config["perm"]
    return shape, perm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the C++ source file (e.g., transpose_36_16_48.cu)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to kernel config",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # e.g., "transpose"
    shapes_str = base_name.replace(".cu", "")
    
    # 尝试从文件名解析 shape（用于验证）
    try:
        shape_from_filename = [int(x) for x in shapes_str.split("_")[1:]]
    except ValueError:
        raise ValueError(f"Invalid filename format: {args.file}")

    # ✅ 从 config 获取真实 shape 和 perm
    input_shape, perm = parse_config(args.config)
    output_shape = [input_shape[i] for i in perm]  # permuted shape

    print(f"🔍 Testing {name.upper()} with input shape {input_shape} -> output shape {output_shape}, perm={perm}")

    # ✅ 生成输入张量
    input_tensor = torch.rand(input_shape, dtype=torch.float32, requires_grad=False)
    input_flat = input_tensor.numpy()  # C-order
    input_ptr = input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # ✅ 黄金标准：PyTorch permute
    expected = input_tensor.permute(*perm).contiguous()
    expected_flat = expected.numpy()

    # ✅ 输出 buffer
    output_numel = expected_flat.size
    result_array = (ctypes.c_float * output_numel)()  # 分配空间

    # 共享库名称
    so_name = args.file.replace(".cu", ".so")

    # 读取并注入宏
    with open(args.file, "r") as f:
        code = f.read()
    code = macro + code

    temp_file_name = args.file.replace(".cu", "_bak.cu")
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
    kernel_func = getattr(lib, name)

    # ✅ 动态设置函数签名，支持 2D/3D/4D
    rank = len(input_shape)

    if rank not in [2, 3, 4]:
        raise NotImplementedError(f"Rank {rank} not supported. Only 2D, 3D, and 4D are supported.")

    # 构建 argtypes: [float*, float*, d0, d1, ..., p0, p1, ...]
    argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    # 添加 shape 维度 (d0, d1, ...)
    argtypes += [ctypes.c_int] * rank
    # 添加 perm 维度 (p0, p1, ...)
    argtypes += [ctypes.c_int] * rank

    kernel_func.argtypes = argtypes

    # ✅ 构建参数列表
    args_list = [input_ptr, result_array] + input_shape + perm

    # ✅ 调用 kernel
    print(f"🚀 Running {name.upper()} kernel with rank-{rank} permute...")
    kernel_func(*args_list)

    # ✅ 获取结果并 reshape
    computed_flat = torch.tensor([result_array[i] for i in range(output_numel)])
    computed_tensor = computed_flat.view(output_shape)

    # ✅ 验证
    is_correct = torch.allclose(
        computed_tensor, expected, rtol=1e-5, atol=1e-6, equal_nan=True
    )

    if is_correct:
        print("✅ Verification successful! C++ permute matches PyTorch.")
    else:
        print("❌ Verification failed!")
        if computed_tensor.dim() >= 2:
            print("Expected (top-left 3x3):")
            print(expected[:min(3, expected.shape[0]), :min(3, expected.shape[1])])
            print("Got (top-left 3x3):")
            print(computed_tensor[:min(3, computed_tensor.shape[0]), :min(3, computed_tensor.shape[1])])
        diff = (computed_tensor - expected).abs()
        print(f"Max error: {diff.max().item():.2e}")

    # 清理
    subprocess.run(["rm", so_name], check=False)
