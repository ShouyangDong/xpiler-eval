import argparse
import ctypes
import json
import os
import subprocess

import torch
from evaluation.utils import run_cuda_compilation as run_compilation
from evaluation.macros import CUDA_MACROS as macro


def parse_config(config_input):
    """Parse config: either a JSON string or a file path."""
    if os.path.isfile(config_input):
        with open(config_input, 'r') as f:
            config = json.load(f)
    else:
        config = json.loads(config_input)
    
    shape = config["args"]
    reduce_dim = config.get("axes", None)
    if reduce_dim is None:
        raise ValueError("Config must contain 'reduce_dim'")
    return shape, reduce_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the CUDA source file (e.g., mean_8_32_64.cu)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to kernel config, must include 'args' and 'reduce_dim'",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # e.g., "mean"
    so_name = args.file.replace(".cu", ".so")

    # 解析配置文件
    input_shape, reduce_dim = parse_config(args.config)

    if not (0 <= reduce_dim < len(input_shape)):
        raise ValueError(f"reduce_dim {reduce_dim} out of range for shape {input_shape}")

    print(f"🔍 Testing {name.upper()} with input shape {input_shape}, reduce_dim={reduce_dim}")

    # 生成随机输入
    input_tensor = torch.rand(input_shape, dtype=torch.float32, requires_grad=False)
    input_ptr = input_tensor.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # PyTorch mean
    expected = torch.mean(input_tensor, dim=reduce_dim).contiguous()
    expected_flat = expected.numpy()
    output_shape = expected.shape
    output_numel = expected_flat.size

    # output buffer
    result_array = (ctypes.c_float * output_numel)()  # 分配output内存

    # 注入宏并生成临时文件
    with open(args.file, "r") as f:
        code = f.read()
    code = macro + code

    temp_file_name = args.file.replace(".cu", "_bak.cu")
    with open(temp_file_name, "w") as f:
        f.write(code)

    # compile
    print(f"⚙️ Compiling {temp_file_name} -> {so_name}")
    success, compile_output = run_compilation(so_name, temp_file_name)
    if not success:
        print("❌ Compilation failed:")
        print(compile_output)
        exit(1)

    os.remove(temp_file_name)

    # load shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    kernel_func = getattr(lib, name + "_kernel")

    # 动态Construct argtypes：支持任意 rank 的 shape 和 reduce_dim
    rank = len(input_shape)

    if rank not in [2, 3, 4]:
        raise NotImplementedError(f"Rank {rank} not supported. Only 2D/3D/4D supported.")

    # Function  signature：void mean(float* input, float* output, int d0, ..., int reduce_dim)
    argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    # Add每个dim大小
    argtypes += [ctypes.c_int] * rank
    # Add reduce_dim 参数
    argtypes.append(ctypes.c_int)

    kernel_func.argtypes = argtypes
    kernel_func.restype = None

    # Construct input arguments
    args_list = [input_ptr, result_array] + input_shape + [reduce_dim]

    # invoke kernel
    print(f"🚀 Running {name.upper()} kernel...")
    kernel_func(*args_list)

    # Get output
    computed_flat = torch.tensor([result_array[i] for i in range(output_numel)])
    computed_tensor = computed_flat.view(output_shape)

    # verification
    is_correct = torch.allclose(
        computed_tensor, expected, rtol=1e-5, atol=1e-5, equal_nan=True
    )

    if is_correct:
        print("✅ Verification successful! C++ mean matches PyTorch.")
    else:
        print("❌ Verification failed!")
        diff = (computed_tensor - expected).abs()
        print(f"Max error: {diff.max().item():.2e}")
        if computed_tensor.numel() > 0:
            print("Expected (first 10):", expected.flatten()[:10].tolist())
            print("Got (first 10):", computed_tensor.flatten()[:10].tolist())

    # clean
    subprocess.run(["rm", so_name], check=False)
