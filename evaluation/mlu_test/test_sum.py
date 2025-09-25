import argparse
import ctypes
import json
import os
import subprocess

import torch

from evaluation.macros import MLU_MACROS as macro
from evaluation.utils import run_mlu_compilation as run_compilation


def parse_config(config_input):
    """
    Parse config: either a JSON string or a file path
    Expected format:
    {
        "op_name": "sum",
        "dtype": "float32",
        "args": [3, 4, 5],        # input shape
        "axes": [0] or 0          # reduction axes
    }
    """
    if os.path.isfile(config_input):
        with open(config_input, "r") as f:
            config = json.load(f)
    else:
        config = json.loads(config_input)

    shape = config["args"]
    axes = config["axes"]
    if isinstance(axes, int):
        axes = [axes]
    return shape, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the C++ source file (e.g., sum_3_4_5.mlu)",
    )
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

    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]  # e.g., "sum"
    shapes_str = base_name.replace(".mlu", "")
    # Dynamically extract shape: parse all numbers from filename
    try:
        shape_from_filename = [int(x) for x in shapes_str.split("_")[1:]]
    except ValueError:
        raise ValueError(
            f"Invalid filename format: {args.file}. Expected: op_M_N_K.mlu"
        )

    # Get the true shape and axes from config
    config_shape, axes = parse_config(args.config)

    print(
        f"🔍 Testing {name.upper()} with input shape {config_shape}, axes={axes}"
    )

    # ✅ 使用 config 中的 shape，而非文件名（更可靠）
    shape = config_shape

    # ✅ Generate input tensor
    A = torch.rand(shape, device="cpu", dtype=torch.float32)

    # ✅ 黄金标准：沿指定 axes 求和，不保留dim（与大多数 kernel 一致）
    expected_tensor = torch.sum(A, dim=axes)  # shape: reduced
    expected_numpy = expected_tensor.numpy()
    expected_flat = expected_numpy.flatten()

    # ✅ 输入指针（展平输入）
    A_flat = A.numpy()  # 自动展平为 C 顺序
    A_ptr = A_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # ✅ output大小
    output_size = expected_flat.size  # int
    result_array = (ctypes.c_float * output_size)()  # 分配空间

    # shared library
    so_name = args.file.replace(".mlu", ".so")

    # Load原始代码
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    # Create tmp file
    temp_file_name = args.file.replace(".mlu", "_bak.mlu")
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

    # ✅ Function  signature
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    kernel_func.restype = None

    # ✅ invoke kernel
    print(f"🚀 Running {name.upper()} kernel...")
    kernel_func(A_ptr, result_array)

    # ✅ Get output
    computed_array = [result_array[i] for i in range(output_size)]
    computed_tensor = torch.tensor(computed_array).view_as(
        torch.from_numpy(expected_numpy)
    )

    # ✅ verification
    abs_diff = torch.abs(computed_tensor - expected_tensor)
    max_error = abs_diff.max().item()

    if max_error <= 1e-3:
        print(f"✅ Verification successful! C++ sum matches PyTorch.")
        print(f"   Output shape: {tuple(expected_tensor.shape)}")
        # 可选：打印前几个值
        if output_size <= 10:
            for i, (exp, got) in enumerate(zip(expected_flat, computed_array)):
                print(f"   Index {i}: Expected: {exp:.6f}, Got: {got:.6f}")
    else:
        print(f"❌ Verification failed! Max error = {max_error:.2e}")
        print(f"   Expected shape: {expected_tensor.shape}")
        print(f"   Computed shape: {computed_tensor.shape}")
        exit(1)

    # clean
    subprocess.run(["rm", so_name], check=False)
