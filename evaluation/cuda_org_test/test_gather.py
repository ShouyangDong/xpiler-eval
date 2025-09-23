# test_gather_cu.py
import argparse
import ctypes
import os
import random
import subprocess
import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def element_wise_gather(params, indices, axis=0):
    """
    黄金标准：支持任意 axis 的 gather
    params: Tensor of shape [N, C]
    indices: LongTensor of shape [M]
    axis: int, along which dim to gather
    返回: shape 为 [M, ...] 的结果（squeeze axis 后）
    """
    dim_size = params.size(axis)
    clamped_indices = torch.clamp(indices, 0, dim_size - 1)
    out_of_bound = (indices < 0) | (indices >= dim_size)

    # 使用 torch.gather
    result = torch.gather(params, dim=axis, index=clamped_indices.unsqueeze(axis)).squeeze(axis)

    # 越界位置设为 0
    if out_of_bound.any():
        result = result.clone()
        result[out_of_bound] = 0.0

    return result


def parse_filename(filename):
    """Parse params_dim0, params_dim1, and axis from filename like gather_4_3_axis1.cu"""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., gather_4_3_axis1

    if not stem.startswith("gather_"):
        raise ValueError(f"Invalid filename: {filename}")

    parts = stem.split('_')[1:]

    # Detect axis
    axis = 0
    if parts[-1].startswith("axis"):
        axis_str = parts[-1][4:]
        try:
            axis = int(axis_str)
        except ValueError:
            raise ValueError(f"Invalid axis in filename: {parts[-1]}")
        parts = parts[:-1]

    try:
        dims = list(map(int, parts))
    except ValueError as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    if len(dims) != 2:
        raise ValueError(f"Expected 2 integers for params shape in filename, got {len(dims)}: {stem}")

    params_dim0, params_dim1 = dims
    print(f"🔍 Parsed from filename: params[{params_dim0}, {params_dim1}], axis={axis}")
    return params_dim0, params_dim1, axis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test C++ GATHER kernel against PyTorch")
    parser.add_argument("--file", type=str, required=True, help="Path to the .cu source file")
    parser.add_argument("--config", required=True, help="JSON string or path (ignored for indices_len)")
    parser.add_argument("--target", required=True, choices=["cuda", "hip", "bang", "cpu"], help="Target platform")
    args = parser.parse_args()

    # === 1. Parse params shape and axis from filename ===
    try:
        PARAMS_DIM0, PARAMS_DIM1, AXIS = parse_filename(args.file)
    except Exception as e:
        print(f"❌ Filename parsing failed: {e}")
        exit(1)

    # === 2. Generate input tensors ===
    params = torch.randn(PARAMS_DIM0, PARAMS_DIM1, dtype=torch.float32, device="cpu")

    # Determine size along the gather axis
    axis_dim_size = params.size(AXIS)

    # Randomly generate indices_len: between 1 and axis_dim_size * 2
    min_len = 1
    max_len = axis_dim_size * 2
    indices_len = random.randint(min_len, max_len)

    print(f"📐 Axis {AXIS} has size {axis_dim_size} → generated random indices_len = {indices_len}")

    # Generate indices: include valid and out-of-bound (-1 or >= axis_dim_size)
    indices = torch.randint(low=-1, high=axis_dim_size + 1, size=(indices_len,), dtype=torch.int64, device="cpu")
    print(f"🧪 params shape: {params.shape}")
    print(f"🧪 indices: {indices.tolist()}")
    print(f"⚙️  axis = {AXIS}")

    # === 3. Golden reference using PyTorch ===
    expected = element_wise_gather(params, indices, axis=AXIS)
    print(f"✅ Expected output shape: {expected.shape}")

    # === 4. Prepare ctypes pointers ===
    def to_ptr(tensor, dtype):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(dtype))

    params_ptr = to_ptr(params, ctypes.c_float)
    indices_ptr = to_ptr(indices, ctypes.c_int64)
    result_ctypes = torch.zeros_like(expected, dtype=torch.float32)
    output_ptr = to_ptr(result_ctypes, ctypes.c_float)

    # === 5. Shared library name and temp file ===
    so_name = args.file.replace(".cu", ".so")
    temp_file = args.file.replace(".cu", "_bak.cu")

    # === 6. Read original code and inject macro ===
    try:
        with open(args.file, "r") as f:
            code = f.read()
    except Exception as e:
        print(f"❌ Failed to read {args.file}: {e}")
        exit(1)

    code = macro + code  # 注入宏定义（如 DEBUG）

    try:
        with open(temp_file, "w") as f:
            f.write(code)
    except Exception as e:
        print(f"❌ Failed to write {temp_file}: {e}")
        exit(1)

    # === 7. Compile C++ to .so ===
    print(f"⚙️ Compiling {temp_file} -> {so_name}")
    success, compile_output = run_compilation(so_name, temp_file)
    if not success:
        print("❌ Compilation failed:")
        print(compile_output)
        exit(1)

    # Clean up temp file
    try:
        os.remove(temp_file)
        print(f"🗑️ Removed {temp_file}")
    except OSError:
        pass

    print(f"✅ Compiled {so_name}")

    # === 8. Load shared library ===
    try:
        lib = ctypes.CDLL(so_name)
    except Exception as e:
        print(f"❌ Failed to load {so_name}: {e}")
        exit(1)

    # Look for 'gather' function
    func_name = "gather"
    try:
        kernel_func = getattr(lib, func_name)
    except AttributeError:
        print(f"❌ Function '{func_name}' not found in compiled library.")
        available = [attr for attr in dir(lib) if attr.isalpha()]
        print(f"Available symbols: {available}")
        exit(1)

    # === 9. Set function signature ===
    # void gather(const float* input, const int64_t* indices, float* output, int N);
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # input
        ctypes.POINTER(ctypes.c_int64),   # indices
        ctypes.POINTER(ctypes.c_float),   # output
        ctypes.c_int,                     # N (number of indices)
    ]
    kernel_func.restype = None

    # === 10. Call C++ kernel ===
    print("🚀 Running C++ GATHER kernel...")
    try:
        kernel_func(
            params_ptr,
            indices_ptr,
            output_ptr,
            indices_len  # only pass N
        )
    except Exception as e:
        print(f"❌ Kernel call failed: {e}")
        exit(1)

    # === 11. Verify result ===
    is_correct = torch.allclose(result_ctypes, expected, rtol=1e-5, atol=1e-5)

    if is_correct:
        print("✅ Verification successful! C++ gather matches PyTorch.")
    else:
        print("❌ Verification failed!")
        print(f"Expected (first 10): {expected.flatten()[:10]}")
        print(f"Got (first 10): {result_ctypes.flatten()[:10]}")
        diff = (result_ctypes - expected).abs()
        print(f"Max error: {diff.max().item():.2e}")
        exit(1)

    # === 12. Cleanup ===
    try:
        subprocess.run(["rm", so_name], check=True)
        print(f"🗑️ Removed {so_name}")
    except subprocess.CalledProcessError:
        print(f"⚠️ Failed to remove {so_name}")
