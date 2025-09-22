# test_gather_cu.py
import argparse
import ctypes
import os
import subprocess
import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def element_wise_gather(params, indices, axis=0):
    """
    黄金标准：支持任意 axis 的 gather
    params: Tensor of shape [N, C, H, W, ...]
    indices: LongTensor of arbitrary shape
    axis: int, along which dim to gather
    """
    # Clamp indices to valid range
    dim_size = params.size(axis)
    clamped_indices = torch.clamp(indices, 0, dim_size - 1)

    # Create out-of-bound mask
    out_of_bound = (indices < 0) | (indices >= dim_size)

    # Use torch.gather for correct behavior
    result = torch.gather(params, dim=axis, index=clamped_indices.unsqueeze(axis)).squeeze(axis)

    # Manually set out-of-bound elements to zero
    if out_of_bound.any():
        result = result.clone()  # Ensure we don't modify in-place
        result[out_of_bound] = 0.0

    return result


def parse_filename(filename):
    """Parse shapes from filename like gather_4_3_2_axis1.cu or gather_4_3_2.cu (default axis=0)"""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., gather_4_3_2 or gather_4_3_2_axis1

    if not stem.startswith("gather_"):
        raise ValueError(f"Invalid filename: {filename}")

    parts = stem.split('_')[1:]

    # Detect axis
    axis = 0
    if parts[-1].startswith("axis"):
        axis = int(parts[-1][4:])
        parts = parts[:-1]

    try:
        parts = list(map(int, parts))
    except ValueError as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    if len(parts) != 3:
        raise ValueError(f"Expected 3 integers in filename, got {len(parts)}: {stem}")

    params_dim0, params_dim1, indices_len = parts
    print(f"🔍 Parsed: params[{params_dim0}, {params_dim1}], indices[{indices_len}], axis={axis}")
    return params_dim0, params_dim1, indices_len, axis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test C++ GATHER kernel against PyTorch (supports axis)")
    parser.add_argument("--file", type=str, required=True, help="Path to the .cu source file")
    parser.add_argument("--config", required=True, help="JSON string or path to kernel config")
    parser.add_argument("--target", required=True, choices=["cuda", "hip", "bang", "cpu"], help="Target platform")
    args = parser.parse_args()

    # === 1. Parse shapes and axis from filename ===
    try:
        PARAMS_DIM0, PARAMS_DIM1, INDICES_LEN, AXIS = parse_filename(args.file)
    except Exception as e:
        print(f"❌ Filename parsing failed: {e}")
        exit(1)

    # === 2. Generate input tensors ===
    params = torch.randn(PARAMS_DIM0, PARAMS_DIM1, dtype=torch.float32, device="cpu")
    # Random indices, include possible out-of-bound (-1 or >= size)
    indices = torch.randint(low=-1, high=PARAMS_DIM0+1, size=(INDICES_LEN,), dtype=torch.int64, device="cpu")

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
    indices_ptr = to_ptr(indices, ctypes.c_int64)  # 注意：PyTorch int64 -> int64_t

    result_ctypes = torch.zeros_like(expected, dtype=torch.float32)
    output_ptr = to_ptr(result_ctypes, ctypes.c_float)

    # === 5. Shared library name and temp file ===
    so_name = args.file.replace(".cu", ".so")
    temp_file = args.file.replace(".cu", "_bak.cu")

    # === 6. Read original code and inject macro ===
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code  # 注入宏定义（如 DEBUG）

    with open(temp_file, "w") as f:
        f.write(code)

    # === 7. Compile C++ to .so ===
    print(f"⚙️ Compiling {temp_file} -> {so_name}")
    success, compile_output = run_compilation(so_name, temp_file)
    if not success:
        print("❌ Compilation failed:")
        print(compile_output)
        exit(1)

    os.remove(temp_file)
    print(f"✅ Compiled {so_name}")

    # === 8. Load shared library ===
    lib = ctypes.CDLL(so_name)

    # 尝试获取 kernel 函数（可扩展命名）
    func_name = "gather"
    try:
        kernel_func = getattr(lib, func_name)
    except AttributeError:
        print(f"❌ Function '{func_name}' not found in compiled library.")
        available = [attr for attr in dir(lib) if attr.isalpha()]
        print(f"Available symbols: {available}")
        exit(1)

    # === 9. Set function signature ===
    # void gather(
    #   const float* params,
    #   const int64_t* indices,
    #   float* output,
    #   int dim0, int dim1,
    #   int indices_len,
    #   int axis
    # );
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # params
        ctypes.POINTER(ctypes.c_int64),   # indices (int64_t)
        ctypes.POINTER(ctypes.c_float),   # output
        ctypes.c_int,                     # dim0
        ctypes.c_int,                     # dim1
        ctypes.c_int,                     # indices_len
        ctypes.c_int,                     # axis
    ]
    kernel_func.restype = None

    # === 10. Call C++ kernel ===
    print("🚀 Running C++ GATHER kernel...")
    kernel_func(
        params_ptr,
        indices_ptr,
        output_ptr,
        PARAMS_DIM0,
        PARAMS_DIM1,
        INDICES_LEN,
        AXIS
    )

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
