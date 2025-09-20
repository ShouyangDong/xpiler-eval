# test_gather_cu.py
import argparse
import ctypes
import os
import subprocess
import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def element_wise_gather(params, indices):
    # 黄金标准：PyTorch gather
    clamped_indices = torch.clamp(indices, 0, params.size(0) - 1)
    result = params[clamped_indices]
    # 处理越界：置零
    out_of_bound = (indices < 0) | (indices >= params.size(0))
    if out_of_bound.any():
        result[out_of_bound] = 0.0
    return result


def parse_filename(filename):
    """Parse shapes from filename like gather_100_32_16.cu"""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., gather_100_32_16

    if not stem.startswith("gather_"):
        raise ValueError(f"Invalid filename: {filename}")

    try:
        parts = list(map(int, stem.split('_')[1:]))
    except ValueError as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    if len(parts) != 3:
        raise ValueError(f"Expected 3 integers in filename, got {len(parts)}: {stem}")

    params_batch, params_len, indices_len = parts
    print(f"🔍 Parsed: params[{params_batch}, {params_len}], indices[{indices_len}]")
    return params_batch, params_len, indices_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test C++ GATHER kernel against PyTorch")
    parser.add_argument("--file", type=str, required=True, help="Path to the .cu source file")
    parser.add_argument("--config", required=True, help="JSON string or path to kernel config")
    parser.add_argument("--target", required=True, choices=["cuda", "hip", "bang", "cpu"], help="Target platform")
    args = parser.parse_args()

    # === 1. Parse shapes from filename ===
    try:
        PARAMS_BATCH, PARAMS_LEN, INDICES_LEN = parse_filename(args.file)
    except Exception as e:
        print(f"❌ Filename parsing failed: {e}")
        exit(1)

    # === 2. Generate input tensors ===
    params = torch.randn(PARAMS_BATCH, PARAMS_LEN, dtype=torch.float32, device="cpu")
    indices = torch.randint(low=-1, high=PARAMS_BATCH+1, size=(INDICES_LEN,), dtype=torch.int32, device="cpu")

    print(f"🧪 params shape: {params.shape}")
    print(f"🧪 indices: {indices.tolist()}")

    # === 3. Golden reference using PyTorch ===
    expected = element_wise_gather(params, indices)
    print(f"✅ Expected output shape: {expected.shape}")

    # === 4. Prepare ctypes pointers ===
    def to_ptr(tensor, dtype):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(dtype))

    params_ptr = to_ptr(params, ctypes.c_float)
    indices_ptr = to_ptr(indices, ctypes.c_int32)

    result_ctypes = torch.zeros_like(expected)
    output_ptr = to_ptr(result_ctypes, ctypes.c_float)

    # === 5. Shared library name and temp file ===
    so_name = args.file.replace(".cu", ".so")
    temp_file = args.file.replace(".cu", "_bak.cu")

    # === 6. Read original code and inject macro ===
    with open(args.file, "r") as f:
        code = f.read()

    # Optional: inject macro (e.g., for debug or config)
    code = macro + code


    # Write temporary modified file
    with open(temp_file, "w") as f:
        f.write(code)

    # === 7. Compile C++ to .so using dlboost ===
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

    # Kernel function name: assume it's 'gather'
    try:
        kernel_func = getattr(lib, "gather")
    except AttributeError:
        print("❌ Function 'gather' not found in compiled library.")
        available = [attr for attr in dir(lib) if attr.isalpha()]
        print(f"Available symbols: {available}")
        exit(1)

    # === 9. Set function signature ===
    # void gather(
    #   const float* params,
    #   const int32_t* indices,
    #   float* output,
    #   int params_batch,
    #   int params_len,
    #   int indices_len
    # );
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # params
        ctypes.POINTER(ctypes.c_int32),   # indices
        ctypes.POINTER(ctypes.c_float),   # output
        ctypes.c_int,                     # params_batch
        ctypes.c_int,                     # params_len
        ctypes.c_int,                     # indices_len
    ]
    kernel_func.restype = None

    # === 10. Call C++ kernel ===
    print("🚀 Running C++ GATHER kernel...")
    kernel_func(
        params_ptr,
        indices_ptr,
        output_ptr,
        PARAMS_BATCH,
        PARAMS_LEN,
        INDICES_LEN
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