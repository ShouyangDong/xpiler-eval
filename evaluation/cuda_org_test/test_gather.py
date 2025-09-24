# test_gather_cu.py
import argparse
import ctypes
import os
import random
import subprocess
import torch
import json

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def element_wise_gather(params, indices, axis=0):

    # invoke gather
    result = torch.gather(params, dim=axis, index=indices)
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test C++ GATHER kernel against PyTorch")
    parser.add_argument("--file", type=str, required=True, help="Path to the .cu source file")
    parser.add_argument("--config", required=True, help="JSON string or path (ignored for indices_len)")
    parser.add_argument("--target", required=True, choices=["cuda", "hip", "bang", "cpu"], help="Target platform")
    args = parser.parse_args()
    # === 1. Parse shape and axis from filename ===
    try:
        if os.path.exists(args.config) and args.config.endswith(".json"):
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            config = json.loads(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to parse config: {e}", file=sys.stderr)
        sys.exit(1)

    op_name = config["op_name"]
    PARAMS_SHAPE = config["args"]
    AXIS = config["axes"]

    # === 2. Generate input tensors ===
    params = torch.randn(*PARAMS_SHAPE, dtype=torch.float32, device="cpu")
    axis_dim_size = params.size(AXIS)

    # Randomly generate indices_len: between 1 and axis_dim_size * 2
    min_len = 1
    max_len = axis_dim_size
    indices_len = random.randint(min_len, max_len)

    print(f"📐 Axis {AXIS} has size {axis_dim_size} → generated random indices_len = {indices_len}")

    # Generate indices: include valid and out-of-bound (-1 or >= axis_dim_size)
    indices = torch.randint(low=0, high=axis_dim_size, size=(indices_len,), dtype=torch.int64, device="cpu")
    print(f"🧪 params shape: {params.shape}")
    print(f"🧪 indices: {indices.tolist()}")
    print(f"⚙️  axis = {AXIS}")

    # 获取目标形状：除了 axis dim是 len(indices)，其余和 params 一样
    output_shape = list(params.shape)
    output_shape[AXIS] = indices.size(0)  # M

    # 将 indices 扩展到目标形状
    # 方法：在 axis dim上 unsqueeze，然后 expand
    indices_expanded = indices.view(*[1 if i != AXIS else -1 for i in range(params.ndim)])
    # 例如 axis=0, ndim=3 → [-1, 1, 1]
    #      axis=1, ndim=3 → [1, -1, 1]

    indices  = indices_expanded.expand(*output_shape)
    # === 3. Golden reference using PyTorch ===
    expected = element_wise_gather(params, indices, axis=AXIS)
    print(f"✅ Expected output shape: {expected.shape}")

    # === 4. Prepare ctypes pointers ===
    def to_ptr(tensor, dtype):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(dtype))
    print("params shape: " ,params.shape)
    print("index : ", indices.shape)
    print("outptu: ", output_shape)
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
    func_name = "gather_kernel"
    try:
        kernel_func = getattr(lib, func_name)
    except AttributeError:
        print(f"❌ Function '{func_name}' not found in compiled library.")
        available = [attr for attr in dir(lib) if attr.isalpha()]
        print(f"Available symbols: {available}")
        exit(1)

    # === 9. Set function signature ===
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
            indices_len
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
