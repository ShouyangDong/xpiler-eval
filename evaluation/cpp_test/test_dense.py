import argparse
import ctypes
import os
import torch

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation


def reference_dense_int16(A_int16: torch.Tensor, B_int16: torch.Tensor, bias_int32: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation: int16 matrix multiplication + bias in int32.
    A: [M, K] int16
    B: [K, N] int16
    bias: [N] int32
    Returns: [M, N] int32
    """
    # MatMul in int32 to avoid overflow
    result = torch.matmul(A_int16, B_int16).to(torch.int32)
    return result + bias_int32.unsqueeze(0)  # Broadcast bias to (1, N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
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
    name = base_name.split("_")[0]  # e.g., 'dense'
    shapes_str = base_name.split(".")[0]  # e.g., "dense_16_512_512"
    shape = [int(x) for x in shapes_str.split("_")[1:]]

    assert len(shape) == 3, "Shape should be M_N_K"
    M, N, K = shape

    print(f"[Test] Running dense_int16_bias_int32 with shape ({M}, {K}) @ ({K}, {N}) + bias[{N}]")

    # Generate random int16 inputs using only PyTorch
    A_int16 = torch.randint(-10, 10, (M, K), dtype=torch.int16)
    B_int16 = torch.randint(-10, 10, (K, N), dtype=torch.int16)
    bias_int32 = torch.randint(-100, 100, (N,), dtype=torch.int32)

    # Compute reference result
    result_ref = reference_dense_int16(A_int16, B_int16, bias_int32)

    # Convert tensors to contiguous .data_ptr() via ctypes
    # Ensure they are contiguous in memory
    A_int16 = A_int16.contiguous()
    B_int16 = B_int16.contiguous()
    bias_int32 = bias_int32.contiguous()

    A_ptr = ctypes.cast(A_int16.data_ptr(), ctypes.POINTER(ctypes.c_int16))
    B_ptr = ctypes.cast(B_int16.data_ptr(), ctypes.POINTER(ctypes.c_int16))
    bias_ptr = ctypes.cast(bias_int32.data_ptr(), ctypes.POINTER(ctypes.c_int32))

    # Prepare output buffer in int32
    result_ctypes = torch.zeros((M, N), dtype=torch.int32).contiguous()
    output_ptr = ctypes.cast(result_ctypes.data_ptr(), ctypes.POINTER(ctypes.c_int32))

    # Compile the C++ source
    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code  # Inject macros (e.g., SIMD flags)

    temp_cpp = args.file.replace(".cpp", "_bak.cpp")
    with open(temp_cpp, "w") as f:
        f.write(code)

    print(f"[Test] Compiling {temp_cpp} -> {so_name}")
    success, output = run_compilation(so_name, temp_cpp)
    if not success:
        print("❌ Compilation failed:")
        print(output)
        exit(1)

    os.remove(temp_cpp)

    # Load shared library
    try:
        lib = ctypes.CDLL(so_name)
        kernel_func = getattr(lib, name)
    except Exception as e:
        print("❌ Failed to load kernel function:")
        print(e)
        exit(1)

    # Set argument types
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_int16),  # A
        ctypes.POINTER(ctypes.c_int16),  # B
        ctypes.POINTER(ctypes.c_int32),  # bias
        ctypes.POINTER(ctypes.c_int32),  # output
    ]
    kernel_func.restype = None

    print("[Test] Running kernel...")
    kernel_func(A_ptr, B_ptr, bias_ptr, output_ptr)

    # Compare results
    abs_diff = torch.abs(result_ctypes - result_ref)
    max_diff = abs_diff.max().item()

    print(f"[Test] Max absolute difference: {max_diff}")

    if max_diff <= 1:
        print("✅ Verification successful! (Max diff <= 1)")
    else:
        print("❌ Verification failed!")
        print(f"Expected max diff ≤ 1, but got {max_diff}")
        exit(1)

    # Cleanup
    try:
        os.remove(so_name)
        print("🗑️ Shared library cleaned up.")
    except Exception as e:
        print(f"Warning: failed to remove .so: {e}")
