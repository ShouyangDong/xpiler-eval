import argparse
import ctypes
import os
import torch


from benchmark.utils import run_dlboost_compilation as run_compilation


# Define the sub function using torch
def sub_op(a, b):
    return a - b  # element-wise subtraction


def parse_filename(filename):
    """Parse shape from filename like sub_3_4.cu"""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., sub_3_4

    if not stem.startswith("sub_"):
        raise ValueError(f"Invalid filename: {filename}")

    try:
        shape = list(map(int, stem.split('_')[1:]))  # skip 'sub'
    except ValueError as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    if len(shape) == 0:
        raise ValueError(f"No shape found in filename: {stem}")

    print(f"🔍 Parsed shape: {shape}")
    return shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test sub CUDA kernel against PyTorch")
    parser.add_argument("--file", type=str, required=True, help="Path to the .cu source file")
    args = parser.parse_args()

    # === 1. Parse shape from filename ===
    try:
        shape = parse_filename(args.file)
    except Exception as e:
        raise RuntimeError(f"❌ Filename parsing failed: {e}")

    # === 2. Generate two random input tensors ===
    a = torch.randn(*shape, dtype=torch.float32, device="cpu")
    b = torch.randn(*shape, dtype=torch.float32, device="cpu")
    print(f"🧪 Input tensor A shape: {a.shape}")
    print(f"🧪 Input tensor B shape: {b.shape}")

    # === 3. Perform sub using PyTorch ===
    result_torch = sub_op(a, b)
    print(f"✅ PyTorch output shape: {result_torch.shape}")

    # === 4. Prepare ctypes pointers ===
    def to_ptr(tensor):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    a_ptr = to_ptr(a)
    b_ptr = to_ptr(b)
    result_cuda = torch.zeros_like(result_torch)
    output_ptr = to_ptr(result_cuda)

    # === 5. Inject macro and compile ===
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    # Inject macro (e.g., for CUDA headers)
    try:
        with open(os.path.join(os.getcwd(), "benchmark/macro/cpp_macro.txt"), "r") as f:
            macro = f.read()
        code = macro + code
    except FileNotFoundError:
        print("⚠️  Macro file not found, proceeding without injection.")

    # Write temporary file
    temp_file = args.file.replace(".cu", "_temp.cu")
    with open(temp_file, "w") as f:
        f.write(code)

    # Compile using dlboost
    success, compile_output = run_compilation(so_name, temp_file)
    if not success:
        print(f"❌ Compilation failed:\n{compile_output}")
        exit(1)
    os.remove(temp_file)
    print(f"✅ Compiled to {so_name}")

    # === 6. Load shared library ===
    lib = ctypes.CDLL(so_name)
    # Kernel name example: sub_kernel_3_4
    kernel_name = f"sub_kernel_{'_'.join(map(str, shape))}"
    try:
        kernel_func = getattr(lib, kernel_name)
    except AttributeError:
        available = [attr for attr in dir(lib) if "sub" in attr.lower()]
        print(f"❌ Kernel function '{kernel_name}' not found in .so")
        print(f"Available functions: {available}")
        exit(1)

    # === 7. Set function signature ===
    # void sub_kernel_N_M(
    #   const float* a, const float* b, float* output,
    #   int* shape, int rank
    # );
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # a
        ctypes.POINTER(ctypes.c_float),  # b
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.POINTER(ctypes.c_int),    # shape
        ctypes.c_int,                    # rank
    ]
    kernel_func.restype = None

    # === 8. Call CUDA kernel ===
    shape_arr = (ctypes.c_int * len(shape))(*shape)
    print(f"🚀 Running CUDA kernel: {kernel_name}")
    kernel_func(a_ptr, b_ptr, output_ptr, shape_arr, len(shape))

    # === 9. Compare results ===
    if torch.allclose(
        result_cuda, result_torch,
        rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! CUDA result matches PyTorch.")
    else:
        print("❌ Verification failed!")
        print(f"PyTorch output (first 5): {result_torch.flatten()[:5]}")
        print(f"CUDA result (first 5): {result_cuda.flatten()[:5]}")
        diff = (result_torch - result_cuda).abs()
        print(f"Max diff: {diff.max().item():.2e}")
        exit(1)

    # Cleanup
    if os.path.exists(so_name):
        os.remove(so_name)
        print(f"🗑️  Removed {so_name}")