import argparse
import ctypes
import os
import torch


from evaluation.utils import run_dlboost_compilation as run_compilation


# Define the reshape function using torch
def reshape_op(x, output_shape):
    return x.reshape(output_shape)


def parse_filename(filename):
    """Parse input_shape and output_shape from filename like reshape_2_3_to_6.cu"""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., reshape_2_3_to_6

    if not stem.startswith("reshape_"):
        raise ValueError(f"Invalid filename: {filename}")

    if '_to_' not in stem:
        raise ValueError(f"Missing '_to_' in filename: {stem}")

    in_part, out_part = stem.split('_to_', 1)  # split on first '_to_'
    try:
        input_shape = list(map(int, in_part.split('_')[1:]))  # skip 'reshape'
        output_shape = list(map(int, out_part.split('_')))
    except ValueError as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    total_in = torch.Size(input_shape).numel()
    total_out = torch.Size(output_shape).numel()

    if total_in != total_out:
        raise ValueError(f"Reshape invalid: {input_shape} -> {output_shape} (size mismatch)")

    print(f"🔍 Parsed input shape: {input_shape}, output shape: {output_shape}")
    return input_shape, output_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reshape CUDA kernel against PyTorch")
    parser.add_argument("--file", type=str, required=True, help="Path to the .cu source file")
    args = parser.parse_args()

    # === 1. Parse input and output shapes from filename ===
    try:
        input_shape, output_shape = parse_filename(args.file)
    except Exception as e:
        raise RuntimeError(f"❌ Filename parsing failed: {e}")

    # === 2. Generate random input tensor ===
    x = torch.randn(*input_shape, dtype=torch.float32, device="cpu")
    print(f"🧪 Input tensor shape: {x.shape}")

    # === 3. Perform reshape using PyTorch ===
    result_torch = reshape_op(x, output_shape)
    print(f"✅ PyTorch output shape: {result_torch.shape}")

    # === 4. Prepare ctypes pointers ===
    def to_ptr(tensor):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    x_ptr = to_ptr(x)
    result_cuda = torch.zeros_like(result_torch)
    output_ptr = to_ptr(result_cuda)

    # === 5. Inject macro and compile ===
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

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
    # Kernel name example: reshape_kernel_2_3_to_6
    kernel_name = f"reshape_kernel_{'_'.join(map(str, input_shape))}_to_{'_'.join(map(str, output_shape))}"
    try:
        kernel_func = getattr(lib, kernel_name)
    except AttributeError:
        available = [attr for attr in dir(lib) if "reshape" in attr.lower()]
        print(f"❌ Kernel function '{kernel_name}' not found in .so")
        print(f"Available functions: {available}")
        exit(1)

    # === 7. Set function signature ===
    # void reshape_kernel_2_3_to_6(
    #   const float* input,
    #   float* output,
    #   int* in_shape, int in_rank,
    #   int* out_shape, int out_rank
    # );
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),      # input
        ctypes.POINTER(ctypes.c_float),      # output
        ctypes.POINTER(ctypes.c_int),        # in_shape
        ctypes.c_int,                        # in_rank
        ctypes.POINTER(ctypes.c_int),        # out_shape
        ctypes.c_int,                        # out_rank
    ]
    kernel_func.restype = None

    # === 8. Call CUDA kernel ===
    in_shape_arr = (ctypes.c_int * len(input_shape))(*input_shape)
    out_shape_arr = (ctypes.c_int * len(output_shape))(*output_shape)

    print(f"🚀 Running CUDA kernel: {kernel_name}")
    kernel_func(
        x_ptr,
        output_ptr,
        in_shape_arr, len(input_shape),
        out_shape_arr, len(output_shape)
    )

    # === 9. Compare results ===
    if torch.equal(result_cuda, result_torch):  # reshape is exact
        print("✅ Verification successful! CUDA result matches PyTorch.")
    else:
        print("❌ Verification failed!")
        print(f"PyTorch output (first 10): {result_torch.flatten()[:10]}")
        print(f"CUDA result (first 10): {result_cuda.flatten()[:10]}")
        diff = (result_torch - result_cuda).abs()
        print(f"Max diff: {diff.max().item():.2e}")
        exit(1)

    # Cleanup
    if os.path.exists(so_name):
        os.remove(so_name)
        print(f"🗑️  Removed {so_name}")