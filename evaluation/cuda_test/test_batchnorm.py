import argparse
import ctypes
import os
import torch
import torch.nn.functional as F

from evaluation.utils import run_dlboost_compilation as run_compilation


# Define the batchnorm function using torch
def batchnorm(x, weight, bias, running_mean, running_var, eps=1e-5, training=False):
    return F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=training,
        eps=eps
    )


def parse_filename(filename):
    """Parse N, C, H, W or N, C from filename like batchnorm_1_3_224_224.cu"""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., batchnorm_1_3_224_224

    if not stem.startswith("batchnorm_"):
        raise ValueError(f"Invalid filename: {filename}")

    try:
        shape_str = stem[len("batchnorm_"):]
        shape = list(map(int, shape_str.split("_")))
    except Exception as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    if len(shape) == 4:
        N, C, H, W = shape
        spatial_dims = 2
        print(f"🔍 Parsed 2D BatchNorm: [N={N}, C={C}, H={H}, W={W}]")
        return shape, spatial_dims
    elif len(shape) == 2:
        N, C = shape
        spatial_dims = 0
        print(f"🔍 Parsed 1D BatchNorm: [N={N}, C={C}]")
        return shape, spatial_dims
    else:
        raise ValueError(f"Unsupported shape length: {len(shape)} (expected 2 or 4)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batchnorm CUDA kernel against PyTorch")
    parser.add_argument("--file", type=str, required=True, help="Path to the .cu source file")
    args = parser.parse_args()

    # === 1. Parse shape from filename ===
    try:
        shape, spatial_dims = parse_filename(args.file)
    except Exception as e:
        raise RuntimeError(f"❌ Filename parsing failed: {e}")

    N, C = shape[0], shape[1]

    # === 2. Generate random input tensors ===
    if spatial_dims == 2:
        x = torch.randn(N, C, shape[2], shape[3], dtype=torch.float32, device="cpu")
    else:
        x = torch.randn(N, C, dtype=torch.float32, device="cpu")

    weight = torch.rand(C, dtype=torch.float32, device="cpu")
    bias = torch.rand(C, dtype=torch.float32, device="cpu")
    running_mean = torch.rand(C, dtype=torch.float32, device="cpu")
    running_var = torch.rand(C, dtype=torch.float32, device="cpu") + 0.5  # avoid zero
    eps = 1e-5

    print(f"🧪 Input tensor shape: {x.shape}")
    print(f"🧪 Parameters: C={C}, eps={eps}")

    # === 3. Perform batchnorm using PyTorch ===
    result_torch = batchnorm(x, weight, bias, running_mean, running_var, eps=eps, training=False)
    print(f"✅ PyTorch output shape: {result_torch.shape}")

    # === 4. Prepare ctypes pointers ===
    def to_ptr(tensor):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    x_ptr = to_ptr(x)
    weight_ptr = to_ptr(weight)
    bias_ptr = to_ptr(bias)
    running_mean_ptr = to_ptr(running_mean)
    running_var_ptr = to_ptr(running_var)

    # === 5. Prepare output buffer for CUDA kernel ===
    result_cuda = torch.zeros_like(result_torch)
    output_ptr = to_ptr(result_cuda)

    # === 6. Inject macro and compile ===
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code
        pass

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

    # === 7. Load shared library ===
    lib = ctypes.CDLL(so_name)
    kernel_name = "batchnorm_kernel"  # Assume kernel is named batchnorm_kernel
    try:
        kernel_func = getattr(lib, kernel_name)
    except AttributeError:
        available = [attr for attr in dir(lib) if "batchnorm" in attr.lower()]
        print(f"❌ Kernel function '{kernel_name}' not found in .so")
        print(f"Available functions: {available}")
        exit(1)

    # === 8. Set function signature ===
    # void batchnorm_kernel(
    #   const float* input, const float* weight, const float* bias,
    #   const float* running_mean, const float* running_var,
    #   float* output,
    #   int N, int C, int H, int W, float eps
    # );
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # weight
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # running_mean
        ctypes.POINTER(ctypes.c_float),  # running_var
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,  # N
        ctypes.c_int,  # C
        ctypes.c_int,  # H (1 if 1D)
        ctypes.c_int,  # W (1 if 1D)
        ctypes.c_float,  # eps
    ]
    kernel_func.restype = None

    # === 9. Call CUDA kernel ===
    H = shape[2] if spatial_dims == 2 else 1
    W = shape[3] if spatial_dims == 2 else 1

    print(f"🚀 Running CUDA kernel: {kernel_name}")
    kernel_func(
        x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
        output_ptr,
        N, C, H, W, eps
    )

    # === 10. Compare results ===
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