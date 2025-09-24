# test_gather.py
import argparse
import ctypes
import os

import torch

from evaluation.utils import run_dlboost_compilation as run_compilation

# Define the gather function using PyTorch


def gather_op(params, indices):
    return params[indices]  # shape: [len(indices), params.shape[1]]


def parse_filename(filename):
    """Parse shapes from filename like gather_100_32_16.cu."""
    base_name = os.path.basename(filename)
    stem = os.path.splitext(base_name)[0]  # e.g., gather_100_32_16

    if not stem.startswith("gather_"):
        raise ValueError(f"Invalid filename: {filename}")

    try:
        parts = list(map(int, stem.split("_")[1:]))  # skip 'gather'
    except ValueError as e:
        raise ValueError(f"Cannot parse shape from {stem}: {e}")

    if len(parts) != 3:
        raise ValueError(
            f"Expected 3 integers in filename, got {len(parts)}: {stem}"
        )

    PARAMS_BATCH, PARAMS_LEN, INDICES_LEN = parts

    print(
        f"🔍 Parsed shapes: params[{PARAMS_BATCH}, {PARAMS_LEN}], indices[{INDICES_LEN}] → output[{INDICES_LEN}, {PARAMS_LEN}]"
    )
    return PARAMS_BATCH, PARAMS_LEN, INDICES_LEN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test gather CUDA kernel against PyTorch"
    )
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

    # === 1. Parse shapes from filename ===
    try:
        PARAMS_BATCH, PARAMS_LEN, INDICES_LEN = parse_filename(args.file)
    except Exception as e:
        raise RuntimeError(f"❌ Filename parsing failed: {e}")

    # === 2. Generate input tensors ===
    params = torch.randn(
        PARAMS_BATCH, PARAMS_LEN, dtype=torch.float32, device="cpu"
    )
    # Random indices in [0, PARAMS_BATCH), including possible out-of-bound for
    # robustness
    indices = torch.randint(
        low=-1,
        high=PARAMS_BATCH + 1,
        size=(INDICES_LEN,),
        dtype=torch.int32,
        device="cpu",
    )
    print(f"🧪 params shape: {params.shape}")
    print(f"🧪 indices: {indices.tolist()} (dtype={indices.dtype})")

    # === 3. Perform gather using PyTorch ===
    # Clamp indices to valid range for reference
    clamped_indices = torch.clamp(indices, 0, PARAMS_BATCH - 1)
    result_torch = gather_op(params, clamped_indices)
    # For out-of-bound indices, PyTorch would error, but our kernel sets to 0
    # So we simulate the same behavior
    out_of_bound_mask = (indices < 0) | (indices >= PARAMS_BATCH)
    if out_of_bound_mask.any():
        result_torch = result_torch.clone()
        result_torch[out_of_bound_mask] = 0.0
    print(f"✅ PyTorch output shape: {result_torch.shape}")

    # === 4. Prepare ctypes pointers ===
    def to_ptr(tensor):
        return tensor.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    params_ptr = to_ptr(params)
    indices_ptr = indices.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32)
    )
    result_cuda = torch.zeros_like(result_torch)
    output_ptr = to_ptr(result_cuda)

    # === 5. Inject macro and compile ===
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    # Optional: inject macro for shape specialization (if needed)
    # But our kernel is already specialized by name, so not needed
    # code = macro + code

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

    # Kernel name: e.g., static_gather_100_32_16_cuda
    kernel_name = (
        f"static_gather_{PARAMS_BATCH}_{PARAMS_LEN}_{INDICES_LEN}_cuda"
    )
    try:
        kernel_func = getattr(lib, kernel_name)
    except AttributeError:
        available = [attr for attr in dir(lib) if "gather" in attr.lower()]
        print(f"❌ Kernel function '{kernel_name}' not found in .so")
        print(f"Available functions: {available}")
        exit(1)

    # === 7. Set function signature ===
    # void static_gather_xxx_cuda(
    #   const float* params,
    #   const int32_t* indices,
    #   float* output
    # );
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # params
        ctypes.POINTER(ctypes.c_int32),  # indices
        ctypes.POINTER(ctypes.c_float),  # output
    ]
    kernel_func.restype = None

    # === 8. Call CUDA kernel ===
    print(f"🚀 Running CUDA kernel: {kernel_name}")
    kernel_func(params_ptr, indices_ptr, output_ptr)

    # Synchronize GPU (if needed, though dlboost may handle it)
    try:
        torch.cuda.synchronize()
    except BaseException:
        pass  # No GPU used

    # === 9. Compare results ===
    if torch.allclose(result_cuda, result_torch, atol=1e-5):
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
