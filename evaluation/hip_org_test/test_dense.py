import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Dense (Linear) HIP kernel output against PyTorch"
    )
    parser.add_argument(
        "--file", type=str, help="Path to the source .hip file"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to kernel configuration file",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform for compilation",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    # Example filename: dense_16_1024_1024.hip → shape = [batch, in_feat,
    # out_feat]
    shapes = base_name.split(".")[0]
    shape = [int(dim) for dim in shapes.split("_")[1:]]

    name = base_name.split("_")[0]  # e.g., 'dense'

    if len(shape) != 3:
        print(
            "[ERROR] Filename should encode: dense_batch_in_features_out_features.hip"
        )
        exit(1)

    batch_size, in_features, out_features = shape

    # Use AMD GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[WARNING] ROCm not available. Running on CPU.")

    # Generate input and parameters
    x = torch.randn(
        batch_size, in_features, dtype=torch.float32, device=device
    )
    weight = torch.randn(
        out_features, in_features, dtype=torch.float32, device=device
    )
    bias = torch.randn(out_features, dtype=torch.float32, device=device)

    # Reference: PyTorch Linear forward
    y_torch = torch.nn.functional.linear(
        x, weight, bias
    )  # shape: (batch_size, out_features)

    # Move reference result to CPU for comparison
    y_torch_cpu = y_torch.cpu().contiguous()

    # Host tensors for kernel input
    x_host = x.cpu().contiguous()
    weight_host = weight.t().cpu().contiguous()
    bias_host = bias.cpu().contiguous()

    # Output buffer (CPU)
    y_kernel = torch.zeros(
        batch_size, out_features, dtype=torch.float32
    ).contiguous()

    # Get raw pointers
    x_ptr = ctypes.cast(x_host.data_ptr(), ctypes.POINTER(ctypes.c_float))
    weight_ptr = ctypes.cast(
        weight_host.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    bias_ptr = ctypes.cast(
        bias_host.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    y_ptr = ctypes.cast(y_kernel.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # Shared library name
    so_name = args.file.replace(".hip", ".so")

    # Read and inject macros
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code  # Inject config constants

    # Write temporary .hip file
    temp_file = base_name + "_bak.hip"
    with open(temp_file, "w") as f:
        f.write(code)

    # Compile kernel
    success, output = run_compilation(so_name, temp_file)
    if not success:
        print("[ERROR] Compilation failed:")
        print(output)
        exit(1)

    os.remove(temp_file)

    # Load shared library
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    dense_func = getattr(lib, name + "_kernel")

    # Define function signature
    dense_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input (x)
        ctypes.POINTER(ctypes.c_float),  # weight (W)
        ctypes.POINTER(ctypes.c_float),  # bias (b)
        ctypes.POINTER(ctypes.c_float),  # output (y)
        ctypes.c_int,  # batch_size
        ctypes.c_int,  # in_features
        ctypes.c_int,  # out_features
    ]
    dense_func.restype = None

    # Call the Dense kernel
    dense_func(
        x_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        batch_size,
        in_features,
        out_features,
    )

    # Verify results
    if torch.allclose(
        y_kernel, y_torch_cpu, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print(
            "✅ Verification successful! Dense layer output matches PyTorch."
        )
    else:
        print("❌ Verification failed! Results do not match.")
        exit(1)

    # Clean up
    subprocess.run(["rm", so_name], check=False)
