import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation

# Or simply use c_uint16 to represent __half*
c_half_p = ctypes.POINTER(ctypes.c_uint16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Dense (Linear) CPP kernel output against PyTorch"
    )
    parser.add_argument(
        "--file", type=str, help="Path to the source .cpp file"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to kernel configuration file",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform for compilation",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    # Example filename: dense_16_1024_1024.cu → shape = [batch, in_feat,
    # out_feat]
    shapes = base_name.split(".")[0]
    shape = [int(dim) for dim in shapes.split("_")[1:]]

    name = base_name.split("_")[0]  # e.g., 'dense'

    if len(shape) != 3:
        print(
            "[ERROR] Filename should encode: dense_batch_in_features_out_features.cu"
        )
        exit(1)

    batch_size, in_features, out_features = shape

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[WARNING] CPP not available. Running on CPU.")
        device = "cpu"

    # -------------------------------------------------------
    # ✅ 1. Generate input and parameters in correct dtypes
    # -------------------------------------------------------
    x = torch.ones(
        batch_size, in_features, dtype=torch.float16, device=device
    )  # fp16
    weight = torch.ones(
        in_features, out_features, dtype=torch.float16, device=device
    )  # fp16
    bias = torch.zeros(
        out_features, dtype=torch.float32, device=device
    )  # fp32

    # -------------------------------------------------------
    # ✅ 2. Reference: PyTorch Linear forward
    # -------------------------------------------------------

    matmul_out = x @ weight
    y_torch = matmul_out + bias
    y_torch = y_torch.float()  # ensure output is fp32 for comparison
    y_torch_cpu = y_torch.cpu().contiguous()

    # -------------------------------------------------------
    # ✅ 3. Host tensors for kernel input (ensure contiguous)
    # -------------------------------------------------------
    x_host = x.cpu().contiguous()  # fp16
    # fp16 (no need to transpose unless kernel expects it)
    weight_host = weight.cpu().contiguous()
    bias_host = bias.cpu().contiguous()  # fp32
    y_kernel = torch.zeros(
        batch_size, out_features, dtype=torch.float32
    ).contiguous()

    # -------------------------------------------------------
    # ✅ 4. Get raw pointers (use c_uint16 for __half*)
    # -------------------------------------------------------
    x_ptr = ctypes.cast(x_host.data_ptr(), c_half_p)
    weight_ptr = ctypes.cast(weight_host.data_ptr(), c_half_p)
    bias_ptr = ctypes.cast(
        bias_host.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )
    y_ptr = ctypes.cast(y_kernel.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # -------------------------------------------------------
    # ✅ 5. Shared library name
    # -------------------------------------------------------
    so_name = args.file.replace(".cu", ".so")

    # -------------------------------------------------------
    # ✅ 6. Read and inject macros
    # -------------------------------------------------------
    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code  # Inject config constants

    temp_file = base_name + "_bak.cu"
    with open(temp_file, "w") as f:
        f.write(code)

    # -------------------------------------------------------
    # ✅ 7. Compile kernel
    # -------------------------------------------------------
    success, output = run_compilation(so_name, temp_file)
    if not success:
        print("[ERROR] Compilation failed:")
        print(output)
        exit(1)

    os.remove(temp_file)

    # -------------------------------------------------------
    # ✅ 8. Load shared library
    # -------------------------------------------------------
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    kernel_func = getattr(lib, name + "_kernel")

    # -------------------------------------------------------
    # ✅ 9. Define function signature
    # -------------------------------------------------------
    kernel_func.argtypes = [
        c_half_p,  # input x: half*
        c_half_p,  # weight W: half*
        ctypes.POINTER(ctypes.c_float),  # bias b: float*
        ctypes.POINTER(ctypes.c_float),  # output y: float*
        ctypes.c_int,  # batch_size (M)
        ctypes.c_int,  # in_features (K)
        ctypes.c_int,  # out_features (N)
    ]
    kernel_func.restype = None

    # -------------------------------------------------------
    # ✅ 10. Call the kernel
    # -------------------------------------------------------
    kernel_func(
        x_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        batch_size,
        in_features,
        out_features,
    )

    # -------------------------------------------------------
    # ✅ 11. Verify results
    # -------------------------------------------------------
    if torch.allclose(
        y_kernel, y_torch_cpu, rtol=1e-2, atol=1e-2, equal_nan=True
    ):
        print(
            "✅ Verification successful! Dense layer output matches PyTorch (FP16 input, FP32 output)."
        )
    else:
        print("❌ Verification failed! Results do not match.")
        # Optional: debug print
        # print("Max diff:", (y_kernel - y_torch_cpu).abs().max().item())
        exit(1)

    # -------------------------------------------------------
    # ✅ 12. Clean up
    # -------------------------------------------------------
    subprocess.run(["rm", so_name], check=False)
