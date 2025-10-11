import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def ref_program(X_fp16, A_fp16, B_fp16):
    """Golden reference using autocast to mimic real inference behavior.

    Inputs: fp16 tensors on CUDA
    Output: fp32 tensor on CUDA
    """
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        O1 = torch.nn.functional.silu(torch.matmul(X_fp16, A_fp16))
        O2 = torch.matmul(X_fp16, B_fp16)
        O = O1 * O2
    return O.float()  # ensure output is fp32


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", type=str, required=True, help="Path to .cu source file"
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to kernel config"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform",
    )
    args = parser.parse_args()

    # Parse shape from filename: gate_mlp_4_4096.cu -> batch=4, dim=4096
    base_name = os.path.basename(args.file)
    name = "gatemlp"
    try:
        shapes = base_name.split(".")[0]
        shape_parts = [int(x) for x in shapes.split("_")[1:]]
        batch, dim_k, dim_n = shape_parts
    except Exception as e:
        raise ValueError(
            f"Invalid filename format: {base_name}. Expected: gate_mlp_batch_dim.cu"
        ) from e

    so_name = args.file.replace(".cu", ".so")
    temp_file = args.file.replace(".cu", "_bak.cu")

    # -------------------------------
    # 1. Inject macros and compile
    # -------------------------------
    with open(args.file, "r") as f:
        code = f.read()
    code = macro + code

    with open(temp_file, "w") as f:
        f.write(code)

    print(f"Compiling {temp_file} -> {so_name}")
    success, log = run_compilation(so_name, temp_file)
    if not success:
        print("Compilation failed:")
        print(log)
        exit(1)

    os.remove(temp_file)

    # -------------------------------
    # 2. Load shared library
    # -------------------------------
    try:
        lib = ctypes.CDLL(so_name)
    except Exception as e:
        print(f"Failed to load {so_name}: {e}")
        exit(1)

    kernel_func = getattr(lib, f"{name}_kernel")
    if kernel_func is None:
        print(f"Function {name}_kernel not found in {so_name}")
        exit(1)

    # -------------------------------
    # 3. Set function signature
    # -------------------------------
    # void gatemlp_kernel(const half*, const half*, const half*, float*, int,
    # int);
    kernel_func.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),  # X (fp16)
        ctypes.POINTER(ctypes.c_uint16),  # A (fp16)
        ctypes.POINTER(ctypes.c_uint16),  # B (fp16)
        ctypes.POINTER(ctypes.c_float),  # O (fp32)
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel_func.restype = None

    # -------------------------------
    # 4. Generate test data (on GPU)
    # -------------------------------
    torch.manual_seed(1234)
    device = torch.device("cuda")

    # Generate fp16 inputs
    X_fp16 = torch.ones(batch, dim_k, dtype=torch.float16, device=device) / 16
    A_fp16 = torch.ones(dim_k, dim_n, dtype=torch.float16, device=device) / 16
    B_fp16 = torch.ones(dim_k, dim_n, dtype=torch.float16, device=device) / 16

    # Reference output (fp32 on CUDA)
    O_ref = ref_program(X_fp16, A_fp16, B_fp16)

    # Output buffer: allocate fp32 tensor on CUDA
    O = torch.zeros(batch, dim_n, dtype=torch.float32, device=device)

    # Ensure contiguous
    X_fp16 = X_fp16.contiguous()
    A_fp16 = A_fp16.contiguous()
    B_fp16 = B_fp16.contiguous()
    O = O.contiguous()

    X_ptr = ctypes.cast(X_fp16.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    A_ptr = ctypes.cast(A_fp16.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    B_ptr = ctypes.cast(B_fp16.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    O_ptr = ctypes.cast(O.data_ptr(), ctypes.POINTER(ctypes.c_float))

    # -------------------------------
    # 5. Launch kernel
    # -------------------------------
    print(f"Running kernel: [{batch}, {dim_k}, {dim_n}] @ fp16")
    kernel_func(X_ptr, A_ptr, B_ptr, O_ptr, batch, dim_k, dim_n)

    # -------------------------------
    # 6. Verification (pure PyTorch, no numpy)
    # -------------------------------
    try:
        torch.testing.assert_close(
            O,
            O_ref,
            rtol=1e-2,  # fp16 计算容忍稍大误差
            atol=1e-3,
            msg="Output does not match reference!",
        )
        print("✅ Verification successful!")
    except AssertionError as e:
        print("❌ Verification failed!")
        print(e)
        exit(1)

    # -------------------------------
    # 7. Cleanup
    # -------------------------------
    subprocess.run(["rm", "-f", so_name], check=False)
