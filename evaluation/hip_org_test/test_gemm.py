import argparse
import ctypes
import os
import subprocess

import torch

from evaluation.macros import HIP_MACROS as macro
from evaluation.utils import run_hip_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate GEMM HIP kernel output against PyTorch"
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
        choices=["cuda", "hip", "bang", "cpu"],
        help="Target platform for compilation",
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(dim) for dim in shapes.split("_")[1:]]
    name = base_name.split("_")[0]

    M, K, N = shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(
            "[ERROR] ROCm not available. Please install PyTorch with ROCm support."
        )
        exit(1)

    A = torch.ones((M, K), dtype=torch.float16, device=device)
    x = torch.ones((K, N), dtype=torch.float16, device=device)

    y_torch = torch.matmul(A, x).to(torch.float32)

    y_torch_cpu = y_torch.cpu().contiguous()
    y_torch_ptr = ctypes.cast(
        y_torch_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)
    )

    A_host = torch.ones((M, K), dtype=torch.float16).contiguous()
    x_host = torch.ones((K, N), dtype=torch.float16).contiguous()

    A_ptr = ctypes.cast(A_host.data_ptr(), ctypes.POINTER(ctypes.c_ushort))
    x_ptr = ctypes.cast(x_host.data_ptr(), ctypes.POINTER(ctypes.c_ushort))

    y_ctypes = torch.zeros((M, N), dtype=torch.float32).contiguous()
    y_ptr = ctypes.cast(y_ctypes.data_ptr(), ctypes.POINTER(ctypes.c_float))

    so_name = args.file.replace(".hip", ".so")

    with open(args.file, "r") as f:
        code = f.read()

    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".hip", ""), base_name + "_bak.hip"
    )
    with open(file_name, "w") as f:
        f.write(code)

    success, output = run_compilation(so_name, file_name)
    if not success:
        print("[ERROR] Compilation failed:")
        print(output)
        exit(1)

    os.remove(file_name)

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")

    function.argtypes = [
        ctypes.POINTER(ctypes.c_ushort),
        ctypes.POINTER(ctypes.c_ushort),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None

    function(A_ptr, x_ptr, y_ptr, M, K, N)

    if torch.allclose(
        y_ctypes, y_torch_cpu, rtol=1e-3, atol=1e-3, equal_nan=True
    ):
        print("✅ Verification successful! Results match.")
    else:
        print("❌ Verification failed! Results do not match.")
        exit(1)

    subprocess.run(["rm", so_name], check=False)
