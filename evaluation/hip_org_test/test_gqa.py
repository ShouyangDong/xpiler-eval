import argparse
import ctypes
import os
import subprocess

import numpy as np
import torch

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation

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
    parser.add_argument(
        "--batch", type=int, default=1, help="Batch size for test"
    )
    args = parser.parse_args()

    # ---------------------------------------------
    # 1.  Parse file name
    # ---------------------------------------------
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    parts = [int(x) for x in shapes.split("_")[1:]]
    batch, seq_q, seq_kv, head_dim = parts

    so_name = args.file.replace(".cpp", ".so")

    # ---------------------------------------------
    # 2. Add macro
    # ---------------------------------------------
    with open(args.file, "r") as f:
        code = f.read()
    code = macro + code

    file_name = args.file.replace(".cpp", "_bak.cpp")
    with open(file_name, "w") as f:
        f.write(code)

    success, output = run_compilation(so_name, file_name)
    if not success:
        print("Compilation failed:")
        print(output)
        exit(1)

    os.remove(file_name)

    # ---------------------------------------------
    # 3. Load shared library
    # ---------------------------------------------
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    gqa_func = getattr(lib, args.name + "_kernel")
    gqa_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # K
        ctypes.POINTER(ctypes.c_float),  # V
        ctypes.POINTER(ctypes.c_float),  # O
        ctypes.c_int,  # batch
        ctypes.c_int,  # num_heads
        ctypes.c_int,  # seq_q
        ctypes.c_int,  # seq_kv
        ctypes.c_int,  # head_dim
    ]
    gqa_func.restype = None

    # ---------------------------------------------
    # 4. Get the refered output using Pytorch
    # ---------------------------------------------
    torch.manual_seed(1234)

    # Generate Q, K, V
    Q = torch.rand([batch, 2, seq_q, 64], dtype=torch.float32, device="cuda")
    K = torch.rand([batch, 2, 64, seq_kv], dtype=torch.float32, device="cuda")
    V = torch.rand([batch, 2, seq_kv, 64], dtype=torch.float32, device="cuda")

    # ✅ Get the referenced ouput
    with torch.no_grad():
        S = torch.matmul(Q, K)  # [b, 2, seq_q, seq_kv]
        S = torch.softmax(S, dim=-1)
        O_ref = torch.matmul(S, V)  # [b, 2, seq_q, 64]

    # ---------------------------------------------
    # 5. Prepare C++ kernel and feed output buffer
    # ---------------------------------------------
    O = torch.zeros_like(O_ref, device="cuda")  # host buffer for output

    Q_cpu = Q.cpu().numpy()
    K_cpu = K.cpu().numpy()
    V_cpu = V.cpu().numpy()

    Q_ptr = Q_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    K_ptr = K_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    V_ptr = V_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # ---------------------------------------------
    # 6. invoke C++/CUDA kernel
    # ---------------------------------------------
    gqa_func(Q_ptr, K_ptr, V_ptr, O_ptr, batch, 2, seq_q, seq_kv, 64)

    # ---------------------------------------------
    # 7. Verification
    # ---------------------------------------------
    np.testing.assert_allclose(
        O,  # C++ kernel output(NumPy)
        O_ref.cpu().numpy(),
        rtol=5e-3,
        atol=5e-3,
        equal_nan=True,
        verbose=True,
    )

    print("✅ GQA verification passed!")

    # ---------------------------------------------
    # 8. clean
    # ---------------------------------------------
    subprocess.run(["rm", so_name], check=False)
