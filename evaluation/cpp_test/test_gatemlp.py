# evaluation/validate_int16_to_int32.py
import argparse
import ctypes
import os
import subprocess

import numpy as np
import torch

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation


def ref_program(X, A, B):
    """Golden reference using PyTorch.
    Inputs: X, A, B as int16 NumPy arrays
    Output: O = SiLU(X @ A) * (X @ B) converted to int32
    """
    # Forward: O = silu(X @ A) * (X @ B)
    C = torch.matmul(X, A).to(torch.float32)
    O2 = torch.matmul(X, B).to(torch.int32)
    O1 = torch.nn.functional.silu(C)
    O_fp32 = O1 * O2
    return O_fp32.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
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

    # Parse file name: gate_mlp_4_4096.cpp -> [4, 4096]
    base_name = os.path.basename(args.file)
    name = "gatemlp"
    shapes = base_name.split(".")[0]
    shape_parts = [int(intg) for intg in shapes.split("_")[1:]]
    batch, dim_k, dim_n = shape_parts
    so_name = args.file.replace(".cpp", ".so")

    # -------------------------------
    # 1. Add macro and compile
    # -------------------------------
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

    # -------------------------------
    # 2. Load kernel
    # -------------------------------
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int16),   # X: int16*
        ctypes.POINTER(ctypes.c_int16),   # A: int16*
        ctypes.POINTER(ctypes.c_int16),   # B: int16*
        ctypes.POINTER(ctypes.c_float),     # O: int32* (output)
    ]
    function.restype = None

    # -------------------------------
    # 3. Generate random inputs (int16)
    # -------------------------------
    torch.manual_seed(1234)

    # -------------------------------
    # Generate input tensors in [-10, 10], int16
    # -------------------------------
    X_int16 = torch.randint(low=-10, high=11, size=(batch, dim_k), dtype=torch.int16)
    A_int16 = torch.randint(low=-10, high=11, size=(dim_k, dim_n), dtype=torch.int16)
    B_int16 = torch.randint(low=-10, high=11, size=(dim_k, dim_n), dtype=torch.int16)
    # Reference: compute in float32, then convert to int32
    O_ref = ref_program(X_int16, A_int16, B_int16)

    # Output buffer: int32
    O = np.zeros((batch, dim_n), dtype=np.float32)

    # Get pointers
    X_ptr = X_int16.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
    A_ptr = A_int16.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
    B_ptr = B_int16.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
    O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # -------------------------------
    # 4. Invoke C++ kernel
    # -------------------------------
    function(X_ptr, A_ptr, B_ptr, O_ptr)

    # -------------------------------
    # 5. Verification
    # -------------------------------
    try:
        np.testing.assert_allclose(
            O,
            O_ref,
            rtol=1e-3,        # relative tolerance (not very meaningful for int)
            atol=2,           # absolute tolerance: allow ±2 due to rounding
            equal_nan=False,
            verbose=True,
        )
        print("✅ Verification successful! (int16 → int32)")
    except AssertionError as e:
        print("❌ Verification failed!")
        print(e)
        # Optional: print diff
        diff = np.abs(O.astype(np.int64) - O_ref.astype(np.int64))
        print(f"Max diff: {diff.max()}")
        exit(1)

    # Clean up
    subprocess.run(["rm", so_name], check=False)
