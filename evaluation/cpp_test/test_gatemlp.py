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

    Inputs: X, A, B as PyTorch tensors (on CPP)
    Output: O = SiLU(X @ A) * (X @ B) as NumPy array
    """
    O1 = torch.nn.functional.silu(torch.matmul(X, A))
    O2 = torch.matmul(X, B)
    O = O1 * O2
    return O.cpu().numpy()


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
    args = parser.parse_args()

    # Parse file name: gate_mlp_4_4096.cpp -> [4, 4096]
    base_name = os.path.basename(args.file)
    name = "gatemlp"
    shapes = base_name.split(".")[0]
    shape_parts = [
        int(intg) for intg in shapes.split("_")[2:]
    ]  # skip "gate", "mlp"
    batch, dim = shape_parts
    so_name = args.file.replace(".cpp", ".so")

    # -------------------------------
    # 1. Add macro
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
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # O
        ctypes.c_int,  # batch
        ctypes.c_int,  # dim
    ]
    function.restype = None

    # -------------------------------
    # ✅ 3. Generate random inputs
    # -------------------------------
    torch.manual_seed(1234)

    #
    X = torch.randn(batch, dim, dtype=torch.float32, device="cpu")
    A = torch.randn(dim, dim, dtype=torch.float32, device="cpu")
    B = torch.randn(dim, dim, dtype=torch.float32, device="cpu")

    # Calculate output
    O_ref = ref_program(X, A, B)

    X_np = X.cpu().numpy()
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()
    O = np.zeros_like(O_ref)  # output buffer

    # Get pointer
    X_ptr = X_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # -------------------------------
    # 4. invoke C++ kernel
    # -------------------------------
    function(X_ptr, A_ptr, B_ptr, O_ptr, batch, dim)

    # -------------------------------
    # 5. Verification
    # -------------------------------
    np.testing.assert_allclose(
        O,
        O_ref,
        rtol=5e-3,
        atol=5e-3,
        equal_nan=True,
        verbose=True,
    )

    print("Verification successful!")

    # clean .so file
    subprocess.run(["rm", so_name], check=False)
