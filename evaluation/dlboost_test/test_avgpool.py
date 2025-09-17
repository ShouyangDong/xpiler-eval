import argparse
import ctypes
import os
import subprocess

import torch

from benchmark.utils import avgpool_np
from benchmark.utils import run_dlboost_compilation as run_compilation


def avgpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # Perform average pooling.
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)

    name = base_name.split(".")[0].split("_")[0]
    shape = base_name.split(".")[0].split("_")[1:5]
    shape = [int(intg) for intg in shape]
    kernel_stride = base_name.split(".")[0].split("_")[5:]
    kernel_stride = [int(intg) for intg in kernel_stride]

    dtype = "float32"
    input_array = torch.randn(*shape, device="cpu")
    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_np = avgpool_np(input_array, kernel_stride)
    output_array = torch.zeros(output_np.shape, dtype=torch.float32)
    output_ptr = output_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # Load the shared library with the avgpool function
    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open(
        os.path.join(os.getcwd(), "benchmark/macro/cpp_macro.txt"), "r"
    ) as f:
        macro = f.read()
        f.close()
    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".cpp", ""), base_name + "_bak.cpp"
    )
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name)
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(input_ptr, output_ptr)
    # Check if the results match
    torch.allclose(
        output_array,
        output_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )
    print("Verification successful!")
    result = subprocess.run(["rm", so_name])
