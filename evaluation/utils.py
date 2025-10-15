import json
import subprocess
import json
import torch
import torch.nn.functional as F
import os

def avgpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # Perform average pooling.
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


def sumpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # Perform average pooling.
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor * kernel_stride[0] * kernel_stride[1]


def maxpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # Perform average pooling.
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


def minpool_np(input_tensor, kernel_stride):
    class MinPool2d(torch.nn.Module):
        def __init__(self, kernel_size, stride=None, padding=0):
            super(MinPool2d, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

        def forward(self, x):
            # Inverted input
            x_neg = -x
            # Perform maximum pooling.
            x_maxpool = F.max_pool2d(
                x_neg,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            # Reversing the result again
            return -x_maxpool

    # Using a custom MinPool2d
    pool = MinPool2d(kernel_size=kernel_stride[:2], stride=kernel_stride[2:])
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    output_tensor = pool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


def conv2d_nchw(input_tensor, kernels, stride, padding=0):
    output = F.conv2d(input_tensor, kernels, stride=stride, padding=padding)
    return output


def conv2d_nhwc(input_nhwc, kernels, stride, padding):
    # Convert the input from NHWC to NCHW.
    input_nchw = input_nhwc.permute(0, 3, 1, 2)

    # Convert the kernel from (O, H, W, I) format to
    # PyTorch's OIHW format.
    weight_oihw = kernels.permute(0, 3, 1, 2)

    # Perform convolution operations using the transformed convolution kernel
    # and input.
    output_nchw = F.conv2d(
        input_nchw, weight_oihw, stride=stride, padding=padding
    )

    # Convert the output from NCHW back to NHWC.
    output_nhwc = output_nchw.permute(0, 2, 3, 1)
    return output_nhwc


def run_cpp_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "g++",
                "-shared",
                "-fPIC",
                "-march=icelake-server",
                "-O3",
                file_name,
                "-o",
                so_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            timeout=40,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_mlu_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "cncc",
                "-shared",
                "-fPIC",
                "--bang-mlu-arch=mtp_592",
                "-O3",
                "-o",
                so_name,
                file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=40,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_cuda_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "nvcc",
                "-Xcompiler",
                "-fPIC",
                "-shared",
                "-arch=sm_80",
                "-o",
                so_name,
                file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=40,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_hip_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "hipcc",
                "-fPIC",
                "-shared",
                "--offload-arch=gfx942",
                "-o",
                so_name,
                file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=40,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_test(file_path, test_script, kernel_config, target):
    """Run a test script for a compiled kernel.

    :param file_path: Path to the .cu/.cpp file (or .so if already compiled)
    :param test_script: Path to the test script (e.g., test_add.py)
    :param kernel_config: Dict containing op_name, args, axes, dtype, etc.
    :param target: One of 'cuda', 'hip', 'bang', 'cpu'
    :return: (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            [
                "python",
                test_script,
                "--file",
                file_path,
                "--config",
                json.dumps(kernel_config),
                "--target",
                target,
            ],
            capture_output=True,
            text=True,
            timeout=400,
        )
        success = result.returncode == 0
        output = result.stdout.strip() + "\n" + result.stderr.strip()
        return success, output
    except Exception as e:
        return False, str(e)


def parse_op_json(json_path, op_name="None", file_type="cpp"):
    # Load config
    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            configs = json.load(f)
    else:
        configs = json.loads(json_path)

    if isinstance(configs, dict):
        configs = [configs]

    # Filter bmm configs
    configs = [c for c in configs if c.get("op_name") == op_name]
    op_configs = []
    for c in configs:
        file_name = f"{c['op_name']}_{'_'.join(map(str, c['args']))}.{file_type}"
        op_configs.append({**c, "file": file_name})
    return op_configs