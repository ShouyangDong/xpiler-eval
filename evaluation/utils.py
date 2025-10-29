import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.macros import CPP_MACROS, CUDA_MACROS, HIP_MACROS, MLU_MACROS

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --- Mapping from operators to test scripts ---
TEST_SCRIPT_MAP = {
    "deformable": "test_deformable_attention.py",
    "layernorm": "test_layer_norm.py",
    "mha": "test_mha.py",
    "rmsnorm": "test_rms_norm.py",
    "gemm": "test_gemm.py",
    "gemv": "test_gemv.py",
    "bmm": "test_bmm.py",
    "conv1d": "test_conv1d.py",
    "conv2d": "test_conv2d.py",
    "conv2dnchw": "test_conv2dNCHW.py",
    "depthwiseconv": "test_depthwise_conv.py",
    "add": "test_add.py",
    "sign": "test_sign.py",
    "avgpool": "test_avgpool.py",
    "maxpool": "test_maxpool.py",
    "minpool": "test_minpool.py",
    "sumpool": "test_sumpool.py",
    "relu": "test_relu.py",
    "sigmoid": "test_sigmoid.py",
    "gelu": "test_gelu.py",
    "softmax": "test_softmax.py",
    "gather": "test_gather.py",
    "transpose": "test_transpose.py",
    "max": "test_max.py",
    "min": "test_min.py",
    "sum": "test_sum.py",
    "mean": "test_mean.py",
    "batchnorm": "test_batchnorm.py",
    "sub": "test_sub.py",
    "sin": "test_sin.py",
    "instancenorm": "test_instancenorm.py",
    "concat": "test_concat.py",
    "scatter": "test_scatter.py",
    "dense": "test_dense.py",
    "gatemlp": "test_gatemlp.py",
    "gqa": "test_gqa.py",
}


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


# Mapping from target to compilation function
COMPILATION_FUNCTIONS = {
    "cpu": run_cpp_compilation,
    "cuda": run_cuda_compilation,
    "hip": run_hip_compilation,
    "mlu": run_mlu_compilation,
}


# Mapping from target to macro
MACRO_FUNCTIONS = {
    "cpu": CPP_MACROS,
    "cuda": CUDA_MACROS,
    "hip": HIP_MACROS,
    "mlu": MLU_MACROS,
}


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
        file_name = (
            f"{c['op_name']}_{'_'.join(map(str, c['args']))}.{file_type}"
        )
        op_configs.append({**c, "file": file_name})
    return op_configs


def patch_source(src_path: str, dst_path: str, target: str) -> bool:
    """Insert macros and save patched source."""
    try:
        with open(src_path, "r") as f:
            code = f.read()
        macro = MACRO_FUNCTIONS.get(target)
        code = macro + code
        with open(dst_path, "w") as f:
            f.write(code)
        return True
    except Exception as e:
        logger.error(f"Failed to patch {src_path}: {e}")
        return False


def compile_kernel(
    op_name: str, config: dict, source_dir: str, target: str
) -> Tuple[dict, bool, str]:
    """Compile one kernel using target-specific compilation function.

    Returns: (config, success, message)
    """
    file_name = config["file"]
    file_path = os.path.join(source_dir, file_name)

    # Extract base name and extension
    base_name, ext = os.path.splitext(file_path)
    so_path = f"{base_name}.so"
    temp_file = f"{base_name}_patched{ext}"

    if not os.path.isfile(file_path):
        return config, False, f"[{op_name}] Source not found: {file_path}"

    # Patch source
    if not patch_source(file_path, temp_file, target):
        return config, False, f"[{op_name}] Patch failed: {file_path}"

    # 🔁 Dynamically select compilation function based on target
    compile_func = COMPILATION_FUNCTIONS.get(target)
    if compile_func is None:
        return (
            config,
            False,
            f"[{op_name}] No compilation function for target: {target}",
        )

    # Run the target-specific compilation
    success, msg = compile_func(so_path, temp_file)

    # Cleanup temp file (always)
    try:
        os.remove(temp_file)
    except OSError:
        pass  # Ignore cleanup failure

    if success:
        return config, True, so_path  # Return path to .so
    else:
        return config, False, f"[{op_name}] Compile failed: {msg}"


def run_tests(
    op_name: str,
    configs: List[dict],
    source_dir: str,
    test_script: str,
    target: str,
    num_workers: int = 4,
) -> List[Tuple[bool, str]]:
    """
    Phase 1: Compile all in parallel.
    Phase 2: Test only successful ones in parallel.
    Phase 3: Clean up .so files.
    """
    logger.info(
        f"[{op_name}] Starting two-phase test for {len(configs)} kernels..."
    )

    compiled_so_map: Dict[str, str] = {}  # file -> so_path
    failed_results = []

    # === PHASE 1: Parallel Compilation ===
    logger.info(
        f"[{op_name}] Phase 1/2: Compiling {len(configs)} kernels with {num_workers} workers..."
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                compile_kernel, op_name, config, source_dir, target
            )
            for config in configs
        ]

        for future in as_completed(futures):
            config, success, msg = future.result()
            if success:
                compiled_so_map[config["file"]] = msg  # msg == so_path
            else:
                failed_results.append((False, msg))
                print(f"❌ Compilation failed: {msg}")

    compiled_count = len(compiled_so_map)
    logger.info(
        f"[{op_name}] Compilation: {compiled_count} succeeded, {len(failed_results)} failed."
    )

    # === PHASE 2: Parallel Testing ===
    if compiled_so_map:
        logger.info(
            f"[{op_name}] Phase 2/2: Testing {compiled_count} compiled kernels..."
        )
        test_configs = [
            (config, compiled_so_map[config["file"]])
            for config in configs
            if config["file"] in compiled_so_map
        ]

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"test_{op_name}", test_script
        )
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(test_module.test_kernel, config, so_path)
                for config, so_path in test_configs
            ]

            for future in as_completed(futures):
                result = future.result()
                if not result[0]:
                    print(f"❌ Test failed: {result[1]}")
                failed_results.append(result)

        passed_count = sum(
            1 for r in failed_results[-len(test_configs):] if r[0]
        )
        logger.info(
            f"[{op_name}] Testing: {passed_count} passed, {len(test_configs) -
                                                           passed_count} failed."
        )

        # === PHASE 3: Clean up .so files ===
        logger.debug(
            f"[{op_name}] Phase 3/3: Cleaning up generated .so files..."
        )
        for _, so_path in test_configs:
            try:
                if os.path.exists(so_path):
                    os.remove(so_path)
            except Exception as e:
                logger.warning(f"[{op_name}] Failed to delete {so_path}: {e}")

    return failed_results


def log_test_results_and_exit(
    results: List[Tuple[bool, str]], op_name: str = "Unknown"
) -> None:
    """Logs individual test outcomes and summarizes the overall result.

    This function processes a list of test results, logs each outcome
    (info for success, error for failure), then prints a final summary.
    It exits the program based on whether all tests passed.

    Args:
        results: A list of tuples, each containing:
                 - bool: True if the test passed, False otherwise.
                 - str:  Message to log (e.g., test description or error).
        op_name: Name of the operator or test group (for logging clarity).

    Exits:
        0 if all tests passed.
        1 if any test failed.
    """
    total = len(results)
    passed = sum(1 for success, _ in results if success)
    failure_count = total - passed
    success_rate = passed / total if total > 0 else 1.0

    # Log individual test results
    for success, message in results:
        if success:
            logger.info("✅ PASS | %s", message)
        else:
            logger.error("❌ FAIL | %s", message)

    # Log final summary
    if success_rate == 1.0:
        logger.info(f"🎉 All {total} tests for '{op_name}' passed.")
        exit(0)
    else:
        logger.error(
            f"❌ {failure_count}/{total} tests failed for '{op_name}'."
        )
        exit(1)


def verify_numpy_tensor(
    output_tensor, ref_tensor, op_name="OP", rtol=1e-3, atol=1e-3
):
    """Check if output_tensor is close to ref_tensor.

    Args:
        output_tensor: Computed result
        ref_tensor: Expected (reference) result
        op_name: Operation name for logging
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        (success: bool, message: str)
    """
    try:
        np.allclose(
            output_tensor, ref_tensor, rtol=1e-3, atol=1e-3, equal_nan=True
        )
        return True, f"[{op_name}] PASSED✅"

    except Exception as e:
        return False, f"[{op_name}] FAILED❌ - Error: {e}"


def verify_torch_tensor(
    output_tensor, ref_tensor, op_name="OP", rtol=1e-3, atol=1e-3
):
    """Check if output_tensor is close to ref_tensor.

    Args:
        output_tensor: Computed result
        ref_tensor: Expected (reference) result
        op_name: Operation name for logging
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        (success: bool, message: str)
    """
    try:
        torch.allclose(
            output_tensor, ref_tensor, rtol=rtol, atol=atol, equal_nan=True
        )
        return True, f"[{op_name}] PASSED✅"

    except Exception as e:
        return False, f"[{op_name}] FAILED❌ - Error: {e}"
