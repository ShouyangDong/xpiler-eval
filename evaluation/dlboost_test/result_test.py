#!/usr/bin/env python3
"""Multi-platform correctness tester for compiled kernels (.so).

Reads kernel configs from kernels.json, maps to test scripts, and runs tests.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

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
    "scatter": "test_scatter.py",
    "instancenorm": "test_instancenorm.py",
    "concate": "test_concate",
}


def run_test_for_config(config, source_dir, test_dir, target):
    """Run test for a single kernel config.

    Returns: (success: bool, message: str)
    """
    op_name = config["op_name"]
    args = config["args"]

    # Construct filename
    file_name = f"{op_name}_{'_'.join(map(str, args))}.{'cu' if target != 'cpu' else 'cpp'}"
    file_path = os.path.join(source_dir, file_name)

    if not os.path.exists(file_path):
        return False, f"[ERROR] File not found: {file_path} (op={op_name})"

    # Find test script
    script_name = TEST_SCRIPT_MAP.get(op_name)
    if not script_name:
        return False, f"[WARN] No test script for op='{op_name}'"

    test_script = os.path.join(test_dir, script_name)
    if not os.path.isfile(test_script):
        return False, f"[ERROR] Test script not found: {test_script}"

    # invoke evaluation.utils.run_test
    try:
        from evaluation.utils import run_test

        success, output = run_test(
            file_path=file_path,
            test_script=test_script,
            kernel_config=config,
            target=target,
        )
        if success:
            return True, "OK"
        else:
            return False, f"[FAIL] {file_name}\n{output}"
    except Exception as e:
        return False, f"[EXCEPTION] {file_name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Run correctness tests on compiled kernels using kernels.json"
    )
    parser.add_argument("json_path", help="Path to kernels.json config file")
    parser.add_argument(
        "source_dir", help="Directory containing compiled .cu/.cpp files"
    )
    parser.add_argument(
        "test_dir",
        help="Directory containing test scripts (e.g., test_add.py)",
    )
    parser.add_argument(
        "--target",
        choices=["cuda", "hip", "bang", "cpu"],
        required=True,
        help="Target platform",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--debug", "-d", type=str, default="", help="debug op name"
    )

    args = parser.parse_args()

    # Load kernels.json
    if not os.path.exists(args.json_path):
        print(
            f"[ERROR] kernels.json not found: {args.json_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(args.json_path, "r") as f:
            kernel_configs = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not kernel_configs:
        print("[INFO] No kernels to test.")
        sys.exit(0)

    total = len(kernel_configs)
    success_count = 0
    if args.debug:
        kernel_configs = [
            config
            for config in kernel_configs
            if config.get("op_name") == args.debug
        ]
    # Test in parallel
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                run_test_for_config,
                config=cfg,
                source_dir=args.source_dir,
                test_dir=args.test_dir,
                target=args.target,
            )
            for cfg in kernel_configs
        ]

        # Get the result
        with tqdm(
            total=total, desc=f"[{args.target.upper()}] Testing"
        ) as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                success, msg = future.result()
                if not success:
                    print(msg, file=sys.stderr)
                else:
                    success_count += 1

    # Summary of Results
    rate = success_count / total
    print(
        f"\n✅ {args.target.upper()} Test Result: {success_count}/{total} = {rate:.2%}"
    )
    sys.exit(0 if rate == 1.0 else 1)


if __name__ == "__main__":
    main()
