"""Multi-platform correctness tester for compiled kernels (.so).

Modified to reduce import overhead by grouping tests per operator. Each test
script is loaded once and runs all its kernels in batch. Runs one operator at a
time (serial) to avoid resource conflicts.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

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
    "concat": "test_concat.py",
    "dense": "test_dense.py",
    "gatemlp": "test_gatemlp.py",
    "gqa": "test_gqa.py",
}


def run_test_for_op(
    op_name, configs, source_dir, test_script, target, job_workers
):
    """Run all configs of one op in a single process (shared import)."""
    try:
        # Dynamically import the test script
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"test_{op_name}", test_script
        )
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)

        # Expect: test_module.run_tests(configs, source_dir, target, num_workers=...)
        if hasattr(test_module, "run_tests"):
            result = test_module.run_tests(
                configs=configs,
                source_dir=source_dir,
                target=target,
                num_workers=job_workers,
            )

            # result: list of (success: bool, message: str)
            passed = sum(1 for r in result if r[0])
            return True, passed
        else:
            return False, f"No run_tests in {test_script}"

    except Exception as e:
        return False, f"Exception in {op_name}: {str(e)}"


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
        choices=["cuda", "hip", "mlu", "cpu"],
        required=True,
        help="Target platform",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel jobs (used inside test scripts)",
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

    # Filter by debug op
    if args.debug:
        kernel_configs = [
            cfg for cfg in kernel_configs if cfg.get("op_name") == args.debug
        ]

    # Group configs by op_name
    op_to_configs = defaultdict(list)
    for cfg in kernel_configs:
        op_name = cfg["op_name"]
        shapes = cfg["args"]
        file_name = f"{op_name}_{'_'.join(map(str, shapes))}.cpp"
        cfg["file"] = file_name
        op_to_configs[op_name].append(cfg)

    total = len(kernel_configs)
    success_count = 0

    # Run each op's tests sequentially to avoid resource conflicts
    pbar = tqdm(
        total=len(op_to_configs), desc=f"[{args.target.upper()}] Testing"
    )

    for op_name, configs in op_to_configs.items():
        script_name = TEST_SCRIPT_MAP.get(op_name)
        if not script_name:
            print(f"[WARN] No test script for op='{op_name}'", file=sys.stderr)
            pbar.update(1)
            continue

        test_script = os.path.join(args.test_dir, script_name)
        if not os.path.isfile(test_script):
            print(
                f"[ERROR] Test script not found: {test_script}",
                file=sys.stderr,
            )
            pbar.update(1)
            continue

        # Run all configs for this op in one process (shared torch import)
        success, count = run_test_for_op(
            op_name=op_name,
            configs=configs,
            source_dir=args.source_dir,
            test_script=test_script,
            target=args.target,
            job_workers=args.jobs,
        )

        pbar.update(1)
        if not success:
            print(f"[FAIL] {count}", file=sys.stderr)
        else:
            success_count += count

    pbar.close()

    # Final summary
    rate = success_count / total
    print(
        f"\n✅ {args.target.upper()} Test Result: {success_count}/{total} = {rate:.2%}"
    )
    sys.exit(0 if rate == 1.0 else 1)


if __name__ == "__main__":
    main()
