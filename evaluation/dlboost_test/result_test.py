#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from benchmark.utils import run_test

# operator → test script filename
TEST_FILE_MAP = {
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
}


def process_file(file_path, test_dir):
    """
    Run the corresponding DLBoost test for a single .cpp file.
    Returns (base_name, success, output).
    """
    base_name = os.path.basename(file_path)
    op_name = base_name.split("_")[0]
    script_name = TEST_FILE_MAP.get(op_name)
    if not script_name:
        return base_name, False, f"[WARN] no test mapping for '{op_name}'"

    test_script = os.path.join(test_dir, script_name)
    if not os.path.isfile(test_script):
        return (
            base_name,
            False,
            f"[ERROR] test script not found: {test_script}",
        )

    success, output = run_test(file_path, test_script)
    return base_name, success, output


def main():
    parser = argparse.ArgumentParser(
        description="Run DLBoost correctness tests on translated .cpp files"
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing translated .cpp files (e.g. translated/cpu_to_dlboost/)",
    )
    parser.add_argument(
        "test_dir",
        help="Directory containing DLBoost test scripts (e.g. benchmark/evaluation/dlboost_test/)",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.source_dir, "*.cpp")
    files = glob.glob(pattern)
    if not files:
        print(
            f"[WARN] no .cpp files found in {args.source_dir}", file=sys.stderr
        )
        sys.exit(0)

    total = len(files)
    success_count = 0

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, fp, args.test_dir): fp
            for fp in files
        }
        for future in tqdm(as_completed(futures), total=total):
            base_name, success, output = future.result()

            if (
                success
                and hasattr(output, "stdout")
                and "Verification successful!" in output.stdout
            ):
                success_count += 1
            else:
                # print failures or mapping warnings
                print(f"--- {base_name} ---")
                print(output)

    rate = success_count / total if total else 0.0
    print(f"Successful tests: {success_count}/{total} ({rate:.2%})")


if __name__ == "__main__":
    main()
