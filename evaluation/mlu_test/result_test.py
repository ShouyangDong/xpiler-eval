#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from benchmark.utils import run_test

# mapping from operator name prefix to its test script
TEST_FILE_MAPPING = {
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
    "depthwiseconv": "test_depthwiseconv.py",
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
    Run the corresponding test for a single .mlu or .cpp file.
    Returns (file_path, output) where output is the subprocess result or error string.
    """
    base_name = os.path.basename(file_path)
    name = base_name.split("_")[0]
    script_name = TEST_FILE_MAPPING.get(name)
    if not script_name:
        return file_path, f"[WARN] No test mapping for prefix '{name}'"

    test_file = os.path.join(test_dir, script_name)
    if not os.path.isfile(test_file):
        return file_path, f"[ERROR] Test script not found: {test_file}"

    success, output = run_test(file_path, test_file)
    return file_path, output


def main():
    parser = argparse.ArgumentParser(
        description="Run functional tests on translated MLU programs"
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing translated .mlu files (e.g. translated/nvidia_to_mlu/)",
    )
    parser.add_argument(
        "test_dir",
        help="Directory containing test scripts (e.g. benchmark/evaluation/mlu_test/)",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.source_dir, "*.mlu")
    files = glob.glob(pattern)
    if not files:
        print(
            f"[WARN] no .mlu files found in {args.source_dir}", file=sys.stderr
        )
        sys.exit(0)

    total = len(files)
    success_count = 0

    with ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, fp, args.test_dir): fp
            for fp in files
        }

        for future in tqdm(as_completed(future_to_file), total=total):
            future_to_file[future]
            file_path, output = future.result()
            if hasattr(output, "stdout") and "Verification successful!" in output.stdout:
                success_count += 1
            else:
                # print failures or warnings
                print(f"--- {os.path.basename(file_path)} ---")
                print(output)

    print(f"Successful tests: {success_count}")
    print(f"Total files: {total}")
    rate = success_count / total if total else 0.0
    print(f"[INFO] MLU test success rate: {rate:.2%}")


if __name__ == "__main__":
    main()
