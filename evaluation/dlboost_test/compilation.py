#!/usr/bin/env python3
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from benchmark.utils import run_dlboost_compilation as run_compilation


def compile_cpp_file(file_path):
    """
    Compile a single .cpp file (with DL Boost macros) into a .so and clean up.
    Returns True on success, False on failure.
    """
    dir_name, base_name = os.path.split(file_path)
    name_no_ext, _ = os.path.splitext(base_name)
    so_path = os.path.join(dir_name, f"{name_no_ext}.so")

    # Read source code
    with open(file_path, "r") as f:
        src = f.read()
    # Read macro
    macro_path = os.path.join("benchmark", "macro", "dlboost_macro.txt")
    with open(macro_path, "r") as f:
        macro = f.read()

    # Write temporary backup file
    bak_path = os.path.join(dir_name, f"{name_no_ext}_bak.cpp")
    with open(bak_path, "w") as f:
        f.write(macro + src)

    # Compile
    success, output = run_compilation(so_path, bak_path)

    # Clean up
    os.remove(bak_path)
    if success and os.path.exists(so_path):
        os.remove(so_path)
    elif not success:
        print(f"[ERROR] Compilation failed for {file_path}", file=sys.stderr)
        print(output, file=sys.stderr)

    return success


def main():
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <dlboost_source_directory>", file=sys.stderr
        )
        sys.exit(1)

    src_dir = sys.argv[1]
    pattern = os.path.join(src_dir, "*.cpp")
    files = glob.glob(pattern)
    if not files:
        print(f"[WARN] No .cpp files found in {src_dir}", file=sys.stderr)
        sys.exit(0)

    # Parallel compilation
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(compile_cpp_file, files), total=len(files))
        )

    total = len(files)
    succ = sum(results)
    print(
        f"[INFO] DLBoost compilation success rate: {succ}/{total} = {succ/total:.2%}"
    )


if __name__ == "__main__":
    main()
