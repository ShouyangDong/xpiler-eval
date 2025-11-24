"""
Unified kernel compiler for multiple targets:
  - cuda   : .cu  → NVCC
  - hip    : .hip → HIPCC / clang++
  - cpp: .cpp → g++ / clang++
  - mlu    : .mlu → cncc (Cambricon)

Each target uses its own macros and compiler function.

Usage:
  python compile_kernels.py --target cuda ./kernels/cuda/
  python compile_kernels.py --target hip ./kernels/hip/
  python compile_kernels.py --target cpp ./kernels/cpp/
  python compile_kernels.py --target mlu ./kernels/mlu/
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# Import target-specific macros and compilation functions
# Ensure these are defined in the respective modules.
from evaluation.macros import CPP_MACROS, CUDA_MACROS, HIP_MACROS, MLU_MACROS
from evaluation.utils import (
    run_cpp_compilation,
    run_cuda_compilation,
    run_hip_compilation,
    run_mlu_compilation,
)

# Supported file extensions for each target.
# Each entry defines the primary extension and allowed variants.
EXTENSION_MAPPING = {
    ".cpp": [".cpp"],
    ".mlu": [".mlu"],
    ".cu": [".cu"],
    ".hip": [".hip"],
}


def get_extensions(primary_ext):
    """Get all allowed extensions that map to the same target.

    Args:
        primary_ext (str): The primary extension (e.g., '.cu', '.mlu').

    Returns:
        list: List of valid extensions for this target.
    """
    return EXTENSION_MAPPING.get(primary_ext, [primary_ext])


def compile_file(file_path, target):
    """Compile a single kernel source file with macros injected.

    Steps:
      1. Read original source
      2. Prepend target-specific macros
      3. Write to _bak.<ext> file
      4. Compile to .so using target compiler
      5. Clean up _bak file and .so (if compilation succeeded)
      6. Return success status

    Args:
        file_path (str): Path to the source file.
        target (str): Backend name ('cuda', 'hip', 'cpp', 'mlu').

    Returns:
        bool: True if compilation succeeded, False otherwise.
    """
    config = BACKEND_CONFIG[target]
    dir_name, base_name = os.path.split(file_path)
    name_no_ext, _ = os.path.splitext(base_name)

    # Temporary and output file paths
    bak_path = os.path.join(dir_name, f"{name_no_ext}_bak{config['ext']}")
    so_path = os.path.join(dir_name, f"{name_no_ext}.so")

    # Read source code
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}", file=sys.stderr)
        return False

    # Write backup file with macros prepended
    try:
        with open(bak_path, "w", encoding="utf-8") as f:
            f.write(config["macros"] + src)
    except Exception as e:
        print(f"[ERROR] Failed to write {bak_path}: {e}", file=sys.stderr)
        return False

    # Run compilation
    success, output = config["compiler"](so_path, bak_path)

    # Clean up temporary files
    try:
        os.remove(bak_path)
        if success and os.path.exists(so_path):
            os.remove(so_path)
    except Exception as e:
        print(f"[WARN] Failed to clean up files: {e}", file=sys.stderr)

    # Report compilation failure
    if not success:
        print(f"[ERROR] Compilation failed for {file_path}:", file=sys.stderr)
        print(output, file=sys.stderr)

    return success


# Backend configuration: maps target name to its settings
BACKEND_CONFIG = {
    "cuda": {
        "ext": ".cu",
        "macros": CUDA_MACROS,
        "compiler": run_cuda_compilation,
        "desc": "NVIDIA CUDA (NVCC)",
        "executor": ThreadPoolExecutor,  # I/O-bound (NVCC calls)
    },
    "hip": {
        "ext": ".hip",
        "macros": HIP_MACROS,
        "compiler": run_hip_compilation,
        "desc": "AMD HIP",
        "executor": ThreadPoolExecutor,  # CPU-bound
    },
    "cpp": {
        "ext": ".cpp",
        "macros": CPP_MACROS,
        "compiler": run_cpp_compilation,
        "desc": "DLBoost (PyTorch C++)",
        "executor": ThreadPoolExecutor,
    },
    "mlu": {
        "ext": ".mlu",
        "macros": MLU_MACROS,
        "compiler": run_mlu_compilation,
        "desc": "Cambricon MLU",
        "executor": ThreadPoolExecutor,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Batch compile kernel files for various targets."
    )
    parser.add_argument(
        "--src_dir", help="Directory containing source files to compile"
    )
    parser.add_argument(
        "--target",
        choices=BACKEND_CONFIG.keys(),
        required=True,
        help="Target target: cuda | hip | cpp | mlu",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: CPU count)",
    )

    args = parser.parse_args()

    # Validate target
    if args.target not in BACKEND_CONFIG:
        print(f"[ERROR] Unknown target: {args.target}", file=sys.stderr)
        sys.exit(1)

    config = BACKEND_CONFIG[args.target]
    exts = get_extensions(config["ext"])

    # Find all source files with allowed extensions
    files = []
    for ext in exts:
        pattern = os.path.join(args.src_dir, f"*{ext}")
        files.extend(glob.glob(pattern))

    if not files:
        print(
            f"[WARN] No files found in {args.src_dir} with extensions {exts}",
            file=sys.stderr,
        )
        sys.exit(0)

    print(
        f"🚀 Compiling {len(files)} file(s) for {config['desc']} ({args.target.upper()})..."
    )

    # Use appropriate executor (thread or process)
    ExecutorClass = config["executor"]
    with ExecutorClass(max_workers=args.jobs) as executor:
        results = list(
            tqdm(
                executor.map(lambda f: compile_file(f, args.target), files),
                total=len(files),
                desc=f"[{args.target.upper()}]",
                unit="file",
            )
        )

    # Print compilation statistics
    total = len(files)
    success_count = sum(results)
    success_rate = success_count / total
    print(
        f"[INFO] {args.target.upper()} success rate: {success_count}/{total} = {success_rate:.2%}"
    )

    # Exit with error code if any compilation failed
    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
