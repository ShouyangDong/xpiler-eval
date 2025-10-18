import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from evaluation.macros import CUDA_MACROS as macro
from evaluation.utils import run_cuda_compilation as run_compilation


def compile_cuda_file(file_path):
    """Compile a single .cuda file into a .so and clean up.

    Returns True if compilation succeeded, False otherwise.
    """
    # Derive names and paths
    dir_name, base_name = os.path.split(file_path)
    name_no_ext, _ = os.path.splitext(base_name)
    so_path = os.path.join(dir_name, f"{name_no_ext}.so")

    # Read source and macro
    with open(file_path, "r") as f:
        src = f.read()

    # Write temporary backed-up file
    bak_path = os.path.join(dir_name, f"{name_no_ext}_bak.cu")
    with open(bak_path, "w") as f:
        f.write(macro + src)

    # Compile
    success, output = run_compilation(so_path, bak_path)
    # Clean up backup
    os.remove(bak_path)

    if success:
        # Remove the .so if it was produced
        if os.path.exists(so_path):
            os.remove(so_path)
        return True
    else:
        print(f"[ERROR] Failed to compile {file_path}", file=sys.stderr)
        print(output, file=sys.stderr)
        return False


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cuda_source_directory>", file=sys.stderr)
        sys.exit(1)

    src_dir = sys.argv[1]
    pattern = os.path.join(src_dir, "*.cu")
    files = glob.glob(pattern)
    if not files:
        print(f"[WARN] No .cu files found in {src_dir}", file=sys.stderr)
        sys.exit(0)

    # Parallel compile
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(compile_cuda_file, files), total=len(files))
        )

    total = len(files)
    succ = sum(results)
    print(
        f"[INFO] cuda compilation success rate: {succ}/{total} = {
            succ / total:.2%}"
    )


if __name__ == "__main__":
    main()
