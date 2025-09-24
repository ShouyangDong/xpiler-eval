#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 导入你已经写好的四个平台的宏和compile函数
from evaluation.macros import (
    CUDA_MACROS,
    HIP_MACROS,
    BANG_MACROS,
    CPU_MACROS,
)

from evaluation.utils import (
    run_cuda_compilation,
    run_hip_compilation,
    run_bang_compilation,
    run_cpp_compilation,
)


# 映射 target → (file_glob, macro, compiler_func)
TARGET_CONFIG = {
    "cuda": {
        "glob": "*.cu",
        "macros": CUDA_MACROS,
        "compiler": run_cuda_compilation,
        "ext": "cu"
    },
    "hip": {
        "glob": "*.hip",
        "macros": HIP_MACROS,
        "compiler": run_hip_compilation,
        "ext": "hip"
    },
    "bang": {
        "glob": "*.mlu",
        "macros": BANG_MACROS,
        "compiler": run_bang_compilation,
        "ext": "mlu"
    },
    "cpu": {
        "glob": "*.cpp",
        "macros": CPU_MACROS,
        "compiler": run_dlboost_compilation,
        "ext": "cpp"
    }
}


def compile_file(file_path, target):
    config = TARGET_CONFIG[target]
    name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)

    # 1️⃣ Load源码 + 插入平台宏
    with open(file_path, "r") as f:
        code = f.read()
    full_code = config["macros"] + code

    # 2️⃣ 生成中间文件 _bak.<ext>
    bak_file = os.path.join(dir_name, f"{name_no_ext}_bak.{config['ext']}")
    with open(bak_file, "w") as f:
        f.write(full_code)

    # 3️⃣ .so output路径
    so_path = os.path.join(dir_name, f"{name_no_ext}.so")

    # 4️⃣ invoke你已实现的compile函数
    success, output = config["compiler"](so_path, bak_file)

    # 5️⃣ 清理中间文件（可选保留 .so）
    try:
        os.remove(bak_file)
        # 可选：失败时保留 .so 用于调试，或加 --keep-so parameters
        if success and os.path.exists(so_path):
            os.remove(so_path)
    except Exception as e:
        print(f"[WARN] cleanup failed: {e}", flush=True)

    if not success:
        print(f"[ERROR] Failed to compile {file_path} (target={target}):\n{output}", flush=True)

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Compile kernels for multiple targets: cuda, hip, bang, cpu"
    )
    parser.add_argument(
        "src_dir",
        help="Directory containing source files to compile"
    )
    parser.add_argument(
        "--target",
        choices=["cuda", "hip", "bang", "cpu"],
        required=True,
        help="Target platform"
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel jobs"
    )

    args = parser.parse_args()

    # 获取目标平台配置
    target_cfg = TARGET_CONFIG[args.target]
    pattern = os.path.join(args.src_dir, target_cfg["glob"])
    files = glob.glob(pattern)

    if not files:
        print(f"[WARN] No {target_cfg['ext']} files found in {args.src_dir}", file=sys.stderr)
        return

    print(f"[INFO] Compiling {len(files)} files for '{args.target}'...")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(
            tqdm(
                executor.map(lambda f: compile_file(f, args.target), files),
                total=len(files),
                desc=f"[{args.target.upper()}]"
            )
        )

    total = len(files)
    succ = sum(results)
    rate = succ / total
    print(f"[INFO] {args.target.upper()} compilation success rate: {succ}/{total} = {rate:.2%}")

    # 可选：非零退出码表示有失败
    if rate < 1.0:
        exit(1)


if __name__ == "__main__":
    main()
