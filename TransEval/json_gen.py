import os
import json
import re
from pathlib import Path

def generate_kernel_json(kernel_dir, output_file="kernels.json", dtype="float32"):
    """
    从 kernel 文件夹生成 JSON 配置文件。

    支持带有 _dim0/_dim1 后缀的 reduce kernels（如 min, max, sum, mean）

    Args:
        kernel_dir: 包含 .cpp kernel 文件的目录
        output_file: 输出的 JSON 文件路径
        dtype: 数据类型，默认 float32
    """
    kernel_dir = Path(kernel_dir)
    result = []

    # 支持的 reduce ops
    reduce_ops = {"min", "max", "sum", "mean"}

    for cpp_file in kernel_dir.glob("*.cu"):
        stem = cpp_file.stem  # 如 add_1_15_64 或 min_2_4_5_6_dim1

        # 检查是否以 _dim0 / _dim1 结尾
        dim_match = re.search(r"_dim([01])$", stem)
        axis = None
        clean_stem = stem
        if dim_match:
            axis = int(dim_match.group(1))
            # 去掉 _dim0/_dim1
            clean_stem = stem[:dim_match.start()]

        parts = clean_stem.split("_")
        if len(parts) < 2:
            print(f"⚠️ 跳过不合规文件名: {cpp_file.name}")
            continue

        op_name = parts[0]
        try:
            # 其余部分应为 shape 尺寸
            args = [int(x) for x in parts[1:]]
        except ValueError:
            print(f"⚠️ 无法解析 shape 参数: {cpp_file.name}")
            continue

        entry = {
            "op_name": op_name,
            "dtype": dtype,
            "args": args
        }

        # 若是 reduce op 且存在 axis，则添加 axes 字段
        if op_name in reduce_ops and axis is not None:
            entry["axes"] = axis  # 可以扩展为列表支持多轴，但目前单轴

        result.append(entry)

    # 写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"✅ 成功生成 {output_file}，共 {len(result)} 个 kernel")

# ================ 使用示例 ===================
if __name__ == "__main__":
    # 修改为你存放 kernel 的实际路径
    kernel_folder = "KernelBench/CUDA"

    generate_kernel_json(
        kernel_dir=kernel_folder,
        output_file="TransEval/cuda.json",
        dtype="float32"
    )
