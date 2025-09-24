import os
import json
import re
from pathlib import Path


def generate_kernel_json(kernel_dir, output_file="kernels.json", dtype="float32"):
    """
    Generate a JSON configuration file from the kernel directory.

    Supports reduce kernels with _dim0/_dim1 suffixes (e.g., min, max, sum, mean).

    Args:
        kernel_dir: Directory containing .cu kernel files
        output_file: Output JSON file path
        dtype: Data type, default is float32
    """
    kernel_dir = Path(kernel_dir)
    result = []

    # Supported reduce operations
    reduce_ops = {"min", "max", "sum", "mean"}

    for cpp_file in kernel_dir.glob("*.cu"):
        stem = cpp_file.stem  # e.g., add_1_15_64 or min_2_4_5_6_dim1

        # Check if filename ends with _dim0 or _dim1
        dim_match = re.search(r"_dim([01])$", stem)
        axis = None
        clean_stem = stem
        if dim_match:
            axis = int(dim_match.group(1))
            # Remove _dim0/_dim1 suffix
            clean_stem = stem[:dim_match.start()]

        parts = clean_stem.split("_")
        if len(parts) < 2:
            print(f"⚠️ Skipping malformed filename: {cpp_file.name}")
            continue

        op_name = parts[0]
        try:
            # Remaining parts should be shape dimensions
            args = [int(x) for x in parts[1:]]
        except ValueError:
            print(f"⚠️ Failed to parse shape parameters: {cpp_file.name}")
            continue

        entry = {
            "op_name": op_name,
            "dtype": dtype,
            "args": args
        }

        # If it's a reduce op and axis is specified, add axes field
        if op_name in reduce_ops and axis is not None:
            entry["axes"] = axis  # Can be extended to list for multi-axis, currently single-axis

        result.append(entry)

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"✅ Successfully generated {output_file} with {len(result)} kernels")


# ================ Example usage ===================
if __name__ == "__main__":
    # Update this to your actual kernel directory path
    kernel_folder = "KernelBench/CUDA"

    generate_kernel_json(
        kernel_dir=kernel_folder,
        output_file="TransEval/cuda.json",
        dtype="float32"
    )
