# mlu_host_generator.py

import re
import sys


def generate_mlu_host(mlu_file: str, cluster_mode: bool = False) -> str:
    """Generates a host-side wrapper for a Cambricon MLU kernel.

    Replaces CUDA-style calls with CNRT (CN Runtime) equivalents.
    """
    try:
        with open(mlu_file, "r") as f:
            code = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {mlu_file}")

    # Extract function (no __global__ in MLU BANG C)
    match = re.search(r"void\s+(\w+)\s*\(([^)]*)\)", code)
    if not match:
        raise ValueError("No 'void func(...)' kernel found.")

    kernel_name = match.group(1)
    param_line = match.group(2)

    # Parse parameters
    params = []
    for p in re.split(r",\s*", param_line):
        if not p.strip():
            continue
        ptr_match = re.match(r"([\w\s*]+)\*\s*(\w+)", p)  # pointer
        val_match = re.match(r"([\w\s]+)\s+(\w+)", p)  # value
        if ptr_match:
            params.append(
                {
                    "type": ptr_match.group(1)
                    .strip()
                    .replace("const", "")
                    .strip(),
                    "name": ptr_match.group(2),
                    "is_ptr": True,
                }
            )
        elif val_match:
            params.append(
                {
                    "type": val_match.group(1).strip(),
                    "name": val_match.group(2),
                    "is_ptr": False,
                }
            )
        else:
            raise ValueError(f"Cannot parse parameter: {p}")

    # Classify
    ptr_params = [p for p in params if p["is_ptr"]]
    val_params = [p for p in params if not p["is_ptr"]]
    inputs = ptr_params[:-1] if len(ptr_params) > 1 else []
    output = ptr_params[-1] if ptr_params else None

    # Find size parameter
    size_param = next(
        (
            p["name"]
            for p in val_params
            if p["name"].lower() in ["size", "n", "len"]
        ),
        "size",
    )

    # Device names
    dev_names = {p["name"]: f"d_{p['name']}" for p in ptr_params}

    # Function name
    host_func_name = f"{kernel_name}_kernel"

    # Host signature
    host_params = ", ".join(
        [f"{p['type']} *{p['name']}" for p in ptr_params]
        + [f"{p['type']} {p['name']}" for p in val_params]
    )

    # Code blocks
    declarations = "\n  ".join(
        f"{p['type']} *{dev_names[p['name']]};" for p in ptr_params
    )

    mallocs = "\n  ".join(
        f"CNRT_CHECK(cnrtMalloc((void**)&{dev_names[p['name']]}, {size_param} * sizeof({p['type']})));"
        for p in ptr_params
    )

    h2d_transfers = "\n  ".join(
        f"CNRT_CHECK(cnrtMemcpy({dev_names[p['name']]}, {p['name']}, "
        f"{size_param} * sizeof({p['type']}), cnrtMemcpyHostToDev));"
        for p in inputs
    )

    d2h_transfer = (
        f"CNRT_CHECK(cnrtMemcpy({output['name']}, {dev_names[output['name']]}, "
        f"{size_param} * sizeof({output['type']}), cnrtMemcpyDevToHost));\n"
        if output
        else ""
    )

    # Launch config
    dim_str = "32, 1, 1" if cluster_mode else "4, 1, 1"  # cluster vs core mode
    func_type = (
        "cnrtFuncTypeBlock"
        if not cluster_mode
        else (
            "cnrtFuncTypeUnion1"
            if not "clusterId" in code
            else "cnrtFuncTypeUnion8"
        )
    )

    launch_config = f"""cnrtQueue_t queue;
    CNRT_CHECK(cnrtCreateQueue(&queue));
    cnrtDim3_t kDim = {{{dim_str}}};
    cnrtFunctionType_t kType = {func_type};"""

    # Kernel launch
    kernel_launch = f"{kernel_name}<<<numBlocks, blockSize>>>({', '.join(dev_names[p['name']] for p in params)});"

    frees = "\n  ".join(
        f"cnrtFree({dev_names[p['name']]});" for p in ptr_params
    )

    # Final MLU code
    generated_code = f"""// Auto-generated MLU host wrapper for {kernel_name}
extern "C" void {host_func_name}({host_params}) {{
  // Device pointers
  {declarations}

  // Allocate MLU memory
  {mallocs}

  // Copy input data to MLU
  {h2d_transfers}

  // Launch configuration
  {launch_config}

  // Launch kernel
  {kernel_launch}

  // Copy result back to host
  {d2h_transfer}
  // Release MLU memory
  {frees}
}}
"""

    # Output file
    output_file = mlu_file.replace(".mlu", "_kernel.cpp").replace(
        ".cu", "_kernel.cpp"
    )
    with open(output_file, "w") as f:
        f.write(generated_code)

    print(f"[✓] MLU host wrapper generated: {output_file}")
    return output_file


# CLI usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mlu_host_generator.py <kernel.mlu>")
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        generate_mlu_host(input_file)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
