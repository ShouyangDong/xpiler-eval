import re
import sys


def generate_hip_host(hip_file: str, block_size: int = 256) -> str:
    """Generates a HIP host-side wrapper from a __global__ kernel in a .hip/.cu
    file.

    Replaces CUDA calls with HIP equivalents.
    """
    try:
        with open(hip_file, "r") as f:
            code = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {hip_file}")

    # Extract kernel
    match = re.search(
        r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", code, re.DOTALL
    )
    if not match:
        raise ValueError("No '__global__ void' kernel found.")

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
        f"hipMalloc(&{dev_names[p['name']]}, {size_param} * sizeof({p['type']}));"
        for p in ptr_params
    )

    h2d_transfers = "\n  ".join(
        f"hipMemcpy({dev_names[p['name']]}, {p['name']}, "
        f"{size_param} * sizeof({p['type']}), hipMemcpyHostToDevice);"
        for p in inputs
    )

    d2h_transfer = (
        f"  hipMemcpy({output['name']}, {dev_names[output['name']]}, "
        f"{size_param} * sizeof({output['type']}), hipMemcpyDeviceToHost);\n"
        if output
        else ""
    )

    # Grid setup
    grid_setup = f"  dim3 blockSize({block_size});"
    grid_calc = (
        f"  dim3 numBlocks(({size_param} + {block_size} - 1) / {block_size});"
    )

    # Use hipLaunchKernelGGL (recommended in HIP)
    kernel_launch = f"  {kernel_name}<<<numBlocks, blockSize>>>({', '.join(dev_names[p['name']] for p in params)});"

    sync = "  hipDeviceSynchronize();"

    frees = "\n  ".join(
        f"hipFree({dev_names[p['name']]});" for p in ptr_params
    )

    # Final HIP code
    generated_code = f"""// Auto-generated HIP host wrapper for {kernel_name}
extern "C" void {host_func_name}({host_params}) {{
  // Device pointers
  {declarations}

  // Allocate GPU memory
  {mallocs}

  // Copy input data to device
  {h2d_transfers}

  // Grid configuration
  {grid_setup}
  {grid_calc}

  // Launch kernel
  {kernel_launch}
  {sync}

  // Copy result back to host
  {d2h_transfer}
  // Release GPU memory
  {frees}
}}
"""

    # Output file
    output_file = hip_file.replace(".cu", "_kernel.hip").replace(
        ".hip", "_kernel.hip"
    )
    with open(output_file, "w") as f:
        f.write(generated_code)

    print(f"[✓] HIP host wrapper generated: {output_file}")
    return output_file


# CLI usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hip_host_generator.py <kernel.hip>")
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        generate_hip_host(input_file)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
