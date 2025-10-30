import re
import sys
from typing import List, Dict, Any


def generate_cuda_host(cu_file: str, block_size: int = 1024) -> str:
    """
    Generates a host-side wrapper function for a CUDA kernel.
    Reads a .cu file, finds the __global__ kernel, and generates a *_kernel.cu file.
    """
    # Read the device kernel file
    try:
        with open(cu_file, 'r') as f:
            code = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {cu_file}")

    # Extract kernel signature: __global__ void kernel_name(...)
    match = re.search(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)', code, re.DOTALL)
    if not match:
        raise ValueError("No '__global__ void' kernel function found in the file.")

    kernel_name = match.group(1)
    param_line = match.group(2)

    # Parse parameters: type and name
    params = []
    for p in re.split(r',\s*', param_line):
        if not p.strip():
            continue
        # Match pointer: float *A, const int *idx
        ptr_match = re.match(r'([\w\s*]+)\*\s*(\w+)', p)
        # Match value: int N, size_t size
        val_match = re.match(r'([\w\s]+)\s+(\w+)', p)
        if ptr_match:
            params.append({
                'type': ptr_match.group(1).strip().replace('const', '').strip(),
                'name': ptr_match.group(2),
                'is_ptr': True
            })
        elif val_match:
            params.append({
                'type': val_match.group(1).strip(),
                'name': val_match.group(2),
                'is_ptr': False
            })
        else:
            raise ValueError(f"Cannot parse parameter: {p}")

    # Separate pointer and value parameters
    ptr_params = [p for p in params if p['is_ptr']]
    val_params = [p for p in params if not p['is_ptr']]

    # Heuristic: last pointer is output, others are inputs
    inputs = ptr_params[:-1] if len(ptr_params) > 1 else []
    output = ptr_params[-1] if ptr_params else None

    # Find size parameter (used for grid calculation)
    size_param = 'size'  # default
    for p in val_params:
        if p['name'].lower() in ['n', 'size', 'len', 'count']:
            size_param = p['name']
            break

    # Device pointer names: A -> d_A
    dev_names = {p['name']: f"d_{p['name']}" for p in ptr_params}

    # Host function name
    host_func_name = f"{kernel_name}_kernel"

    # Full host parameter list
    host_params = ", ".join(
        [f"{p['type']} *{p['name']}" for p in ptr_params] +
        [f"{p['type']} {p['name']}" for p in val_params]
    )

    # Device pointer declarations
    declarations = "\n  ".join(
        f"{p['type']} *{dev_names[p['name']]};"
        for p in ptr_params
    )

    # cudaMalloc calls
    mallocs = "\n  ".join(
        f"cudaMalloc(&{dev_names[p['name']]}, {size_param} * sizeof({p['type']}));"
        for p in ptr_params
    )

    # cudaMemcpy Host to Device (inputs only)
    h2d_transfers = "\n  ".join(
        f"cudaMemcpy({dev_names[p['name']]}, {p['name']}, "
        f"{size_param} * sizeof({p['type']}), cudaMemcpyHostToDevice);"
        for p in inputs
    )

    # cudaMemcpy Device to Host (output only)
    d2h_transfer = (
        f"cudaMemcpy({output['name']}, {dev_names[output['name']]}, "
        f"{size_param} * sizeof({output['type']}), cudaMemcpyDeviceToHost);\n"
        if output else ""
    )

    # Kernel launch config
    grid_setup = f"dim3 blockSize({block_size});"
    grid_calc = f"dim3 numBlocks(({size_param} + {block_size} - 1) / {block_size});"
    kernel_launch = f"{kernel_name}<<<numBlocks, blockSize>>>({', '.join(dev_names[p['name']] for p in params)});"

    # Free device memory
    frees = "\n  ".join(
        f"cudaFree({dev_names[p['name']]});"
        for p in ptr_params
    )

    # Final generated code
    generated_code = f'''// Auto-generated host wrapper for {kernel_name}
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

  // Copy result back to host
  {d2h_transfer}
  // Release GPU memory
  {frees}
}}
'''

    # Output file
    output_file = cu_file.replace(".cu", "_kernel.cu")
    with open(output_file, 'w') as f:
        f.write(generated_code)

    print(f"[✓] Host wrapper generated: {output_file}")
    return output_file


# CLI usage: python cuda_host_generator.py kernel.cu
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cuda_host_generator.py <kernel.cu>")
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        generate_cuda_host(input_file)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)