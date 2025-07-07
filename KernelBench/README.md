# Cross-Platform Kernel Sources for Deep Learning Operators

This directory contains source code implementations of deep learning operator kernels for multiple hardware platforms.  
Each subdirectory corresponds to a specific backend and includes platform-optimized operator code.

## Directory Structure

- `BANG/` – Source code for **Cambricon MLU** platform (written in BANG-C)
- `C/` – Source code for **CPU** platform (written in standard C)
- `CUDA/` – Source code for **NVIDIA GPUs** (written in CUDA C)
- `HIP/` – Source code for **AMD GPUs** (written in HIP)

## Description

Each platform directory contains hand-optimized or auto-generated kernels for a variety of tensor operators,  
such as `conv2d`, `matmul`, `add`, and more. These implementations are designed to take advantage of  
each platform's specific compute model, memory hierarchy, and instruction set.

This source code can be used for:
- Performance benchmarking across heterogeneous platforms
- Compiler backend development and validation
- Cross-platform operator translation
- Reference implementations for operator-level optimizations

## Notes

- Kernels are organized by operator name or function.
- Code style and dependencies may vary slightly across platforms.
- All implementations are focused on performance and correctness in deep learning workloads.

