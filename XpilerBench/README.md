# Cross-Platform Kernel Sources for Deep Learning Operators
This directory contains source code implementations of deep learning operator kernels for multiple hardware platforms.

Each subdirectory corresponds to a specific backend and includes platform-specific code used in conjunction with the JSON configuration files in the `TransEval/` directory.

The JSON files (e.g., cpp.json, cuda.json, mlu.json, hip.json) define the list of operators to be tested and their input configurations.

The source code here provides the ground-truth implementations for each platform, which are used as references during cross-platform translation and correctness verification.

## Directory Structure

- MLU/ – Source code for Cambricon MLU platform (written in BANG-C)

    Corresponds to kernels defined in `TransEval/mlu.json`
- CPP/ – Source code for CPU platform (written in C with VNNI)

    Corresponds to kernels defined in `TransEval/c.json`
- CUDA/ – Source code for NVIDIA GPUs (written in CUDA C with Tensor Core)

    Includes both standard CUDA kernels and Tensor Core-optimized variants

    Corresponds to kernels defined in `TransEval/cuda.json`
- HIP/ – Source code for AMD GPUs (written in HIP with Matrix Core)

    Corresponds to kernels defined in `TransEval/hip.json`


## Description
Each platform directory contains hand-optimized or auto-generated kernels for a variety of tensor operators, such as conv2d, matmul, add, relu, and more. The set of operators implemented in each directory exactly matches the operator list specified in its corresponding JSON configuration file in `TransEval/`.

These implementations are designed to:

- Leverage each platform’s compute model, memory hierarchy, and instruction set 
- Serve as reference implementations for correctness validation
- Support end-to-end testing of kernel translation frameworks

## Usage
This repository structure is designed for:

- Cross-platform kernel translation testing
- Correctness verification via comparison against golden outputs defined in JSON configs
- Performance benchmarking across heterogeneous hardware
- Compiler and backend development for deep learning frameworks

## Notes
Kernels are organized by operator name (e.g., matmul_*, conv2d_*) within each platform directory.
Code style, dependencies, and build requirements may vary across platforms.
All implementations prioritize correctness and performance in deep learning workloads.
The `TransEval/*.json` files are the authoritative source for which operators and shapes are tested.
