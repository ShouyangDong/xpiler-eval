# Operator Metadata for Heterogeneous Deep Learning Platforms

This directory contains JSON metadata files that describe deep learning operators across multiple hardware platforms.  
Each JSON file corresponds to a specific backend and includes detailed information for all supported operators on that platform.

## Contents

- `bang.json` – Operators for **Cambricon MLU** platform  
- `c.json` – Operators for **CPU** backend  
- `cuda.json` – Operators for **NVIDIA GPU (CUDA)** backend  
- `hip.json` – Operators for **AMD GPU (HIP/ROCm)** backend  

## Operator Metadata

Each JSON file provides a list of operator entries. For each operator, the following attributes are included:

- **Operator name**  
- **Input and output shapes**  
- **Data types** (e.g., `float32`, `int8`)  
- **Input/output tensor count**  
- **Platform-specific features or constraints**

This metadata is useful for tasks such as:
- Cross-platform operator translation
- Compiler optimization
- Performance analysis
- Model conversion or verification

## Usage

These files can be loaded and parsed using any standard JSON library in Python, C++, etc. They are intended for use in deep learning compilers, transpilers, and evaluation frameworks.

## Example

```json
{
  "op_name": "conv2d",
  "inputs": [
    {"shape": [1, 3, 224, 224], "dtype": "float32"},
    {"shape": [64, 3, 7, 7], "dtype": "float32"}
  ],
  "outputs": [
    {"shape": [1, 64, 112, 112], "dtype": "float32"}
  ]
}

