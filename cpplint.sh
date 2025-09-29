#!/bin/bash

echo "Formatting C, HIP, MLU and CUDA files ..."

find KernelBench/CPP -name "*.cpp" -exec clang-format -i {} \;

find KernelBench/CUDA -name "*.cu" -exec clang-format -i {} \;

find KernelBench/MLU -name "*.mlu" -exec clang-format -i {} \;

find KernelBench/HIP -name "*.hip" -exec clang-format -i {} \;

echo "✅ Done! All files formatted."
