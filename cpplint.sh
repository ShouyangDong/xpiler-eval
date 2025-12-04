#!/bin/bash

echo "Formatting C, HIP, MLU and CUDA files ..."

find XpilerBench/CPP -name "*.cpp" -exec clang-format -i {} \;

find XpilerBench/CUDA -name "*.cu" -exec clang-format -i {} \;

find XpilerBench/MLU -name "*.mlu" -exec clang-format -i {} \;

find XpilerBench/HIP -name "*.hip" -exec clang-format -i {} \;

echo "✅ Done! All files formatted."
