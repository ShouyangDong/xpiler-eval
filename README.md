# Xpiler-eval: A Benchmark for Evaluating LLMs on Deep Learning Systems Programming

**Xpiler-eval** is a benchmark for testing large language models (LLMs) on **cross-system code translation** tasks in deep learning compilers.

It focuses on translating tensor operators between different hardware platforms, such as Nvidia GPU with Tensor Core, AMD GPU with Matrix Core, Intel CPU, and Cambricon MLU.

## Evaluation Metrics

- **Compilation Success Rate**  
  Whether the generated code compiles correctly.

- **Execution Success Rate**  
  Whether the compiled code runs without errors.

- **Performance Comparison**  
  Runtime performance compared to vendor-provided library implementations.

# How to Run
## Method 1: Using Command-Line Arguments (Recommended)
``./run_benchmark.sh -b /path/to/benchmarks [-o /output/path]``

Example:

``
./run_benchmark.sh -b ./benchmarks -o ./results
``

## Method 2: Using Environment Variables (Legacy)
``
export BENCH_DIR=/path/to/benchmarks
export OUT_ROOT_DIR=./results  # optional
./run_benchmark.sh
``
