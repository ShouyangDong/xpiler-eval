#!/usr/bin/env bash
# run_benchmark.sh - Cross-platform benchmarking script for code translation testing

set -euo pipefail

# === Default values (can be overridden via command line) ===
BENCH_DIR=""
OUT_ROOT_DIR="."

# === Help function ===
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

This script compiles and tests translated code across different hardware platforms (CPU, MLU, CUDA, HIP).
It runs compilation and correctness tests for each source -> target platform pair.

OPTIONS:
    -b, --bench-dir DIR       Root directory containing source code tests (required)
    -o, --out-root DIR        Output root directory for results (default: .)
    -h, --help                Show this help message and exit

EXAMPLES:
    $0 -b /path/to/benchmarks
    $0 -b ./benchmarks -o ./output_results

ENVIRONMENT VARIABLES:
    You can also set:
      BENCH_DIR     - Path to benchmark directory (required)
      OUT_ROOT_DIR  - Output directory (optional, default: current dir)

NOTE: Command-line arguments take precedence over environment variables.
EOF
    exit 1
}

# === Parse command-line arguments ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--bench-dir)
            if [[ -n "${2:-}" && "${2}" != -* ]]; then
                BENCH_DIR="$2"
                shift 2
            else
                echo "Error: Argument for --bench-dir is missing" >&2
                usage
            fi
            ;;
        -o|--out-root)
            if [[ -n "${2:-}" && "${2}" != -* ]]; then
                OUT_ROOT_DIR="$2"
                shift 2
            else
                echo "Error: Argument for --out-root is missing" >&2
                usage
            fi
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# === Validate required inputs ===
if [[ -z "$BENCH_DIR" ]]; then
    echo "Error: BENCH_DIR must be provided via -b or environment variable." >&2
    usage
fi

if [[ ! -d "$BENCH_DIR" ]]; then
    echo "Error: Benchmark directory does not exist: $BENCH_DIR" >&2
    exit 1
fi

# === Platform configuration ===
declare -A COMPILE_SCRIPTS=(
    ["cpu"]="benchmark/evaluation/dlboost_test/compilation.py"
    ["mlu"]="benchmark/evaluation/mlu_test/compilation.py"
    ["cuda"]="benchmark/evaluation/cuda_test/compilation.py"
    ["hip"]="benchmark/evaluation/hip_test/compilation.py"
)

declare -A TEST_SCRIPTS=(
    ["cpu"]="benchmark/evaluation/dlboost_test/result_test.py"
    ["mlu"]="benchmark/evaluation/mlu_test/result_test.py"
    ["cuda"]="benchmark/evaluation/cuda_test/result_test.py"
    ["hip"]="benchmark/evaluation/hip_test/result_test.py"
)

# === Direction pairs: source:target ===
DIRECTIONS=(
    "mlu:cpu"
    "cpu:mlu"
    "mlu:hip"
    "mlu:cuda"
    "cpu:hip"
    "cpu:cuda"
    "cuda:mlu"
    "cuda:hip"
    "cuda:cpu"
    "hip:mlu"
    "hip:cuda"
    "hip:cpu"
)

# === Main execution loop ===
for dir_pair in "${DIRECTIONS[@]}"; do
    src_plat="${dir_pair%%:*}"
    dst_plat="${dir_pair##*:}"

    src_dir="$BENCH_DIR/${src_plat}_code_test"
    out_dir="$OUT_ROOT_DIR/${src_plat}_${dst_plat}"

    # Ensure output directory exists
    mkdir -p "$out_dir"

    # 1) Compilation Test
    compile_py="${COMPILE_SCRIPTS[$dst_plat]}"
    if [[ ! -f "$compile_py" ]]; then
        echo "Error: Compilation script not found: $compile_py" >&2
        exit 1
    fi

    echo "-> Compiling translated code for $dst_plat using $compile_py"
    python3 "$compile_py" "$out_dir"

    # 2) Correctness Test
    test_py="${TEST_SCRIPTS[$dst_plat]}"
    if [[ ! -f "$test_py" ]]; then
        echo "Error: Test script not found: $test_py" >&2
        exit 1
    fi

    ref_data_dir="benchmark/evaluation/${dst_plat}_test"
    if [[ ! -d "$ref_data_dir" ]]; then
        echo "Warning: Reference data directory not found: $ref_data_dir" >&2
    fi

    echo "-> Running correctness tests on $dst_plat with $test_py"
    python3 "$test_py" "$out_dir" "$ref_data_dir"

	# TODO: add performance test
done

echo "All benchmark directions completed."
