#!/usr/bin/env bash
# run_benchmark.sh - Cross-platform kernel compilation and correctness testing with success rate summary

set -euo pipefail

# === Default values ===
BENCH_DIR=""
OUT_ROOT_DIR="."

# === Help message ===
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

This script compiles and tests translated kernels across multiple hardware backends.
It runs compilation and correctness verification for each source → target platform pair.

OPTIONS:
    -b, --bench-dir DIR       Root directory containing source test code (required)
    -o, --out-root DIR        Output root for results and logs (default: .)
    -h, --help                Show this help message and exit

EXAMPLES:
    $0 -b ./benchmarks
    $0 -b ./benchmarks -o ./results

ENVIRONMENT VARIABLES:
    BENCH_DIR      - Path to benchmarks (required)
    OUT_ROOT_DIR   - Output directory (optional)

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
                echo "Error: Missing value for --bench-dir" >&2
                usage
            fi
            ;;
        -o|--out-root)
            if [[ -n "${2:-}" && "${2}" != -* ]]; then
                OUT_ROOT_DIR="$2"
                shift 2
            else
                echo "Error: Missing value for --out-root" >&2
                usage
            fi
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# === Validate inputs ===
if [[ -z "$BENCH_DIR" ]]; then
    echo "Error: BENCH_DIR is required. Use -b or set environment variable." >&2
    usage
fi

if [[ ! -d "$BENCH_DIR" ]]; then
    echo "Error: Benchmark directory does not exist: $BENCH_DIR" >&2
    exit 1
fi

# === Platform mappings ===
declare -A TARGET_TO_BACKEND=(
    ["cpu"]="dlboost"
    ["mlu"]="mlu"
    ["cuda"]="cuda"
    ["hip"]="hip"
)

declare -A TARGET_TO_JSON=(
    ["cpu"]="c.json"
    ["mlu"]="mlu.json"
    ["cuda"]="cuda.json"
    ["hip"]="hip.json"
)

# === Test directions: source:target ===
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

# === Script and config paths ===
COMPILE_SCRIPT="EvalEngine/eval_compilation.py"
TEST_SCRIPT="EvalEngine/eval_computation.py"
JSON_DIR="TransEval"

# === Validate dependencies ===
if [[ ! -f "$COMPILE_SCRIPT" ]]; then
    echo "Error: Compiler script not found: $COMPILE_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "Error: Test script not found: $TEST_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$JSON_DIR" ]]; then
    echo "Error: JSON config directory not found: $JSON_DIR" >&2
    exit 1
fi

# === Initialize counters ===
TOTAL_DIRS=${#DIRECTIONS[@]}
COMPILE_SUCCESS=0
CORRECTNESS_SUCCESS=0

# === Main execution loop ===
for dir_pair in "${DIRECTIONS[@]}"; do
    src_plat="${dir_pair%%:*}"
    dst_plat="${dir_pair##*:}"

    backend="${TARGET_TO_BACKEND[$dst_plat]}"
    json_file="$JSON_DIR/${TARGET_TO_JSON[$dst_plat]}"
    src_dir="$BENCH_DIR/${src_plat}_code_test"
    out_dir="$OUT_ROOT_DIR/${src_plat}_${dst_plat}"

    # Validate platform support
    if [[ -z "${backend:-}" ]]; then
        echo "❌ Unsupported target platform: $dst_plat"
        continue
    fi

    if [[ ! -f "$json_file" ]]; then
        echo "❌ Config file not found: $json_file"
        continue
    fi

    mkdir -p "$out_dir"

    echo "=> $src_plat → $dst_plat [Backend: $backend]"

    # --- 1) Compilation ---
    echo "   🛠️  Compiling kernel..."
    if python3 "$COMPILE_SCRIPT" --backend "$backend" "$src_dir" --output-dir "$out_dir" >"$out_dir/compile.log" 2>&1; then
        echo "   ✅ Compilation succeeded"
        ((COMPILE_SUCCESS++))
    else
        echo "   ❌ Compilation failed (log: $out_dir/compile.log)"
    fi

    # --- 2) Correctness Test ---
    echo "   🧪 Running correctness test..."
    if python3 "$TEST_SCRIPT" "$json_file" "$out_dir" "$src_dir" --target "$dst_plat" --jobs 4 >"$out_dir/test.log" 2>&1; then
        echo "   ✅ Test passed"
        ((CORRECTNESS_SUCCESS++))
    else
        echo "   ❌ Test failed (log: $out_dir/test.log)"
    fi

    echo ""
done

# === 📊 Final Summary ===
echo "📊 Benchmark Summary"
echo "────────────────────────────────────"
printf "  Total directions:    %2d\n" "$TOTAL_DIRS"
printf "  Compilation:         %2d / %d" "$COMPILE_SUCCESS" "$TOTAL_DIRS"
compile_rate=$(echo "scale=1; $COMPILE_SUCCESS * 100 / $TOTAL_DIRS" | bc -l)
printf " (%.1f%%)\n" "$compile_rate"

printf "  Correctness:         %2d / %d" "$CORRECTNESS_SUCCESS" "$TOTAL_DIRS"
correctness_rate=$(echo "scale=1; $CORRECTNESS_SUCCESS * 100 / $TOTAL_DIRS" | bc -l)
printf " (%.1f%%)\n" "$correctness_rate"
echo "────────────────────────────────────"
echo "🎉 All benchmark directions completed."
