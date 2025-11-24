# Copilot instructions for xpiler-eval

This file orients AI coding agents to the xpiler-eval benchmark repository so they can be immediately productive. Keep guidance concrete and codebase-specific.

1. Big picture
   - Purpose: this repo benchmarks LLMs on cross-system tensor operator translation and evaluation.
   - Major components:
     - `CodeGen/` : code generation utilities for kernel source creation (Python scripts like `kernel_gen.py`).
     - `EvalEngine/` : orchestration scripts (`eval_compilation.py`, `eval_computation.py`, `eval_performance.py`) used by the top-level runner.
     - `evaluation/` : per-backend test harnesses and helpers. Key files: `evaluation/utils.py` (compile/test pipeline, macros), `evaluation/macros.py`.
     - `TransEval/` and JSON files (`cuda.json`, `mlu.json`, etc.): per-target operator configs used by the runner.
     - `KernelBench/` : benchmark documentation and sample kernels.

2. Primary developer workflows (explicit commands)
   - Setup: source `env.sh` to prepare PATH/vars used by native compilers.
     - Example: `source env.sh`
   - Run the full benchmark: `./evaluate.sh -b /path/to/benchmarks -o ./results` (this calls `EvalEngine/eval_compilation.py` and `EvalEngine/eval_computation.py`).
   - Quick local compile+test for one operator:
     - Use `python evaluation/utils.py` indirectly via the scripts in `evaluation/` — see `evaluate.sh` flow. For single-operator runs use the test scripts under `evaluation/<target>_test/` (e.g. `evaluation/mlu_test/test_mean.py`).

3. How tests and compilation work (concrete examples)
   - Configs: operator configs come from `TransEval/*.json`. `parse_op_json` in `evaluation/utils.py` accepts either a path or a JSON string and yields configs with a `file` name like `mean_1_2_3.mlu`.
   - Patch + compile flow: `compile_kernel` patches source by prepending macros from `evaluation/macros.py`, writes a `_patched` file, then calls a target-specific compiler (`nvcc`, `hipcc`, `cncc`, or `g++`) via `COMPILATION_FUNCTIONS`.
   - Test flow: `run_tests` compiles in parallel, imports the corresponding `test_*` module dynamically, calls `test_kernel(config, so_path)` for each compiled `.so`, then removes `.so` files.
   - Example test file pattern: `evaluation/mlu_test/test_mean.py` constructs ctypes argtypes dynamically based on input rank, loads the shared object, calls `<op_name>_kernel`, and verifies results with `verify_torch_tensor`.

4. Project-specific conventions & patterns
   - File naming: generated sources use `{op_name}_{'_'.join(args)}.{ext}` (see `parse_op_json`). The compiled artifact becomes `{base}.so`.
   - Macros: sources are patched by prefixing backend macros; do not assume sources are standalone.
   - Targets: canonical targets are `cpu`, `cuda`, `hip`, `mlu`. Map to compilers via `COMPILATION_FUNCTIONS`.
   - Tests: each op has corresponding `test_<op>.py` under `evaluation/*_test/`. They expect a shared object exporting `<op_name>_kernel` with an implicit C ABI.
   - Parallelism: `run_tests` uses ThreadPoolExecutor; be mindful of side-effects (file creation/removal) when modifying parallel phases.

5. Integration points and external dependencies
   - Native compilers required: `nvcc`, `hipcc`, `cncc`, `g++` — ensure paths in `env.sh` and host toolchain.
   - Python dependencies: see `requirements.txt` (PyTorch, numpy). Tests use `python` to run test scripts (scripts call `torch`).
   - Dynamic import: `run_tests` dynamically loads test modules via `importlib.util.spec_from_file_location`.

6. Typical change patterns for contributors
   - Adding an operator: add config to `TransEval/*.json`, implement a generator in `CodeGen/` (if needed) producing the source named per `parse_op_json`, and add/modify `evaluation/*_test/test_<op>.py`.
   - Changing a compiler flag: update the appropriate function in `evaluation/utils.py` (e.g., `run_cuda_compilation`). Keep consistent timeouts and error capture.

7. Quick examples to mention in PRs or prompts
   - “To add a new mean kernel for MLU, add a config entry to `TransEval/mlu.json` and ensure `CodeGen/` can emit `mean_<args>.mlu`; tests will be picked up by `evaluation/mlu_test/test_mean.py` and executed by `evaluate.sh`.”

8. Safety and verification hints for AI agents
   - Always read `evaluation/utils.py` before editing compilation/test logic; it centralizes macros, compilers, and test orchestration.
   - When changing test harnesses, keep the ctypes ABI stable: tests expect `<op>_kernel` and arguments `(float* input, float* output, int d0, ..., int reduce_dim)` in `test_mean.py` style.

If any section is unclear or you want more examples (e.g., a sample JSON config and the generated file name), tell me which part to expand.
