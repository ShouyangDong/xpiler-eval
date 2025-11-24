# eval_performance.py
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

# -------------------------------
# 1. 算子实现：PyTorch 版本
# -------------------------------


def torch_scatter(
    input: torch.Tensor, indices: torch.Tensor, axis: int
) -> torch.Tensor:
    # 创建输出（全零）
    output = input.clone()  # 或 torch.zeros_like(input)
    # 简化：沿 axis scatter 数值（这里只是示例，可根据需求改）
    index_tensors = [slice(None)] * input.ndim
    index_tensors[axis] = indices.long()
    index_tensors = tuple(index_tensors)
    output[index_tensors] = input[index_tensors]  # 简化操作
    return output


def torch_gatemlp(
    X: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor
) -> torch.Tensor:
    gate = torch.nn.functional.silu(X @ W_gate)
    up = X @ W_up
    return gate * up


# -------------------------------
# 2. 构造输入张量
# -------------------------------


def create_input(op: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    op_name = op["op_name"]
    dtype_str = op["dtype"]
    dtype = torch.float32 if dtype_str == "float32" else torch.int16
    args = op["args"]
    kwargs = {
        k: v for k, v in op.items() if k not in ["op_name", "dtype", "args"]
    }

    if op_name == "scatter":
        batch, dim, seq_len, kv_len = args
        shape = [batch, dim, seq_len, kv_len]
        input_tensor = torch.randn(shape, dtype=torch.float32, device=device)
        num_picked = seq_len // 2
        indices = torch.randint(0, seq_len, (num_picked,), device=device)
        return {
            "inputs": [input_tensor, indices],
            "kwargs": {"axis": kwargs.get("axis")},
            "output_shape": shape,
            "desc": f"scatter{shape}_axis{kwargs.get('axis')}",
        }

    elif op_name == "gatemlp":
        M, K, N = args
        X = torch.randn(M, K, dtype=torch.float32, device=device)
        W_gate = torch.randn(K, N, dtype=torch.float32, device=device)
        W_up = torch.randn(K, N, dtype=torch.float32, device=device)
        if dtype == torch.int16:
            X = X.to(torch.int16)
            W_gate = W_gate.to(torch.int16)
            W_up = W_up.to(torch.int16)
        return {
            "inputs": [X, W_gate, W_up],
            "kwargs": {},
            "output_shape": [M, N],
            "desc": f"gatemlp[{M},{K},{N}]",
        }

    else:
        raise ValueError(f"Unknown op: {op_name}")


# -------------------------------
# 3. 执行并计时
# -------------------------------


def benchmark_op(
    op: Dict[str, Any], device: torch.device, runs: int = 100, warmup: int = 5
) -> Dict[str, Any]:
    op_name = op["op_name"]

    # 构造输入
    try:
        data = create_input(op, device)
        inputs = data["inputs"]
        kwargs = data["kwargs"]
        output_shape = data["output_shape"]
        desc = data["desc"]
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "op_name": op_name,
            "args": op["args"],
        }

    # 选择函数
    func = {"scatter": torch_scatter, "gatemlp": torch_gatemlp}.get(op_name)

    if not func:
        return {
            "success": False,
            "error": f"No PyTorch implementation for {op_name}",
            "op_name": op_name,
        }

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = func(*inputs, **kwargs)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # 正式计时
    times = []
    for _ in range(runs):
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            func(*inputs, **kwargs)
        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        "success": True,
        "op_name": op_name,
        "args": op["args"],
        "dtype": op["dtype"],
        "desc": desc,
        "output_shape": output_shape,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "runs": runs,
        "device": str(device),
    }


# -------------------------------
# 4. 主函数
# -------------------------------


def main(targets: List[str], runs: int = 100):
    base_dir = "TransEval"
    results = []

    # 自动选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")

    print(f"🚀 Using device: {device}")

    for target in targets:
        json_path = os.path.join(base_dir, target, "operators.json")
        if not os.path.exists(json_path):
            print(f"⚠️  {target}: {json_path} not found, skipping.")
            continue

        print(f"\n📁 Loading {target.upper()} operators...")
        with open(json_path, "r") as f:
            ops = json.load(f)

        target_results = []
        for op in ops:
            print(
                f"  ⏱️  Running: {op['op_name']} {op['args']} ({op['dtype']})"
            )
            result = benchmark_op(op, device, runs=runs)
            target_results.append(result)
            results.append({"target": target, "result": result})
            if result["success"]:
                print(
                    f"    ✅ Mean: {result['mean_ms']:.4f} ms ± {result['std_ms']:.4f}"
                )
            else:
                print(f"    ❌ Error: {result['error']}")

        print(f"✅ {target.upper()}: {len(target_results)} ops evaluated.")

    # 保存结果
    with open("torch_baseline_perf.json", "w") as f:
        json.dump(results, f, indent=2)

    # 打印汇总表
    print("\n" + "=" * 80)
    print(
        f"{'Platform':<8} {'Op':<12} {'Args':<20} {'Mean (ms)':<12} {'Device'}"
    )
    print("=" * 80)
    for item in results:
        r = item["result"]
        if r["success"]:
            print(
                f"{item['target']:<8} {r['op_name']:<12} {str(r['args']):<20} {r['mean_ms']:<12.4f} {r['device']}"
            )
    print("=" * 80)
    print("📊 Full results saved to 'torch_baseline_perf.json'")


# -------------------------------
# 5. CLI 入口
# -------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate PyTorch performance for operators across targets."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="cuda,hip,mlu,cpu",
        help="Platforms to evaluate (comma-separated): cuda,hip,mlu,cpu",
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of runs per op"
    )

    args = parser.parse_args()
    targets = [p.strip() for p in args.target.split(",") if p.strip()]

    main(targets, runs=args.runs)
