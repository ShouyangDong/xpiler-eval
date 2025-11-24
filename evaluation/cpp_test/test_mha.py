"""Batch correctness tester for Multi-Head Attention (MHA) kernels with
parallel compilation and testing."""

import argparse
import ctypes
import logging
import math
from typing import Tuple

import torch
import torch.nn.functional as F

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
    verify_torch_tensor,
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def reference_mha(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Reference MHA forward pass using PyTorch."""
    # q: [B, H, L, D], k/v: [B, H, S, D]
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
        q.size(-1)
    )  # [B,H,L,S]
    if causal:
        mask = torch.triu(torch.ones(L, S, device=q.device), diagonal=1).bool()
        score = score.masked_fill(mask, torch.finfo(score.dtype).min)
    attn = F.softmax(score, dim=-1)
    output = torch.matmul(attn, v)  # [B,H,L,D]
    return output


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled MHA kernel."""
    shape = config["args"]
    causal = config.get("causal", False)
    dtype_str = config.get("dtype", "float32")

    op_name = config["op_name"]
    # Load shared library
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, op_name, None)
    if not func:
        return False, f"[{op_name}] Function 'mha' not found in {so_path}"

    # Set function signature
    ctype = ctypes.c_float if dtype_str == "float32" else ctypes.c_ushort
    torch_dtype = torch.float32 if dtype_str == "float32" else torch.float16

    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    func.restype = None

    # Generate input
    torch.manual_seed(1234)
    q = torch.randn(shape, dtype=torch_dtype)
    k = torch.randn(shape, dtype=torch_dtype)
    v = torch.randn(shape, dtype=torch_dtype)

    expected = reference_mha(q, k, v, causal=causal).cpu()

    # Flatten and get pointers
    q_flat = q.flatten().numpy()
    k_flat = k.flatten().numpy()
    v_flat = v.flatten().numpy()
    out_flat = torch.zeros(expected.shape, dtype=torch_dtype).flatten().numpy()

    ptr_q = q_flat.ctypes.data_as(ctypes.POINTER(ctype))
    ptr_k = k_flat.ctypes.data_as(ctypes.POINTER(ctype))
    ptr_v = v_flat.ctypes.data_as(ctypes.POINTER(ctype))
    ptr_out = out_flat.ctypes.data_as(ctypes.POINTER(ctype))

    # Call kernel
    func(ptr_q, ptr_k, ptr_v, ptr_out, causal)

    # Reshape and compare
    result = torch.from_numpy(out_flat).reshape(expected.shape).cpu()
    return verify_torch_tensor(result, expected, op_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (CPU)")
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the operator to test (used to filter configs).",
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
    )
    parser.add_argument(
        "--source_dir", default="./", help="Directory containing .cpp files"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["cuda", "hip", "mlu", "cpu"],
        help="Target platform",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid 'mha' kernels found in config.")
        exit(0)

    # Run tests
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Log individual results
    log_test_results_and_exit(results, op_name=args.name)
