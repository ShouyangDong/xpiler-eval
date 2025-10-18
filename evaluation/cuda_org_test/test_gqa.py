import argparse
import ctypes
import logging
import os
from typing import Tuple

import torch

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
    verify_numpy_tensor,
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


def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on a successfully compiled kernel."""
    batch, seq_q, seq_kv, head_dim = config["args"]
    op_name = config["op_name"]
    # ---------------------------------------------
    # 3. Load shared library
    # ---------------------------------------------
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_path))
    gqa_func = getattr(lib, op_name + "_kernel")
    gqa_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # K
        ctypes.POINTER(ctypes.c_float),  # V
        ctypes.POINTER(ctypes.c_float),  # O
        ctypes.c_int,  # batch
        ctypes.c_int,  # num_heads
        ctypes.c_int,  # seq_q
        ctypes.c_int,  # seq_kv
        ctypes.c_int,  # head_dim
    ]
    gqa_func.restype = None

    # ---------------------------------------------
    # 4. Get the refered output using Pytorch
    # ---------------------------------------------
    torch.manual_seed(1234)

    # Generate Q, K, V
    Q = torch.rand([batch, 2, seq_q, 64], dtype=torch.float32, device="cuda")
    K = torch.rand([batch, 2, 64, seq_kv], dtype=torch.float32, device="cuda")
    V = torch.rand([batch, 2, seq_kv, 64], dtype=torch.float32, device="cuda")

    # ✅ Get the referenced ouput
    with torch.no_grad():
        S = torch.matmul(Q, K)  # [b, 2, seq_q, seq_kv]
        S = torch.softmax(S, dim=-1)
        O_ref = torch.matmul(S, V)  # [b, 2, seq_q, 64]

    # ---------------------------------------------
    # 5. Prepare C++ kernel and feed output buffer
    # ---------------------------------------------
    O = torch.zeros_like(O_ref, device="cuda")  # host buffer for output

    Q_cpu = Q.cpu().numpy()
    K_cpu = K.cpu().numpy()
    V_cpu = V.cpu().numpy()

    Q_ptr = Q_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    K_ptr = K_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    V_ptr = V_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    O_ptr = O.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # ---------------------------------------------
    # 6. invoke C++/CUDA kernel
    # ---------------------------------------------
    gqa_func(Q_ptr, K_ptr, V_ptr, O_ptr, batch, 2, seq_q, seq_kv, 64)

    # ---------------------------------------------
    # 7. Verification
    # ---------------------------------------------
    return verify_numpy_tensor(result_reshaped, expected, op_name=op_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels (HIP)")
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the operator to test (used to filter configs).",
    )
    parser.add_argument(
        "--config", required=True, help="JSON string or path to config file"
    )
    parser.add_argument(
        "--source_dir", default="./", help="Directory with .cpp files"
    )
    parser.add_argument(
        "--target",
        default="cpu",
        choices=["cuda", "cpu", "mlu", "hip"],
        help="Target platform",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name, file_type="cu")

    if not configs:
        logger.warning(f"No {args.name} kernels found in config.")
        exit(0)

    # Run two-phase test
    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )

    # Summary
    log_test_results_and_exit(results, op_name=args.name)
