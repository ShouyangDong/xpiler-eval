"""Batch correctness tester for scatter kernels with parallel compilation and
testing."""

import argparse
import ctypes
import logging
from typing import Tuple

import numpy as np
import torch

from evaluation.utils import (
    log_test_results_and_exit,
    parse_op_json,
    run_tests,
)

# ----------------- Logger -----------------
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


# ----------------- Reference Implementation -----------------
def reference_scatter(
    self_tensor: torch.Tensor,
    indices_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    dim: int,
):
    """Mimic torch.Tensor.scatter_ semantics."""
    result = self_tensor.clone()
    result.scatter_(dim=dim, index=indices_tensor, src=src_tensor)
    return result


# ----------------- Core: Kernel Test -----------------
def test_kernel(config: dict, so_path: str) -> Tuple[bool, str]:
    """Run correctness test on compiled scatter kernel."""
    try:
        dim = config["axis"]
        shape = config["args"]
        config["op_name"]
        file_name = config["file"]
        op_name = config["op_name"]
        # ----------- Prepare tensors -----------
        self_tensor = torch.rand(*shape, dtype=torch.float32)
        src_tensor = torch.rand(*shape, dtype=torch.float32)
        size_dim = shape[dim]
        indices_tensor = torch.randint(0, size_dim, shape, dtype=torch.int64)

        expected = reference_scatter(
            self_tensor, indices_tensor, src_tensor, dim
        )

        # Convert to contiguous numpy arrays
        self_np = (
            self_tensor.contiguous().numpy().astype(np.float32, copy=False)
        )
        src_np = src_tensor.contiguous().numpy().astype(np.float32, copy=False)
        indices_np = (
            indices_tensor.contiguous().numpy().astype(np.int32, copy=False)
        )
        out_np = np.zeros_like(self_np, dtype=np.float32)

        # ----------- Load shared lib -----------
        lib = ctypes.CDLL(so_path)
        kernel_func = getattr(lib, op_name, None)
        kernel_func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # self
            ctypes.POINTER(ctypes.c_int32),  # indices
            ctypes.POINTER(ctypes.c_float),  # src
            ctypes.POINTER(ctypes.c_float),  # output
        ]
        kernel_func.restype = None

        # ----------- Call kernel -----------
        kernel_func(
            self_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            src_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        # Compare
        result_reshaped = torch.from_numpy(out_np).reshape(expected.shape)
        try:
            torch.allclose(result_reshaped, expected, rtol=1e-4, atol=1e-4)
            return True, f"[{op_name}] PASSED✅: {file_name}"
        except Exception as e:
            return False, f"[{op_name}] FAILED❌: {file_name} | {str(e)}"

    except Exception as e:
        return False, f"[{op_name}] Exception in test {file_name}: {str(e)}"


# ----------------- CLI -----------------
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
        "--target", required=True, choices=["cuda", "hip", "mlu", "cpu"]
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel workers")

    args = parser.parse_args()

    # Parse config
    configs = parse_op_json(args.config, args.name)

    if not configs:
        logger.warning("No valid scatter kernels found.")
        exit(0)

    results = run_tests(
        args.name, configs, args.source_dir, args.target, num_workers=args.jobs
    )
    log_test_results_and_exit(results, op_name=args.name)
