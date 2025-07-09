import logging
import tempfile

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.target import Target
import tempfile
import tvm
from tvm import tir, topi, te
from tvm.target import Target
import tvm.meta_schedule as ms
import numpy as np
from tvm.runtime import ndarray as nd

def test_tune_matmul_cuda(op_name, order=2):
    mod = create_te_workload(op_name, order)
    with tempfile.TemporaryDirectory() as work_dir:
        # 使用 A100 GPU 作为目标
        target = Target("nvidia/nvidia-a100", host="llvm")
        database = ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir=work_dir,
            max_trials_global=1024,
            num_trials_per_iter=64,
        )
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            args = ArgInfo.from_prim_func(mod)
            dev = tvm.device("cuda", 0)
            # 构建 CUDA 内核
            myfunc = tvm.build(sch.mod, target=target, name=op_name)
            
            # 打印 CUDA 源代码
            cuda_src = myfunc.imported_modules[0].get_source()
            print("==== CUDA Kernel Source ====")
            print(cuda_src)
            filename = f"{op_name}_{order}.cu"
            with open(filename, "w") as f:
                f.write(cuda_src)
            # 准备输入数据
            inputs = []
            for arg in args:
                shape = arg.shape
                dtype = arg.dtype
                buffer = tvm.nd.array(np.zeros(shape, dtype=dtype), dev)
                inputs.append(buffer)

            evaluator = myfunc.time_evaluator(
                myfunc.entry_name, dev, repeat=1, number=10
            )
            eval_time = evaluator(*inputs).mean * 1e3  # 转换为 ms
            print(f"The time of {op_name} is {eval_time:.3f} ms")

if __name__ == "__main__":
    tensor_ir = ["C1D", "C2D", "C3D", "GMM", "GRP", "DIL", "DEP","T2D", "CAP", "NRM", "SFM", "CBR", "TBG"]
    for order in range(4):
        print("===========================")
        for op_name in tensor_ir:
            test_tune_matmul_cuda(op_name, order)
