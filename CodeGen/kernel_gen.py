import json
from pathlib import Path

import tvm
from tvm import te, topi

# output目录
OUTPUT_DIR = Path("generated_kernels")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_shapes(op_name, args):
    """根据算子推断输入 shape."""
    if "conv2d" in op_name:
        N, C, H, W = 1, args[0], args[1], args[2]
        if "NHWC" in op_name:
            return (N, H, W, C), (3, 3, C, args[0] // 2)  # NHWC
        else:
            return (N, C, H, W), (args[0] // 2, C, 3, 3)  # NCHW
    elif op_name in ["gemm", "bmm"]:
        return (args[0], args[1]), (args[1], args[0] // 2)
    elif op_name == "gemv":
        return (args[0], args[1]), (args[1],)
    elif "pool2d" in op_name:
        return ((1, args[0], args[1], args[2]),)  # NCHW
    elif "norm" in op_name:
        return ((2, args[0], 128),)  # B, S, D
    elif op_name in ["self-atten", "DAT"]:
        D = args[0]
        return (1, 8, 128, D), (1, 8, D, 128), (1, 8, 128, D)  # QKV
    else:
        return (tuple(args),)


def get_tvm_op(op_name, placeholders):
    """Construct TVM 计算图."""
    A = placeholders[0]
    if len(placeholders) > 1:
        B = placeholders[1]

    if op_name == "add":
        return topi.add(A, B)
    elif op_name == "sub":
        return topi.subtract(A, B)
    elif op_name == "mul":
        return topi.multiply(A, B)
    elif op_name == "div":
        return topi.divide(A, B)
    elif op_name == "relu":
        return topi.nn.relu(A)
    elif op_name == "sigmoid":
        return topi.sigmoid(A)
    elif op_name == "softmax":
        return topi.nn.softmax(A)
    elif op_name == "gelu":
        return topi.nn.gelu(A)
    elif "conv" in op_name:
        layout = "NHWC" if "NHWC" in op_name else "NCHW"
        out_channels = placeholders[1].shape[-2 if layout == "NHWC" else 0]
        return topi.nn.conv2d(
            A,
            B,
            strides=1,
            padding=1,
            layout=layout,
            out_channels=int(out_channels),
        )
    elif "pool2d" in op_name:
        pool_type = op_name.split("pool2d")[0]
        return topi.nn.pool2d(
            A, kernel=(2, 2), stride=(2, 2), pool_type=pool_type, layout="NCHW"
        )
    elif op_name == "gemm":
        return topi.nn.dense(A, B)
    elif op_name == "bmm":
        return topi.nn.batch_matmul(A, B)
    elif op_name == "gemv":
        return topi.nn.dense(A, B)
    elif "norm" in op_name:
        if "batch" in op_name:
            return topi.nn.batch_norm(
                A, tvm.te.placeholder((A.shape[1],), dtype=A.dtype)
            )[0]
        elif "layer" in op_name:
            return topi.nn.layer_norm(A, axes=[-1])
        elif "RMS" in op_name:
            variance = (
                topi.sum(te.power(A, 2), axis=-1, keepdims=True) / A.shape[-1]
            )
            return A / topi.sqrt(variance + 1e-6)
    elif op_name in ["self-atten", "DAT"]:
        Q, K, V = placeholders
        scale = 1.0 / (Q.shape[-1] ** 0.5)
        attn = topi.nn.batch_matmul(Q, topi.transpose(K, [0, 1, 3, 2])) * scale
        prob = topi.nn.softmax(attn)
        return topi.nn.batch_matmul(prob, V)
    else:
        # 默认：逐元素加法（兜底）
        return topi.add(A, A) if len(placeholders) == 1 else topi.add(A, B)


def generate_kernel(op_name, args, dtype="float32"):
    """生成多后端 kernel."""
    shapes = create_shapes(op_name, args)
    placeholders = [
        te.placeholder(s, name=f"input{i}", dtype=dtype)
        for i, s in enumerate(shapes)
    ]

    # Construct计算
    C = get_tvm_op(op_name, placeholders)
    s = te.create_schedule(C.op)

    inputs = placeholders + ([C] if isinstance(C, te.tensor.Tensor) else [])
    func_name = f"{op_name}_{'_'.join(map(str, args))}"

    # ==================== 1. Generate C++ Kernel ====================
    with tvm.target.Target("c"):
        c_mod = tvm.build(s, inputs, target="c", name=func_name)
        c_code = c_mod.get_source()
        (OUTPUT_DIR / f"{func_name}.cpp").write_text(c_code)
        print(f"✅ {func_name}.cpp generated")

    # ==================== 2. Generate CUDA Kernel ====================
    try:
        with tvm.target.Target("cuda"):
            cuda_mod = tvm.build(s, inputs, target="cuda", name=func_name)
            cuda_code = cuda_mod.get_source()
            (OUTPUT_DIR / f"{func_name}.cu").write_text(cuda_code)
            print(f"✅ {func_name}.cu generated")
    except Exception as e:
        print(f"❌ CUDA failed for {func_name}: {e}")

    # ==================== 3. Generate HIP Kernel ====================
    try:
        with tvm.target.Target("rocm"):
            hip_mod = tvm.build(s, inputs, target="rocm", name=func_name)
            hip_code = hip_mod.get_source()
            (OUTPUT_DIR / f"{func_name}.hip").write_text(hip_code)
            print(f"✅ {func_name}.hip generated")
    except Exception as e:
        print(f"❌ HIP failed for {func_name}: {e}")


def main(json_file="kernels.json"):
    try:
        with open(json_file, "r") as f:
            kernels = json.load(f)
    except FileNotFoundError:
        print(f"{json_file} not found, using demo data.")
        kernels = [
            {"op_name": "add", "args": [3, 4]},
            {"op_name": "relu", "args": [2, 3]},
            {"op_name": "conv2dNCHW", "args": [3, 32, 32]},
            {"op_name": "maxpool2d", "args": [3, 224, 224]},
            {"op_name": "gemm", "args": [128, 64]},
            {"op_name": "self-atten", "args": [128, 64]},
        ]

    print("🚀 Generating multi-backend kernels...")
    for item in kernels:
        op_name = item["op_name"]
        args = item["args"]
        dtype = item.get("dtype", "float32")
        generate_kernel(op_name, args, dtype)


if __name__ == "__main__":
    main()
