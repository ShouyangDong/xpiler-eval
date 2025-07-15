__global__ void gemm(half *A, half *B, float *C) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    // 计算当前线程块处理的C矩阵块位置
    const int c_row_base = blockIdx.y * 16;  // 行分块：32/16=2块
    const int c_col_base = blockIdx.x * 16;  // 列分块：1024/16=8块

    // 初始化累加器
    floatx4 d = {0.0f};

    // K维度分块循环（32/16=2次）
    for(int k_step = 0; k_step < 32; k_step += 16) {
        float16x4 a, b;

        // 向量化加载数据（每个线程处理4个元素）
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            // 加载A的子矩阵块（行方向连续）
            int a_row = c_row_base + threadIdx.x;
            int a_col = k_step + threadIdx.y * 4 + i;
            a[i] = A[a_row * 32 + a_col];

            // 加载B的子矩阵块（列方向连续）
            int b_row = k_step + threadIdx.y * 4 + i;
            int b_col = c_col_base + threadIdx.x;
            b[i] = B[b_row * 1024 + b_col];
        }

        // 执行矩阵核心指令（16x16x16 MFMA）
        d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
    }

    // 写回结果到全局内存
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        int c_row = c_row_base + threadIdx.x;
        int c_col = c_col_base + threadIdx.y * 4 + i;
        if(c_row < 32 && c_col < 1024) {  // 边界检查
            C[c_row * 1024 + c_col] = d[i];
        }
    }
}

extern "C" void gemm_kernel(half *A, half *B, float *C,  int m, int k, int n) {
  half *d_A;
  half *d_B;
  float *d_C;

  hipMalloc(&d_A, m * k * sizeof(half));
  hipMalloc(&d_B, k * n * sizeof(half));
  hipMalloc(&d_C, m * n * sizeof(float));

  hipMemcpy(d_A, A, m * k * sizeof(half), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, k * n * sizeof(half), hipMemcpyHostToDevice);

  dim3 blockSize(64);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16);
  gemm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
