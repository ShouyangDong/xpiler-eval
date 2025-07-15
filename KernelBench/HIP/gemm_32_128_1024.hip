__global__ void gemm(half *A, half *B, float *C) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    // 矩阵维度参数
    const int M = 32, K = 128, N = 1024;
    const int LDA = K;    // A的列维度
    const int LDB = N;    // B的列维度
    const int LDC = N;    // C的列维度
    const int TILE_SIZE = 16; // 矩阵核心分块尺寸

    // 计算线程块映射
    const int c_row_base = blockIdx.y * TILE_SIZE;    // 行分块 (32/16=2块)
    const int c_col_base = blockIdx.x * TILE_SIZE;    // 列分块 (1024/16=64块)

    // 初始化累加器
    floatx4 d = {0.0f};

    // K维度分块循环 (128/16=8次)
    for(int k_step = 0; k_step < K; k_step += TILE_SIZE) {
        float16x4 a, b;

        // 向量化加载数据 (每个线程加载4个连续元素)
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            // A矩阵索引: [行][k_step+列偏移]
            int a_row = c_row_base + threadIdx.x;
            int a_col = k_step + threadIdx.y * 4 + i;
            a[i] = A[a_row * LDA + a_col];

            // B矩阵索引: [k_step+行偏移][列]
            int b_row = k_step + threadIdx.y * 4 + i;
            int b_col = c_col_base + threadIdx.x;
            b[i] = B[b_row * LDB + b_col];
        }

        // 执行矩阵核心指令 (16x16x16 MFMA)
        d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
    }

    // 结果写回全局内存
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        int c_row = c_row_base + threadIdx.x;
        int c_col = c_col_base + threadIdx.y * 4 + i;
        C[c_row * LDC + c_col] = d[i];
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
