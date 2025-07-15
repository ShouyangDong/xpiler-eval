__global__ void gemm(half *A, half *B, float *C) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    // 矩阵维度参数
    const int M = 1024, K = 16, N = 1024;
    const int LDA = K;   // A的列维度
    const int LDB = N;   // B的列维度
    const int LDC = N;   // C的列维度

    // 线程块处理的C矩阵块位置
    const int c_row_base = blockIdx.y * 16;  // 行分块 (1024/16=64块)
    const int c_col_base = blockIdx.x * 16;  // 列分块 (1024/16=8块)

    // 寄存器声明
    float16x4 a, b;
    floatx4 d = {0.0f};

    // 向量化加载A矩阵数据（K=16只需单次加载）
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        int a_row = c_row_base + threadIdx.x;
        int a_col = threadIdx.y * 4 + i;  // K维度全覆盖
        a[i] = A[a_row * LDA + a_col];
    }

    // 向量化加载B矩阵数据
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        int b_row = threadIdx.y * 4 + i;
        int b_col = c_col_base + threadIdx.x;
        b[i] = B[b_row * LDB + b_col];
    }

    // 执行单次MFMA计算（16x16x16完整计算）
    d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);

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
