__global__ void gemm(half *A, half *B, float *C) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    // 矩阵维度参数
    const int M = 32, K = 128, N = 128;
    const int LDA = K;   // A的leading dimension
    const int LDB = N;   // B的leading dimension
    const int LDC = N;   // C的leading dimension

    // 计算当前线程块处理的C矩阵块位置
    const int c_row_base = blockIdx.y * 16;  // 行分块：32/16=2块
    const int c_col_base = blockIdx.x * 16;  // 列分块：128/16=8块

    // 初始化累加器（4个float）
    floatx4 d = {0.0f};

    // K维度分块循环（128/16=8次）
    for(int k_step = 0; k_step < K; k_step += 16) {
        float16x4 a, b;

        // 向量化加载A/B子矩阵（带边界检查）
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            // A矩阵索引：行由C块行决定，列按K分块
            const int a_row = c_row_base + threadIdx.x;
            const int a_col = k_step + threadIdx.y * 4 + i;
            a[i] = (a_row < M && a_col < K) ? A[a_row * LDA + a_col] : 0.0h;

            // B矩阵索引：行按K分块，列由C块列决定
            const int b_row = k_step + threadIdx.y * 4 + i;
            const int b_col = c_col_base + threadIdx.x;
            b[i] = (b_row < K && b_col < N) ? B[b_row * LDB + b_col] : 0.0h;
        }

        // 执行MFMA指令（16x16x16矩阵乘加）
        d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
    }

    // 结果写回全局内存（带边界保护）
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        const int c_row = c_row_base + threadIdx.x;
        const int c_col = c_col_base + threadIdx.y * 4 + i;
        if(c_row < M && c_col < N) {
            C[c_row * LDC + c_col] = d[i];
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
