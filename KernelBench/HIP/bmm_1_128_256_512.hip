__global__ void bmm(half *A, half *B, float *C) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4 d = {0.0f};   // 初始化累加器为0

    // 当前线程块处理的C块起始位置
    int c_row_base = blockIdx.x * 16;   // M维度：128/16=8块
    int c_col_base = blockIdx.y * 16;   // N维度：512/16=32块

    // 遍历K维度（256），分块步长16
    for (int k_step = 0; k_step < 256; k_step += 16) {
        float16x4 a, b;

        // 加载当前K块的A和B子矩阵
        for (int i = 0; i < 4; ++i) {
            // A的索引：行由当前C块行决定，列按K分块
            int a_row = c_row_base + threadIdx.x;          // 行范围[0,127]
            int a_col = k_step + threadIdx.y * 4 + i;      // 列范围[0,255]
            int a_idx = a_row * 256 + a_col;
            a[i] = A[a_idx];

            // B的索引：行按K分块，列由当前C块列决定
            int b_row = k_step + threadIdx.y * 4 + i;      // 行范围[0,255]
            int b_col = c_col_base + threadIdx.x;           // 列范围[0,511]
            int b_idx = b_row * 512 + b_col;
            b[i] = B[b_idx];
        }

        // 执行MFMA并累加结果
        d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
    }

    // 将结果写回C矩阵
    for (int i = 0; i < 4; ++i) {
        int c_row = c_row_base + threadIdx.x;
        int c_col = c_col_base + threadIdx.y * 4 + i;
        if (c_row < 128 && c_col < 512) {  // 边界保护
            C[c_row * 512 + c_col] = d[i];
        }
    }
}

extern "C" void bmm_kernel(half *A, half *B, float *C,  int b, int m, int k,
                           int n) {
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
  bmm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
