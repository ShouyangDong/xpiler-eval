__global__ void bmm(half *A, half *B, float *C) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    const int LDA = 128;  // 单个矩阵的列数
    const int LDB = 128;
    const int LDC = 128;
    const int BATCH_SIZE = 4;

    // 计算批次索引和矩阵块索引
    int batch_id = blockIdx.z;  // 使用三维网格处理批次
    int c_row_base = (blockIdx.x % 8) * 16;  // 每个矩阵分为8x8个块
    int c_col_base = (blockIdx.x / 8) * 16;  // 合并x维度处理矩阵内位置

    // 计算全局内存偏移量
    float16_t* batch_A = A + batch_id * 128 * 128;
    float16_t* batch_B = B + batch_id * 128 * 128;
    float* batch_C = C + batch_id * 128 * 128;

    floatx4 d = {0.0f};

    // 主计算循环（K=128）
    for(int k_step = 0; k_step < 128; k_step += 16) {
        float16x4 a, b;

        // 向量化加载数据
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            // 加载A矩阵数据
            int a_row = c_row_base + threadIdx.x;
            int a_col = k_step + threadIdx.y * 4 + i;
            a[i] = batch_A[a_row * LDA + a_col];

            // 加载B矩阵数据
            int b_row = k_step + threadIdx.y * 4 + i;
            int b_col = c_col_base + threadIdx.x;
            b[i] = batch_B[b_row * LDB + b_col];
        }

        // 矩阵核心计算
        d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
    }

    // 结果写回
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        int c_row = c_row_base + threadIdx.x;
        int c_col = c_col_base + threadIdx.y * 4 + i;
        batch_C[c_row * LDC + c_col] = d[i];
    }
}

extern "C" void bmm_kernel(half *A, half *B, float *C,  int b, int m, int k,
                           int n) {
  half *d_A;
  half *d_B;
  float *d_C;

  hipMalloc(&d_A, b * m * k * sizeof(half));
  hipMalloc(&d_B, b * k * n * sizeof(half));
  hipMalloc(&d_C, b * m * n * sizeof(float));

  hipMemcpy(d_A, A, b * m * k * sizeof(half), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, b * k * n * sizeof(half), hipMemcpyHostToDevice);

  dim3 blockSize(64, 1, 4);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16);
  bmm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, b * m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
