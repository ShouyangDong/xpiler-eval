__global__ void bmm(half *A, half *B, float *C) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    // 矩阵维度参数
    const int M = 128, K = 256, N = 512;
    const int LDA = K;   // A的列数（行主序）
    const int LDB = N;   // B的列数
    const int LDC = N;   // C的列数
    const int BATCH = 4; // 批次数

    // 计算三维网格中的位置
    const int batch_id = blockIdx.z;                   // 批次索引 [0~3]
    const int c_row_base = blockIdx.y * 16;            // C块行起始 [0,16,...,112]
    const int c_col_base = blockIdx.x * 16;            // C块列起始 [0,16,...,496]

    // 获取当前批次的矩阵指针
    float16_t* batch_A = A + batch_id * M * K;
    float16_t* batch_B = B + batch_id * K * N;
    float* batch_C = C + batch_id * M * N;

    // 初始化累加器
    floatx4 d = {0.0f};

    // 主计算循环（K=256，分块步长16）
    for (int k_step = 0; k_step < K; k_step += 16) {
        float16x4 a, b;

        // 向量化加载A和B的子矩阵
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            // A的索引：行由C块行决定，列按K分块
            int a_row = c_row_base + threadIdx.x;
            int a_col = k_step + threadIdx.y * 4 + i;
            a[i] = batch_A[a_row * LDA + a_col];

            // B的索引：行按K分块，列由C块列决定
            int b_row = k_step + threadIdx.y * 4 + i;
            int b_col = c_col_base + threadIdx.x;
            b[i] = batch_B[b_row * LDB + b_col];
        }

        // 执行矩阵核心运算（16x16x16 MFMA）
        d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
    }

    // 写回结果到C矩阵
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int c_row = c_row_base + threadIdx.x;
        int c_col = c_col_base + threadIdx.y * 4 + i;
        if (c_row < M && c_col < N) { // 边界保护
            batch_C[c_row * LDC + c_col] = d[i];
        }
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

  dim3 blockSize(64);
  dim3 numBlocks((n + 16 - 1) / 16, (m + 16 - 1) / 16, 4);
  bmm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  hipMemcpy(C, d_C, b * m * n * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
}
