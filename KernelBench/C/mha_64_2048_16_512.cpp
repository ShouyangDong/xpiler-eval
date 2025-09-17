extern "C" void mha(float *Q, float *K, float *V, float *output) {
  // 定义所需的数组，使用不同的变量名以避免重定义
  int8_t arr_a_64[64];
  int8_t arr_b_64[64];
  int32_t arr_d_16[16]; // AVX-512 寄存器能同时处理 16 个 int32 元素
  float score[16 * 16]; // 假设heads为16

  int8_t arr_a_16[16];
  int8_t arr_b_16[16];
  int32_t arr_d_4[4];

  const int batch = 64;
  const int seq_len = 2048;
  const int heads = 16;
  const int dim = 512;
  const float scale = 1.0f / sqrt(dim);

  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < seq_len; j++) {
      for (int m = 0; m < heads; m++) {
        for (int n = 0; n < heads; n++) {
          int32_t sum = 0;

          for (int local_s = 0; local_s < dim / 64;
               local_s++) { // 每次处理 64 个元素
            for (int local_i = 0; local_i < 64; local_i++) {
              arr_a_64[local_i] = static_cast<int8_t>(
                  Q[i * seq_len * heads * dim + j * heads * dim + m * dim +
                    local_s * 64 + local_i]);
              arr_b_64[local_i] = static_cast<int8_t>(
                  K[i * seq_len * heads * dim + j * heads * dim + n * dim +
                    local_s * 64 + local_i]);
            }

            __m512i acc = _mm512_setzero_si512();

            __m512i _a =
                _mm512_loadu_si512(reinterpret_cast<const __m512i *>(arr_a_64));
            __m512i _b =
                _mm512_loadu_si512(reinterpret_cast<const __m512i *>(arr_b_64));

            acc = _mm512_dpbusd_epi32(acc, _a, _b);

            _mm512_storeu_si512(reinterpret_cast<__m512i *>(arr_d_16), acc);

            for (int k = 0; k < 16; ++k) { // 处理 16 个累积结果
              sum += arr_d_16[k];
            }
          }

          score[m * heads + n] = static_cast<float>(sum) * scale;
        }
      }

      // Softmax
      for (int m = 0; m < heads; ++m) {
        float max_val = -INFINITY;
        for (int n = 0; n < heads; ++n) {
          max_val = std::max(max_val, score[m * heads + n]);
        }
        float sum_exp = 0.0f;
        for (int n = 0; n < heads; ++n) {
          score[m * heads + n] = expf(score[m * heads + n] - max_val);
          sum_exp += score[m * heads + n];
        }
        for (int n = 0; n < heads; ++n) {
          score[m * heads + n] /= sum_exp;
        }
      }

      // 计算输出
      for (int j_dl = 0; j_dl < heads; j_dl++) {
        for (int k_dl = 0; k_dl < dim; k_dl++) {
          int32_t sum = 0;
          // 将浮点数组量化到int8类型
          for (int local_i = 0; local_i < heads; ++local_i) {
            arr_a_16[local_i] = static_cast<int8_t>(
                heads * score[j_dl * heads + local_i]); // 假设量化到[-128,127]
            arr_b_16[local_i] =
                static_cast<int8_t>(V[i * seq_len * heads * dim +
                                      j * heads * dim + local_i * dim + k_dl]);
          }

          // 使用VNNI指令进行乘加操作
          __m128i acc = _mm_setzero_si128(); // 初始化累加器为0

          // 加载量化后的数据到SIMD寄存器中
          __m128i _a =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_a_16));
          __m128i _b =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_b_16));

          // 使用_mm_dpbusds_epi32进行乘加操作 (VNNI)
          acc = _mm_dpbusds_epi32(acc, _a, _b); // 执行乘加操作：acc += a * b

          // 将累加结果存储到arr_d_4中
          _mm_storeu_si128(reinterpret_cast<__m128i *>(arr_d_4), acc);

          // 将arr_d_4中的值累加得到最终的结果
          for (int i_dl = 0; i_dl < 4; ++i_dl) {
            sum += arr_d_4[i_dl];
          }

          // 反量化并存储到输出矩阵output中
          output[i * seq_len * heads * dim + j * heads * dim + j_dl * dim +
                 k_dl] = static_cast<float>(sum) / heads; // 恢复到浮点数
        }
      }
    }
  }
}
