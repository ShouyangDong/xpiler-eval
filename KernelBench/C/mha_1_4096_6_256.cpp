extern "C" void mha(float *Q, float *K, float *V, float *output) {
  int8_t arr_a[64];
  int8_t arr_b[64];
  int32_t arr_d[16]; // AVX-512 寄存器能同时处理 16 个 int32 元素
  float score[6 * 6];

  const int batch = 1;
  const int seq_len = 4096;
  const int heads = 6;
  const int dim = 256;
  const float scale = 1.0f / sqrt(dim);

  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < seq_len; j++) {
      for (int m = 0; m < heads; m++) {
        for (int n = 0; n < heads; n++) {
          int32_t sum = 0;

          for (int local_s = 0; local_s < dim / 64;
               local_s++) { // 每次处理 64 个元素
            for (int local_i = 0; local_i < 64; local_i++) {
              arr_a[local_i] = static_cast<int8_t>(
                  Q[i * seq_len * heads * dim + j * heads * dim + m * dim +
                    local_s * 64 + local_i]);
              arr_b[local_i] = static_cast<int8_t>(
                  K[i * seq_len * heads * dim + j * heads * dim + n * dim +
                    local_s * 64 + local_i]);
            }

            __m512i acc = _mm512_setzero_si512();

            __m512i _a =
                _mm512_loadu_si512(reinterpret_cast<const __m512i *>(arr_a));
            __m512i _b =
                _mm512_loadu_si512(reinterpret_cast<const __m512i *>(arr_b));

            acc = _mm512_dpbusd_epi32(acc, _a, _b);

            _mm512_storeu_si512(reinterpret_cast<__m512i *>(arr_d), acc);

            for (int k = 0; k < 16; ++k) { // 处理 16 个累积结果
              sum += arr_d[k];
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

      // Final Matmul
      for (int m = 0; m < heads; ++m) {
        for (int n = 0; n < dim; ++n) {
          output[i * seq_len * heads * dim + j * heads * dim + m * dim + n] =
              0.0f;
          for (int k = 0; k < heads; ++k) {
            output[i * seq_len * heads * dim + j * heads * dim + m * dim + n] +=
                score[m * heads + k] *
                V[i * seq_len * heads * dim + j * heads * dim + k * dim + n];
          }
        }
      }
    }
  }
}
