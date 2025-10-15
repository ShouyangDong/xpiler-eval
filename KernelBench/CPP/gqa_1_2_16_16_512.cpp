extern "C" void gqa(
    const int16_t* Q,
    const int16_t* K,
    const int16_t* V,
    int32_t* O)
{
  const int B = 1;
  const int H = 2;
  const int Sq = 16;
  const int Skv = 512;
  const int D = 16;
  const int W_SCALE = 100; // quantize scale

  alignas(64) int32_t score_i32[Skv];
  alignas(64) float score_f[Skv];
  alignas(64) int16_t weight_q[Skv];
  alignas(64) int16_t kcol[D];
  alignas(64) int32_t tmp32[16];
  alignas(64) int16_t vtmp[32];

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const int16_t* Q_base = Q + ((b * H + h) * Sq * D);
      const int16_t* K_base = K + ((b * H + h) * D * Skv);
      const int16_t* V_base = V + ((b * H + h) * Skv * D);
      int32_t* O_base = O + ((b * H + h) * Sq * D);

      for (int qi = 0; qi < Sq; ++qi) {
        const int16_t* q_ptr = Q_base + qi * D;

        // Step 1: Compute QK^T
        for (int j = 0; j < Skv; ++j) {
          for (int d = 0; d < D; ++d)
            kcol[d] = K_base[d * Skv + j];

          __m512i acc = _mm512_setzero_si512();
          __m512i a_vec = _mm512_loadu_si512((const __m512i*)q_ptr);
          __m512i b_vec = _mm512_loadu_si512((const __m512i*)kcol);
          acc = _mm512_dpwssds_epi32(acc, a_vec, b_vec);
          _mm512_storeu_si512((__m512i*)tmp32, acc);

          int64_t sum64 = 0;
          for (int t = 0; t < 16; ++t) sum64 += tmp32[t];
          score_i32[j] = (int32_t)sum64;
        }

        // Step 2: Softmax (float)
        float maxv = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < Skv; ++j) {
          score_f[j] = (float)score_i32[j];
          if (score_f[j] > maxv) maxv = score_f[j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < Skv; ++j) {
          score_f[j] = std::exp(score_f[j] - maxv);
          sum_exp += score_f[j];
        }
        for (int j = 0; j < Skv; ++j) score_f[j] /= sum_exp;

        // Step 3: Quantize softmax weights
        for (int j = 0; j < Skv; ++j) {
          int32_t wq = (int32_t)std::lround(score_f[j] * (float)W_SCALE);
          if (wq > 32767) wq = 32767;
          if (wq < -32768) wq = -32768;
          weight_q[j] = (int16_t)wq;
        }

        // Step 4: Compute weighted sum S @ V
        for (int d = 0; d < D; ++d) {
          __m512i acc = _mm512_setzero_si512();

          for (int j0 = 0; j0 < Skv; j0 += 32) {
            __m512i wvec = _mm512_loadu_si512((const __m512i*)(weight_q + j0));
            for (int t = 0; t < 32; ++t)
              vtmp[t] = V_base[(j0 + t) * D + d];
            __m512i vvec = _mm512_loadu_si512((const __m512i*)vtmp);
            acc = _mm512_dpwssds_epi32(acc, wvec, vvec);
          }

          _mm512_storeu_si512((__m512i*)tmp32, acc);
          int64_t sum64 = 0;
          for (int t = 0; t < 16; ++t) sum64 += tmp32[t];
          int32_t out = (int32_t)((sum64 + W_SCALE / 2) / W_SCALE);
          O_base[qi * D + d] = out;
        }
      }
    }
  }
}
