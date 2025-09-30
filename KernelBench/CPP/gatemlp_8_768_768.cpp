float silu(float x) { return x / (1.0f + expf(-x)); }

extern "C" void gatemlp(int16_t *A, int16_t *B, int16_t *C, float *D) {
  int M = 8;
  int K = 768;
  int N = 768;
  int16_t b_tmp[32];
  int16_t c_tmp[32];
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      __m512i abcc = _mm512_setzero_si512();
      __m512i accc = _mm512_setzero_si512();
      for (int k = 0; k < K; k += 32) {
        __m512i a_vec = _mm512_loadu_si512((__m512i *)&A[i * K + k]);
        for (int t = 0; t < 32; ++t)
          b_tmp[t] = B[(k + t) * N + j]; // column-major
        for (int tc = 0; tc < 32; ++tc)
          c_tmp[tc] = C[(k + tc) * N + j]; // column-major
        __m512i b_vec = _mm512_load_si512((__m512i *)b_tmp);
        __m512i c_vec = _mm512_load_si512((__m512i *)c_tmp);
        abcc = _mm512_dpwssds_epi32(abcc, a_vec, b_vec);
        accc = _mm512_dpwssds_epi32(accc, a_vec, c_vec);
      }

      // 水平加法 reduce 32个int32 -> 1个int32
      int32_t tmp_b[16];
      int32_t tmp_c[16];
      _mm512_store_si512((__m512i *)tmp_b, abcc);
      _mm512_store_si512((__m512i *)tmp_c, accc);
      float sum_b = 0.0f, sum_c = 0.0f;
      for (int t = 0; t < 16; ++t) {
        sum_b += tmp_b[t];
        sum_c += tmp_c[t];
      }
      float gate = silu(sum_b);
      D[i * N + j] = gate * sum_c;
    }
  }
}
