extern "C" void gemv(float *A, float *x, float *y) {
  uint8_t arr_a[64];
  uint8_t arr_b[64];
  uint32_t arr_d[16];

  for (int i = 0; i < 112; ++i) {
    uint32_t sum = 0;
    __m512i acc = _mm512_setzero_si512();

    for (int local_s = 0; local_s < 2; ++local_s) {

      for (int j = 0; j < 64; ++j) {
        arr_a[j] = A[i * 128 + local_s * 64 + j];
        arr_b[j] = x[local_s * 64 + j];
      }

      __m512i _a = _mm512_loadu_si512(arr_a);
      __m512i _b = _mm512_loadu_si512(arr_b);
      acc = _mm512_dpbusd_epi32(acc, _a, _b);
    }

    _mm512_storeu_si512(arr_d, acc);

    for (int k = 0; k < 16; ++k) {
      sum += arr_d[k];
    }

    y[i] = static_cast<float>(sum);
  }
}
