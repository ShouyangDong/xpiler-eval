extern "C" void gemv(float *A, float *x, float *y) {
  int8_t arr_a[16];
  int8_t arr_b[16];
  int32_t arr_d[4];

  for (int i = 0; i < 112; ++i) {
    int32_t sum = 0;

    __m128i acc = _mm_setzero_si128();
    for (int local_s = 0; local_s < 14; ++local_s) {

      for (int j = 0; j < 16; ++j) {
        arr_a[j] = A[i * 224 + local_s * 16 + j];
        arr_b[j] = x[local_s * 16 + j];
      }

      __m128i _a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_a));
      __m128i _b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_b));

      acc = _mm_dpbusds_epi32(acc, _a, _b);
    }

    _mm_storeu_si128(reinterpret_cast<__m128i *>(arr_d), acc);

    for (int k = 0; k < 4; ++k) {
      sum += arr_d[k];
    }

    y[i] = static_cast<float>(sum);
  }
}
