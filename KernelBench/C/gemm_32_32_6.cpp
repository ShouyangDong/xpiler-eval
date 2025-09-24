extern "C" void gemm(float *A, float *B, float *result) {
  int8_t arr_a[16];
  int8_t arr_b[16];
  int32_t arr_d[4];

  for (int j = 0; j < 32; j++) {
    for (int k = 0; k < 6; k++) {
      int32_t sum = 0;

      for (int local_s = 0; local_s < 2; local_s++) {

        for (int local_i = 0; local_i < 16; local_i++) {
          arr_a[local_i] =
              static_cast<int8_t>(A[j * 32 + local_s * 16 + local_i]);
          arr_b[local_i] =
              static_cast<int8_t>(B[(local_s * 16 + local_i) * 6 + k]);
        }

        __m128i acc = _mm_setzero_si128();

        __m128i _a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_a));
        __m128i _b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_b));

        acc = _mm_dpbusds_epi32(acc, _a, _b);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(arr_d), acc);

        for (int i = 0; i < 4; ++i) {
          sum += arr_d[i];
        }
      }

      result[j * 6 + k] = static_cast<float>(sum);
    }
  }
}