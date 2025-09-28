extern "C" void bmm(float *A, float *B, float *result) {
  uint8_t arr_a[64];
  uint8_t arr_b[64];
  uint32_t arr_d[16];

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 128; k++) {
        uint32_t sum = 0;

        __m512i acc = _mm512_setzero_si512();

        for (int local_s = 0; local_s < 2; local_s++) {

          for (int local_i = 0; local_i < 64; local_i++) {
            arr_a[local_i] = static_cast<uint8_t>(
                A[i * 4 * 128 + j * 128 + local_s * 64 + local_i]);
            arr_b[local_i] = static_cast<uint8_t>(
                B[i * 128 * 128 + (local_s * 64 + local_i) * 128 + k]);
          }

          __m512i _a =
              _mm512_loadu_si512(reinterpret_cast<const void *>(arr_a));
          __m512i _b =
              _mm512_loadu_si512(reinterpret_cast<const void *>(arr_b));

          acc = _mm512_dpbusd_epi32(acc, _a, _b);
        }

        _mm512_storeu_si512(reinterpret_cast<void *>(arr_d), acc);

        for (int i = 0; i < 16; ++i) {
          sum += arr_d[i];
        }

        result[i * 4 * 128 + j * 128 + k] = static_cast<float>(sum);
      }
    }
  }
}