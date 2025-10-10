extern "C" void dense(int16_t* A, int16_t* B,
                      int32_t* bias, int32_t* C) {
    int M = 32;
    int K = 768;
    int N = 768;
    int16_t b_tmp[32];
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m512i acc = _mm512_setzero_si512();

            for (int k = 0; k < K; k += 32) {
                __m512i a_vec = _mm512_loadu_si512((__m512i*)&A[i*K + k]);
                for (int t = 0; t < 32; ++t)
                    b_tmp[t] = B[(k + t)*N + j]; // column-major
                __m512i b_vec = _mm512_load_si512((__m512i*)b_tmp);
                acc = _mm512_dpwssds_epi32(acc, a_vec, b_vec);
            }

            // 水平加法 reduce 32个int32 -> 1个int32
            int32_t tmp[16];
            _mm512_store_si512((__m512i*)tmp, acc);
            int32_t sum = 0;
            for (int t = 0; t < 16; ++t) sum += tmp[t];

            C[i*N + j] = sum + bias[j];
        }
    }
}
