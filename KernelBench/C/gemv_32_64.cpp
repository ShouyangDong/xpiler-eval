extern "C" void gemv(float *A, float *x, float *y) {
  uint8_t arr_a[64];
  uint8_t arr_b[64];
  uint32_t arr_d[16];

  for (int i = 0; i < 32; ++i) {
    uint32_t sum = 0;
    __m512i acc = _mm512_setzero_si512(); // 累加器初始化为0

    // 加载A和x到arr_a和arr_b
    for (int j = 0; j < 64; ++j) {
      arr_a[j] = A[i * 64 + j];
      arr_b[j] = x[j];
    }

    // 使用VNNI指令进行乘加操作
    __m512i _a = _mm512_loadu_si512(arr_a);
    __m512i _b = _mm512_loadu_si512(arr_b);
    acc = _mm512_dpbusd_epi32(acc, _a, _b);

    // 将累加结果存储到arr_d中
    _mm512_storeu_si512(arr_d, acc);

    // 将arr_d中的值累加得到最终的结果
    for (int k = 0; k < 16; ++k) {
      sum += arr_d[k];
    }

    y[i] = static_cast<float>(sum);
  }
}