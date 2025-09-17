extern "C" void gemv(float *A, float *x, float *y) {
  int8_t arr_a[16];
  int8_t arr_b[16];
  int32_t arr_d[4];

  for (int i = 0; i < 112; ++i) {
    int32_t sum = 0;
    // 使用VNNI指令进行乘加操作
    __m128i acc = _mm_setzero_si128(); // 初始化累加器为0
    for (int local_s = 0; local_s < 14; ++local_s) {
      // 将浮点数组A和x量化到int8类型
      for (int j = 0; j < 16; ++j) {
        arr_a[j] = A[i * 224 + local_s * 16 + j];
        arr_b[j] = x[local_s * 16 + j];
      }
      // 加载量化后的数据到SIMD寄存器中
      __m128i _a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_a));
      __m128i _b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_b));

      // 使用_mm_dpbusds_epi32进行乘加操作 (VNNI)
      acc = _mm_dpbusds_epi32(acc, _a, _b); // 执行乘加操作：acc += a * b
    }
    // 将累加结果存储到arr_d中
    _mm_storeu_si128(reinterpret_cast<__m128i *>(arr_d), acc);

    // 将arr_d中的值累加得到最终的结果
    for (int k = 0; k < 4; ++k) {
      sum += arr_d[k];
    }

    // 反量化并存储到输出向量y中
    y[i] = static_cast<float>(sum);
  }
}
