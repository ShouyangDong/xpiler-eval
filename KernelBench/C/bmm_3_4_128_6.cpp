extern "C" void bmm(float *A, float *B, float *result) {
  uint8_t arr_a[64];
  uint8_t arr_b[64];
  uint32_t arr_d[16];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 6; k++) {
        uint32_t sum = 0;
        // 使用VNNI指令进行乘加操作
        __m512i acc = _mm512_setzero_si512(); // 初始化累加器为0
        // 遍历128个元素，每次处理64个，以适应AVX512
        for (int local_s = 0; local_s < 2; local_s++) {
          // 将浮点数组A和B量化到int8类型
          for (int local_i = 0; local_i < 64; local_i++) {
            arr_a[local_i] = static_cast<uint8_t>(
                A[i * 4 * 128 + j * 128 + local_s * 64 + local_i]);
            arr_b[local_i] = static_cast<uint8_t>(
                B[i * 128 * 6 + (local_s * 64 + local_i) * 6 + k]);
          }

          // 加载量化后的数据到512位SIMD寄存器中
          __m512i _a =
              _mm512_loadu_si512(reinterpret_cast<const void *>(arr_a));
          __m512i _b =
              _mm512_loadu_si512(reinterpret_cast<const void *>(arr_b));

          // 使用_mm512_dpbusd_epi32进行乘加操作 (AVX512 VNNI)
          acc = _mm512_dpbusd_epi32(acc, _a, _b); // 执行乘加操作：acc += a * b
        }

        // 将累加结果存储到arr_d中
        _mm512_storeu_si512(reinterpret_cast<void *>(arr_d), acc);

        // 将arr_d中的值累加得到最终的结果
        for (int i = 0; i < 16; ++i) {
          sum += arr_d[i];
        }

        // 反量化并存储到输出矩阵result中
        result[i * 4 * 6 + j * 6 + k] = static_cast<float>(sum);
      }
    }
  }
}