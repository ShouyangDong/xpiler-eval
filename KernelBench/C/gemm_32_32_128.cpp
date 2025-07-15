extern "C" void gemm(float *A, float *B, float *result) {
  for (int j = 0; j < 32; j++) {
    for (int k = 0; k < 128; k++) {
      result[j * 128 + k] = 0;
      for (int l = 0; l < 32; l++) {
        result[j * 128 + k] += A[j * 32 + l] * B[l * 128 + k];
      }
    }
  }
}
