extern "C" void gemm(float *A, float *B, float *result) {
  for (int j = 0; j < 32; j++) {
    for (int k = 0; k < 6; k++) {
      result[j * 6 + k] = 0;
      for (int l = 0; l < 128; l++) {
        result[j * 6 + k] += A[j * 128 + l] * B[l * 6 + k];
      }
    }
  }
}
