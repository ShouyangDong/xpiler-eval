extern "C" void gemm(float *A, float *B, float *result) {
  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 6; k++) {
      result[j * 6 + k] = 0;
      for (int l = 0; l < 5; l++) {
        result[j * 6 + k] += A[j * 5 + l] * B[l * 6 + k];
      }
    }
  }
}