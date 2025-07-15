extern "C" void bmm(float *A, float *B, float *result) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 32; j++) {
      for (int k = 0; k < 6; k++) {
        result[i * 32 * 6 + j * 6 + k] = 0;
        for (int l = 0; l < 5; l++) {
          result[i * 32 * 6 + j * 6 + k] +=
              A[i * 32 * 5 + j * 5 + l] * B[i * 5 * 6 + l * 6 + k];
        }
      }
    }
  }
}