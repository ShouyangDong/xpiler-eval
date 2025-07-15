extern "C" void bmm(float *A, float *B, float *result) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 128; k++) {
        result[i * 4 * 128 + j * 128 + k] = 0;
        for (int l = 0; l < 5; l++) {
          result[i * 4 * 128 + j * 128 + k] +=
              A[i * 4 * 5 + j * 5 + l] * B[i * 5 * 128 + l * 128 + k];
        }
      }
    }
  }
}