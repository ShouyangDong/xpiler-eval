extern "C" void bmm(float *A, float *B, float *result) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 128; j++) {
      for (int k = 0; k < 128; k++) {
        for (int l = 0; l < 128; l++) {
          result[i * 128 * 128 + j * 128 + k] +=
              A[i * 128 * 128 + j * 128 + l] * B[i * 128 * 128 + l * 128 + k];
        }
      }
    }
  }
}
