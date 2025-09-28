

extern "C" void rmsnorm(float *A, float *B) {
  float eps = 1e-5f;

  for (int i = 0; i < 4096; i++) {
    float mean = 0.0;
    for (int j = 0; j < 2048; j++) {
      mean += A[i * 2048 + j] * A[i * 2048 + j];
    }

    mean /= 2048;
    float scale = 1.0 / sqrt(mean + eps);

    for (int k = 0; k < 2048; k++)
      B[i * 2048 + k] = A[i * 2048 + k] * scale;
  }
}