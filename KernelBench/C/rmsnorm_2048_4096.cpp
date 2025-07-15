

extern "C" void rmsnorm(float *A, float *B) {
  float eps = 1e-5f;

  for (int i = 0; i < 2048; i++) {
    float mean = 0.0;
    for (int j = 0; j < 4096; j++) {
      mean += A[i * 4096 + j] * A[i * 4096 + j];
    }

    mean /= 4096;
    float scale = 1.0 / sqrt(mean + eps);

    for (int k = 0; k < 4096; k++)
      B[i * 4096 + k] = A[i * 4096 + k] * scale;
  }
}