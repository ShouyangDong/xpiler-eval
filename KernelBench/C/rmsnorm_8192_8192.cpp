

extern "C" void rmsnorm(float *A, float *B) {
  float eps = 1e-5f;

  for (int i = 0; i < 8192; i++) {
    float mean = 0.0;
    for (int j = 0; j < 8192; j++) {
      mean += A[i * 8192 + j] * A[i * 8192 + j];
    }

    mean /= 8192;
    float scale = 1.0 / sqrt(mean + eps);

    for (int k = 0; k < 8192; k++)
      B[i * 8192 + k] = A[i * 8192 + k] * scale;
  }
}