extern "C" void gemv(float *A, float *x, float *y) {
  for (int i = 0; i < 112; i++) {
    y[i] = 0;
    for (int j = 0; j < 224; j++) {
      y[i] += A[i * 224 + j] * x[j];
    }
  }
}