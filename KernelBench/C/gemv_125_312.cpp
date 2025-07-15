extern "C" void gemv(float *A, float *x, float *y) {
  for (int i = 0; i < 125; i++) {
    y[i] = 0;
    for (int j = 0; j < 312; j++) {
      y[i] += A[i * 312 + j] * x[j];
    }
  }
}