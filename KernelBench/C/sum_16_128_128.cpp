extern "C" void sum(float *input, float *output) {
  int d0 = 16;
  int d1 = 128;
  int d2 = 128;

  int output_size = d1 * d2;

  for (int i = 0; i < output_size; i++) {
    output[i] = 0.0f;
  }

  for (int k = 0; k < d1; k++) {
    for (int l = 0; l < d2; l++) {
      for (int i = 0; i < d0; i++) {

        output[k * d2 + l] += input[i * d1 * d2 + k * d2 + l];
      }
    }
  }
}
