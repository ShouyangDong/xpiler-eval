extern "C" void max(float *input, float *output) {
  float max_val = -INFINITY;

  for (int i = 0; i < 128; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  output[0] = max_val;
}
