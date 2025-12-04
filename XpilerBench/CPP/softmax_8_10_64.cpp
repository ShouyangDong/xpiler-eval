extern "C" void softmax(float *input, float *output) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 10; j++) {

      float max_val = input[i * 10 * 64 + j * 64];
      for (int k = 1; k < 64; k++) {
        int index = i * 10 * 64 + j * 64 + k;
        if (input[index] > max_val) {
          max_val = input[index];
        }
      }

      float sum = 0.0;
      for (int k = 0; k < 64; k++) {
        int index = i * 10 * 64 + j * 64 + k;
        input[index] = expf(input[index] - max_val);
        sum += input[index];
      }

      for (int k = 0; k < 64; k++) {
        int index = i * 10 * 64 + j * 64 + k;
        output[index] = input[index] / sum;
      }
    }
  }
}
