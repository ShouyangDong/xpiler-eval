extern "C" void softmax(float *input, float *output) {
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 3; j++) {

      float max_val = input[i * 3 * 128 + j * 128];
      for (int k = 1; k < 128; k++) {
        int index = i * 3 * 128 + j * 128 + k;
        if (input[index] > max_val) {
          max_val = input[index];
        }
      }

      float sum = 0.0;
      for (int k = 0; k < 128; k++) {
        int index = i * 3 * 128 + j * 128 + k;
        input[index] = expf(input[index] - max_val);
        sum += input[index];
      }

      for (int k = 0; k < 128; k++) {
        int index = i * 3 * 128 + j * 128 + k;
        output[index] = input[index] / sum;
      }
    }
  }
}
