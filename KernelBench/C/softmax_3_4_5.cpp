extern "C" void softmax(float *input, float *output) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {

      float max_val = input[i * 4 * 5 + j * 5];
      for (int k = 1; k < 5; k++) {
        int index = i * 4 * 5 + j * 5 + k;
        if (input[index] > max_val) {
          max_val = input[index];
        }
      }

      float sum = 0.0;
      for (int k = 0; k < 5; k++) {
        int index = i * 4 * 5 + j * 5 + k;
        input[index] = expf(input[index] - max_val);
        sum += input[index];
      }

      for (int k = 0; k < 5; k++) {
        int index = i * 4 * 5 + j * 5 + k;
        output[index] = input[index] / sum;
      }
    }
  }
}
