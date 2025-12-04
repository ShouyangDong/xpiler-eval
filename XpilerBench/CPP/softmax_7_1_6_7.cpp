extern "C" void softmax(float *input, float *output) {
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 6; k++) {

        float max_val = input[i * 1 * 6 * 7 + j * 6 * 7 + k * 7];
        for (int l = 1; l < 7; l++) {
          int index = i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l;
          if (input[index] > max_val) {
            max_val = input[index];
          }
        }

        float sum = 0.0;
        for (int l = 0; l < 7; l++) {
          int index = i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l;
          input[index] = expf(input[index] - max_val);
          sum += input[index];
        }

        for (int l = 0; l < 7; l++) {
          int index = i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l;
          output[index] = input[index] / sum;
        }
      }
    }
  }
}
