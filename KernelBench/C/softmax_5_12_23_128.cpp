extern "C" void softmax(float *input, float *output) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 12; j++) {
      for (int k = 0; k < 23; k++) {

        float max_val = input[i * 12 * 23 * 128 + j * 23 * 128 + k * 128];
        for (int l = 1; l < 128; l++) {
          int index = i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l;
          if (input[index] > max_val) {
            max_val = input[index];
          }
        }

        float sum = 0.0;
        for (int l = 0; l < 128; l++) {
          int index = i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l;
          input[index] = expf(input[index] - max_val);
          sum += input[index];
        }

        for (int l = 0; l < 128; l++) {
          int index = i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l;
          output[index] = input[index] / sum;
        }
      }
    }
  }
}
