extern "C" void softmax(float *input, float *output) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 7; j++) {
      for (int k = 0; k < 3; k++) {

        float max_val = input[i * 7 * 3 * 32 + j * 3 * 32 + k * 32];
        for (int l = 1; l < 32; l++) {
          int index = i * 7 * 3 * 32 + j * 3 * 32 + k * 32 + l;
          if (input[index] > max_val) {
            max_val = input[index];
          }
        }

        float sum = 0.0;
        for (int l = 0; l < 32; l++) {
          int index = i * 7 * 3 * 32 + j * 3 * 32 + k * 32 + l;
          input[index] = expf(input[index] - max_val);
          sum += input[index];
        }

        for (int l = 0; l < 32; l++) {
          int index = i * 7 * 3 * 32 + j * 3 * 32 + k * 32 + l;
          output[index] = input[index] / sum;
        }
      }
    }
  }
}
