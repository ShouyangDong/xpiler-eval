
float reluf(float input) { return input > 0 ? input : 0; }
extern "C" void relu(float *input, float *output) {
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 6; k++) {
        for (int l = 0; l < 7; l++) {
          output[i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l] =
              reluf(input[i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l]);
        }
      }
    }
  }
}