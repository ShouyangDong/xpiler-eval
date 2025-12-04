
float reluf(float input) { return input > 0 ? input : 0; }
extern "C" void relu(float *input, float *output) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 12; j++) {
      for (int k = 0; k < 23; k++) {
        for (int l = 0; l < 128; l++) {
          output[i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l] =
              reluf(input[i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l]);
        }
      }
    }
  }
}