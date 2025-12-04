
float reluf(float input) { return input > 0 ? input : 0; }
extern "C" void relu(float *input, float *output) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 5; k++) {
        output[i * 4 * 5 + j * 5 + k] = reluf(input[i * 4 * 5 + j * 5 + k]);
      }
    }
  }
}