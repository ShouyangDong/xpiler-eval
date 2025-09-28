
float reluf(float input) { return input > 0 ? input : 0; }
extern "C" void relu(float *input, float *output) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < 64; k++) {
        output[i * 10 * 64 + j * 64 + k] =
            reluf(input[i * 10 * 64 + j * 64 + k]);
      }
    }
  }
}